#!/usr/bin/env python3
"""Entry point: wires the live transports, intent/policy modules, and
SystemArbitrator together per configs/system.yaml, then runs the
orchestrator loop against the sim (or hardware speaking the same protocol).

Usage: python run.py [path/to/system.yaml]
       python run.py [path/to/system.yaml] --trials 10 --timeout 90 [--name v4]
           evaluation: N autonomous trials of --timeout seconds each, arms homed
           and a new scene loaded between them (teleop_orchestrator/evaluation.py).
           Score afterwards with scripts/evaluate_policy.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import yaml

from teleop_orchestrator.arbitrator import EVALUATE_MODE, INTERVENTION_MODE, SystemArbitrator
from teleop_orchestrator.evaluation import EvalSession, RoutingLogger
from teleop_orchestrator.intent_factory import build_intent_model
from teleop_orchestrator.live import wire
from teleop_orchestrator.live.arm_channel import ArmChannel
from teleop_orchestrator.live.camera_geometry import load_intrinsics
from teleop_orchestrator.live.gaze_receiver import GazeReceiver
from teleop_orchestrator.live.head_channel import HeadChannel
from teleop_orchestrator.live.live_source import LiveSource
from teleop_orchestrator.live.object_source import SimObjectSource
from teleop_orchestrator.live.shared_frame_reader import SharedFrameReader
from teleop_orchestrator.live.sys_state_client import SysStateClient
from teleop_orchestrator.logging import RunLogger
from teleop_orchestrator.operator_console import OperatorConsole
from teleop_orchestrator.orchestrator import Orchestrator
from teleop_orchestrator.playback_module import PlaybackModule
from teleop_orchestrator.policy_module import PolicyModule


def load_config(path: str) -> dict:
    """Loads and returns system.yaml as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_live_channels(net: dict, host: str):
    """Constructs and starts every UDP/SHM live channel from system.yaml's network block."""
    sys_state = SysStateClient(remote_ip=host, send_port=net["avatar"]["send_port"],
                                receive_port=net["avatar"]["receive_port"])
    arm_left = ArmChannel("arm_left", wire.DeviceId.LEFT_ARM, host,
                           net["arm_left"]["send_port"], net["arm_left"]["receive_port"],
                           absolute_send_port=net["arm_left"].get("absolute_send_port"))
    arm_right = ArmChannel("arm_right", wire.DeviceId.RIGHT_ARM, host,
                            net["arm_right"]["send_port"], net["arm_right"]["receive_port"],
                            absolute_send_port=net["arm_right"].get("absolute_send_port"))
    head = HeadChannel(host, net["head"]["send_port"], net["head"]["receive_port"],
                        absolute_send_port=net["head"].get("absolute_send_port"))
    gaze = GazeReceiver(receive_port=net["gaze"]["receive_port"])
    objects = SimObjectSource(receive_port=net["scene_objects"]["receive_port"])
    cameras = {name: SharedFrameReader(shm_name) for name, shm_name in net["cameras"].items()}

    # SharedFrameReader has no background thread -- it attaches lazily on
    # first read() (try_open()), unlike the UDP channels below, so it's
    # deliberately excluded from this start() loop.
    for c in (sys_state, arm_left, arm_right, head, gaze, objects):
        c.start()
    return sys_state, arm_left, arm_right, head, gaze, objects, cameras


def _scene_snapshot(objects: SimObjectSource):
    """Most recent SceneObjectsMsg, or None if the feed has not produced one.

    Carries both the avatar's SysState and its per-arm authority. Read from here
    rather than from SysStateClient because the avatar's reliable command channel
    is point-to-point: with the VR interface connected it talks to the interface,
    and this process would see no state at all.
    """
    return objects.latest()


def resolve_mode(cfg: dict) -> str:
    """Resolves the one mode the session runs in, from the policy and playback
    blocks. Exactly one of them may command the arms: they both emit an
    ActionOutput and the actuator can only send one pose per arm per tick, so
    enabling both is a config error rather than something to arbitrate at
    runtime."""
    policy, playback = cfg["modules"]["policy"], cfg["modules"].get("playback", {})
    policy_on = policy["enabled"] and policy["mode"] != "off"
    if playback.get("enabled") and policy_on:
        raise ValueError("modules.policy and modules.playback are both enabled -- "
                         "only one may command the arms; set policy.mode: off or playback.enabled: false")
    return "playback" if playback.get("enabled") else policy["mode"]


def build_modules(cfg: dict, mode: str, arm_left: ArmChannel, arm_right: ArmChannel,
                  head: HeadChannel) -> dict:
    """Constructs the enabled intent/policy/playback modules per system.yaml's modules block."""
    modules = {}
    mcfg = cfg["modules"]
    if mcfg["intent"]["enabled"]:
        modules["intent"] = build_intent_model(mcfg["intent"]["family"], mcfg["intent"]["checkpoint"])
    if mode == "playback":
        pbcfg = mcfg["playback"]
        modules["playback"] = PlaybackModule(pbcfg["episode"], arm_left, arm_right,
                                              speed=pbcfg.get("speed", 1.0),
                                              start_ramp_s=pbcfg.get("start_ramp_s", 2.0),
                                              replay_head=pbcfg.get("replay_head", True))
    elif mcfg["policy"]["enabled"] and mcfg["policy"]["mode"] != "off":
        # PolicyModule reads its required camera names/order from the
        # checkpoint itself (see policy_module.py) -- system.yaml's
        # network.cameras must have matching keys, but nothing here needs
        # to tell it which ones or in what order.
        dbg = mcfg["policy"].get("debug_dump_dir")
        stored = mcfg["policy"].get("training_stored_hw")
        # head is passed for completeness, not because PolicyModule reads it:
        # a head-enabled checkpoint needs the measured neck angles in proprio,
        # and it takes those off SensorFrame, which LiveSource fills from this
        # channel (now that it has its own port via transmission_absolute) and
        # from SceneObjectsMsg if that ever goes quiet. Commanding the head
        # happens in make_actuator's on_tick, off the head_pan/head_tilt extras.
        modules["policy"] = PolicyModule(mcfg["policy"]["checkpoint"], arm_left, arm_right, head,
                                          debug_dump_dir=dbg,
                                          debug_dump_every=mcfg["policy"].get("debug_dump_every", 30),
                                          stored_hw=stored,
                                          resume_ramp_s=mcfg["policy"].get("resume_ramp_s", 0.5),
                                          max_rot_lead_deg=mcfg["policy"].get("max_rot_lead_deg", 30.0),
                                          yaw_clamp=mcfg["policy"].get("yaw_clamp", True),
                                          ensemble_m=mcfg["policy"].get("ensemble_m", 0.01),
                                          gripper_ensemble_m=mcfg["policy"].get("gripper_ensemble_m", 0.01))
    return modules


# teleop-simulator arm_control.cpp: kGripperMaxWidth = 0.08, and gripper_cmd is
# logged as exactly 0.0 (closed) or 0.08 (open) -- a binary signal. 0.04 is the
# midpoint; ACT's L1 output is continuous so it needs thresholding somewhere.
_GRIPPER_OPEN_WIDTH_M = 0.08
_GRIPPER_CLOSE_THRESHOLD_M = 0.04


class GripperLatch:
    """Close flag with hysteresis and a minimum hold, one per arm.

    The ensembled width is an average of ~40 chunk predictions. Near a grasp it
    hovers around the 0.04 m midpoint, so a single threshold flips every tick or
    two: logs 095/096 show 45 closes in two runs, mostly bursts 0.1-0.2 s apart,
    many before the fingers reached grasp depth, each one shoving the parcel.

    Close when the width drops below close_below; re-open only once it rises
    above open_above AND the gripper has been closed for min_hold_s. The human
    demos hold a successful grasp 1.9-4.2 s, so a 1 s hold never cuts a real
    carry short. close_below == open_above with min_hold_s 0 is the old
    single-threshold behaviour.
    """

    def __init__(self, close_below: float = 0.03, open_above: float = 0.06, min_hold_s: float = 1.0):
        self.close_below, self.open_above, self.min_hold_s = close_below, open_above, min_hold_s
        self._closed: dict = {}
        self._t_closed: dict = {}

    def __call__(self, side: str, width: float, now: float) -> float:
        closed = self._closed.get(side, False)
        if not closed and width < self.close_below:
            closed = True
            self._t_closed[side] = now
        elif closed and width > self.open_above and now - self._t_closed.get(side, now) >= self.min_hold_s:
            closed = False
        self._closed[side] = closed
        return 1.0 if closed else 0.0

    def reset(self) -> None:
        """Forget both arms' latched state (a new trial starts with the gripper open)."""
        self._closed.clear()
        self._t_closed.clear()


def agree_from_spread(spread: float, spread_full: float, spread_zero: float) -> float:
    """Maps ensemble_spread onto the HUD's AGREE scale: 1 at or below spread_full,
    0 at or above spread_zero, linear between."""
    return float(min(1.0, max(0.0, (spread_zero - spread) / (spread_zero - spread_full))))


def make_status_sender(sys_state: SysStateClient, status_hz: float, agree_spread: tuple):
    """Returns a per-tick callback that sends policy_status (INF, AGREE) to the
    HUD at status_hz, via the avatar."""
    period = 1.0 / status_hz
    last = [0.0]

    def send(extras: dict) -> None:
        now = time.monotonic()
        if now - last[0] < period or "inference_ms" not in extras:
            return
        last[0] = now
        agree = agree_from_spread(extras.get("ensemble_spread", 0.0), *agree_spread)
        sys_state.send_policy_status(extras["inference_ms"], agree)
    return send


def make_actuator(arbitrator: SystemArbitrator, arm_left: ArmChannel, arm_right: ArmChannel,
                   head: HeadChannel, head_pin_absolute: dict,
                   logger=None, action_key: str = "policy", status_sender=None,
                   gripper_latch: Optional["GripperLatch"] = None,
                   gate=None):
    """Returns the Orchestrator.run(on_tick=...) callback that sends the acting
    module's ActionOutput to the arms -- only when autonomous_allowed, so
    supportive/off mode never issues an ArmCommandMsg regardless of what the
    module predicts. action_key names that module ("policy" or "playback");
    everything below this line is identical for both, which is the point.
    gate, if given, is a zero-argument callable; nothing is sent (arms or head)
    on ticks where it returns False -- the evaluation loop uses it to keep the
    policy off the arms while they are homed between trials.
    Uses send_absolute_command: the retrained checkpoint predicts absolute
    world-frame poses (see teleop-policy/configs/dataset.yaml), which the sim
    only interprets correctly on the absolute channel (worldAbsoluteToBase),
    not send_command's delta-from-origin/VR path.

    Also holds the head at a fixed look-down offset every tick while
    autonomous: ACT was trained on operator-driven head motion (following
    gaze down toward the grasp), but nothing here reproduces that, so
    head_cam_left drifts out-of-distribution over a session unless pinned
    somewhere reasonable. Deterministic/scripted rather than learned --
    see system.yaml's geometry.head_pin_absolute comment for tuning.
    Playback overrides it per tick with the head track it recorded
    (extras head_pan/head_tilt), which is strictly better when available.
    """
    # Last SysState reported by each arm itself, for printing transitions once.
    last_arm_state: dict = {}

    def arm_is_engaged(side: str, channel: ArmChannel) -> bool:
        """True only if the arm's own ArmStateMsg says ENGAGED.

        The arbitrator's authority gate says who MAY command an arm; it does not
        know whether the arm can move. After a FAULT (e.g. a wrist joint limit)
        the arm is frozen, but the policy kept predicting and this kept sending,
        which looked like the policy hesitating over a parcel (runs
        20260924_191355/191451). Commands to a non-ENGAGED arm do nothing, so
        stop sending them and say so, once per transition.
        """
        st = channel.latest()
        state = None if st is None else int(st.state)
        prev = last_arm_state.get(side, wire.SysState.ENGAGED)
        if state != prev:
            name = "no state" if state is None else wire.SysState.NAMES.get(state, str(state))
            if state == wire.SysState.ENGAGED:
                print(f"[run] {side} is ENGAGED again; resuming commands")
            else:
                detail = ""
                if st is not None and state == wire.SysState.FAULT:
                    detail = (f" fault_code={st.fault_code} q=" +
                              " ".join(f"{q:+.3f}" for q in st.joints))
                print(f"[run] {side} is {name.upper()}, not commanding it{detail}")
            last_arm_state[side] = state
        return state == wire.SysState.ENGAGED

    def on_tick(frame, outputs) -> None:
        action = outputs.get(action_key)
        if action is None:
            return
        extras = action.extras or {}
        # Before the authority check: the HUD keeps showing inference time and
        # agreement while the operator holds the robot, which is when AGREE matters.
        if status_sender is not None:
            status_sender(extras)
        if not action.ee_pose or not arbitrator.autonomous_allowed:
            return
        if gate is not None and not gate():
            return

        # The head is pinned only while this process is actually driving
        # something. It is NOT in the policy's action space -- it is pinned so
        # head_cam_left stays where the training data had it -- but it IS an
        # observation, so it cannot simply follow the operator either.
        #
        # The resolution: whoever holds the arms holds the head. While the
        # policy drives, the head stays pinned and the policy's view stays in
        # distribution. The moment the operator has taken every arm, this stops
        # sending and their head is their own again -- which is exactly when
        # they need to look around, because they are fixing something.
        #
        # There is still no authority ON the head itself -- both processes send
        # to the same port and the last writer wins -- so "stop sending" remains
        # the whole handover mechanism. What changed on 2026-09-22 is that the
        # other side now plays by the same rule: the interface gates
        # SendHeadCommand on the operator holding authority
        # (OperatorPawn::IsOperatorHoldingHead), so the two processes are never
        # both writing. Before that, the operator's HMD was writing over this
        # pin at 90 Hz whenever their headset moved, which is why the pin
        # "worked" only while nobody was wearing it.
        #
        # extras head_pan/head_tilt are no longer just playback's recorded
        # track: the retrained policy emits head as real action dims, and this
        # forwards them unchanged. head_pin_absolute stays as the fallback
        # for a checkpoint that does not predict head -- a pin is still better
        # than a head that drifts, and the dict lookup is what tells the two
        # cases apart without a config flag.
        held = [s for s in ("arm_left", "arm_right")
                if s in action.ee_pose and arbitrator.arm_allowed(frame, s)]
        if held:
            # One frame everywhere in this process: ABSOLUTE joint targets, on
            # the head's absolute channel. The policy predicts absolute because
            # head.csv logs absolute, and head_pin_absolute is configured in the
            # same frame, so nothing here converts anything.
            #
            # The home-relative channel still exists and still belongs to the VR
            # interface, which has a captured origin to be relative to. This
            # process simply never speaks that frame -- which is the whole point
            # of adding a second channel instead of converting in the sender.
            pan, tilt = extras.get("head_pan"), extras.get("head_tilt")
            if pan is None or tilt is None:
                pan, tilt = head_pin_absolute["pan"], head_pin_absolute["tilt"]
            head.send_absolute_command(wire.SysState.ENGAGED, pan, tilt)

        for side, channel in (("arm_left", arm_left), ("arm_right", arm_right)):
            if side not in action.ee_pose:
                continue
            # Per arm, inside the loop rather than once above it: during an
            # intervention the operator holds one arm while the policy keeps
            # driving the other, so one check for the whole robot would either
            # freeze the arm the policy still owns or keep commanding the one a
            # hand is already on. The head deliberately stays pinned either way
            # -- the wrist view of the arm still running has to stay in
            # distribution while the other is being corrected.
            if not arbitrator.arm_allowed(frame, side):
                continue
            if not arm_is_engaged(side, channel):
                continue
            pos = action.ee_pose[side][:3]
            quat = action.ee_pose[side][3:7]
            # ArmCommandMsg.gripper is a BOOLEAN CLOSE FLAG on the sim side, not a
            # width: arm_control.cpp does `desired_gripper_closed_ = cmd.gripper > 0.5f`
            # for both the VR and the absolute channel. PolicyModule outputs a
            # gripper WIDTH in metres (action dim 9, 0.0 closed .. 0.08 open --
            # data_logger.hpp writes gripper_cmd = closed ? 0.0 : 0.08, so that is
            # what ACT was trained to reproduce).
            #
            # Feeding the width straight through meant the flag was `0.08 > 0.5`
            # -> false on every single tick, so the gripper could never close in
            # autonomous mode no matter what the policy predicted. Note the two
            # conventions are also INVERTED (small width = closed, large flag =
            # close), so this is a threshold-and-invert, not a rescale.
            width = action.gripper.get(side, _GRIPPER_OPEN_WIDTH_M)
            if gripper_latch is not None:
                close_flag = gripper_latch(side, width, time.monotonic())
            else:
                close_flag = 1.0 if width < _GRIPPER_CLOSE_THRESHOLD_M else 0.0
            channel.send_absolute_command(wire.SysState.ENGAGED, pos, quat, close_flag)
            if logger is not None:
                logger.log_gripper(side, width=width, close_flag=close_flag)
                logger.log_arm_state(side, channel.latest())
    return on_tick


def build_evaluation(cfg: dict, args, config_path: str, sys_state, arm_left, arm_right,
                     run_logger, policy, latch) -> EvalSession:
    """EvalSession from system.yaml's modules.policy.evaluation block, with
    --trials/--timeout/--name/--out taking precedence."""
    pcfg = cfg["modules"]["policy"]
    ecfg = dict(pcfg.get("evaluation") or {})
    for key, val in (("trials", args.trials), ("timeout_s", args.timeout),
                     ("name", args.name), ("out_dir", args.out)):
        if val is not None:
            ecfg[key] = val
    ckpt = pcfg["checkpoint"]
    # Default name: the checkpoint's folder, e.g. act_sorting_relpos_v4.
    name = ecfg.get("name") or os.path.basename(os.path.dirname(os.path.normpath(ckpt))) or "eval"
    run_id = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(ecfg.get("out_dir", "logs/eval"), run_id)
    policy_snapshot = {k: v for k, v in pcfg.items() if k not in ("evaluation",)}

    def on_trial_start() -> None:
        policy.reset()
        if latch is not None:
            latch.reset()

    return EvalSession(
        n_trials=int(ecfg.get("trials", 10)), timeout_s=float(ecfg.get("timeout_s", 90.0)),
        sys_state=sys_state, arms={"arm_left": arm_left, "arm_right": arm_right},
        logger=run_logger, out_dir=out_dir, run_id=run_id, on_trial_start=on_trial_start,
        settle_s=float(ecfg.get("settle_s", 1.5)),
        reset_timeout_s=float(ecfg.get("reset_timeout_s", 45.0)),
        meta={"checkpoint": ckpt, "config_path": config_path,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "policy": policy_snapshot})


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Run the orchestrator against the sim.")
    ap.add_argument("config", nargs="?", default="configs/system.yaml")
    ap.add_argument("--trials", type=int, default=None,
                    help="evaluation: number of trials (switches policy.mode to 'evaluate')")
    ap.add_argument("--timeout", type=float, default=None,
                    help="evaluation: length of each trial in seconds")
    ap.add_argument("--name", default=None,
                    help="evaluation: run name prefix (default: the checkpoint's folder name)")
    ap.add_argument("--out", default=None, help="evaluation: parent folder (default logs/eval)")
    return ap.parse_args(argv)


def main(args=None) -> None:
    args = args if args is not None else parse_args([])
    config_path = args.config
    cfg = load_config(config_path)
    if args.trials is not None or args.timeout is not None:
        cfg["modules"]["policy"]["mode"] = EVALUATE_MODE
    net, geo = cfg["network"], cfg["geometry"]
    host = net["host"]

    sys_state, arm_left, arm_right, head, gaze, objects, cameras = build_live_channels(net, host)
    threaded_channels = (sys_state, arm_left, arm_right, head, gaze, objects)

    def shutdown() -> None:
        """Stops the threaded UDP channels and closes the SHM camera readers."""
        for c in threaded_channels:
            c.stop()
        for c in cameras.values():
            c.close()

    gaze_intrinsics = None
    try:
        gaze_intrinsics = load_intrinsics(geo["camera_params_path"], geo["camera_name"])
    except (OSError, KeyError) as e:
        print(f"[run] WARNING: gaze normalization unavailable ({e}); gaze_valid will stay False this session.")

    live_source = LiveSource(
        object_source=objects, arm_left=arm_left, arm_right=arm_right, head=head, gaze=gaze,
        cameras=cameras, head_position=tuple(geo["head_position"]), camera_position=tuple(geo["camera_position"]),
        gaze_intrinsics=gaze_intrinsics, n_candidates=geo["n_candidates"],
    )

    mode = resolve_mode(cfg)
    modules = build_modules(cfg, mode, arm_left, arm_right, head)
    # state_source: the avatar's SysState comes in on SceneObjectsMsg, not on
    # the command channel -- see _scene_snapshot and SystemArbitrator's
    # constructor. Without it engage() waits on a value that never changes.
    arbitrator = SystemArbitrator(sys_state, policy_mode=mode,
                                  require_confirmation_for=cfg["arbitrator"]["require_confirmation_for"],
                                  state_source=lambda: _scene_snapshot(objects))

    console = OperatorConsole(arbitrator, sys_state)
    console.start()

    print(f"[run] modules: {list(modules)}  mode: {arbitrator.policy_mode}")
    if mode == EVALUATE_MODE and "policy" not in modules:
        raise ValueError("evaluate mode needs modules.policy.enabled: true")
    evaluation: Optional[EvalSession] = None
    try:
        if mode == INTERVENTION_MODE:
            # Park the arms BEFORE anything else can command them, and before we
            # know whether the headset is even up. Until this lands, authority is
            # UNSET, which gates nothing -- so without it there is a window where
            # the policy has both arms and possibly nobody is watching.
            arbitrator.claim_hold()
            print("[run] intervention: both arms parked in HOLD. Engage from the VR "
                  "interface, arm the intervention panel, then press RESUME.")
            # No terminal prompt here on purpose. The operator is in a headset and
            # cannot see this window; their RESUME press is the confirmation, and
            # it is given by someone looking at the robot. If it never comes, this
            # waits forever and nothing moves -- which is the safe outcome.
            if not arbitrator.await_operator_handover(lambda: _scene_snapshot(objects)):
                print("[run] no handover from the operator -- exiting without running")
                return
        elif not arbitrator.engage():
            # engage() has already printed which of the two it was.
            print("[run] engage aborted -- exiting without running")
            return

        orchestrator = Orchestrator(live_source, modules)
        # Constructed here rather than inside run() so the actuator can log what
        # it actually put on the wire (see RunLogger.log_gripper) alongside what
        # the policy predicted.
        run_logger = RoutingLogger() if mode == EVALUATE_MODE else RunLogger()
        action_key = "playback" if mode == "playback" else "policy"
        pcfg = cfg["modules"]["policy"]
        status_sender = None
        if action_key == "policy":
            status_sender = make_status_sender(sys_state, pcfg.get("status_hz", 5.0),
                                               tuple(pcfg.get("agree_spread", [0.01, 0.05])))
        # Policy only: playback replays recorded 0/0.08 widths, which need no latch
        # and should reproduce the recording exactly.
        latch = None
        if action_key == "policy":
            g = pcfg.get("gripper", {}) or {}
            latch = GripperLatch(close_below=g.get("close_below", 0.03),
                                 open_above=g.get("open_above", 0.06),
                                 min_hold_s=g.get("min_hold_s", 1.0))
            print(f"[run] gripper latch: close < {latch.close_below} m, open > {latch.open_above} m "
                  f"after >= {latch.min_hold_s} s")
        if mode == EVALUATE_MODE:
            evaluation = build_evaluation(cfg, args, config_path, sys_state, arm_left, arm_right,
                                          run_logger, modules["policy"], latch)
        actuator = make_actuator(arbitrator, arm_left, arm_right, head, geo["head_pin_absolute"],
                                  logger=run_logger, action_key=action_key, status_sender=status_sender,
                                  gripper_latch=latch,
                                  gate=(lambda: evaluation.acting) if evaluation is not None else None)
        on_tick = actuator
        if evaluation is not None:
            def on_tick(frame, outputs):
                evaluation.tick(frame)     # trial timer, start/end, logger swap
                actuator(frame, outputs)
        # Playback ends itself when the recording runs out; a policy run only
        # ever ends on the console or Ctrl+C.
        playback = modules.get("playback")
        should_stop = (console.stop_requested if playback is None
                       else lambda: console.stop_requested() or playback.finished)
        if evaluation is not None:
            should_stop = lambda: console.stop_requested() or evaluation.finished
            evaluation.begin()
        logger = orchestrator.run(logger=run_logger, on_tick=on_tick, should_stop=should_stop)
        if logger.n_ticks == 0:
            # LiveSource ticks on SceneObjectsMsg and gives up after a couple of
            # seconds of silence, so a run that ends here never saw the sim at
            # all -- and every module is innocent. Say which feed was missing
            # rather than leaving an empty log to explain it.
            print(f"[run] WARNING: no frames -- nothing arrived on scene_objects port "
                  f"{net['scene_objects']['receive_port']} within LiveSource's tick timeout. "
                  f"Check the avatar's robot_config avatar.scene_objects block is present and "
                  f"enabled, and that it points at this host/port.")
        if evaluation is not None:
            # Each trial was saved on its own as it ended; there is no whole-run log.
            evaluation.close(stopped=console.stop_requested())
            evaluation = None
            return
        log_path = f"logs/run_{time.strftime('%Y%m%d_%H%M%S')}.hdf5"
        meta = {"policy_mode": arbitrator.policy_mode, "config_path": config_path}
        if playback is not None:
            meta["playback_episode"] = playback.path
        logger.save(log_path, meta=meta)
        print(f"[run] saved {log_path}")
    except KeyboardInterrupt:
        if evaluation is not None:
            evaluation.close(stopped=True)
            evaluation = None
        # Ctrl+C is the emergency-stop path -- same cleanup as a graceful
        # 'stop', just triggered from the keyboard instead of the console
        # (also covers Ctrl+C while still waiting on the engage confirm).
        print("\n[run] interrupted -- disengaging and shutting down")
    finally:
        if mode == INTERVENTION_MODE:
            # Park the arms rather than disengaging. The operator owns SysState
            # here, and requesting IDLE would end THEIR episode the moment this
            # process exits -- including on a Ctrl+C they did not press. The
            # avatar's watchdog would reach HOLD on its own 250 ms after we stop
            # sending anyway; this just makes it immediate and explicit.
            arbitrator.claim_hold()
        else:
            arbitrator.disengage()
        shutdown()


if __name__ == "__main__":
    main(parse_args())
