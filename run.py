#!/usr/bin/env python3
"""Entry point: wires the live transports, intent/policy modules, and
SystemArbitrator together per configs/system.yaml, then runs the
orchestrator loop against the sim (or hardware speaking the same protocol).

Usage: python run.py [path/to/system.yaml]
"""

from __future__ import annotations

import sys
import time

import yaml

from teleop_orchestrator.arbitrator import SystemArbitrator
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
    head = HeadChannel(host, net["head"]["send_port"], net["head"]["receive_port"])
    gaze = GazeReceiver(receive_port=net["gaze"]["receive_port"])
    objects = SimObjectSource(receive_port=net["scene_objects"]["receive_port"])
    cameras = {name: SharedFrameReader(shm_name) for name, shm_name in net["cameras"].items()}

    # SharedFrameReader has no background thread -- it attaches lazily on
    # first read() (try_open()), unlike the UDP channels below, so it's
    # deliberately excluded from this start() loop.
    for c in (sys_state, arm_left, arm_right, head, gaze, objects):
        c.start()
    return sys_state, arm_left, arm_right, head, gaze, objects, cameras


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


def build_modules(cfg: dict, mode: str, arm_left: ArmChannel, arm_right: ArmChannel) -> dict:
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
        modules["policy"] = PolicyModule(mcfg["policy"]["checkpoint"], arm_left, arm_right,
                                          debug_dump_dir=dbg,
                                          debug_dump_every=mcfg["policy"].get("debug_dump_every", 30),
                                          stored_hw=stored)
    return modules


# teleop-simulator arm_control.cpp: kGripperMaxWidth = 0.08, and gripper_cmd is
# logged as exactly 0.0 (closed) or 0.08 (open) -- a binary signal. 0.04 is the
# midpoint; ACT's L1 output is continuous so it needs thresholding somewhere.
_GRIPPER_OPEN_WIDTH_M = 0.08
_GRIPPER_CLOSE_THRESHOLD_M = 0.04


def make_actuator(arbitrator: SystemArbitrator, arm_left: ArmChannel, arm_right: ArmChannel,
                   head: HeadChannel, head_look_down_offset: dict, logger=None,
                   action_key: str = "policy"):
    """Returns the Orchestrator.run(on_tick=...) callback that sends the acting
    module's ActionOutput to the arms -- only when autonomous_allowed, so
    supportive/off mode never issues an ArmCommandMsg regardless of what the
    module predicts. action_key names that module ("policy" or "playback");
    everything below this line is identical for both, which is the point.
    Uses send_absolute_command: the retrained checkpoint predicts absolute
    world-frame poses (see teleop-policy/configs/dataset.yaml), which the sim
    only interprets correctly on the absolute channel (worldAbsoluteToBase),
    not send_command's delta-from-origin/VR path.

    Also holds the head at a fixed look-down offset every tick while
    autonomous: ACT was trained on operator-driven head motion (following
    gaze down toward the grasp), but nothing here reproduces that, so
    head_cam_left drifts out-of-distribution over a session unless pinned
    somewhere reasonable. Deterministic/scripted rather than learned --
    see system.yaml's geometry.head_look_down_offset comment for tuning.
    Playback overrides it per tick with the head track it recorded
    (extras head_pan/head_tilt), which is strictly better when available.
    """
    def on_tick(frame, outputs) -> None:
        action = outputs.get(action_key)
        if action is None or not action.ee_pose or not arbitrator.autonomous_allowed:
            return
        extras = action.extras or {}
        head.send_command(wire.SysState.ENGAGED,
                           extras.get("head_pan", head_look_down_offset["pan"]),
                           extras.get("head_tilt", head_look_down_offset["tilt"]))
        for side, channel in (("arm_left", arm_left), ("arm_right", arm_right)):
            if side not in action.ee_pose:
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
            close_flag = 1.0 if width < _GRIPPER_CLOSE_THRESHOLD_M else 0.0
            channel.send_absolute_command(wire.SysState.ENGAGED, pos, quat, close_flag)
            if logger is not None:
                logger.log_gripper(side, width=width, close_flag=close_flag)
                logger.log_arm_state(side, channel.latest())
    return on_tick


def main(config_path: str = "configs/system.yaml") -> None:
    cfg = load_config(config_path)
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
    modules = build_modules(cfg, mode, arm_left, arm_right)
    arbitrator = SystemArbitrator(sys_state, policy_mode=mode, require_confirmation_for=cfg["arbitrator"]["require_confirmation_for"])

    console = OperatorConsole(arbitrator, sys_state)
    console.start()

    print(f"[run] modules: {list(modules)}  mode: {arbitrator.policy_mode}")
    try:
        if not arbitrator.engage():
            print("[run] engage aborted (refused confirmation or timed out) -- exiting without running")
            return

        orchestrator = Orchestrator(live_source, modules)
        # Constructed here rather than inside run() so the actuator can log what
        # it actually put on the wire (see RunLogger.log_gripper) alongside what
        # the policy predicted.
        run_logger = RunLogger()
        action_key = "playback" if mode == "playback" else "policy"
        actuator = make_actuator(arbitrator, arm_left, arm_right, head, geo["head_look_down_offset"],
                                  logger=run_logger, action_key=action_key)
        # Playback ends itself when the recording runs out; a policy run only
        # ever ends on the console or Ctrl+C.
        playback = modules.get("playback")
        should_stop = (console.stop_requested if playback is None
                       else lambda: console.stop_requested() or playback.finished)
        logger = orchestrator.run(logger=run_logger, on_tick=actuator, should_stop=should_stop)
        if logger.n_ticks == 0:
            # LiveSource ticks on SceneObjectsMsg and gives up after a couple of
            # seconds of silence, so a run that ends here never saw the sim at
            # all -- and every module is innocent. Say which feed was missing
            # rather than leaving an empty log to explain it.
            print(f"[run] WARNING: no frames -- nothing arrived on scene_objects port "
                  f"{net['scene_objects']['receive_port']} within LiveSource's tick timeout. "
                  f"Check the avatar's robot_config avatar.scene_objects block is present and "
                  f"enabled, and that it points at this host/port.")
        log_path = f"logs/run_{time.strftime('%Y%m%d_%H%M%S')}.hdf5"
        meta = {"policy_mode": arbitrator.policy_mode, "config_path": config_path}
        if playback is not None:
            meta["playback_episode"] = playback.path
        logger.save(log_path, meta=meta)
        print(f"[run] saved {log_path}")
    except KeyboardInterrupt:
        # Ctrl+C is the emergency-stop path -- same cleanup as a graceful
        # 'stop', just triggered from the keyboard instead of the console
        # (also covers Ctrl+C while still waiting on the engage confirm).
        print("\n[run] interrupted -- disengaging and shutting down")
    finally:
        arbitrator.disengage()
        shutdown()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/system.yaml")
