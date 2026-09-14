"""Interactive stdin console for controlling a live session while
Orchestrator.run() is executing: graceful stop, pause/resume, re-home
(disengage + re-engage), and per-arm reset/recovery -- everything
SysStateClient/SystemArbitrator already expose, just not reachable once the
run loop owns the main thread. Runs on a background daemon thread so it
never blocks the run loop; commands take effect on the next tick boundary
(see Orchestrator.run's should_stop docstring for the same latency note).
"""

from __future__ import annotations

import queue
import threading
import time

from .arbitrator import SystemArbitrator
from .live import wire
from .live.sys_state_client import SysStateClient

_HELP = """\
Commands:
  stop          gracefully stop the run loop (takes effect next tick)
  pause         request PAUSED (policy stops acting, ENGAGED resumes with 'resume')
  resume        request ENGAGED (resume after pause)
  home          disengage then re-engage (re-homes the arms; SAME scene)
  next_episode  end the episode, request a NEW scene from the episode config
                server, then re-home. Use this between policy rollouts --
                'home' alone leaves the objects untouched. (alias: restart)
  reset_left    request recovery for arm_left
  reset_right   request recovery for arm_right
  resume_left   confirm resume for arm_left after its reset completes
  resume_right  confirm resume for arm_right after its reset completes
  help          show this message
"""


class OperatorConsole:
    """Reads commands from stdin on a background thread and dispatches them
    against SystemArbitrator/SysStateClient. Pass stop_requested as
    Orchestrator.run()'s should_stop to wire the 'stop' command through."""

    def __init__(self, arbitrator: SystemArbitrator, sys_state: SysStateClient):
        self.arbitrator = arbitrator
        self.sys_state = sys_state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._pending_confirm: "queue.Queue[str] | None" = None
        # Commands are executed on their own thread, NOT on the stdin reader.
        # A command that needs more input (engage()'s confirm prompt) blocks
        # until confirm() receives a line, and only the reader thread can
        # supply one -- so dispatching inline deadlocks the console the first
        # time 'home' is used in a mode that requires confirmation. That went
        # unnoticed because the initial engage() is called from the main
        # thread, leaving the reader free; only re-engaging from the console
        # hits it.
        self._cmd_q: "queue.Queue[str]" = queue.Queue()
        self._reader_ident: int | None = None
        # Lines typed while a command is still running that aren't themselves
        # commands. Almost always someone answering a confirm prompt slightly
        # before it appears: the worker hasn't reached confirm() yet, so
        # _pending_confirm is still None and a naive dispatch would report
        # "unknown command 'yes'" and leave the prompt hanging forever.
        self._typeahead: "queue.Queue[str]" = queue.Queue()

    #: everything _dispatch understands; anything else typed mid-command is
    #: treated as type-ahead for a confirm prompt rather than a bad command.
    COMMANDS = frozenset({
        "stop", "pause", "resume", "home", "next_episode", "restart",
        "reset_left", "reset_right", "resume_left", "resume_right", "help",
    })

    def start(self) -> None:
        """Starts the background stdin-reading thread. Also becomes
        arbitrator's confirmation callback (see confirm()) -- it must be the
        only thing that ever calls input(), or engage()'s confirm prompt and
        this loop's command reads race for the same terminal input."""
        self.arbitrator._confirm = self.confirm
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        print("[console] ready -- type 'help' for commands")

    def stop_requested(self) -> bool:
        """Pass as Orchestrator.run(should_stop=...)."""
        return self._stop.is_set()

    def confirm(self, mode: str) -> bool:
        """SystemArbitrator's confirmation callback, routed through this
        console's single input()-reading thread instead of calling input()
        itself -- otherwise a concurrent input() here and in engage() race
        for the same stdin line and neither reliably sees it.

        Must never be called ON the reader thread: it waits for a line that
        only that thread can deliver. Guarded rather than left to hang, since
        the symptom (console silently stops accepting input) gives no clue.
        """
        if self._reader_ident is not None and threading.get_ident() == self._reader_ident:
            raise RuntimeError(
                "OperatorConsole.confirm() called on the stdin reader thread -- this would "
                "deadlock. Dispatch the command via _cmd_q instead of calling it inline."
            )
        print(f"[console] About to engage in '{mode}' mode -- policy will command "
              f"the arms directly. Type 'yes' to confirm:")
        # Someone may already have answered before the prompt printed.
        try:
            reply = self._typeahead.get_nowait()
            print(f"[console] (using already-typed '{reply.strip()}')")
            return reply.strip().lower() == "yes"
        except queue.Empty:
            pass
        q: "queue.Queue[str]" = queue.Queue()
        self._pending_confirm = q
        try:
            reply = q.get()
        finally:
            self._pending_confirm = None
        return reply.strip().lower() == "yes"

    def _loop(self) -> None:
        """Reads stdin and does nothing slow -- everything else is handed to
        the worker so this thread is always available to answer a confirm
        prompt or an emergency stop."""
        self._reader_ident = threading.get_ident()
        while True:
            try:
                line = input()
            except EOFError:
                return
            if self._pending_confirm is not None:
                self._pending_confirm.put(line)
                continue
            cmd = line.strip().lower()
            if not cmd:
                continue
            if cmd == "stop":
                # Handled inline, deliberately: stop must work even if the
                # worker is wedged on something.
                self._stop.set()
                print("[console] stop requested -- run loop will exit after the current tick")
                continue
            if cmd not in self.COMMANDS:
                # Not a command; hold it for a confirm prompt that may be about
                # to appear rather than rejecting it (see _typeahead).
                self._typeahead.put(line)
                continue
            self._cmd_q.put(cmd)

    def _worker_loop(self) -> None:
        while True:
            cmd = self._cmd_q.get()
            try:
                self._dispatch(cmd)
            except Exception as e:  # a bad command must not kill the console
                print(f"[console] command '{cmd}' failed: {e!r}")

    def _dispatch(self, cmd: str) -> None:
        if cmd == "stop":
            self._stop.set()
            print("[console] stop requested -- run loop will exit after the current tick")
        elif cmd == "pause":
            self.sys_state.request_state(wire.SysState.PAUSED)
        elif cmd == "resume":
            self.sys_state.request_state(wire.SysState.ENGAGED)
        elif cmd == "home":
            if not self.arbitrator.disengage():
                print("[console] sim did not reach IDLE within 5s -- not re-engaging "
                      "(a HOMING request sent while still ENGAGED is silently dropped)")
                return
            ok = self.arbitrator.engage()
            print(f"[console] re-engage {'succeeded' if ok else 'failed or refused'}")
        elif cmd in ("next_episode", "restart"):
            # Re-homing is NOT enough to get a new scene: Avatar only calls
            # requestEpisodeConfig at startup and in its episode_restart
            # handler, so without this the objects stay exactly where they are
            # (and an episode-config server in --replay mode keeps serving the
            # same episode). Restart first, then re-home onto the new scene.
            self.sys_state.request_episode_restart(label="operator_next_episode")
            print("[console] episode_restart sent -- new scene requested from the config server")
            time.sleep(1.0)  # let the Avatar rebuild the scene before homing into it
            if not self.arbitrator.disengage():
                print("[console] sim did not reach IDLE within 5s -- re-home manually with 'home'")
                return
            ok = self.arbitrator.engage()
            print(f"[console] re-engage {'succeeded' if ok else 'failed or refused'}")
        elif cmd == "reset_left":
            self.sys_state.request_arm_reset("arm_left")
        elif cmd == "reset_right":
            self.sys_state.request_arm_reset("arm_right")
        elif cmd == "resume_left":
            self.sys_state.request_arm_resume("arm_left")
        elif cmd == "resume_right":
            self.sys_state.request_arm_resume("arm_right")
        elif cmd == "help":
            print(_HELP)
        else:
            print(f"[console] unknown command '{cmd}' -- type 'help' for commands")
