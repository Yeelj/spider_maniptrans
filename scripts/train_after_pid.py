#!/usr/bin/env python3
"""Wait for a training PID, then run the next training command in tmux.

Defaults: PID 24826, tmux session 0 / window 0 / pane 0.
Run this script from another terminal; the tmux pane keeps its working directory
and activated Python environment. A file lock prevents duplicate watchers.
"""

import argparse
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


TRAIN_COMMAND = shlex.join([
    "python", "main/rl/train.py", "task=ResDexHand", "dexhand=artimano",
    "side=BiH", "headless=true", "num_envs=4096", "learning_rate=2e-4",
    "test=false", "randomStateInit=true", "dataIndices=[9fc3e@0]",
    "early_stop_epochs=10000", "actionsMovingAverage=0.4",
    "experiment=OakInk_9fc3e@0_kni@1_artimano", "new_obj=kni@1", "correct=true",
])


def log(message):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + message, flush=True)


def process_info(pid):
    """Return Linux /proc fields, including process identity and foreground PGID."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # The parenthesized process name may contain spaces or closing parentheses.
    return raw[raw.rfind(")") + 2:].split()


def tmux(*args):
    return subprocess.run(
        ["tmux", *args], check=True, text=True, capture_output=True,
    ).stdout.strip()


def pane_info(target):
    return tmux(
        "display-message", "-p", "-t", target,
        "#{pane_id}\t#{pane_pid}\t#{pane_current_path}\t#{pane_in_mode}\t#{pane_dead}",
    ).split("\t")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=24826, help="PID to watch")
    parser.add_argument("--target", default="=0:0.0", help="tmux target pane")
    parser.add_argument("--interval", type=float, default=5, help="poll interval in seconds")
    parser.add_argument("--command", default=TRAIN_COMMAND, help="next shell command")
    parser.add_argument("--dry-run", action="store_true", help="validate and print; do not wait or send")
    args = parser.parse_args()
    if args.pid <= 0 or not (0 < args.interval < float("inf")):
        parser.error("--pid and --interval must be positive (and finite)")
    if not args.command.strip() or "\n" in args.command or "\r" in args.command:
        parser.error("--command must be a nonempty single line")

    # Keep this handle open for the entire run. Kernel releases the lock on exit.
    lock_path = Path(tempfile.gettempdir()) / f"train_after_pid_{os.getuid()}_{args.pid}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"PID {args.pid} already has a running watcher") from None

        watched = process_info(args.pid)
        if watched is None or watched[0] in ("Z", "X"):
            raise RuntimeError(f"PID {args.pid} is not running; no command was sent")
        start_time = watched[19]  # /proc stat field 22: protects against PID reuse.
        pane_id, shell_pid, cwd, _, dead = pane_info(args.target)
        shell = process_info(int(shell_pid))
        if shell is None or dead != "0" or shell[4] == "0" or watched[4] != shell[4]:
            raise RuntimeError(f"PID {args.pid} is not attached to live pane {args.target}")

        # Validate ancestry as well as the terminal, so an unrelated PID cannot
        # cause a command to be inserted into this pane.
        ancestor = args.pid
        while ancestor != int(shell_pid):
            info = process_info(ancestor)
            if info is None or int(info[1]) <= 1 or int(info[1]) == ancestor:
                raise RuntimeError(f"PID {args.pid} is not a child of pane {pane_id}'s shell")
            ancestor = int(info[1])
        shell_start = shell[19]

        log(f"Watching PID {args.pid}, tmux {args.target} ({pane_id}), cwd={cwd}")
        log(f"Next command: {args.command}")
        if args.dry_run:
            log("Dry run complete; no command was sent")
            return

        while True:
            current = process_info(args.pid)
            if current is None or current[19] != start_time or current[0] in ("Z", "X"):
                break
            time.sleep(args.interval)

        log(f"PID {args.pid} exited; waiting for the pane's shell to regain the terminal")
        while True:
            current_id, current_shell, _, in_mode, dead = pane_info(pane_id)
            shell = process_info(int(shell_pid))
            if (current_id != pane_id or current_shell != shell_pid or dead != "0"
                    or shell is None or shell[19] != shell_start or shell[0] in ("Z", "X")):
                raise RuntimeError("The original tmux pane/shell is gone; no command was sent")
            # /proc stat fields 5 and 8: shell process group and foreground group.
            if shell[2] == shell[5] and in_mode == "0":
                break
            time.sleep(args.interval)

        # Literal mode preserves Hydra brackets and shell quoting. Enter is a
        # separate key, sent exactly once. No Ctrl-C is sent to the old training.
        tmux("send-keys", "-t", pane_id, "-l", "--", args.command)
        tmux("send-keys", "-t", pane_id, "Enter")
        log(f"Training command submitted to {pane_id}; watcher finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Watcher cancelled")
        sys.exit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        log(f"ERROR: {detail}")
        sys.exit(1)
