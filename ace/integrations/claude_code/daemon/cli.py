"""Command-line interface for ACE daemon.

Provides commands for starting, stopping, and managing the ACE daemon.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from .service import DaemonConfig, DaemonService, is_daemon_running, stop_daemon


def cmd_start(args) -> int:
    """Start the daemon."""
    pid = is_daemon_running()
    if pid:
        print(f"Daemon already running (PID: {pid})")
        return 1

    config = DaemonConfig(
        max_workers=args.workers,
        log_level="DEBUG" if args.verbose else "INFO",
    )

    if args.foreground:
        # Run in foreground (for debugging)
        print("Starting ACE daemon in foreground (CLI subscription mode)...")
        daemon = DaemonService(config)
        return daemon.run()
    else:
        # Daemonize
        print("Starting ACE daemon (CLI subscription mode)...")
        _daemonize()
        daemon = DaemonService(config)
        return daemon.run()


def cmd_stop(args) -> int:
    """Stop the daemon."""
    pid = is_daemon_running()
    if not pid:
        print("Daemon not running")
        return 1

    print(f"Stopping daemon (PID: {pid})...")
    if stop_daemon():
        print("Daemon stopped")
        return 0
    else:
        print("Failed to stop daemon")
        return 1


def cmd_restart(args) -> int:
    """Restart the daemon."""
    stop_daemon()
    return cmd_start(args)


def cmd_status(args) -> int:
    """Show daemon status."""
    pid = is_daemon_running()
    if pid:
        print(f"Daemon running (PID: {pid})")

        # Show queue stats
        queue_dir = Path.home() / ".ace" / "queue"
        if queue_dir.exists():
            pending = len(list(queue_dir.glob("hook-*.json")))
            failed_dir = queue_dir / "failed"
            failed = len(list(failed_dir.glob("*.json"))) if failed_dir.exists() else 0
            print(f"Queue: {pending} pending, {failed} failed")

        # Show log location
        log_file = Path.home() / ".ace" / "logs" / "daemon.log"
        if log_file.exists():
            print(f"Log: {log_file}")

        return 0
    else:
        print("Daemon not running")
        return 1


def cmd_install(args) -> int:
    """Install launchd service (macOS)."""
    import platform

    if platform.system() != "Darwin":
        print("launchd is only available on macOS")
        print("On Linux, use systemd or create a service manually")
        return 1

    plist_content = _generate_launchd_plist()
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.ace.daemon.plist"

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    print(f"Installed: {plist_path}")
    print()
    print("To load and start the service:")
    print(f"  launchctl load {plist_path}")
    print(f"  launchctl start com.ace.daemon")
    print()
    print("The daemon will auto-start on login.")
    print()
    print("To check status:")
    print("  ace-daemon status")

    return 0


def cmd_uninstall(args) -> int:
    """Uninstall launchd service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.ace.daemon.plist"

    if not plist_path.exists():
        print("Service not installed")
        return 1

    # Unload first
    import subprocess

    subprocess.run(
        ["launchctl", "unload", str(plist_path)],
        capture_output=True,
    )

    plist_path.unlink()
    print("Service uninstalled")
    return 0


def cmd_logs(args) -> int:
    """Show daemon logs."""
    log_file = Path.home() / ".ace" / "logs" / "daemon.log"

    if not log_file.exists():
        print("No log file found")
        return 1

    if args.follow:
        # Tail -f equivalent
        import subprocess

        try:
            subprocess.run(["tail", "-f", str(log_file)])
        except KeyboardInterrupt:
            pass
        return 0
    else:
        # Show last N lines
        lines = log_file.read_text().splitlines()
        for line in lines[-args.lines :]:
            print(line)
        return 0


def _daemonize() -> None:
    """Daemonize the process (Unix double-fork)."""
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent exits
        sys.exit(0)

    # Decouple from parent
    os.setsid()
    os.chdir("/")

    # Second fork
    pid = os.fork()
    if pid > 0:
        # First child exits
        sys.exit(0)

    # Redirect stdio to /dev/null
    sys.stdin = open("/dev/null", "r")
    sys.stdout = open("/dev/null", "w")
    sys.stderr = open("/dev/null", "w")


def _generate_launchd_plist() -> str:
    """Generate launchd plist file content.

    ACE uses subscription-only mode via Claude CLI. No API keys required.
    """
    # Use the current Python interpreter (the one running ace-daemon)
    # This ensures we use the Python that has ace-framework installed
    python_path = sys.executable

    # Build program arguments (CLI subscription mode is always used)
    prog_args = [
        f"        <string>{python_path}</string>",
        "        <string>-m</string>",
        "        <string>ace.daemon.cli</string>",
        "        <string>start</string>",
        "        <string>--foreground</string>",
    ]

    args_str = "\n".join(prog_args)
    home = Path.home()

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ace.daemon</string>

    <key>ProgramArguments</key>
    <array>
{args_str}
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>{home}/.ace/logs/daemon.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{home}/.ace/logs/daemon.stderr.log</string>

    <key>WorkingDirectory</key>
    <string>{home}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>
"""


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="ace-daemon",
        description="ACE Learning Daemon - Background processing for Claude Code hooks",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the daemon")
    start_parser.add_argument(
        "--foreground",
        "-f",
        action="store_true",
        help="Run in foreground (don't daemonize)",
    )
    start_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=2,
        help="Max concurrent learning tasks (default: 2)",
    )
    start_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Stop command
    subparsers.add_parser("stop", help="Stop the daemon")

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart the daemon")
    restart_parser.add_argument("--workers", "-w", type=int, default=2)
    restart_parser.add_argument("--verbose", "-v", action="store_true")
    restart_parser.add_argument("--foreground", "-f", action="store_true")

    # Status command
    subparsers.add_parser("status", help="Show daemon status")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show daemon logs")
    logs_parser.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow log output (like tail -f)",
    )
    logs_parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=50,
        help="Number of lines to show (default: 50)",
    )

    # Install command
    subparsers.add_parser(
        "install", help="Install launchd service (macOS)"
    )

    # Uninstall command
    subparsers.add_parser("uninstall", help="Uninstall launchd service")

    args = parser.parse_args()

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "restart": cmd_restart,
        "status": cmd_status,
        "logs": cmd_logs,
        "install": cmd_install,
        "uninstall": cmd_uninstall,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
