#!/usr/bin/env python3
"""
k3_mrl_watcher.py — Multi-Path Auto-Indexing Watcher Daemon for K3 Ecosystem.
Monitors Knowledge Items, Skills, Brain artifacts, and Agent blueprints.
Automatically triggers incremental MRL re-indexing with debouncing and Secret Manager fallback.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DEFAULT_WATCH_DIRS = [
    Path.home() / ".gemini" / "antigravity" / "knowledge",
    Path(r"C:\K3_Firehose\.agent\skills"),
    Path(r"C:\K3_Firehose\Proto_book\.agents"),
]

DEFAULT_INDEX = Path(__file__).resolve().parent / "mrl_index.pkl"
DEFAULT_REBUILD_SCRIPT = Path(r"C:\tmp\mrl_index_all.py")


def _resolve_api_key() -> Optional[str]:
    """Retrieves Gemini API key from environment or Google Cloud Secret Manager."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    try:
        cmd = [
            "gcloud", "secrets", "versions", "access", "latest",
            "--secret=GEMINI_API_KEY",
            "--project=gen-lang-client-0778100894",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
        if res.returncode == 0 and res.stdout.strip():
            fetched = res.stdout.strip()
            os.environ["GEMINI_API_KEY"] = fetched
            return fetched
    except Exception:
        pass
    return None


class MultiPathIndexTriggerHandler(FileSystemEventHandler):
    def __init__(
        self,
        debounce_seconds: int = 10,
        rebuild_script: Optional[Path] = None,
        index_file: Optional[Path] = None,
    ):
        super().__init__()
        self.debounce_seconds = debounce_seconds
        self.rebuild_script = rebuild_script or DEFAULT_REBUILD_SCRIPT
        self.index_file = index_file or DEFAULT_INDEX
        self.last_trigger_time = 0.0
        self._extensions = {".md", ".py", ".js", ".ts", ".json", ".txt", ".docx"}
        self._process: Optional[subprocess.Popen] = None

    def on_modified(self, event):
        self._handle_event(event)

    def on_created(self, event):
        self._handle_event(event)

    def _handle_event(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix not in self._extensions:
            return

        # Ignore changes to index files, lock files, and tmp files
        if path.name.endswith(".pkl") or path.name.endswith(".tmp") or path.name.endswith(".lock"):
            return

        current_time = time.time()
        if (current_time - self.last_trigger_time) > self.debounce_seconds:
            self.last_trigger_time = current_time
            print(f"\n[WATCHER] Change detected in {path.name} ({path.parent.name})")

            # Check if previous process is still running
            if self._process and self._process.poll() is None:
                print("[WATCHER] Previous indexing pass still active. Skipping concurrent trigger.")
                return

            api_key = _resolve_api_key()
            env = os.environ.copy()
            if api_key:
                env["GEMINI_API_KEY"] = api_key

            if self.rebuild_script.exists():
                print(f"[WATCHER] Spawning incremental re-indexer: {self.rebuild_script.name}...")
                self._process = subprocess.Popen(
                    [sys.executable, str(self.rebuild_script)],
                    env=env,
                    cwd=str(self.rebuild_script.parent),
                )
            else:
                indexer_fallback = Path(__file__).resolve().parent / "k3_mrl_indexer.py"
                if indexer_fallback.exists():
                    print(f"[WATCHER] Spawning fallback indexer: {indexer_fallback.name}...")
                    self._process = subprocess.Popen(
                        [sys.executable, str(indexer_fallback), "--path", str(path.parent), "--index", str(self.index_file)],
                        env=env,
                        cwd=str(indexer_fallback.parent),
                    )
                else:
                    print(f"[WATCHER ERROR] Neither {self.rebuild_script} nor {indexer_fallback} found.")


def run_watcher(
    watch_dirs: List[Path],
    rebuild_script: Path = DEFAULT_REBUILD_SCRIPT,
    index_file: Path = DEFAULT_INDEX,
    debounce: int = 10,
):
    valid_dirs = [d.resolve() for d in watch_dirs if d.exists()]
    if not valid_dirs:
        print("[ERROR] None of the specified watch directories exist.")
        return

    print("==================================================")
    print("  K3 MRL Auto-Indexing Watcher Daemon")
    print(f"  Target Index   : {index_file}")
    print(f"  Rebuild Script : {rebuild_script}")
    print(f"  Debounce Delay : {debounce}s")
    print(f"  Monitoring {len(valid_dirs)} Directories:")
    for d in valid_dirs:
        print(f"    • {d}")
    print("==================================================")

    # Pre-verify API key availability
    key = _resolve_api_key()
    if key:
        print(f"[WATCHER AUTH] API key resolved (prefix: {key[:8]}...)")
    else:
        print("[WATCHER AUTH WARNING] No API key detected in environment or Secret Manager.")

    event_handler = MultiPathIndexTriggerHandler(
        debounce_seconds=debounce,
        rebuild_script=rebuild_script,
        index_file=index_file,
    )

    observer = Observer()
    for d in valid_dirs:
        observer.schedule(event_handler, str(d), recursive=True)

    observer.start()
    print("\n[WATCHER] Active and waiting for file modifications. Press Ctrl+C to terminate.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[WATCHER] Shutting down...")
        observer.stop()
    observer.join()
    print("[WATCHER] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K3 MRL Multi-Path Auto-Indexing Watcher")
    parser.add_argument(
        "--dirs",
        nargs="+",
        type=str,
        help="Directories to watch. Defaults to knowledge, skills, and Proto_book/.agents",
    )
    parser.add_argument(
        "--script",
        type=str,
        default=str(DEFAULT_REBUILD_SCRIPT),
        help="Rebuild script to invoke on change (default: c:\\tmp\\mrl_index_all.py)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(DEFAULT_INDEX),
        help="Target index pickle path",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=10,
        help="Debounce interval in seconds (default: 10)",
    )

    args = parser.parse_args()
    dirs = [Path(p) for p in args.dirs] if args.dirs else DEFAULT_WATCH_DIRS
    run_watcher(
        watch_dirs=dirs,
        rebuild_script=Path(args.script),
        index_file=Path(args.index),
        debounce=args.debounce,
    )
