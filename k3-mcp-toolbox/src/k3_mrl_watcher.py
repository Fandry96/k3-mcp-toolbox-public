import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class IndexTriggerHandler(FileSystemEventHandler):
    def __init__(
        self, debounce_seconds: int = 5, target_dir: str = "", index_file: str = ""
    ):
        self.debounce_seconds = debounce_seconds
        self.last_trigger_time = 0
        self.target_dir = target_dir
        self.index_file = index_file
        self._extensions = {".md", ".py", ".js", ".ts", ".json", ".txt"}
        self._process = None

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

        # Ignore changes to the index file itself
        if path.name == Path(self.index_file).name or path.name.endswith(".tmp"):
            return

        current_time = time.time()
        if (current_time - self.last_trigger_time) > self.debounce_seconds:
            self.last_trigger_time = current_time
            print(
                f"\n[WATCHER] Detected changes in {path.name}. Triggering incremental MRL re-indexing..."
            )

            # Don't spawn multiple running processes if one is already running
            if self._process and self._process.poll() is None:
                print(
                    "[WATCHER] Previous indexing still running, skipping this trigger."
                )
                return

            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "k3_mrl_indexer.py",
                    "--path",
                    self.target_dir,
                    "--index",
                    self.index_file,
                ]
            )


def run_watcher(target_dir: str, index_file: str, debounce: int = 5):
    target_path = Path(target_dir).resolve()
    if not target_path.exists():
        print(f"[ERROR] Target directory {target_path} does not exist.")
        return

    event_handler = IndexTriggerHandler(
        debounce_seconds=debounce, target_dir=target_dir, index_file=index_file
    )
    observer = Observer()
    observer.schedule(event_handler, str(target_path), recursive=True)
    observer.start()

    print(f"[WATCHER] Active. Monitoring {target_path} for code/docs changes...")
    print(f"[WATCHER] Target index: {index_file}")
    print("[WATCHER] Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRL Auto-Indexing Watcher Daemon")
    parser.add_argument(
        "--path", type=str, default=".", help="Directory to watch and index"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="mrl_index.pkl",
        help="Location of the index pickle file",
    )
    parser.add_argument(
        "--debounce",
        type=int,
        default=5,
        help="Seconds to wait between trigger executions",
    )

    args = parser.parse_args()
    run_watcher(args.path, args.index, args.debounce)
