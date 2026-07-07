from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp_md_loop.active_learning_loop import run_active_learning_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run/resume the MLP-MD active-learning loop.")
    parser.add_argument("--conf", required=True, help="TOML configuration file")
    parser.add_argument("--once", action="store_true", help="Run one loop iteration and exit")
    parser.add_argument("--dry-run", action="store_true", help="Prepare files/state without submitting or running commands")
    parser.add_argument("--quiet", action="store_true", help="Hide progress messages")
    args = parser.parse_args()

    state = run_active_learning_loop(
        args.conf,
        once=args.once,
        dry_run=args.dry_run,
        progress=not args.quiet,
    )
    print(f"iteration={state.iteration}")
    print(f"completed_dft_logs={state.completed_dft_logs}")
    if state.stopped_reason:
        print(f"stopped_reason={state.stopped_reason}")


if __name__ == "__main__":
    main()
