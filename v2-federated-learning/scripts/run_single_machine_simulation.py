"""Convenience wrapper: run single-machine FL simulation with verbose output.

This script is a thin wrapper around run_federated_training.py that sets
sensible defaults for local development / debugging.

Usage:
    python scripts/run_single_machine_simulation.py
    python scripts/run_single_machine_simulation.py --rounds 5 --strategy fedprox
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main() -> None:
    # Forward any extra CLI args to the main training script
    extra = sys.argv[1:]
    cmd = [
        sys.executable,
        str(BASE / "run_federated_training.py"),
        "--mode", "simulation",
        *extra,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
