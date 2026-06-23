from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    service_root = Path(__file__).resolve().parents[1]
    contracts_root = service_root.parent / "manganarrator_contracts" / "python"

    if not contracts_root.exists():
        print(f"[contracts] Skipping sync; not found: {contracts_root}")
        return 0

    print(f"[contracts] Syncing mn-contracts from {contracts_root}")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        str(contracts_root),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())