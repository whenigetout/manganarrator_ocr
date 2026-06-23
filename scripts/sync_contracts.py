from __future__ import annotations

import subprocess
import sys


GITHUB_PACKAGE = (
    "git+https://github.com/whenigetout/manganarrator_contracts.git"
    "@main#subdirectory=python"
)


def main() -> int:
    print(f"[contracts] Installing mn-contracts from GitHub...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        GITHUB_PACKAGE,
    ])
    print("[contracts] mn-contracts is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())