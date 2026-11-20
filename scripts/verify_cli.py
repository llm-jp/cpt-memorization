"""Check every public command's help without GPU dependencies or model downloads."""

import os
import subprocess
import sys

from cpt_memorization.__main__ import COMMANDS

for command in COMMANDS:
    result = subprocess.run(
        [sys.executable, "-m", "cpt_memorization", command, "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode:
        raise SystemExit(f"{command} failed:\n{result.stderr}")
print(f"All {len(COMMANDS)} command help checks passed")
