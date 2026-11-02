"""Write a pip constraints file pinning every dependency in pyproject.toml to its declared minimum.

Usage: python ci/minimum_constraints.py > constraints-min.txt
       pip install -e ".[dev]" -c constraints-min.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
project = pyproject["project"]
requirements = list(project["dependencies"]) + [r for group in project["optional-dependencies"].values() for r in group]
for requirement in requirements:
    # name>=version, optionally followed by an environment marker such as "; python_version < '3.11'".
    match = re.fullmatch(r"([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9A-Za-z.]*)\s*(;.+)?", requirement.strip())
    if not match:
        sys.exit(f"cannot derive a minimum from requirement {requirement!r}; declare it as name>=version")
    marker = f" {match[3]}" if match[3] else ""
    print(f"{match[1]}=={match[2]}{marker}")
