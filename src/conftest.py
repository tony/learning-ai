"""Keep pytest from importing lessons whose dependencies the project environment lacks.

PyPI carries no torch wheel for Python 3.15, so torch is not a project
dependency. A lesson that needs it installs it from its own PEP 723 header and
runs with ``uv run --script``; collecting it here would fail on ``import torch``.
"""

import importlib.util

collect_ignore_glob: list[str] = []
if importlib.util.find_spec("torch") is None:
    collect_ignore_glob.append("053_earn_pytorch.py")
