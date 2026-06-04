"""Pytest bootstrap.

Puts the repository root on sys.path so tests can `import constitutional_bioguard`
and `from scripts import ...` whether or not the package is pip-installed and without
relying on per-script `sys.path` hacks.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
