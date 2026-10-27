"""Installed-package import check for the shared rules, approved row S-15."""

import subprocess
import sys
from pathlib import Path

import deriv_mt5

# -I keeps the current directory and PYTHONPATH off sys.path, so the child imports only the
# installed package and cannot pick up a source copy from the working directory.
CHILD_CODE = """
import sys

BLOCKED = ("vectorbt", "numba", "numpy", "pandas", "nautilus_trader")


class BlockEngineImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in BLOCKED:
            raise ImportError("engine import blocked: " + fullname)
        return None


sys.meta_path.insert(0, BlockEngineImports())

import deriv_mt5.rules
import deriv_mt5.ticks

assert deriv_mt5.rules.stake_cents(1000000) == 500
print(deriv_mt5.__file__)
"""


def test_s15_shared_rules_import_with_no_engines():
    """Shared rules import with no vectorbt or Nautilus installed"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", CHILD_CODE],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    # The package file must sit in this project's package folder or in the active environment.
    package_file = Path(deriv_mt5.__file__).resolve()
    project_package = (Path(__file__).resolve().parents[1] / "deriv_mt5").resolve()
    environment = Path(sys.prefix).resolve()
    inside_project = package_file.is_relative_to(project_package)
    inside_environment = package_file.is_relative_to(environment)
    assert inside_project or inside_environment, package_file

    # The child, with engine imports blocked, must load that same file.
    assert Path(result.stdout.strip()).resolve() == package_file
