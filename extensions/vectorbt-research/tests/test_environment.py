import importlib.metadata
import sys
import tomllib
from pathlib import Path

import numba
import vectorbt as vbt

LOCK = Path(__file__).resolve().parents[1] / "uv.lock"


@numba.njit(fastmath=False)
def add_one(value):
    return value + 1


def test_vectorbt_imports_and_reports_version_1_2_0():
    assert vbt is not None
    assert importlib.metadata.version("vectorbt") == "1.2.0"


def test_python_is_3_13():
    assert sys.version_info[:2] == (3, 13)


def test_imported_versions_match_the_lock():
    locked = {
        package["name"]: package["version"]
        for package in tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]
        if "version" in package
    }
    for name in ("vectorbt", "numba", "numpy", "pandas", "pyarrow"):
        assert importlib.metadata.version(name) == locked[name], name


def test_numba_integer_function_compiles_and_returns_expected_value():
    assert add_one(2) == 3
    assert len(add_one.signatures) == 1
