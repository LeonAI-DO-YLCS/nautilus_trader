import importlib.metadata

import numba
import vectorbt as vbt


@numba.njit(fastmath=False)
def double_plus_one(value):
    return value * 2 + 1


def test_vectorbt_imports_and_reports_version_1_2_0():
    assert vbt is not None
    assert importlib.metadata.version("vectorbt") == "1.2.0"


def test_numba_integer_function_compiles_and_returns_expected_value():
    assert double_plus_one(20) == 41
    assert len(double_plus_one.signatures) == 1
