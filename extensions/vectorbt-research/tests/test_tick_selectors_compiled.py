"""Installed-package Python and Numba parity for the shared tick selectors and checked expiry.

The same installed `deriv_mt5.ticks` functions run in plain Python and compiled with Numba, on the
same literal cases the shared package tests use (`tick_selector_cases.json`). The adapter side owns
the Numba setup: it registers the original helpers (`checked_cents`, `checked_add`, `expiry_ns`)
with `register_jitable`, then compiles the original public functions with `fastmath=False`. The
shared package stays stdlib-only. Callers pass int64 arrays, plain-int times and plain-int
ordinals; plain-Python callers convert array scalars with `int(value)`.
"""

import json
import random
import types
from pathlib import Path

import numba
import numpy as np
import pytest
from deriv_mt5 import rules, ticks
from numba.extending import register_jitable

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "deriv-mt5"
    / "tests"
    / "fixtures"
    / "tick_selector_cases.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))
OVERFLOW = CASES["overflow"]
RANDOM_CASES = 300

register_jitable(rules.checked_cents)
register_jitable(rules.checked_add)
register_jitable(ticks.expiry_ns)
_jit = numba.njit(fastmath=False)

PYTHON_API = types.SimpleNamespace(
    expiry_ns=ticks.expiry_ns,
    entry=ticks.entry_index_in_range,
    exit=ticks.exit_index_in_range,
    is_fresh=ticks.exit_is_fresh,
)
COMPILED_API = types.SimpleNamespace(
    expiry_ns=_jit(ticks.expiry_ns),
    entry=_jit(ticks.entry_index_in_range),
    exit=_jit(ticks.exit_index_in_range),
    is_fresh=_jit(ticks.exit_is_fresh),
)


@pytest.fixture(params=["python", "compiled"])
def api(request):
    return PYTHON_API if request.param == "python" else COMPILED_API


def _array(values):
    return np.array(values, dtype=np.int64)


def test_compiled_functions_use_fastmath_off():
    for name in ("expiry_ns", "entry", "exit", "is_fresh"):
        assert not COMPILED_API.__dict__[name].targetoptions.get("fastmath", False)


@pytest.mark.parametrize("case", CASES["entry"], ids=lambda case: case["name"])
def test_entry_literal_cases(api, case):
    got = api.entry(
        _array(case["timestamps"]),
        _array(case["ordinals"]),
        case["begin"],
        case["end"],
        case["start_ns"],
        case["availability"],
    )
    assert got == case["expected"]


@pytest.mark.parametrize("case", CASES["exit"], ids=lambda case: case["name"])
def test_exit_literal_cases(api, case):
    got = api.exit(
        _array(case["timestamps"]), case["begin"], case["end"], case["expiry_ns"]
    )
    assert got == case["expected"]


@pytest.mark.parametrize("case", CASES["freshness"], ids=lambda case: case["name"])
def test_exit_freshness_literal_cases(api, case):
    timestamps = _array(case["timestamps"])
    index = api.exit(timestamps, 0, len(timestamps), case["expiry_ns"])
    assert index == case["expected_index"]
    assert api.is_fresh(int(timestamps[index]), case["expiry_ns"]) is case["fresh"]


def test_expiry_is_exact_at_both_int64_limits(api):
    assert (
        api.expiry_ns(OVERFLOW["last_start_that_fits"])
        == OVERFLOW["last_expiry_that_fits"]
    )
    assert api.expiry_ns(OVERFLOW["int64_min"]) == OVERFLOW["expiry_of_int64_min"]
    assert api.expiry_ns(CASES["contract_start_ns"]) == CASES["expiry_ns"]


@pytest.mark.parametrize(
    "start_ns",
    [
        OVERFLOW["first_start_that_overflows"],
        OVERFLOW["int64_max"] - 1,
        OVERFLOW["int64_max"],
    ],
    ids=repr,
)
def test_expiry_overflow_raises_instead_of_wrapping(api, start_ns):
    with pytest.raises(OverflowError, match="signed 64-bit"):
        api.expiry_ns(start_ns)


def test_entry_overflow_raises_instead_of_skipping_the_contract(api):
    start_ns = OVERFLOW["int64_max"] - 1  # the case that wrapped to a skipped contract
    timestamps = _array([OVERFLOW["int64_max"]])
    ordinals = _array([1])
    with pytest.raises(OverflowError, match="signed 64-bit"):
        api.entry(timestamps, ordinals, 0, 1, start_ns, 0)
    with pytest.raises(OverflowError, match="signed 64-bit"):
        api.entry(_array([]), _array([]), 0, 0, start_ns, 0)


@pytest.mark.parametrize(("begin", "end"), [(-1, 2), (0, 4), (3, 5)], ids=repr)
def test_range_outside_the_arrays_raises_index_error(api, begin, end):
    timestamps = _array([1, 2, 3])
    ordinals = _array([1, 2, 3])
    with pytest.raises(IndexError, match="outside the tick arrays"):
        api.entry(timestamps, ordinals, begin, end, 0, 0)
    with pytest.raises(IndexError, match="outside the tick arrays"):
        api.exit(timestamps, begin, end, 100)


def test_entry_range_longer_than_the_ordinals_raises_index_error(api):
    with pytest.raises(IndexError, match="outside the tick arrays"):
        api.entry(_array([1, 2]), _array([1]), 0, 2, 0, 0)


def test_random_cases_compiled_matches_python():
    rng = random.Random(20261028)
    start = CASES["contract_start_ns"]
    second = 1_000_000_000
    for case_number in range(RANDOM_CASES):
        count = rng.randint(0, 12)
        timestamps = sorted(
            start + rng.randint(-5, 130) * second + rng.randint(-1, 1)
            for _ in range(count)
        )
        ordinals = []
        ordinal = rng.randint(-3, 3)
        for _ in range(count):
            ordinal += rng.randint(1, 3)
            ordinals.append(ordinal)
        begin = rng.randint(0, count)
        end = rng.randint(begin, count)
        start_ns = start + rng.randint(-3, 8) * second + rng.randint(-1, 1)
        availability = rng.randint(-4, (ordinals[-1] if ordinals else 0) + 4)
        expiry = start_ns + rng.choice([60 * second, 0, 1, -1])
        ts, od = _array(timestamps), _array(ordinals)
        label = f"case {case_number}: {timestamps} {ordinals} [{begin},{end})"
        assert COMPILED_API.entry(
            ts, od, begin, end, start_ns, availability
        ) == PYTHON_API.entry(ts, od, begin, end, start_ns, availability), label
        assert COMPILED_API.exit(ts, begin, end, expiry) == PYTHON_API.exit(
            ts, begin, end, expiry
        ), label
