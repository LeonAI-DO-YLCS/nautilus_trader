"""Installed-package Python and Numba parity for the shared Deriv money rules (approved row R-02).

The same installed `deriv_mt5.rules` functions run in plain Python and compiled with Numba. The
adapter side owns the Numba setup: it registers the original `checked_cents` helper with
`register_jitable`, then compiles the original public functions with `fastmath=False`. The shared
package stays stdlib-only. Compiled callers unwrap the optional stake with `int(stake)` after the
`None` check, and plain-Python callers convert array scalars with `int(value)`.
"""

import importlib.metadata
import json
import types
from pathlib import Path

import numba
import numpy as np
import pytest
from deriv_mt5 import rules
from numba.extending import register_jitable

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "deriv-mt5"
    / "tests"
    / "fixtures"
    / "money_outcome_cases.json"
)
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))
BOUNDS = CASES["S-08"]

register_jitable(rules.checked_cents)
_jit = numba.njit(fastmath=False)


def _identity(function):
    return function


def _make_callers(stake_cents, gross_credit_cents, is_win, checked_add, jit):
    """Build the callers once so plain Python and Numba run the same source."""

    def settle_trade(available, direction, entry, exit_):
        stake = stake_cents(available)
        if stake is None:
            return available
        concrete_stake = int(stake)
        credit = gross_credit_cents(concrete_stake, is_win(direction, entry, exit_))
        return checked_add(available - concrete_stake, credit)

    settle_trade = jit(settle_trade)

    def run_sequence(available, entries, exits):
        for i in range(entries.shape[0]):
            available = settle_trade(available, "CALL", int(entries[i]), int(exits[i]))
        return available

    def accumulate(start, steps):
        total = start
        for i in range(steps.shape[0]):
            total = checked_add(total, int(steps[i]))
        return total

    return settle_trade, jit(run_sequence), jit(accumulate)


def _api(stake_cents, gross_credit_cents, is_win, checked_cents, checked_add, jit):
    settle_trade, run_sequence, accumulate = _make_callers(
        stake_cents, gross_credit_cents, is_win, checked_add, jit
    )
    return types.SimpleNamespace(
        stake_cents=stake_cents,
        gross_credit_cents=gross_credit_cents,
        is_win=is_win,
        checked_cents=checked_cents,
        checked_add=checked_add,
        settle_trade=settle_trade,
        run_sequence=run_sequence,
        accumulate=accumulate,
    )


PYTHON_API = _api(
    rules.stake_cents,
    rules.gross_credit_cents,
    rules.is_win,
    rules.checked_cents,
    rules.checked_add,
    _identity,
)
COMPILED_API = _api(
    _jit(rules.stake_cents),
    _jit(rules.gross_credit_cents),
    _jit(rules.is_win),
    _jit(rules.checked_cents),
    _jit(rules.checked_add),
    _jit,
)


@pytest.fixture(params=["python", "compiled"])
def api(request):
    return PYTHON_API if request.param == "python" else COMPILED_API


def test_deriv_mt5_comes_from_an_installed_distribution():
    assert importlib.metadata.version("deriv-mt5") == "0.1.0"


def test_compiled_functions_use_fastmath_off():
    for name in (
        "stake_cents",
        "gross_credit_cents",
        "is_win",
        "checked_cents",
        "checked_add",
    ):
        assert not COMPILED_API.__dict__[name].targetoptions.get("fastmath", False)


@pytest.mark.parametrize("row", ["S-01", "S-02", "S-03", "S-04"])
def test_stake_rows_s01_to_s04(api, row):
    for case in CASES[row]:
        assert api.stake_cents(case["available"]) == case["stake"]


def test_stake_s04_below_minimum_is_none(api):
    assert api.stake_cents(34) is None


def test_gross_rows_s05_including_68_cent_gross(api):
    for case in CASES["S-05"]:
        assert api.gross_credit_cents(case["stake"], True) == case["gross"]
    assert api.gross_credit_cents(35, True) == 68


def test_gross_rows_s06_loss_and_tie_credit_nothing(api):
    for case in CASES["S-06"]:
        won = api.is_win(case["direction"], case["entry"], case["exit"])
        assert api.gross_credit_cents(case["stake"], won) == case["gross"]


def test_all_six_outcomes_s07(api):
    for case in CASES["S-07"]:
        assert api.is_win(case["direction"], case["entry"], case["exit"]) is case["win"]


def test_signed_bounds_are_kept_exactly(api):
    assert api.checked_cents(BOUNDS["int64_min"]) == BOUNDS["int64_min"]
    assert api.checked_cents(BOUNDS["int64_max"]) == BOUNDS["int64_max"]
    assert (
        api.checked_add(BOUNDS["free_before_credit_at_max"], BOUNDS["credit"])
        == BOUNDS["int64_max"]
    )
    assert api.checked_add(BOUNDS["int64_min"], 1) == -9223372036854775807
    assert api.checked_add(-9223372036854775807, -1) == BOUNDS["int64_min"]
    assert (
        api.gross_credit_cents(BOUNDS["max_safe_stake"], True)
        == BOUNDS["gross_of_max_safe_stake"]
    )
    assert api.gross_credit_cents(-BOUNDS["max_safe_stake"], True) == -9223372036854775


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (9223372036854775807, 1),
        (9223372036854774832, 976),
        (-9223372036854775808, -1),
        (-9223372036854775000, -1000),
    ],
    ids=repr,
)
def test_checked_add_overflow_on_both_signs(api, a, b):
    with pytest.raises(OverflowError):
        api.checked_add(a, b)


@pytest.mark.parametrize(
    "stake", [4725088133634619, 9223372036854775807, -4725088133634619], ids=repr
)
def test_gross_credit_cents_unsafe_win_stake_overflows(api, stake):
    with pytest.raises(OverflowError):
        api.gross_credit_cents(stake, True)


def test_gross_credit_cents_unsafe_stake_still_wins_nothing_on_a_loss(api):
    assert api.gross_credit_cents(4725088133634619, False) == 0


@pytest.mark.parametrize("value", [1.5, 1.0, True, False], ids=repr)
def test_checked_cents_rejects_float_and_bool(api, value):
    with pytest.raises(TypeError):
        api.checked_cents(value)


@pytest.mark.parametrize("available", [1000000.0, 34.0, True], ids=repr)
def test_stake_cents_rejects_float_and_bool(api, available):
    with pytest.raises(TypeError):
        api.stake_cents(available)


@pytest.mark.parametrize("stake", [500.0, True], ids=repr)
@pytest.mark.parametrize("won", [True, False])
def test_gross_credit_cents_rejects_float_and_bool_stake(api, stake, won):
    with pytest.raises(TypeError):
        api.gross_credit_cents(stake, won)


@pytest.mark.parametrize(
    ("entry", "exit_cents"),
    [(100.1, 100.2), (100, 101.0), (True, 2), (100, True)],
    ids=repr,
)
def test_is_win_rejects_float_and_bool_prices_before_direction(api, entry, exit_cents):
    with pytest.raises(TypeError):
        api.is_win("CALL", entry, exit_cents)
    with pytest.raises(TypeError):
        api.is_win("XX", entry, exit_cents)


@pytest.mark.parametrize(
    ("a", "b"), [(1.5, 2), (2, 1.5), (True, 1), (1, False)], ids=repr
)
def test_checked_add_rejects_float_and_bool_operands(api, a, b):
    with pytest.raises(TypeError):
        api.checked_add(a, b)


@pytest.mark.parametrize("direction", ["call", "put", "Rise", "XX"], ids=repr)
def test_is_win_rejects_unknown_direction(api, direction):
    with pytest.raises(ValueError):
        api.is_win(direction, 100, 101)


def test_caller_winning_trade_credits_gross(api):
    assert api.settle_trade(1000000, "CALL", 100, 101) == 1000476


def test_caller_no_trade_path_keeps_cash(api):
    assert api.settle_trade(34, "CALL", 100, 101) == 34


@pytest.mark.parametrize(
    ("direction", "entry", "exit_cents", "after"),
    [
        ("CALL", 100, 99, 999500),
        ("CALL", 100, 100, 999500),
        ("PUT", 100, 101, 999500),
        ("PUT", 100, 99, 1000476),
    ],
    ids=repr,
)
def test_caller_loss_tie_and_put_win(api, direction, entry, exit_cents, after):
    assert api.settle_trade(1000000, direction, entry, exit_cents) == after


def test_caller_sequence_over_int64_arrays(api):
    entries = np.array([100, 100, 101], dtype=np.int64)
    exits = np.array([101, 100, 99], dtype=np.int64)
    assert api.run_sequence(1000000, entries, exits) == 999477


def test_caller_accumulate_reaches_the_signed_limits_exactly(api):
    assert (
        api.accumulate(9223372036854774831, np.array([976], dtype=np.int64))
        == BOUNDS["int64_max"]
    )
    assert (
        api.accumulate(-9223372036854775807, np.array([-1], dtype=np.int64))
        == BOUNDS["int64_min"]
    )


def test_caller_accumulate_raises_overflow_at_min_plus_minus_one(api):
    with pytest.raises(OverflowError):
        api.accumulate(BOUNDS["int64_min"], np.array([-1], dtype=np.int64))


def test_caller_accumulate_raises_overflow_at_max_plus_one(api):
    with pytest.raises(OverflowError):
        api.accumulate(BOUNDS["int64_max"], np.array([1], dtype=np.int64))
