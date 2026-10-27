"""Regression guards: integer-only inputs and signed 64-bit limits for the shared rules.

Extends tests/test_rules.py, which stays unchanged. Some tests are red until the guards exist.
"""

import pytest
from deriv_mt5 import rules


@pytest.mark.parametrize("value", [1.5, 1.0, True, False, "500"], ids=repr)
def test_checked_cents_rejects_non_int(value):
    """Cent amounts must be int. Floats, bools and strings raise TypeError, never coerce."""
    with pytest.raises(TypeError):
        rules.checked_cents(value)


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1], ids=repr)
def test_checked_cents_rejects_out_of_signed_64_bit_range(value):
    """Values outside the signed 64-bit range raise OverflowError."""
    with pytest.raises(OverflowError):
        rules.checked_cents(value)


@pytest.mark.parametrize("available", [1000000.0, 999500.0, 34.0, True], ids=repr)
def test_stake_cents_rejects_non_int(available):
    """A float such as 1000000.0 raises TypeError: no float stake, no early None."""
    with pytest.raises(TypeError):
        rules.stake_cents(available)


@pytest.mark.parametrize("available", [2**63, -(2**63) - 1], ids=repr)
def test_stake_cents_rejects_out_of_signed_64_bit_range(available):
    """Out-of-range cash raises OverflowError, including on the below-minimum path."""
    with pytest.raises(OverflowError):
        rules.stake_cents(available)


@pytest.mark.parametrize("stake", [499.5, 500.0, True], ids=repr)
@pytest.mark.parametrize("won", [True, False])
def test_gross_credit_cents_rejects_non_int_stake(stake, won):
    """The stake is checked before the win/loss branch, so a loss cannot hide a bad stake."""
    with pytest.raises(TypeError):
        rules.gross_credit_cents(stake, won=won)


@pytest.mark.parametrize("stake", [2**63, -(2**63) - 1], ids=repr)
def test_gross_credit_cents_rejects_out_of_range_stake_on_loss(stake):
    """An out-of-range stake raises OverflowError even when the trade lost (early return)."""
    with pytest.raises(OverflowError):
        rules.gross_credit_cents(stake, won=False)


def test_gross_credit_cents_keeps_largest_safe_win_stake():
    """The largest stake whose win credit fits in signed 64-bit gives the exact credit."""
    assert rules.gross_credit_cents(4725088133634618, won=True) == 9223372036854774


@pytest.mark.parametrize("stake", [4725088133634619, 2**63 - 1, 2**63], ids=repr)
def test_gross_credit_cents_rejects_unsafe_win_stake(stake):
    """One cent above the safe stake limit raises OverflowError."""
    with pytest.raises(OverflowError):
        rules.gross_credit_cents(stake, won=True)


def test_gross_credit_cents_keeps_largest_safe_negative_stake():
    """The safe range is symmetric: the largest safe negative stake gives the exact floored credit."""
    assert rules.gross_credit_cents(-4725088133634618, won=True) == -9223372036854775


@pytest.mark.parametrize("stake", [-4725088133634619, -(2**63)], ids=repr)
def test_gross_credit_cents_rejects_unsafe_negative_stake(stake):
    """The safe range is symmetric, so a large negative stake also raises OverflowError."""
    with pytest.raises(OverflowError):
        rules.gross_credit_cents(stake, won=True)


@pytest.mark.parametrize(
    ("entry", "exit_cents"),
    [(100.1, 100.2), (100, 101.0), (True, 2), (100, True)],
    ids=repr,
)
def test_is_win_rejects_non_int_prices(entry, exit_cents):
    """Prices are integer cents. A float or bool price raises TypeError, never compares."""
    with pytest.raises(TypeError):
        rules.is_win("CALL", entry, exit_cents)


@pytest.mark.parametrize(
    ("entry", "exit_cents"), [(0, 2**63), (-(2**63) - 1, 0)], ids=repr
)
def test_is_win_rejects_out_of_signed_64_bit_range(entry, exit_cents):
    """Prices outside the signed 64-bit range raise OverflowError."""
    with pytest.raises(OverflowError):
        rules.is_win("CALL", entry, exit_cents)


def test_is_win_checks_prices_before_direction():
    """Price checks run first: a bad price raises TypeError even with an unknown direction."""
    with pytest.raises(TypeError):
        rules.is_win("XX", 100.1, 100)


@pytest.mark.parametrize("direction", ["call", "put", "Rise"], ids=repr)
def test_is_win_rejects_unknown_direction(direction):
    """Only the exact strings CALL and PUT are accepted; other values raise ValueError."""
    with pytest.raises(ValueError):
        rules.is_win(direction, 100, 101)


@pytest.mark.parametrize("a, b", [(1.5, 2), (2, 1.5), (True, 1), (1, False)], ids=repr)
def test_checked_add_rejects_non_int_operands(a, b):
    """Both addition operands must be int. A float operand raises TypeError, never a float sum."""
    with pytest.raises(TypeError):
        rules.checked_add(a, b)


@pytest.mark.parametrize("a, b", [(2**63, -1), (1, 2**63), (-(2**63) - 1, 1)], ids=repr)
def test_checked_add_rejects_out_of_range_operands(a, b):
    """Operands outside the signed 64-bit range raise OverflowError."""
    with pytest.raises(OverflowError):
        rules.checked_add(a, b)


def test_checked_add_keeps_signed_upper_limit():
    """A sum that lands exactly on the signed 64-bit maximum is returned."""
    assert rules.checked_add(9223372036854774831, 976) == 9223372036854775807


@pytest.mark.parametrize(
    "a, b",
    [(9223372036854774832, 976), (9223372036854775807, 1)],
    ids=repr,
)
def test_checked_add_rejects_sum_above_signed_max(a, b):
    """A sum above the signed 64-bit maximum raises OverflowError."""
    with pytest.raises(OverflowError):
        rules.checked_add(a, b)


def test_checked_add_keeps_signed_lower_limit():
    """A sum that lands exactly on the signed 64-bit minimum is returned."""
    assert rules.checked_add(-9223372036854775808, 1) == -9223372036854775807
    assert rules.checked_add(-9223372036854775807, -1) == -9223372036854775808


@pytest.mark.parametrize(
    "a, b",
    [(-9223372036854775808, -1), (-9223372036854775000, -1000)],
    ids=repr,
)
def test_checked_add_rejects_sum_below_signed_min(a, b):
    """A sum below the signed 64-bit minimum raises OverflowError."""
    with pytest.raises(OverflowError):
        rules.checked_add(a, b)
