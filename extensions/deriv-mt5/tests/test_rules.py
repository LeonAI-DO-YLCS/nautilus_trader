"""Shared money and outcome rules, approved rows S-01 to S-08. Red until implemented."""

import json
from pathlib import Path

import pytest

from deriv_mt5 import rules

FIXTURE = Path(__file__).parent / "fixtures" / "money_outcome_cases.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_s01_stake_for_10000_free():
    """Given 10,000.00 USD free, when a 5.00 Rise trade is placed (0.05% of free cash)"""
    for case in CASES["S-01"]:
        assert rules.stake_cents(case["available"]) == case["stake"]


def test_s02_stake_for_9995_free_rounds_down():
    """Given 9,995.00 free, the stake is 4.99 (rounded down to cents)"""
    for case in CASES["S-02"]:
        assert rules.stake_cents(case["available"]) == case["stake"]


def test_s03_stake_minimum_is_0_35():
    """given 699.99 free, the stake is 0.35 (the minimum)"""
    for case in CASES["S-03"]:
        assert rules.stake_cents(case["available"]) == case["stake"]


def test_s04_no_trade_below_minimum():
    """given 0.34 free, no trade is placed and the reason is logged"""
    # Pure rule: no stake means no trade. The logged reason is checked at engine and report level.
    for case in CASES["S-04"]:
        assert rules.stake_cents(case["available"]) is None


def test_s05_win_credits_gross():
    """A win on a 4.99 stake credits 9.74 gross"""
    for case in CASES["S-05"]:
        assert rules.gross_credit_cents(case["stake"], won=True) == case["gross"]


def test_s06_loss_or_tie_credits_zero():
    """a loss or tie credits 0.00"""
    for case in CASES["S-06"]:
        won = rules.is_win(case["direction"], case["entry"], case["exit"])
        assert won is False
        assert rules.gross_credit_cents(case["stake"], won=won) == case["gross"]


def test_s07_direction_and_ties_lose():
    """CALL=Rise: exit > entry; PUT=Fall: exit < entry; ties lose"""
    for case in CASES["S-07"]:
        assert rules.is_win(case["direction"], case["entry"], case["exit"]) is case["win"]


def test_s08_guard_range_raises_no_wrap():
    """integer value beyond the guard range raises an error, no wrapped value"""
    g = CASES["S-08"]
    assert rules.checked_cents(g["int64_max"]) == g["int64_max"]
    assert rules.checked_cents(g["int64_min"]) == g["int64_min"]
    with pytest.raises(OverflowError):
        rules.checked_cents(g["int64_max_plus_1"])
    with pytest.raises(OverflowError):
        rules.checked_cents(g["int64_min_minus_1"])
    assert rules.gross_credit_cents(g["max_safe_stake"], won=True) == g["gross_of_max_safe_stake"]
    with pytest.raises(OverflowError):
        rules.gross_credit_cents(g["first_unsafe_stake"], won=True)
    assert rules.checked_add(g["free_before_credit_at_max"], g["credit"]) == g["int64_max"]
    with pytest.raises(OverflowError):
        rules.checked_add(g["free_one_cent_over"], g["credit"])
