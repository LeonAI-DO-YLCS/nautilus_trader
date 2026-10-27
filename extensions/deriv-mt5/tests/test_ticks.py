"""Shared timing rules, approved rows S-09 to S-14. Red until implemented."""

from deriv_mt5.ticks import entry_ordinal, exit_is_fresh, exit_ordinal, expiry_ns

SECOND_NS = 1_000_000_000
# 2026-08-18 12:00:00 PM UTC, the contract start (1787054400 s since the Unix epoch).
START_NS = 1_787_054_400_000_000_000


def at(offset_s):
    """UTC nanoseconds at t+offset_s seconds from the start (t-1 is offset -1)."""
    return START_NS + offset_s * SECOND_NS


def test_s09_entry_first_price_at_or_after_start_and_expiry_one_minute_later():
    """entry is the first price at or after the start (12:00:02 PM), expiry is 12:01:00 PM"""
    ticks = [
        (at(-1), 1),  # before the start: never the entry
        (at(2), 2),  # 12:00:02 PM: first price at or after the start
        (at(3), 3),  # later price: not the entry
        (at(62), 4),  # after expiry: never used
    ]
    assert expiry_ns(START_NS) == at(60)  # 12:01:00 PM
    assert entry_ordinal(ticks, START_NS) == 2


def test_s10_exit_is_last_price_at_or_before_expiry():
    """exit is the last price at or before expiry"""
    expiry = at(60)
    ticks = [
        (at(58), 1),
        (at(60), 2),  # at expiry: eligible, and the last eligible price
        (at(61), 3),  # after expiry: never the exit
    ]
    assert exit_ordinal(ticks, expiry) == 2


def test_s11_expiry_price_ten_seconds_old_or_less_settles():
    """an expiry price 10 s old or less settles"""
    expiry = at(60)
    assert exit_is_fresh(at(60), expiry) is True  # age 0 s
    assert exit_is_fresh(at(50), expiry) is True  # age 10 s, the limit


def test_s12_expiry_price_older_than_ten_seconds_stays_unresolved():
    """one older than 10 s stays unresolved, locked and flagged"""
    expiry = at(60)
    ticks = [(at(49), 1)]  # last price at or before expiry, age 11 s
    assert exit_ordinal(ticks, expiry) == 1
    assert exit_is_fresh(at(49), expiry) is False
    # Engine-side part (stake stays locked, flag is kept) is tested at engine level, not here.


def test_s13_first_price_after_expiry_means_no_trade():
    """a first price after expiry means no trade"""
    ticks = [
        (at(-1), 1),  # last price before the start
        (at(61), 2),  # first price after expiry (12:01:00 PM)
    ]
    assert entry_ordinal(ticks, START_NS) is None
    # Engine-side part (no purchase, no debit, persisted late-entry reason) is tested at engine level.


def test_s14_equal_time_rows_keep_source_order():
    """equal-time rows keep source order"""
    # Entry: same-time rows at the start; the first in source order, ordinal 7, is chosen.
    entry_ticks = [(at(2), 7), (at(2), 8), (at(2), 9)]
    assert entry_ordinal(entry_ticks, START_NS) == 7
    # Exit: same-time rows at expiry; the last in source order, ordinal 9, is chosen.
    exit_ticks = [(at(60), 7), (at(60), 8), (at(60), 9)]
    assert exit_ordinal(exit_ticks, at(60)) == 9
