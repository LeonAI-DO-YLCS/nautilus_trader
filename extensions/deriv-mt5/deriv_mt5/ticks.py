"""Timing rules for Deriv tick data: entry, exit and exit freshness.

Times are integer UTC nanoseconds. A tick is a (timestamp_ns, ordinal) pair. Ticks come in
source order: timestamps never decrease, and rows with equal timestamps keep their source order.
"""

_CONTRACT_NS = 60_000_000_000  # 1-minute contract
_FRESH_LIMIT_NS = 10_000_000_000  # an exit up to 10 s before expiry is fresh


def expiry_ns(start_ns: int) -> int:
    """Return the expiry time: the contract start plus 60 s."""
    return start_ns + _CONTRACT_NS


def entry_ordinal(ticks: list[tuple[int, int]], start_ns: int) -> int | None:
    """Return the ordinal of the first tick at or after the start.

    Return None when no tick is at or after the start, or when that tick is after expiry.
    """
    expiry = expiry_ns(start_ns)
    for timestamp_ns, ordinal in ticks:
        if timestamp_ns >= start_ns:
            return ordinal if timestamp_ns <= expiry else None
    return None


def exit_ordinal(ticks: list[tuple[int, int]], expiry_ns: int) -> int | None:
    """Return the ordinal of the last tick at or before expiry.

    On equal times the last row in source order wins.
    Return None when no tick is at or before expiry.
    """
    exit_row = None
    for timestamp_ns, ordinal in ticks:
        if timestamp_ns > expiry_ns:
            break
        exit_row = ordinal
    return exit_row


def exit_is_fresh(exit_ns: int, expiry_ns: int) -> bool:
    """Return True when the exit is at most 10 s before expiry.

    An exit after expiry is never fresh: it is not a valid expiry price.
    """
    age_ns = expiry_ns - exit_ns
    return 0 <= age_ns <= _FRESH_LIMIT_NS
