"""Timing rules for Deriv tick data: entry, exit and exit freshness.

Times are integer UTC nanoseconds. A tick is a (timestamp_ns, ordinal) pair. Ticks come in
source order: timestamps never decrease, and rows with equal timestamps keep their source order.

`entry_index_in_range` and `exit_index_in_range` hold the selection rules. They scan parallel
sequences (Python lists or int64 arrays) over a half-open row range with plain loops, never sort
or copy, and return a row index, or -1 when no row qualifies. Exception messages are constant
strings so Numba can compile these same functions: the vectorbt adapter registers `expiry_ns`
(and `checked_cents` and `checked_add` from `rules`) with `register_jitable` and compiles the
selectors with `fastmath=False`. The package stays stdlib-only.

`entry_ordinal` and `exit_ordinal` are the list-based wrappers over the same selectors.
"""

from collections.abc import Sequence

from deriv_mt5.rules import checked_add

_CONTRACT_NS = 60_000_000_000  # 1-minute contract
_FRESH_LIMIT_NS = 10_000_000_000  # an exit up to 10 s before expiry is fresh


def expiry_ns(start_ns: int) -> int:
    """Return the expiry time: the contract start plus 60 s.

    Raise OverflowError instead of wrapping when the expiry does not fit a signed 64-bit integer.
    Like the money rules, floats and bools raise TypeError.
    """
    return checked_add(start_ns, _CONTRACT_NS)


def entry_index_in_range(
    timestamps: Sequence[int],
    ordinals: Sequence[int],
    begin: int,
    end: int,
    start_ns: int,
    availability_ordinal: int,
) -> int:
    """Return the entry row in the half-open range [begin, end), or -1 when there is none.

    The entry is the first row with timestamp at or after the start, ordinal at or after
    `availability_ordinal`, and timestamp at or before the start's expiry. Only rows in the range
    are read. Raise OverflowError when the expiry does not fit a signed 64-bit integer and
    IndexError when the range is outside either array.
    """
    if begin < 0 or end > len(timestamps) or end > len(ordinals):
        raise IndexError("tick range is outside the tick arrays")
    expiry = expiry_ns(start_ns)
    for row in range(begin, end):
        timestamp_ns = timestamps[row]
        if timestamp_ns < start_ns:
            continue
        if timestamp_ns > expiry:
            return -1  # timestamps never decrease: no later row can be in time
        if ordinals[row] >= availability_ordinal:
            return row
    return -1


def exit_index_in_range(
    timestamps: Sequence[int], begin: int, end: int, expiry_ns: int
) -> int:
    """Return the exit row in the half-open range [begin, end), or -1 when there is none.

    The exit is the last row with timestamp at or before expiry; on equal times the last row in
    source order wins. Only rows in the range are read. Raise IndexError when the range is outside
    the array.
    """
    if begin < 0 or end > len(timestamps):
        raise IndexError("tick range is outside the tick arrays")
    exit_row = -1
    for row in range(begin, end):
        if timestamps[row] > expiry_ns:
            break
        exit_row = row
    return exit_row


def entry_ordinal(ticks: list[tuple[int, int]], start_ns: int) -> int | None:
    """Return the ordinal of the first tick at or after the start.

    Return None when no tick is at or after the start, or when that tick is after expiry.
    """
    timestamps, ordinals = _split(ticks)
    # This API has no availability limit, so use the smallest ordinal: no row is filtered out.
    index = entry_index_in_range(
        timestamps, ordinals, 0, len(ordinals), start_ns, min(ordinals, default=0)
    )
    return None if index < 0 else ordinals[index]


def exit_ordinal(ticks: list[tuple[int, int]], expiry_ns: int) -> int | None:
    """Return the ordinal of the last tick at or before expiry.

    On equal times the last row in source order wins.
    Return None when no tick is at or before expiry.
    """
    timestamps, ordinals = _split(ticks)
    index = exit_index_in_range(timestamps, 0, len(timestamps), expiry_ns)
    return None if index < 0 else ordinals[index]


def exit_is_fresh(exit_ns: int, expiry_ns: int) -> bool:
    """Return True when the exit is at most 10 s before expiry.

    An exit after expiry is never fresh: it is not a valid expiry price.
    """
    age_ns = expiry_ns - exit_ns
    return 0 <= age_ns <= _FRESH_LIMIT_NS


def _split(ticks: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Return the timestamps and the ordinals of (timestamp_ns, ordinal) pairs, in order."""
    timestamps = []
    ordinals = []
    for timestamp_ns, ordinal in ticks:
        timestamps.append(timestamp_ns)
        ordinals.append(ordinal)
    return timestamps, ordinals
