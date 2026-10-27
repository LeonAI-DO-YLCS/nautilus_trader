"""Shared tick selectors and the checked expiry.

The literal cases live in `fixtures/tick_selector_cases.json`; the research side compiles the same
selectors with Numba and runs the same cases. Expected values there are written by hand, never
computed by the code under test.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
from deriv_mt5.ticks import (
    entry_index_in_range,
    entry_ordinal,
    exit_index_in_range,
    exit_is_fresh,
    exit_ordinal,
    expiry_ns,
)

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "tick_selector_cases.json").read_text(
        encoding="utf-8"
    )
)
START_NS = CASES["contract_start_ns"]
EXPIRY_NS = CASES["expiry_ns"]
OVERFLOW = CASES["overflow"]
SECOND_NS = 1_000_000_000
CONTRACT_NS = 60 * SECOND_NS
RANDOM_CASES = 3000


class ReadLog(Sequence):
    """A sequence that records every index read, to prove a scan stays inside its range."""

    def __init__(self, values):
        self.values = list(values)
        self.reads = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.reads.append(index)
        return self.values[index]


def _legacy_entry(ticks, start_ns):
    """The list-based entry rule as it was before the selectors (kept here as the reference)."""
    expiry = start_ns + CONTRACT_NS
    for timestamp_ns, ordinal in ticks:
        if timestamp_ns >= start_ns:
            return ordinal if timestamp_ns <= expiry else None
    return None


def _legacy_exit(ticks, expiry):
    """The list-based exit rule as it was before the selectors (kept here as the reference)."""
    exit_row = None
    for timestamp_ns, ordinal in ticks:
        if timestamp_ns > expiry:
            break
        exit_row = ordinal
    return exit_row


@pytest.mark.parametrize("case", CASES["entry"], ids=lambda case: case["name"])
def test_entry_literal_cases(case):
    got = entry_index_in_range(
        case["timestamps"],
        case["ordinals"],
        case["begin"],
        case["end"],
        case["start_ns"],
        case["availability"],
    )
    assert got == case["expected"]


@pytest.mark.parametrize("case", CASES["exit"], ids=lambda case: case["name"])
def test_exit_literal_cases(case):
    got = exit_index_in_range(
        case["timestamps"], case["begin"], case["end"], case["expiry_ns"]
    )
    assert got == case["expected"]


@pytest.mark.parametrize("case", CASES["freshness"], ids=lambda case: case["name"])
def test_exit_freshness_literal_cases(case):
    timestamps = case["timestamps"]
    index = exit_index_in_range(timestamps, 0, len(timestamps), case["expiry_ns"])
    assert index == case["expected_index"]
    assert exit_is_fresh(timestamps[index], case["expiry_ns"]) is case["fresh"]


def test_literal_cases_are_the_documented_contract():
    assert expiry_ns(START_NS) == EXPIRY_NS == START_NS + 60 * SECOND_NS
    assert len(CASES["entry"]) >= 19
    assert len(CASES["exit"]) >= 10


@pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
def test_selectors_accept_any_sequence(container):
    timestamps = container([START_NS - 1, START_NS + 2, START_NS + 2])
    ordinals = container([1, 2, 3])
    assert entry_index_in_range(timestamps, ordinals, 0, 3, START_NS, 3) == 2
    assert exit_index_in_range(timestamps, 0, 3, START_NS + 2) == 2


def test_selectors_read_only_rows_inside_the_range():
    timestamps = ReadLog([START_NS + 1 + i for i in range(8)])
    ordinals = ReadLog(range(8))
    assert entry_index_in_range(timestamps, ordinals, 2, 6, START_NS, 4) == 4
    assert set(timestamps.reads) <= {2, 3, 4, 5}
    assert set(ordinals.reads) <= {2, 3, 4, 5}
    timestamps = ReadLog([START_NS + i for i in range(8)])
    assert exit_index_in_range(timestamps, 2, 6, EXPIRY_NS) == 5
    assert set(timestamps.reads) <= {2, 3, 4, 5}


def test_selectors_do_not_change_their_input():
    timestamps = [START_NS + 3, START_NS + 1, START_NS + 2]  # out of order on purpose
    ordinals = [3, 1, 2]
    entry_index_in_range(timestamps, ordinals, 0, 3, START_NS, 0)
    exit_index_in_range(timestamps, 0, 3, EXPIRY_NS)
    assert timestamps == [START_NS + 3, START_NS + 1, START_NS + 2]
    assert ordinals == [3, 1, 2]


@pytest.mark.parametrize(("begin", "end"), [(-1, 2), (0, 4), (3, 5)], ids=repr)
def test_selectors_reject_a_range_outside_the_arrays(begin, end):
    timestamps = [START_NS, START_NS, START_NS]
    ordinals = [1, 2, 3]
    with pytest.raises(IndexError, match="outside the tick arrays"):
        entry_index_in_range(timestamps, ordinals, begin, end, START_NS, 0)
    with pytest.raises(IndexError, match="outside the tick arrays"):
        exit_index_in_range(timestamps, begin, end, EXPIRY_NS)


def test_entry_rejects_a_range_longer_than_the_ordinals():
    with pytest.raises(IndexError, match="outside the tick arrays"):
        entry_index_in_range([START_NS, START_NS], [1], 0, 2, START_NS, 0)


# --- checked expiry -------------------------------------------------------------------------


def test_expiry_at_the_largest_start_that_fits_is_the_largest_int64():
    assert (
        expiry_ns(OVERFLOW["last_start_that_fits"]) == OVERFLOW["last_expiry_that_fits"]
    )
    assert OVERFLOW["last_expiry_that_fits"] == OVERFLOW["int64_max"]


def test_expiry_of_the_smallest_start_is_exact():
    assert expiry_ns(OVERFLOW["int64_min"]) == OVERFLOW["expiry_of_int64_min"]


@pytest.mark.parametrize(
    "start_ns",
    [
        OVERFLOW["first_start_that_overflows"],
        OVERFLOW["int64_max"] - 1,
        OVERFLOW["int64_max"],
    ],
    ids=repr,
)
def test_expiry_overflow_raises_instead_of_wrapping(start_ns):
    with pytest.raises(OverflowError, match="signed 64-bit"):
        expiry_ns(start_ns)


def test_expiry_overflow_message_is_constant():
    messages = set()
    for start_ns in (OVERFLOW["first_start_that_overflows"], OVERFLOW["int64_max"]):
        with pytest.raises(OverflowError) as raised:
            expiry_ns(start_ns)
        messages.add(str(raised.value))
    assert len(messages) == 1
    assert str(OVERFLOW["int64_max"]) not in messages.pop()


@pytest.mark.parametrize("start_ns", [1.5, 1.0, True, False], ids=repr)
def test_expiry_rejects_floats_and_bools_like_the_money_rules(start_ns):
    with pytest.raises(TypeError):
        expiry_ns(start_ns)


def test_entry_index_overflow_raises_even_with_no_rows():
    start_ns = OVERFLOW["first_start_that_overflows"]
    with pytest.raises(OverflowError, match="signed 64-bit"):
        entry_index_in_range([], [], 0, 0, start_ns, 0)
    with pytest.raises(OverflowError, match="signed 64-bit"):
        entry_index_in_range([OVERFLOW["int64_max"]], [1], 0, 1, start_ns, 0)


def test_entry_ordinal_overflow_raises_through_the_list_api():
    with pytest.raises(OverflowError, match="signed 64-bit"):
        entry_ordinal([(OVERFLOW["int64_max"], 1)], OVERFLOW["int64_max"])


# --- public list API keeps its behaviour ---------------------------------------------------


def test_entry_ordinal_returns_none_when_absent():
    assert entry_ordinal([], START_NS) is None
    assert entry_ordinal([(START_NS - 1, 1)], START_NS) is None
    assert entry_ordinal([(EXPIRY_NS + 1, 1)], START_NS) is None


def test_exit_ordinal_returns_none_when_absent():
    assert exit_ordinal([], EXPIRY_NS) is None
    assert exit_ordinal([(EXPIRY_NS + 1, 1)], EXPIRY_NS) is None


def test_entry_ordinal_puts_no_limit_on_existing_ordinals():
    assert entry_ordinal([(START_NS, -5)], START_NS) == -5
    assert entry_ordinal([(START_NS, 2**70)], START_NS) == 2**70
    assert entry_ordinal([(START_NS, -(2**70))], START_NS) == -(2**70)
    assert entry_ordinal([(START_NS, 9), (START_NS, 3)], START_NS) == 9


def test_exit_ordinal_puts_no_limit_on_existing_ordinals():
    assert exit_ordinal([(EXPIRY_NS, 9), (EXPIRY_NS, 3)], EXPIRY_NS) == 3
    assert exit_ordinal([(EXPIRY_NS, 2**70)], EXPIRY_NS) == 2**70


# --- seeded random cases --------------------------------------------------------------------


def _random_ticks(rng, sort):
    """Return (timestamps, ordinals): many equal times, some 1 ns off the contract edges."""
    count = rng.randint(0, 12)
    edges = [START_NS, EXPIRY_NS]
    timestamps = []
    for _ in range(count):
        if rng.random() < 0.3:
            timestamps.append(rng.choice(edges) + rng.randint(-1, 1))
        else:
            timestamps.append(START_NS + rng.randint(-5, 130) * SECOND_NS)
    if sort:
        timestamps.sort()
    ordinals = []
    ordinal = rng.randint(-3, 3)
    for _ in range(count):
        ordinal += rng.randint(1, 3)
        ordinals.append(ordinal)
    return timestamps, ordinals


def _random_start(rng, timestamps):
    if timestamps and rng.random() < 0.6:
        return rng.choice(timestamps) + rng.randint(-1, 1)
    return START_NS + rng.randint(-3, 8) * SECOND_NS + rng.randint(-1, 1)


def test_random_cases_match_the_preserved_list_api():
    rng = random.Random(20261026)
    for case_number in range(RANDOM_CASES):
        timestamps, ordinals = _random_ticks(rng, sort=rng.random() < 0.75)
        ticks = list(zip(timestamps, ordinals, strict=True))
        start_ns = _random_start(rng, timestamps)
        expiry = start_ns + CONTRACT_NS
        exit_expiry = rng.choice(
            [expiry, _random_start(rng, timestamps), START_NS + 60 * SECOND_NS]
        )
        label = f"case {case_number}: {ticks} start={start_ns} expiry={exit_expiry}"
        expected_entry = _legacy_entry(ticks, start_ns)
        expected_exit = _legacy_exit(ticks, exit_expiry)
        assert entry_ordinal(ticks, start_ns) == expected_entry, label
        assert exit_ordinal(ticks, exit_expiry) == expected_exit, label
        # The numeric selectors, with no availability limit, give the same rows.
        no_limit = min(ordinals, default=0)
        entry_index = entry_index_in_range(
            timestamps, ordinals, 0, len(ticks), start_ns, no_limit
        )
        entry_got = None if entry_index < 0 else ordinals[entry_index]
        assert entry_got == expected_entry, label
        exit_index = exit_index_in_range(timestamps, 0, len(ticks), exit_expiry)
        exit_got = None if exit_index < 0 else ordinals[exit_index]
        assert exit_got == expected_exit, label


def test_random_cases_match_the_availability_oracle():
    rng = random.Random(20261027)
    for case_number in range(RANDOM_CASES):
        timestamps, ordinals = _random_ticks(rng, sort=True)
        count = len(timestamps)
        begin = rng.randint(0, count)
        end = rng.randint(begin, count)
        start_ns = _random_start(rng, timestamps)
        availability = rng.randint(-4, (ordinals[-1] if ordinals else 0) + 4)
        exit_expiry = rng.choice(
            [start_ns + CONTRACT_NS, _random_start(rng, timestamps)]
        )
        label = (
            f"case {case_number}: ts={timestamps} ord={ordinals} range=[{begin},{end}) "
            f"start={start_ns} availability={availability} expiry={exit_expiry}"
        )
        entry_oracle = [
            i
            for i in range(begin, end)
            if timestamps[i] >= start_ns
            and ordinals[i] >= availability
            and timestamps[i] <= start_ns + CONTRACT_NS
        ]
        assert entry_index_in_range(
            timestamps, ordinals, begin, end, start_ns, availability
        ) == (entry_oracle[0] if entry_oracle else -1), label
        exit_oracle = [i for i in range(begin, end) if timestamps[i] <= exit_expiry]
        assert exit_index_in_range(timestamps, begin, end, exit_expiry) == (
            exit_oracle[-1] if exit_oracle else -1
        ), label
