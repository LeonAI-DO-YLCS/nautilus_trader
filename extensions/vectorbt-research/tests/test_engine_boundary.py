"""Regression tests for the ledger engine boundary and tick-input findings (Engram #6790).

They cover an entry exactly at expiry, bool values in any tick field, and an expiry past the
signed 64-bit range, all on the Python path. The compiled path gets its own overflow test
in the compiled-loop tests.
"""

import pytest
from vectorbt_research.engine import run_ledger

_EXPIRY_NS = 60_000_000_000  # start 0 plus 60 s


def _signal(start_time_ns=0, availability_ordinal=1):
    """A CALL signal named "s"."""
    return {
        "signal_id": "s",
        "direction": "CALL",
        "start_time_ns": start_time_ns,
        "availability_ordinal": availability_ordinal,
    }


def _run(ticks, signals=None):
    """Run the ledger on the bid scenario with 10,000.00 USD of starting cash."""
    return run_ledger(
        ticks,
        [_signal()] if signals is None else signals,
        scenario="bid",
        starting_cash_cents=1_000_000,
    )


def _cash_row(event, signal_id, before, after):
    """Expected cash row; `before` and `after` are (available, locked) cents."""
    return {
        "event": event,
        "signal_id": signal_id,
        "before_available_cents": before[0],
        "before_locked_cents": before[1],
        "available_cents": after[0],
        "locked_cents": after[1],
    }


def _assert_single_settled(result, contract, cash):
    """Exact equality on one settled contract and its cash rows.

    Reason wording is not approved yet, so each reason is only required to be nonempty.
    """
    assert len(result.contracts) == 1
    assert result.contracts[0]["reason"].strip()
    assert {k: v for k, v in result.contracts[0].items() if k != "reason"} == contract
    assert result.cash == cash
    assert result.skipped_count == 0
    assert result.unresolved_count == 0
    assert result.incomplete is False
    assert result.incomplete_reason == ""


# Entry and exit both at bid 20000: the tie loses. Stake 500, no credit, net -500.
_TIE_CONTRACT = {
    "signal_id": "s",
    "direction": "CALL",
    "price_reference": "bid",
    "status": "settled",
    "outcome": "loss",
    "entry_ordinal": 1,
    "exit_ordinal": 1,
    "entry_timestamp_ns": _EXPIRY_NS,
    "expiry_timestamp_ns": _EXPIRY_NS,
    "exit_timestamp_ns": _EXPIRY_NS,
    "entry_spot": 20_000,
    "exit_spot": 20_000,
    "stake_cents": 500,
    "gross_credit_cents": 0,
    "net_cents": -500,
}
_TIE_CASH = [
    _cash_row("initial", "", (0, 0), (1_000_000, 0)),
    _cash_row("entry", "s", (1_000_000, 0), (999_500, 500)),
    _cash_row("settlement", "s", (999_500, 500), (999_500, 0)),
]

# Entry at bid 20000, exit at the group's last row, bid 20200: a win. Stake 500, gross 976.
_WIN_CONTRACT = {
    "signal_id": "s",
    "direction": "CALL",
    "price_reference": "bid",
    "status": "settled",
    "outcome": "win",
    "entry_ordinal": 1,
    "exit_ordinal": 3,
    "entry_timestamp_ns": _EXPIRY_NS,
    "expiry_timestamp_ns": _EXPIRY_NS,
    "exit_timestamp_ns": _EXPIRY_NS,
    "entry_spot": 20_000,
    "exit_spot": 20_200,
    "stake_cents": 500,
    "gross_credit_cents": 976,
    "net_cents": 476,
}
_WIN_CASH = [
    _cash_row("initial", "", (0, 0), (1_000_000, 0)),
    _cash_row("entry", "s", (1_000_000, 0), (999_500, 500)),
    _cash_row("settlement", "s", (999_500, 500), (1_000_476, 0)),
]


def test_an_entry_at_expiry_on_the_only_tick_settles_as_a_loss():
    """An entry exactly at expiry settles on its own tick, and a tie loses."""
    result = _run([(_EXPIRY_NS, 1, 20_000, 20_000, 20_000)])

    _assert_single_settled(result, _TIE_CONTRACT, _TIE_CASH)


def test_a_later_tick_leaves_an_entry_at_expiry_result_unchanged():
    """A tick after expiry must not change the result of an entry at expiry."""
    tick = (_EXPIRY_NS, 1, 20_000, 20_000, 20_000)
    later_tick = (61_000_000_000, 2, 20_000, 20_000, 20_000)
    alone = _run([tick])
    with_later = _run([tick, later_tick])

    assert with_later == alone
    _assert_single_settled(with_later, _TIE_CONTRACT, _TIE_CASH)


def test_an_entry_at_expiry_exits_on_the_last_row_of_its_group():
    """Entry at expiry takes the first eligible row, exit the group's last row."""
    # Only the last row (20200) beats the entry (20000). Exiting on the first row ties,
    # and exiting on the middle row (19900) loses.
    result = _run(
        [
            (_EXPIRY_NS, 1, 20_000, 20_000, 20_000),
            (_EXPIRY_NS, 2, 19_900, 19_900, 19_900),
            (_EXPIRY_NS, 3, 20_200, 20_200, 20_200),
        ]
    )

    _assert_single_settled(result, _WIN_CONTRACT, _WIN_CASH)


@pytest.mark.parametrize(
    "tick",
    [
        pytest.param((True, 1, 20_000, 20_000, 20_000), id="timestamp"),
        pytest.param((_EXPIRY_NS, True, 20_000, 20_000, 20_000), id="ordinal"),
        pytest.param((_EXPIRY_NS, 1, True, 20_000, 20_000), id="bid"),
        pytest.param((_EXPIRY_NS, 1, 20_000, True, 20_000), id="mid"),
        pytest.param((_EXPIRY_NS, 1, 20_000, 20_000, True), id="ask"),
    ],
)
def test_a_bool_in_any_tick_field_raises_type_error(tick):
    """A bool in any tick field raises TypeError, not a silent 0 or 1."""
    with pytest.raises(TypeError):
        _run([tick])


def test_an_expiry_past_the_signed_64_bit_range_raises_overflow_error():
    """An expiry past the int64 range raises OverflowError instead of wrapping."""
    # Python path only for now. The compiled path gets its own overflow test later.
    start = 9_223_372_036_854_775_806  # 2**63 - 2: start plus 60 s does not fit int64
    ticks = [(start, 1, 20_000, 20_000, 20_000)]

    with pytest.raises(OverflowError):
        _run(ticks, [_signal(start_time_ns=start)])
