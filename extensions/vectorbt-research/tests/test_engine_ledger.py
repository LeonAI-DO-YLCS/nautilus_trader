"""Ledger loop tests R-06, R-07 and R-08 (red until `vectorbt_research.engine` exists).

Engine API these tests set (later ledger tests R-09 to R-16 and the runner reuse it):

    from vectorbt_research.engine import run_ledger

    run_ledger(ticks, signals, *, scenario, starting_cash_cents) -> LedgerResult

- `ticks`: list of `(timestamp_ns, ordinal, bid, mid, ask)` in source order. All values are
  plain ints: UTC nanoseconds, the source row ordinal, and the three prices in the integer
  comparison units of `vectorbt_research.input` (`bid_half_cents`, `mid_half_cents`,
  `ask_half_cents`). Timestamps never decrease; equal-time rows keep source order.
- `signals`: list of dicts in frozen-list order with the R-03 keys `signal_id`, `direction`
  (`"CALL"` or `"PUT"`), `start_time_ns` and `availability_ordinal`. A signal may enter only
  on a tick whose ordinal is at least its `availability_ordinal` (it cannot see earlier rows).
- `scenario`: `"bid"`, `"mid"` or `"ask"` picks which price column is used for entry and exit.
- `starting_cash_cents`: int. The run starts with this much free cash and 0 locked.
- Returns `LedgerResult` with attributes:
  - `contracts`: one dict per signal in signal order, with the R-17 contract-row keys
    `signal_id`, `direction`, `price_reference` (the scenario), `status`, `reason`, `outcome`,
    `entry_ordinal`, `exit_ordinal`, `entry_timestamp_ns`, `expiry_timestamp_ns`,
    `exit_timestamp_ns`, `entry_spot`, `exit_spot` (the picked price, unchanged),
    `stake_cents`, `gross_credit_cents`, `net_cents`. `entry_spot` and `exit_spot` stay in
    the input half-cent units: the engine never converts them.
  - `cash`: one dict per cash transition, in order, with the R-17 cash-row keys `event`
    (`"initial"`, `"entry"` or `"settlement"`), `signal_id` (`""` for `"initial"`),
    `before_available_cents`, `before_locked_cents`, `available_cents`, `locked_cents`.
  - `skipped_count`, `unresolved_count`: ints. `incomplete`: bool. `incomplete_reason`: str,
    empty when `incomplete` is false (the R-14 test sets its content).
- Every number in `contracts` and `cash`, and both counts, is a plain Python `int`.
- Per timestamp group: consume every tick, then settle contracts that expire at or before
  the group (in admission order), then admit due entries in signal order. Exact reason strings are not approved;
  these tests only require a nonempty `reason` string.
"""

from vectorbt_research.engine import run_ledger

_T0 = 1_787_023_200_000_000_000  # 2026-08-18 03:20:00 UTC (S8 window start)
_SEC = 1_000_000_000


def _tick(offset_s, ordinal, price):
    """An aligned tick: bid, mid and ask carry the same price."""
    return (_T0 + offset_s * _SEC, ordinal, price, price, price)


def _signal(signal_id, direction, start_s, availability_ordinal):
    return {
        "signal_id": signal_id,
        "direction": direction,
        "start_time_ns": _T0 + start_s * _SEC,
        "availability_ordinal": availability_ordinal,
    }


def _settled(
    signal_id,
    direction,
    outcome,
    *,
    entry_ordinal,
    exit_ordinal,
    entry_s,
    expiry_s,
    exit_s,
    entry_spot,
    exit_spot,
    stake,
    gross,
    net,
):
    """Expected contract row without `reason`, built only from literal arguments."""
    return {
        "signal_id": signal_id,
        "direction": direction,
        "price_reference": "bid",
        "status": "settled",
        "outcome": outcome,
        "entry_ordinal": entry_ordinal,
        "exit_ordinal": exit_ordinal,
        "entry_timestamp_ns": _T0 + entry_s * _SEC,
        "expiry_timestamp_ns": _T0 + expiry_s * _SEC,
        "exit_timestamp_ns": _T0 + exit_s * _SEC,
        "entry_spot": entry_spot,
        "exit_spot": exit_spot,
        "stake_cents": stake,
        "gross_credit_cents": gross,
        "net_cents": net,
    }


def _cash(event, signal_id, before, after):
    """Expected cash row; `before` and `after` are (available, locked) cents."""
    return {
        "event": event,
        "signal_id": signal_id,
        "before_available_cents": before[0],
        "before_locked_cents": before[1],
        "available_cents": after[0],
        "locked_cents": after[1],
    }


def _assert_ledger(result, *, contracts, cash, label=""):
    """Exact equality on the full contract and cash lists, with plain-int types.

    `label` names the case in a failure message.
    """
    for row in result.contracts:
        assert isinstance(row["reason"], str)
        assert row["reason"].strip()
    for row in [*result.contracts, *result.cash]:
        for key, value in row.items():
            assert type(value) in (int, str), (key, type(value))
    assert type(result.skipped_count) is int
    assert type(result.unresolved_count) is int
    assert result.incomplete_reason == ""
    assert [
        {key: value for key, value in row.items() if key != "reason"}
        for row in result.contracts
    ] == contracts, label
    assert result.cash == cash, label
    assert result.skipped_count == 0
    assert result.unresolved_count == 0
    assert result.incomplete is False


def test_the_compiled_loop_debits_and_credits_each_contract_once():
    """the compiled loop debits and credits each contract once"""
    # C1: 1000000 free -> 5.00 stake -> 999500 free / 500 locked -> win 976 gross.
    # Ticks after expiry (ordinals 4 to 6) must add no second debit or credit.
    cases = [
        # (case, exit price, outcome, gross, net, free and locked cash after settlement)
        ("win", 20_400, "win", 976, 476, (1_000_476, 0)),
        ("loss", 19_800, "loss", 0, -500, (999_500, 0)),
        ("tie", 20_000, "loss", 0, -500, (999_500, 0)),
    ]
    for case, exit_price, outcome, gross, net, final in cases:
        ticks = [
            _tick(2, 1, 20_000),
            _tick(30, 2, 20_200),
            _tick(60, 3, exit_price),
            _tick(61, 4, 21_000),
            _tick(62, 5, 21_000),
            _tick(62, 6, 22_000),
        ]
        signals = [_signal("signal-1", "CALL", 0, 1)]

        result = run_ledger(
            ticks, signals, scenario="bid", starting_cash_cents=1_000_000
        )

        _assert_ledger(
            result,
            contracts=[
                _settled(
                    "signal-1",
                    "CALL",
                    outcome,
                    entry_ordinal=1,
                    exit_ordinal=3,
                    entry_s=2,
                    expiry_s=60,
                    exit_s=60,
                    entry_spot=20_000,
                    exit_spot=exit_price,
                    stake=500,
                    gross=gross,
                    net=net,
                )
            ],
            cash=[
                _cash("initial", "", (0, 0), (1_000_000, 0)),
                _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
                _cash("settlement", "signal-1", (999_500, 500), final),
            ],
            label=case,
        )


def test_ticks_then_due_settlements_then_entries_at_the_same_timestamp():
    """ticks, then due settlements, then entries, at the same timestamp"""
    # Signal 1 (CALL) expires at +60 s. Three ticks share that timestamp: ordinals 2, 3
    # and 4 at prices 19800, 20000 and 20600. Its exit is the last source ordinal (4,
    # a win); ordinal 2 would lose and ordinal 3 would tie. Signals 2 (PUT) and 3 (CALL)
    # both start at +60 s, listed in that order. Signal 2's stake must come from the cash
    # released by signal 1's win (1000476 -> 500), not from the earlier 999500 (which would
    # give 499). Signal 3 is admitted second (999976 -> 499), so reversed signal order
    # would swap the two stakes. Both expire at +120 s and settle in admission order.
    ticks = [
        _tick(2, 1, 20_000),
        _tick(60, 2, 19_800),
        _tick(60, 3, 20_000),
        _tick(60, 4, 20_600),
        _tick(120, 5, 19_600),
        _tick(121, 6, 19_000),
    ]
    signals = [
        _signal("signal-1", "CALL", 0, 1),
        _signal("signal-2", "PUT", 60, 1),
        _signal("signal-3", "CALL", 60, 1),
    ]

    result = run_ledger(ticks, signals, scenario="bid", starting_cash_cents=1_000_000)

    _assert_ledger(
        result,
        contracts=[
            _settled(
                "signal-1",
                "CALL",
                "win",
                entry_ordinal=1,
                exit_ordinal=4,
                entry_s=2,
                expiry_s=60,
                exit_s=60,
                entry_spot=20_000,
                exit_spot=20_600,
                stake=500,
                gross=976,
                net=476,
            ),
            _settled(
                "signal-2",
                "PUT",
                "win",
                entry_ordinal=2,
                exit_ordinal=5,
                entry_s=60,
                expiry_s=120,
                exit_s=120,
                entry_spot=19_800,
                exit_spot=19_600,
                stake=500,
                gross=976,
                net=476,
            ),
            _settled(
                "signal-3",
                "CALL",
                "loss",
                entry_ordinal=2,
                exit_ordinal=5,
                entry_s=60,
                expiry_s=120,
                exit_s=120,
                entry_spot=19_800,
                exit_spot=19_600,
                stake=499,
                gross=0,
                net=-499,
            ),
        ],
        cash=[
            _cash("initial", "", (0, 0), (1_000_000, 0)),
            _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
            _cash("settlement", "signal-1", (999_500, 500), (1_000_476, 0)),
            _cash("entry", "signal-2", (1_000_476, 0), (999_976, 500)),
            _cash("entry", "signal-3", (999_976, 500), (999_477, 999)),
            _cash("settlement", "signal-2", (999_477, 999), (1_000_453, 499)),
            _cash("settlement", "signal-3", (1_000_453, 499), (1_000_453, 0)),
        ],
    )


def test_tick_generated_signals_cannot_see_earlier_rows():
    """tick-generated signals cannot see earlier rows"""
    # Three rows share the start timestamp: ordinals 7, 8 and 9 at prices 20000, 20100
    # and 20200. A signal available on ordinal N enters on ordinal N (the first eligible
    # row), never on an earlier row of the same timestamp and never on a later one. The
    # exit price 20050 makes the outcome depend on the entry row: a win only from 7.
    cases = [
        # (availability ordinal, entry spot, outcome, gross, net, final cash)
        (7, 20_000, "win", 976, 476, (1_000_476, 0)),
        (8, 20_100, "loss", 0, -500, (999_500, 0)),
        (9, 20_200, "loss", 0, -500, (999_500, 0)),
    ]
    for availability_ordinal, entry_spot, outcome, gross, net, final in cases:
        ticks = [
            _tick(-1, 6, 19_900),
            _tick(0, 7, 20_000),
            _tick(0, 8, 20_100),
            _tick(0, 9, 20_200),
            _tick(30, 10, 20_500),
            _tick(60, 11, 20_050),
            _tick(61, 12, 21_000),
        ]
        signals = [_signal("signal-1", "CALL", 0, availability_ordinal)]

        result = run_ledger(
            ticks, signals, scenario="bid", starting_cash_cents=1_000_000
        )

        _assert_ledger(
            result,
            contracts=[
                _settled(
                    "signal-1",
                    "CALL",
                    outcome,
                    entry_ordinal=availability_ordinal,
                    exit_ordinal=11,
                    entry_s=0,
                    expiry_s=60,
                    exit_s=60,
                    entry_spot=entry_spot,
                    exit_spot=20_050,
                    stake=500,
                    gross=gross,
                    net=net,
                )
            ],
            cash=[
                _cash("initial", "", (0, 0), (1_000_000, 0)),
                _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
                _cash("settlement", "signal-1", (999_500, 500), final),
            ],
            label=f"availability ordinal {availability_ordinal}",
        )
