"""Ledger loop tests R-06 to R-16.

Engine API these tests set (the runner reuses it; R-13 to R-16 add `window_end_ns`,
`compiled` and `run_ledger_chunks`, described above R-13 below):

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


# --- R-09 to R-12: capacity, full-win balance, unresolved contracts, scenarios --------------
#
# Row shapes these tests set for contracts that never settled (the ledger fixes the 0 and ""
# placeholders; exact `reason` wording is not approved, only that it is a nonempty string):
#
# - Skipped contract (no purchase): `status` "skipped", `outcome` "", every ordinal, timestamp,
#   spot, stake, gross and net is 0. It adds no cash row and counts once in `skipped_count`.
# - Unresolved contract (stale expiry price): `status` "unresolved", `outcome` "", its real
#   entry ordinal, entry time, entry spot and stake, its real expiry time, and 0 for the exit
#   ordinal, exit time, exit spot, gross and net. It adds no cash row and counts once in
#   `unresolved_count`.


def _skipped(signal_id, direction, scenario="bid"):
    """Expected contract row without `reason` for a signal that was not purchased."""
    return {
        "signal_id": signal_id,
        "direction": direction,
        "price_reference": scenario,
        "status": "skipped",
        "outcome": "",
        "entry_ordinal": 0,
        "exit_ordinal": 0,
        "entry_timestamp_ns": 0,
        "expiry_timestamp_ns": 0,
        "exit_timestamp_ns": 0,
        "entry_spot": 0,
        "exit_spot": 0,
        "stake_cents": 0,
        "gross_credit_cents": 0,
        "net_cents": 0,
    }


def _unresolved(
    signal_id, direction, *, entry_ordinal, entry_s, expiry_s, entry_spot, stake
):
    """Expected contract row without `reason` for a stake locked after a stale expiry price."""
    return {
        "signal_id": signal_id,
        "direction": direction,
        "price_reference": "bid",
        "status": "unresolved",
        "outcome": "",
        "entry_ordinal": entry_ordinal,
        "exit_ordinal": 0,
        "entry_timestamp_ns": _T0 + entry_s * _SEC,
        "expiry_timestamp_ns": _T0 + expiry_s * _SEC,
        "exit_timestamp_ns": 0,
        "entry_spot": entry_spot,
        "exit_spot": 0,
        "stake_cents": stake,
        "gross_credit_cents": 0,
        "net_cents": 0,
    }


def _assert_run(result, *, contracts, cash, skipped_count, unresolved_count):
    """Exact equality on the full contract and cash lists and both counts, plain-int types.

    Unlike `_assert_ledger`, it allows skipped and unresolved contracts. It leaves
    `incomplete` open but requires `incomplete_reason` to be empty exactly when it is false.
    """
    for row in result.contracts:
        assert isinstance(row["reason"], str)
        assert row["reason"].strip()
    for row in [*result.contracts, *result.cash]:
        for key, value in row.items():
            assert type(value) in (int, str), (key, type(value))
    assert type(result.skipped_count) is int
    assert type(result.unresolved_count) is int
    assert type(result.incomplete) is bool
    assert (result.incomplete_reason == "") == (not result.incomplete)
    assert [
        {key: value for key, value in row.items() if key != "reason"}
        for row in result.contracts
    ] == contracts
    assert result.cash == cash
    assert result.skipped_count == skipped_count
    assert result.unresolved_count == unresolved_count


# Five CALL contracts open at once. Signal k starts at k-1 s and enters on the tick at that
# second (ordinal k, price 20000). Each expires 60 s later on a tick of its own with a higher
# price (all five win). The tick at 5 s (ordinal 6) is the start tick of a sixth signal.
# Stakes come from free cash only: 1000000 // 2000 = 500, then 499 four times (the locked
# stakes are excluded). Wins credit 500 -> 976 and 499 -> 974 (499 * 1952 // 1000).
_OVERLAP_TICKS = [
    _tick(0, 1, 20_000),
    _tick(1, 2, 20_000),
    _tick(2, 3, 20_000),
    _tick(3, 4, 20_000),
    _tick(4, 5, 20_000),
    _tick(5, 6, 20_000),
    _tick(60, 7, 20_100),
    _tick(61, 8, 20_200),
    _tick(62, 9, 20_300),
    _tick(63, 10, 20_400),
    _tick(64, 11, 20_500),
    _tick(70, 12, 21_000),
]
# (signal id, entry ordinal, entry s, exit ordinal, expiry s, exit spot, stake, gross, net)
_OVERLAP_WINS = [
    ("signal-1", 1, 0, 7, 60, 20_100, 500, 976, 476),
    ("signal-2", 2, 1, 8, 61, 20_200, 499, 974, 475),
    ("signal-3", 3, 2, 9, 62, 20_300, 499, 974, 475),
    ("signal-4", 4, 3, 10, 63, 20_400, 499, 974, 475),
    ("signal-5", 5, 4, 11, 64, 20_500, 499, 974, 475),
]


def _overlap_signals(count):
    return [_signal(f"signal-{k}", "CALL", k - 1, 1) for k in range(1, count + 1)]


def _overlap_contracts(scenario="bid"):
    """The five settled winners; the scenario only changes `price_reference`."""
    return [
        {
            **_settled(
                signal_id,
                "CALL",
                "win",
                entry_ordinal=entry_ordinal,
                exit_ordinal=exit_ordinal,
                entry_s=entry_s,
                expiry_s=expiry_s,
                exit_s=expiry_s,
                entry_spot=20_000,
                exit_spot=exit_spot,
                stake=stake,
                gross=gross,
                net=net,
            ),
            "price_reference": scenario,
        }
        for (
            signal_id,
            entry_ordinal,
            entry_s,
            exit_ordinal,
            expiry_s,
            exit_spot,
            stake,
            gross,
            net,
        ) in _OVERLAP_WINS
    ]


def _overlap_cash():
    """Five entries (free 997504, locked 2496), then five settlements to 1002376 and 0."""
    return [
        _cash("initial", "", (0, 0), (1_000_000, 0)),
        _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
        _cash("entry", "signal-2", (999_500, 500), (999_001, 999)),
        _cash("entry", "signal-3", (999_001, 999), (998_502, 1_498)),
        _cash("entry", "signal-4", (998_502, 1_498), (998_003, 1_997)),
        _cash("entry", "signal-5", (998_003, 1_997), (997_504, 2_496)),
        _cash("settlement", "signal-1", (997_504, 2_496), (998_480, 1_996)),
        _cash("settlement", "signal-2", (998_480, 1_996), (999_454, 1_497)),
        _cash("settlement", "signal-3", (999_454, 1_497), (1_000_428, 998)),
        _cash("settlement", "signal-4", (1_000_428, 998), (1_001_402, 499)),
        _cash("settlement", "signal-5", (1_001_402, 499), (1_002_376, 0)),
    ]


def test_a_sixth_signal_is_skipped_and_logged_while_five_contracts_are_open():
    """a sixth signal is skipped and logged while five contracts are open"""
    # The sixth signal starts at 5 s, while all five contracts are open until 60 s. Free
    # cash is 997504, so a stake of 498 is affordable: only the five-contract limit stops it.
    # It must leave cash and the accepted count unchanged, and must not be queued: the first
    # slot frees at 60 s, and a queued sixth would add a cash row and change every balance.
    # Its reason must differ from the reason of an accepted contract and from the
    # insufficient-cash reason (a run with 0.34 USD), so the report can tell them apart.
    signals = [*_overlap_signals(5), _signal("signal-6", "PUT", 5, 1)]

    result = run_ledger(
        _OVERLAP_TICKS, signals, scenario="bid", starting_cash_cents=1_000_000
    )

    _assert_run(
        result,
        contracts=[*_overlap_contracts(), _skipped("signal-6", "PUT")],
        cash=_overlap_cash(),
        skipped_count=1,
        unresolved_count=0,
    )
    assert result.incomplete is False
    capacity_reason = result.contracts[5]["reason"]
    assert capacity_reason != result.contracts[0]["reason"]

    short_ticks = [_tick(0, 1, 20_000), _tick(60, 2, 20_100), _tick(61, 3, 21_000)]
    short = run_ledger(
        short_ticks,
        [_signal("signal-1", "CALL", 0, 1)],
        scenario="bid",
        starting_cash_cents=34,
    )
    _assert_run(
        short,
        contracts=[_skipped("signal-1", "CALL")],
        cash=[_cash("initial", "", (0, 0), (34, 0))],
        skipped_count=1,
        unresolved_count=0,
    )
    assert short.incomplete is False
    assert short.contracts[0]["reason"] != capacity_reason


def test_when_all_five_win_the_balance_ends_at_10_023_76_with_locked_cash_0_00():
    """when all five win the balance ends at 10,023.76 with locked cash 0.00"""
    # C3: gross credits [976, 974, 974, 974, 974] add up to 4872; free cash goes from
    # 997504 to 1002376 and the 2496 locked stakes are all released.
    result = run_ledger(
        _OVERLAP_TICKS,
        _overlap_signals(5),
        scenario="bid",
        starting_cash_cents=1_000_000,
    )

    _assert_ledger(result, contracts=_overlap_contracts(), cash=_overlap_cash())
    assert [row["gross_credit_cents"] for row in result.contracts] == [
        976,
        974,
        974,
        974,
        974,
    ]
    last = result.cash[-1]
    assert (last["available_cents"], last["locked_cents"]) == (1_002_376, 0)


def test_an_unresolved_contract_keeps_its_stake_locked_and_keeps_its_slot():
    """an unresolved contract keeps its stake locked and keeps its slot"""
    # Signal 1 enters at 0 s and expires at 60 s. The last tick at or before 60 s is at 49 s
    # (ordinal 6, price 20300, a winning price), so the exit is 11 s old: unresolved. Signals
    # 2 to 5 enter at 20 to 23 s and expire at 80 to 83 s, so four contracts are still open at
    # 60 s. The next tick is at 65 s, also the start of signal 6: five slots are busy (four
    # open and one unresolved), so signal 6 is skipped. The stale price must not settle
    # signal 1, and a later tick must not repair it. Signal 1 keeps 500 locked to the end:
    # no credit, refund or release, and it adds no cash row. Signals 2 to 5 settle normally
    # (win, loss, tie, win), so the end balance is 999452 free and 500 locked.
    ticks = [
        _tick(0, 1, 20_000),
        _tick(20, 2, 20_000),
        _tick(21, 3, 20_000),
        _tick(22, 4, 20_000),
        _tick(23, 5, 20_000),
        _tick(49, 6, 20_300),
        _tick(65, 7, 20_000),
        _tick(80, 8, 20_100),
        _tick(81, 9, 19_900),
        _tick(82, 10, 20_000),
        _tick(83, 11, 20_200),
        _tick(90, 12, 21_000),
    ]
    signals = [
        _signal("signal-1", "CALL", 0, 1),
        _signal("signal-2", "CALL", 20, 1),
        _signal("signal-3", "CALL", 21, 1),
        _signal("signal-4", "CALL", 22, 1),
        _signal("signal-5", "CALL", 23, 1),
        _signal("signal-6", "CALL", 65, 1),
    ]

    result = run_ledger(ticks, signals, scenario="bid", starting_cash_cents=1_000_000)

    _assert_run(
        result,
        contracts=[
            _unresolved(
                "signal-1",
                "CALL",
                entry_ordinal=1,
                entry_s=0,
                expiry_s=60,
                entry_spot=20_000,
                stake=500,
            ),
            _settled(
                "signal-2",
                "CALL",
                "win",
                entry_ordinal=2,
                exit_ordinal=8,
                entry_s=20,
                expiry_s=80,
                exit_s=80,
                entry_spot=20_000,
                exit_spot=20_100,
                stake=499,
                gross=974,
                net=475,
            ),
            _settled(
                "signal-3",
                "CALL",
                "loss",
                entry_ordinal=3,
                exit_ordinal=9,
                entry_s=21,
                expiry_s=81,
                exit_s=81,
                entry_spot=20_000,
                exit_spot=19_900,
                stake=499,
                gross=0,
                net=-499,
            ),
            _settled(
                "signal-4",
                "CALL",
                "loss",
                entry_ordinal=4,
                exit_ordinal=10,
                entry_s=22,
                expiry_s=82,
                exit_s=82,
                entry_spot=20_000,
                exit_spot=20_000,
                stake=499,
                gross=0,
                net=-499,
            ),
            _settled(
                "signal-5",
                "CALL",
                "win",
                entry_ordinal=5,
                exit_ordinal=11,
                entry_s=23,
                expiry_s=83,
                exit_s=83,
                entry_spot=20_000,
                exit_spot=20_200,
                stake=499,
                gross=974,
                net=475,
            ),
            _skipped("signal-6", "CALL"),
        ],
        cash=[
            _cash("initial", "", (0, 0), (1_000_000, 0)),
            _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
            _cash("entry", "signal-2", (999_500, 500), (999_001, 999)),
            _cash("entry", "signal-3", (999_001, 999), (998_502, 1_498)),
            _cash("entry", "signal-4", (998_502, 1_498), (998_003, 1_997)),
            _cash("entry", "signal-5", (998_003, 1_997), (997_504, 2_496)),
            _cash("settlement", "signal-2", (997_504, 2_496), (998_478, 1_997)),
            _cash("settlement", "signal-3", (998_478, 1_997), (998_478, 1_498)),
            _cash("settlement", "signal-4", (998_478, 1_498), (998_478, 999)),
            _cash("settlement", "signal-5", (998_478, 999), (999_452, 500)),
        ],
        skipped_count=1,
        unresolved_count=1,
    )
    unresolved_reason = result.contracts[0]["reason"]
    skipped_reason = result.contracts[5]["reason"]
    assert unresolved_reason != skipped_reason


def test_each_bid_mid_and_ask_scenario_starts_independently_at_10_000_00_usd():
    """each bid, mid and ask scenario starts independently at 10,000.00 USD"""
    # Aligned fixture (bid = mid = ask): every scenario gives the same C3 cash, 1002376 free
    # and 0 locked, and its contract rows carry that scenario as `price_reference`.
    for scenario in ("bid", "mid", "ask"):
        result = run_ledger(
            _OVERLAP_TICKS,
            _overlap_signals(5),
            scenario=scenario,
            starting_cash_cents=1_000_000,
        )

        _assert_ledger(
            result,
            contracts=_overlap_contracts(scenario),
            cash=_overlap_cash(),
            label=f"aligned {scenario}",
        )

    # Asymmetric fixture, one CALL: bid/mid/ask entry 100/102/104 and exit 101/102/103.
    # Bid enters 100 and exits 101 (win), mid 102 and 102 (tie, loses), ask 104 and 103
    # (loses). Each scenario uses its own column for both entry and exit and starts again at
    # 1000000 free and 0 locked, whatever an earlier scenario ended with.
    ticks = [
        (_T0, 1, 100, 102, 104),
        (_T0 + 60 * _SEC, 2, 101, 102, 103),
        (_T0 + 61 * _SEC, 3, 90, 92, 94),
    ]
    signals = [_signal("signal-1", "CALL", 0, 1)]
    cases = [
        # (scenario, entry spot, exit spot, outcome, gross, net, final free cash)
        ("bid", 100, 101, "win", 976, 476, 1_000_476),
        ("mid", 102, 102, "loss", 0, -500, 999_500),
        ("ask", 104, 103, "loss", 0, -500, 999_500),
    ]
    for scenario, entry_spot, exit_spot, outcome, gross, net, final_free in cases:
        result = run_ledger(
            ticks, signals, scenario=scenario, starting_cash_cents=1_000_000
        )

        _assert_ledger(
            result,
            contracts=[
                {
                    **_settled(
                        "signal-1",
                        "CALL",
                        outcome,
                        entry_ordinal=1,
                        exit_ordinal=2,
                        entry_s=0,
                        expiry_s=60,
                        exit_s=60,
                        entry_spot=entry_spot,
                        exit_spot=exit_spot,
                        stake=500,
                        gross=gross,
                        net=net,
                    ),
                    "price_reference": scenario,
                }
            ],
            cash=[
                _cash("initial", "", (0, 0), (1_000_000, 0)),
                _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
                _cash("settlement", "signal-1", (999_500, 500), (final_free, 0)),
            ],
            label=f"asymmetric {scenario}",
        )


# --- R-13 to R-16: continuation, data exhaustion, chunks, overflow --------------------------
#
# Engine API these tests add (names fixed by the navigator review, story-1-ledger-red-review.md):
#
# - `run_ledger(..., window_end_ns=None)`: the end of the selected window, in UTC nanoseconds
#   (the first nanosecond after the window, so the end itself is outside).
#   A signal whose `start_time_ns` is at or after it is a continuation signal: it is never
#   admitted and its contract row is a skipped row (the R-09 shape) with a nonempty reason.
#   Ticks after the window are still read, so contracts that entered inside the window settle
#   from them (S3). `None` means there is no window end (R-06 to R-12 behavior).
# - `run_ledger(..., compiled=False)`: `False` (the default) runs the Python loop and
#   `compiled=True` runs the compiled loop. Both must give the same `LedgerResult`.
# - `run_ledger_chunks(chunks, signals, *, scenario, starting_cash_cents, window_end_ns=None,
#   compiled=False)`: `chunks` is an iterable of tick lists whose concatenation is the whole
#   tick list. It returns the same `LedgerResult` as `run_ledger` on the concatenation. A chunk
#   may end inside a timestamp group: nothing settles or enters until that group is complete
#   (a later timestamp arrives or the chunks end). Empty chunks are allowed.
#
# R-13 to R-16 pass `compiled=False` and `compiled=True` explicitly, so changing the engine's
# default path cannot turn the Python half of a comparison into a second compiled run.


def _run(
    ticks,
    signals,
    *,
    starting_cash_cents,
    scenario="bid",
    compiled=None,
    window_end_ns=None,
):
    """Call `run_ledger`, passing `compiled` and `window_end_ns` only when they are given."""
    options = {}
    if compiled is not None:
        options["compiled"] = compiled
    if window_end_ns is not None:
        options["window_end_ns"] = window_end_ns
    return run_ledger(
        ticks,
        signals,
        scenario=scenario,
        starting_cash_cents=starting_cash_cents,
        **options,
    )


def _assert_overflows(ticks, signals, *, starting_cash_cents, compiled):
    """The run raises `OverflowError`, returns nothing, and leaves its inputs unchanged."""
    ticks_before = [tuple(tick) for tick in ticks]
    signals_before = [dict(signal) for signal in signals]
    result = None
    try:
        result = _run(
            ticks,
            signals,
            starting_cash_cents=starting_cash_cents,
            compiled=compiled,
        )
    except OverflowError:
        pass
    else:
        raise AssertionError(f"no OverflowError (compiled={compiled}): {result}")
    assert result is None
    assert ticks == ticks_before
    assert signals == signals_before


def test_continuation_past_the_window_settles_existing_contracts_only():
    """continuation past the window settles existing contracts only"""
    # The window ends at 30 s. Signal 1 (CALL) starts at 0 s inside the window, enters at 2 s
    # (500 stake) and expires at 60 s, after the window. Continuation data settles it: the
    # last row at 60 s is 20400, above the entry 20000, so it wins 976 and the run ends at
    # 1000476 free and 0 locked. Signal 2 starts exactly at the window end (30 s) and signal
    # 3 at 45 s. Both have a tick to enter on (ordinals 2 and 3), so only the window stops
    # them: they must admit no contract and add no cash row. A tick after expiry (61 s) must
    # change nothing.
    ticks = [
        _tick(2, 1, 20_000),
        _tick(30, 2, 20_100),
        _tick(45, 3, 20_200),
        _tick(60, 4, 20_400),
        _tick(61, 5, 21_000),
    ]
    signals = [
        _signal("signal-1", "CALL", 0, 1),
        _signal("signal-2", "CALL", 30, 1),
        _signal("signal-3", "PUT", 45, 1),
    ]

    for compiled in (False, True):
        result = _run(
            ticks,
            signals,
            starting_cash_cents=1_000_000,
            window_end_ns=_T0 + 30 * _SEC,
            compiled=compiled,
        )

        _assert_run(
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
                    exit_spot=20_400,
                    stake=500,
                    gross=976,
                    net=476,
                ),
                _skipped("signal-2", "CALL"),
                _skipped("signal-3", "PUT"),
            ],
            cash=[
                _cash("initial", "", (0, 0), (1_000_000, 0)),
                _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
                _cash("settlement", "signal-1", (999_500, 500), (1_000_476, 0)),
            ],
            skipped_count=2,
            unresolved_count=0,
        )
        assert result.incomplete is False

        # Control: without a window end the same signals do enter, so the window is what stops them.
        control = _run(ticks, signals, starting_cash_cents=1_000_000, compiled=compiled)
        assert [
            row["signal_id"] for row in control.cash if row["event"] == "entry"
        ] == [
            "signal-1",
            "signal-2",
            "signal-3",
        ]


def test_true_data_exhaustion_marks_the_run_incomplete_and_keeps_locked_stakes():
    """true data exhaustion marks the run incomplete and keeps locked stakes"""
    # Signal 1 (CALL) enters at 2 s (stake 500) and expires at 60 s. The data ends at 55 s
    # (ordinal 3, price 20400, a winning price only 5 s before expiry), so no row at or after
    # expiry exists. The contract stays unresolved: 500 locked, free 999500, no settlement
    # row, no refund, no gross or net, and the run is incomplete with a reason. Settling on
    # the last available price would be an invented settlement.
    ticks = [_tick(2, 1, 20_000), _tick(30, 2, 20_200), _tick(55, 3, 20_400)]
    signals = [_signal("signal-1", "CALL", 0, 1)]

    for compiled in (False, True):
        result = _run(ticks, signals, starting_cash_cents=1_000_000, compiled=compiled)

        _assert_run(
            result,
            contracts=[
                _unresolved(
                    "signal-1",
                    "CALL",
                    entry_ordinal=1,
                    entry_s=2,
                    expiry_s=60,
                    entry_spot=20_000,
                    stake=500,
                )
            ],
            cash=[
                _cash("initial", "", (0, 0), (1_000_000, 0)),
                _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
            ],
            skipped_count=0,
            unresolved_count=1,
        )
        assert result.incomplete is True
        assert result.incomplete_reason.strip()

        # Control: one more row at expiry settles the same contract and the run is complete, so
        # the data ending before expiry is what makes the first run incomplete.
        control = _run(
            [*ticks, _tick(60, 4, 20_400)],
            signals,
            starting_cash_cents=1_000_000,
            compiled=compiled,
        )
        _assert_ledger(
            control,
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
                    exit_spot=20_400,
                    stake=500,
                    gross=976,
                    net=476,
                )
            ],
            cash=[
                _cash("initial", "", (0, 0), (1_000_000, 0)),
                _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
                _cash("settlement", "signal-1", (999_500, 500), (1_000_476, 0)),
            ],
        )


# The R-07 timestamp-group fixture again (the R-07 test keeps its copy inline): the group at
# +60 s holds ordinals 2, 3 and 4. R-15 splits it between chunks.
_R07_TICKS = [
    _tick(2, 1, 20_000),
    _tick(60, 2, 19_800),
    _tick(60, 3, 20_000),
    _tick(60, 4, 20_600),
    _tick(120, 5, 19_600),
    _tick(121, 6, 19_000),
]
_R07_SIGNALS = [
    _signal("signal-1", "CALL", 0, 1),
    _signal("signal-2", "PUT", 60, 1),
    _signal("signal-3", "CALL", 60, 1),
]


def _r07_contracts():
    return [
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
    ]


def _r07_cash():
    return [
        _cash("initial", "", (0, 0), (1_000_000, 0)),
        _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
        _cash("settlement", "signal-1", (999_500, 500), (1_000_476, 0)),
        _cash("entry", "signal-2", (1_000_476, 0), (999_976, 500)),
        _cash("entry", "signal-3", (999_976, 500), (999_477, 999)),
        _cash("settlement", "signal-2", (999_477, 999), (1_000_453, 499)),
        _cash("settlement", "signal-3", (1_000_453, 499), (1_000_453, 0)),
    ]


def test_a_chunked_run_equals_the_whole_array_run_including_a_same_time_chunk_boundary():
    """a chunked run equals the whole-array run, including a same-time chunk boundary"""
    # The R-07 fixture split into chunks. Cuts after 2 and after 3 ticks fall inside the +60 s
    # group (ordinals 2 | 3, 4 and 2, 3 | 4). Nothing may settle or enter at a chunk end: an
    # early settlement would exit on ordinal 2 (a loss) or 3 (a tie) instead of 4 (a win), and
    # an early entry would enter signal 2 before signal 1's win released its cash. Every split
    # must give the literal R-07 contracts, cash rows and counts, and equal the whole-array
    # result, in the Python loop and in the compiled loop.
    # The chunk API does not exist yet, so it is imported here: a module-level import would
    # stop every ledger test above from being collected.
    from vectorbt_research.engine import run_ledger_chunks

    ticks = _R07_TICKS
    splits = {
        "cut inside the group after ordinal 2": [ticks[:2], ticks[2:]],
        "cut inside the group after ordinal 3": [ticks[:3], ticks[3:]],
        "cut before the group": [ticks[:1], ticks[1:]],
        "cut after the group": [ticks[:4], ticks[4:]],
        "cut between the last two ticks": [ticks[:5], ticks[5:]],
        "one tick per chunk": [[tick] for tick in ticks],
        "empty chunks around a group cut": [
            [],
            ticks[:2],
            [],
            ticks[2:3],
            ticks[3:],
            [],
        ],
    }
    whole_results = []
    for compiled in (False, True):
        whole = _run(
            ticks, _R07_SIGNALS, starting_cash_cents=1_000_000, compiled=compiled
        )
        _assert_ledger(
            whole,
            contracts=_r07_contracts(),
            cash=_r07_cash(),
            label=f"whole array, compiled={compiled}",
        )
        whole_results.append(whole)
        for name, chunks in splits.items():
            chunked = run_ledger_chunks(
                iter(chunks),
                _R07_SIGNALS,
                scenario="bid",
                starting_cash_cents=1_000_000,
                compiled=compiled,
            )
            _assert_ledger(
                chunked,
                contracts=_r07_contracts(),
                cash=_r07_cash(),
                label=f"{name}, compiled={compiled}",
            )
            assert chunked == whole, (name, compiled)
    assert whole_results[0] == whole_results[1]

    # The window end passed to chunks. The window ends at the +60 s group, so signals 2 and 3
    # (they start at +60 s) are continuation signals: skipped, no cash row. Signal 1 still wins
    # from that group's last row even when a chunk ends inside the group.
    window_end_ns = _T0 + 60 * _SEC
    for compiled in (False, True):
        whole = _run(
            ticks,
            _R07_SIGNALS,
            starting_cash_cents=1_000_000,
            compiled=compiled,
            window_end_ns=window_end_ns,
        )
        _assert_run(
            whole,
            contracts=[
                _r07_contracts()[0],
                _skipped("signal-2", "PUT"),
                _skipped("signal-3", "CALL"),
            ],
            cash=_r07_cash()[:3],
            skipped_count=2,
            unresolved_count=0,
        )
        assert whole.incomplete is False
        for name, chunks in splits.items():
            chunked = run_ledger_chunks(
                iter(chunks),
                _R07_SIGNALS,
                scenario="bid",
                starting_cash_cents=1_000_000,
                window_end_ns=window_end_ns,
                compiled=compiled,
            )
            assert chunked == whole, ("window end", name, compiled)


# Q1 literals of the shared rule: 976 is the gross credit of a 500-cent stake, and the signed
# 64-bit maximum minus 976 plus 976 reaches the maximum exactly.
_INT64_MAX = 9_223_372_036_854_775_807
_Q1_FREE_AT_MAX = 9_223_372_036_854_774_831  # max - 976
_Q1_FREE_ONE_OVER = 9_223_372_036_854_774_832  # max - 975
_Q1_CREDIT = 976

# The ledger sets the stake from free cash (0.05%), so a 976 credit never meets a free balance
# near the maximum. These starting cash values reproduce the same boundary through the ledger.
# A CALL wins in both: free cash S gives stake S // 2000 = 4609491900282853 and gross
# 4609491900282853 * 1952 // 1000 = 8997728189352129. After the entry free cash is
# S - 4609491900282853 and the win adds the gross. With S = 9218983800565706531 the sum is
# exactly the maximum; one cent more (S = 9218983800565706532) is one over it.
_AT_MAX_START = 9_218_983_800_565_706_531
_OVER_MAX_START = 9_218_983_800_565_706_532
_BIG_STAKE = 4_609_491_900_282_853
_BIG_GROSS = 8_997_728_189_352_129
_BIG_NET = 4_388_236_289_069_276  # gross - stake
_AT_MAX_AFTER_ENTRY = 9_214_374_308_665_423_678  # _AT_MAX_START - _BIG_STAKE
# Starting at the maximum itself, the stake is 4611686018427387; a loss leaves the rest free.
_MAX_START_STAKE = 4_611_686_018_427_387
_MAX_START_AFTER_LOSS = 9_218_760_350_836_348_420  # max - 4611686018427387
_NEAR_MAX_NS = _INT64_MAX - 1  # a start whose expiry (start + 60 s) does not fit


def _big_ticks(exit_price):
    return [_tick(0, 1, 20_000), _tick(60, 2, exit_price), _tick(61, 3, 21_000)]


def test_integer_overflow_in_the_compiled_loop_fails_visibly():
    """integer overflow in the compiled loop fails visibly"""
    # The Q1 literals hold in the shared rule that the loop uses (they cannot be reached
    # through the ledger itself, see the comment above the constants). Imported here only
    # for these two literal anchors.
    from deriv_mt5.rules import checked_add

    assert checked_add(_Q1_FREE_AT_MAX, _Q1_CREDIT) == _INT64_MAX
    try:
        checked_add(_Q1_FREE_ONE_OVER, _Q1_CREDIT)
    except OverflowError:
        pass
    else:
        raise AssertionError("max - 975 + 976 did not raise OverflowError")

    signals = [_signal("signal-1", "CALL", 0, 1)]
    for compiled in (False, True):
        label = f"compiled={compiled}"
        # Safe addition reaches the maximum: the win credit lands exactly on 2**63 - 1.
        at_max = _run(
            _big_ticks(20_400),
            signals,
            starting_cash_cents=_AT_MAX_START,
            compiled=compiled,
        )
        _assert_ledger(
            at_max,
            contracts=[
                _settled(
                    "signal-1",
                    "CALL",
                    "win",
                    entry_ordinal=1,
                    exit_ordinal=2,
                    entry_s=0,
                    expiry_s=60,
                    exit_s=60,
                    entry_spot=20_000,
                    exit_spot=20_400,
                    stake=_BIG_STAKE,
                    gross=_BIG_GROSS,
                    net=_BIG_NET,
                )
            ],
            cash=[
                _cash("initial", "", (0, 0), (_AT_MAX_START, 0)),
                _cash(
                    "entry",
                    "signal-1",
                    (_AT_MAX_START, 0),
                    (_AT_MAX_AFTER_ENTRY, _BIG_STAKE),
                ),
                _cash(
                    "settlement",
                    "signal-1",
                    (_AT_MAX_AFTER_ENTRY, _BIG_STAKE),
                    (_INT64_MAX, 0),
                ),
            ],
            label=label,
        )

        # One cent more free cash makes the credit overflow (max + 1): the run fails with
        # no report and no wrapped balance.
        _assert_overflows(
            _big_ticks(20_400),
            signals,
            starting_cash_cents=_OVER_MAX_START,
            compiled=compiled,
        )
        # Same start, but the contract loses: no credit, so no overflow and a normal report.
        lost = _run(
            _big_ticks(19_800),
            signals,
            starting_cash_cents=_OVER_MAX_START,
            compiled=compiled,
        )
        assert [row["status"] for row in lost.contracts] == ["settled"]
        assert [row["outcome"] for row in lost.contracts] == ["loss"]
        assert (lost.cash[-1]["available_cents"], lost.cash[-1]["locked_cents"]) == (
            _OVER_MAX_START - _BIG_STAKE,
            0,
        )

        # An input one above the maximum fails; the maximum itself is a valid input.
        _assert_overflows(
            _big_ticks(19_800),
            signals,
            starting_cash_cents=_INT64_MAX + 1,
            compiled=compiled,
        )
        at_input_max = _run(
            _big_ticks(19_800),
            signals,
            starting_cash_cents=_INT64_MAX,
            compiled=compiled,
        )
        assert at_input_max.cash == [
            _cash("initial", "", (0, 0), (_INT64_MAX, 0)),
            _cash(
                "entry",
                "signal-1",
                (_INT64_MAX, 0),
                (_INT64_MAX - _MAX_START_STAKE, _MAX_START_STAKE),
            ),
            _cash(
                "settlement",
                "signal-1",
                (_INT64_MAX - _MAX_START_STAKE, _MAX_START_STAKE),
                (_MAX_START_AFTER_LOSS, 0),
            ),
        ]

        # A start whose expiry does not fit in 64 bits fails; it is not a skipped contract.
        _assert_overflows(
            [(_NEAR_MAX_NS, 1, 20_000, 20_000, 20_000)],
            [
                {
                    "signal_id": "signal-1",
                    "direction": "CALL",
                    "start_time_ns": _NEAR_MAX_NS,
                    "availability_ordinal": 1,
                }
            ],
            starting_cash_cents=1_000_000,
            compiled=compiled,
        )

        # Normal money stays exact: the R-06 win, 1000000 -> 999500/500 -> 1000476/0.
        normal = _run(
            [
                _tick(2, 1, 20_000),
                _tick(30, 2, 20_200),
                _tick(60, 3, 20_400),
                _tick(61, 4, 21_000),
            ],
            signals,
            starting_cash_cents=1_000_000,
            compiled=compiled,
        )
        _assert_ledger(
            normal,
            contracts=[
                _settled(
                    "signal-1",
                    "CALL",
                    "win",
                    entry_ordinal=1,
                    exit_ordinal=3,
                    entry_s=2,
                    expiry_s=60,
                    exit_s=60,
                    entry_spot=20_000,
                    exit_spot=20_400,
                    stake=500,
                    gross=976,
                    net=476,
                )
            ],
            cash=[
                _cash("initial", "", (0, 0), (1_000_000, 0)),
                _cash("entry", "signal-1", (1_000_000, 0), (999_500, 500)),
                _cash("settlement", "signal-1", (999_500, 500), (1_000_476, 0)),
            ],
            label=label,
        )
