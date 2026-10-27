"""Ledger loop tests R-06 to R-12.

Engine API these tests set (later ledger tests R-13 to R-16 and the runner reuse it):

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
