from __future__ import annotations

import pytest
from vectorbt_research import engine

_T0 = 1_787_023_200_000_000_000
_SEC = 1_000_000_000


def _tick(offset_s: int, ordinal: int, price: int = 20_000):
    return (_T0 + offset_s * _SEC, ordinal, price, price, price)


def _signal(signal_id: str = "s", *, start_s: int = 0, availability: int = 1):
    return {
        "signal_id": signal_id,
        "direction": "CALL",
        "start_time_ns": _T0 + start_s * _SEC,
        "availability_ordinal": availability,
    }


def _run(ticks, signals, *, window_end_ns=None):
    return engine.run_ledger(
        ticks,
        signals,
        scenario="bid",
        starting_cash_cents=1_000_000,
        window_end_ns=window_end_ns,
        compiled=False,
    )


def test_first_eligible_tick_at_window_end_is_skipped_but_one_before_opens():
    ticks = [_tick(1, 1), _tick(2, 2), _tick(60, 3, 20_100), _tick(61, 4)]
    signal = [_signal()]

    at_end = _run(ticks, signal, window_end_ns=_T0 + _SEC)
    before_end = _run(ticks, signal, window_end_ns=_T0 + _SEC + 1)

    assert at_end.contracts[0]["status"] == "skipped"
    assert len(at_end.cash) == 1
    assert at_end.skipped_count == 1
    assert before_end.contracts[0]["status"] == "settled"
    assert [row["event"] for row in before_end.cash] == [
        "initial",
        "entry",
        "settlement",
    ]


def test_signal_waiting_for_availability_at_window_end_is_skipped():
    ticks = [_tick(1, 1), _tick(2, 2), _tick(3, 3), _tick(61, 4)]
    result = _run(
        ticks,
        [_signal(availability=3)],
        window_end_ns=_T0 + 2 * _SEC,
    )

    assert result.contracts[0]["status"] == "skipped"
    assert len(result.cash) == 1
    assert result.skipped_count == 1


def test_open_contract_settles_from_ticks_after_window_end():
    ticks = [_tick(1, 1), _tick(60, 2, 20_100), _tick(61, 3)]
    result = _run(ticks, [_signal()], window_end_ns=_T0 + 2 * _SEC)

    assert result.contracts[0]["status"] == "settled"
    assert result.contracts[0]["exit_ordinal"] == 2
    assert [row["event"] for row in result.cash] == [
        "initial",
        "entry",
        "settlement",
    ]


def test_chunk_splits_at_every_position_match_one_run_including_same_time_group():
    ticks = [
        _tick(1, 1, 19_900),
        _tick(1, 2, 20_000),
        _tick(60, 3, 20_100),
        _tick(60, 4, 20_200),
        _tick(61, 5, 20_200),
    ]
    signals = [_signal(availability=2)]
    expected = _run(ticks, signals)
    assert expected.contracts[0]["status"] == "settled"
    assert expected.contracts[0]["exit_ordinal"] == 4

    for split in range(len(ticks) + 1):
        actual = engine.run_ledger_chunks(
            [ticks[:split], ticks[split:]],
            signals,
            scenario="bid",
            starting_cash_cents=1_000_000,
            compiled=False,
        )
        assert actual == expected, f"split at {split}"


@pytest.mark.parametrize("compiled", [False, True])
def test_chunk_processing_does_not_buffer_the_full_tick_history(monkeypatch, compiled):
    credits = []
    gross_credit_cents = engine.gross_credit_cents

    def record_credit(stake, won):
        credit = gross_credit_cents(stake, won)
        credits.append(credit)
        return credit

    processed_cash_rows = []
    if compiled:
        run_core = engine._run_core

        def observe_core(*args, **kwargs):
            before_rows = kwargs["state"].cash_rows
            state = run_core(*args, **kwargs)
            if kwargs.get("compiled") and state.cash_rows > before_rows:
                processed_cash_rows.append(state.cash_rows)
            return state

        monkeypatch.setattr(engine, "_run_core", observe_core)
    else:
        monkeypatch.setattr(engine, "gross_credit_cents", record_credit)

    def chunks():
        yield [_tick(1, 1)]
        yield [_tick(60, 2, 20_100)]
        # The next timestamp completes the expiry group, so settlement must occur
        # before the consumer requests another chunk. No allocator threshold is needed.
        yield [_tick(61, 3, 20_100)]
        if compiled:
            assert processed_cash_rows == [2, 3], (
                "compiled ticks were buffered instead of settling incrementally"
            )
        else:
            assert credits == [976], (
                "tick history was buffered instead of settling incrementally"
            )
        yield [_tick(62, 4, 20_100)]

    result = engine.run_ledger_chunks(
        chunks(),
        [_signal()],
        scenario="bid",
        starting_cash_cents=1_000_000,
        compiled=compiled,
    )

    assert result.contracts[0]["status"] == "settled"
    assert [row["event"] for row in result.cash] == [
        "initial",
        "entry",
        "settlement",
    ]
    if compiled:
        assert processed_cash_rows == [2, 3]
    else:
        assert credits == [976]


def test_short_window_without_fresh_expiry_tick_keeps_stake_locked():
    chunks = [
        [_tick(1, 1)],
        [_tick(48, 2, 20_400)],
        [_tick(61, 3, 20_100)],
    ]
    result = engine.run_ledger_chunks(
        chunks,
        [_signal()],
        scenario="bid",
        starting_cash_cents=1_000_000,
        window_end_ns=_T0 + 2 * _SEC,
        compiled=False,
    )

    contract = result.contracts[0]
    assert contract["status"] == "unresolved"
    assert contract["stake_cents"] == 500
    assert contract["gross_credit_cents"] == 0
    assert contract["net_cents"] == 0
    assert contract["exit_timestamp_ns"] == 0
    assert [row["event"] for row in result.cash] == ["initial", "entry"]
    assert result.cash[-1]["locked_cents"] == 500
