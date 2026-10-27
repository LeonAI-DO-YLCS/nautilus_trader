import pandas as pd
import pytest
import vectorbt as vbt
from vectorbt_research.signals import frozen_signals_to_vectorbt, signals_from_vectorbt


def test_frozen_hand_written_signals_pass_through_vectorbt_in_same_order():
    """frozen hand-written signals pass through vectorbt in the same order"""
    # S7 is a fixed list; starts lie within S8 (2026-08-18 03:20-04:20 UTC).
    # Preserve list order, not timestamp or availability-ordinal sorted order.
    frozen_signals = [
        {
            "signal_id": "signal-01",
            "direction": "CALL",
            "start_time_ns": 1787023320000000000,
            "availability_ordinal": 4,
        },
        {
            "signal_id": "signal-02",
            "direction": "PUT",
            "start_time_ns": 1787023260000000000,
            "availability_ordinal": 2,
        },
        {
            "signal_id": "signal-03",
            "direction": "PUT",
            "start_time_ns": 1787023260000000000,
            "availability_ordinal": 7,
        },
        {
            "signal_id": "signal-04",
            "direction": "CALL",
            "start_time_ns": 1787023440000000000,
            "availability_ordinal": 1,
        },
        {
            "signal_id": "signal-05",
            "direction": "CALL",
            "start_time_ns": 1787023380000000000,
            "availability_ordinal": 9,
        },
    ]
    quote_times_ns = [
        1787023200000000000,
        1787023260000000000,
        1787023320000000000,
        1787023380000000000,
        1787023440000000000,
    ]

    vectorbt_signals = frozen_signals_to_vectorbt(frozen_signals, quote_times_ns)

    # One vectorbt signal column per frozen signal: equal-time signals never merge.
    assert isinstance(vectorbt_signals, pd.DataFrame)
    assert isinstance(
        vectorbt_signals.vbt.signals, vbt.signals.accessors.SignalsDFAccessor
    )
    assert vectorbt_signals.vbt.signals.total().sum() == len(frozen_signals)

    assert vectorbt_signals.index.tolist() == quote_times_ns
    assert all(pd.api.types.is_bool_dtype(dtype) for dtype in vectorbt_signals.dtypes)
    assert vectorbt_signals.to_numpy().tolist() == [
        [False, False, False, False, False],
        [False, True, True, False, False],
        [True, False, False, False, False],
        [False, False, False, False, True],
        [False, False, False, True, False],
    ]

    records = signals_from_vectorbt(vectorbt_signals)

    assert [record["signal_id"] for record in records] == [
        "signal-01",
        "signal-02",
        "signal-03",
        "signal-04",
        "signal-05",
    ]
    assert [record["direction"] for record in records] == [
        "CALL",
        "PUT",
        "PUT",
        "CALL",
        "CALL",
    ]
    assert [record["start_time_ns"] for record in records] == [
        1787023320000000000,
        1787023260000000000,
        1787023260000000000,
        1787023440000000000,
        1787023380000000000,
    ]
    assert [record["availability_ordinal"] for record in records] == [4, 2, 7, 1, 9]
    for record in records:
        assert type(record["start_time_ns"]) is int
        assert type(record["availability_ordinal"]) is int


def test_duplicate_quote_times_are_rejected():
    """duplicate quote times make frozen_signals_to_vectorbt raise ValueError"""
    frozen_signals = [
        {
            "signal_id": "signal-01",
            "direction": "CALL",
            "start_time_ns": 1787023260000000000,
            "availability_ordinal": 1,
        }
    ]
    quote_times_ns = [1787023200000000000, 1787023260000000000, 1787023260000000000]

    with pytest.raises(ValueError, match="quote times must be unique"):
        frozen_signals_to_vectorbt(frozen_signals, quote_times_ns)
