"""
EMA crossover strategy: 1-minute closes, three averages, Rise and Fall on the crossing bar.

E-01 a bar's close is the last quote of its UTC minute, at the run's own price.
E-02 the averages equal pandas' exponential mean (span n, no adjustment, n bars warm-up).
E-03 Rise fires on the bar where the close moves above all averages, Fall below all.
E-04 a signal starts at its bar's close, after every quote of that bar.
E-05 no signal fires while any average is still warming up.
E-06 a real V75 window saves a run whose contracts come from the strategy's signals.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt
from vectorbt_research.dataset import read_quote_window
from vectorbt_research.ema_cross import (
    DEFAULT_WINDOWS,
    ema_cross_signals,
    minute_closes,
    run_ema_cross_window,
)
from vectorbt_research.input import parse_quote_rows
from vectorbt_research.reports import read_run
from vectorbt_research.runner import run_story_1

DATASET_DIR = Path(
    "/mnt/c/LeonAI-DO/dev/TRADING/Datasets/Volatility75Index_yearly_deduped"
)
_BASE = datetime(2026, 8, 18, 0, 0, tzinfo=UTC)
_SMALL = (2, 3, 4)


def _timestamp(seconds):
    return (_BASE + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _rows(closes, spread="0.00"):
    """
    One quote at the start of each minute, with bid = close and ask = close + spread.

    A signal starting at a minute enters on that minute's quote and exits on the next
    minute's quote, exactly at expiry.
    """
    return [
        {
            "timestamp": _timestamp(minute * 60),
            "ordinal": minute,
            "bid": f"{close:.2f}",
            "ask": f"{close + float(spread):.2f}",
        }
        for minute, close in enumerate(closes)
    ]


def _quotes(rows):
    quotes, _ = parse_quote_rows(rows)
    return quotes


def test_e01_close_is_last_quote_of_the_minute_at_the_run_price():
    rows = [
        {"timestamp": _timestamp(5), "ordinal": 0, "bid": "100.00", "ask": "101.00"},
        {"timestamp": _timestamp(59), "ordinal": 1, "bid": "102.00", "ask": "103.00"},
        {"timestamp": _timestamp(60), "ordinal": 2, "bid": "104.00", "ask": "105.00"},
    ]
    quotes = _quotes(rows)

    bid = minute_closes(quotes, "bid")
    ask = minute_closes(quotes, "ask")
    mid = minute_closes(quotes, "mid")

    # Half-cents: 102.00 is 20400. The quote at 60 s opens the next minute.
    assert bid.tolist() == [20400, 20800]
    assert ask.tolist() == [20600, 21000]
    assert mid.tolist() == [20500, 20900]
    assert list(bid.index) == [
        pd.Timestamp("2026-08-18T00:00:00Z"),
        pd.Timestamp("2026-08-18T00:01:00Z"),
    ]


def test_e02_averages_match_the_pandas_exponential_mean():
    close = pd.Series(np.linspace(100.0, 130.0, 40) + np.sin(np.arange(40)))
    emas = vbt.MA.run(close, window=list(_SMALL), ewm=True).ma

    for column, window in enumerate(_SMALL):
        expected = close.ewm(span=window, adjust=False, min_periods=window).mean()
        np.testing.assert_allclose(emas.iloc[:, column], expected, equal_nan=True)


def test_e03_rise_and_fall_fire_only_on_the_crossing_bar():
    # Bars 0-4 sit on the averages; bar 5 jumps above all three and bar 6 stays above;
    # bar 7 drops below all three and bar 8 stays below.
    closes = [100, 100, 100, 100, 100, 110, 111, 90, 89]

    signals = ema_cross_signals(_quotes(_rows(closes)), "bid", _SMALL)

    assert signals == [
        {
            "signal_id": "ema-20260818T0006-CALL",
            "direction": "CALL",
            "start_time": "2026-08-18T00:06:00Z",
        },
        {
            "signal_id": "ema-20260818T0008-PUT",
            "direction": "PUT",
            "start_time": "2026-08-18T00:08:00Z",
        },
    ]


def test_e03_close_above_only_some_averages_does_not_fire():
    # Bar 5 falls below all three. Bar 6 (94) is above the 2-bar average (93.78) but
    # below the 3-bar one (94.50), so nothing fires. Bar 7 (97) is above all three.
    closes = [100, 100, 100, 100, 100, 90, 94, 97]

    signals = ema_cross_signals(_quotes(_rows(closes)), "bid", _SMALL)

    assert [(s["direction"], s["start_time"]) for s in signals] == [
        ("PUT", "2026-08-18T00:06:00Z"),
        ("CALL", "2026-08-18T00:08:00Z"),
    ]


def test_e04_signal_starts_after_every_quote_of_its_bar():
    closes = [100, 100, 100, 100, 100, 110]
    quotes = _quotes(_rows(closes))

    (signal,) = ema_cross_signals(quotes, "bid", _SMALL)

    start = pd.Timestamp(signal["start_time"]).value
    bar_quotes = [q for q in quotes if q.timestamp_ns < start]
    assert bar_quotes[-1].bid_half_cents == 22000
    assert all(q.timestamp_ns >= start for q in quotes[len(bar_quotes) :])


def test_e05_no_signal_while_any_average_warms_up():
    # The jump at bar 2 comes before the slowest average (4 bars) is ready.
    closes = [100, 100, 110, 110, 110, 110]

    assert ema_cross_signals(_quotes(_rows(closes)), "bid", _SMALL) == []
    # The default slowest average needs 200 bars, so 200 bars never fire.
    rising = [100 + i for i in range(200)]
    assert ema_cross_signals(_quotes(_rows(rising)), "bid", DEFAULT_WINDOWS) == []


@pytest.mark.parametrize("windows", [(), (9, 0, 200), (9, 2.5, 200)])
def test_e05_bad_windows_are_rejected(windows):
    with pytest.raises(ValueError, match="EMA windows"):
        ema_cross_signals(_quotes(_rows([100, 101])), "bid", windows)


def test_e03_signals_drive_the_shared_ledger(tmp_path):
    # The last quote (87) comes after the final expiry, so every contract settles.
    closes = [100, 100, 100, 100, 100, 110, 111, 90, 89, 88, 87]
    rows = _rows(closes)

    run_dir = run_story_1(
        quote_rows=rows,
        frozen_signals=ema_cross_signals(_quotes(rows), "bid", _SMALL),
        scenario="bid",
        output_dir=tmp_path,
    )

    contracts = read_run(run_dir)["contracts"]
    # Rise enters at 111 (bar 6) and exits at 90 (bar 7): a loss.
    # Fall enters at 89 (bar 8) and exits at 88 (bar 9): a win.
    assert contracts[["direction", "status", "outcome"]].values.tolist() == [
        ["CALL", "settled", "loss"],
        ["PUT", "settled", "win"],
    ]


@pytest.mark.skipif(
    not DATASET_DIR.is_dir(), reason="Volatility75 dataset folder is absent"
)
def test_e06_real_window_saves_a_strategy_run(tmp_path):
    start_ns = int(datetime(2026, 8, 18, 0, 0, tzinfo=UTC).timestamp()) * 10**9
    end_ns = int(datetime(2026, 8, 18, 4, 0, tzinfo=UTC).timestamp()) * 10**9

    run_dir = run_ema_cross_window(
        dataset_folder=DATASET_DIR,
        window_start_ns=start_ns,
        window_end_ns=end_ns,
        scenario="mid",
        output_dir=tmp_path,
    )

    run = read_run(run_dir)
    quotes, _ = parse_quote_rows(
        read_quote_window(
            DATASET_DIR / "Volatility75Index_2026.parquet", start_ns, end_ns
        )
    )
    expected = ema_cross_signals(quotes, "mid")
    manifest = run["manifest"]
    assert manifest["label"] == "MT5 simulation"
    assert manifest["purpose"] == "EMA crossover backtest"
    assert manifest["strategy"]["parameters"]["ema_windows"] == [9, 21, 200]
    assert manifest["strategy"]["source"] == "vectorbt_research.ema_cross"
    assert expected, "a 4-hour window should give at least one crossing"
    assert run["signals"]["signal_id"].tolist() == [s["signal_id"] for s in expected]
    counts = manifest["counts"]
    assert counts["contracts"] + counts["skipped_signals"] == len(expected)
    assert (run_dir / "chart.html").is_file()
