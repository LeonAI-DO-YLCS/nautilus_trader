"""
Run charts: every saved run gets chart.html with price and entries, equity and drawdown.

C-01 a run writes a self-contained chart.html next to its unchanged report files.
C-02 the price panel shows every quote at the run's own price (bid, mid or ask).
C-03 each opened contract has one entry and one exit marker; skipped signals have none.
C-04 the equity points follow the saved cash log, timed by contract entry and expiry.
C-05 drawdown is the fall of account value below its highest value so far.
"""

from datetime import UTC, datetime, timedelta

import pytest
from vectorbt_research.charts import chart_data
from vectorbt_research.input import parse_quote_rows
from vectorbt_research.reports import read_run
from vectorbt_research.runner import run_story_1

_BASE = datetime(2026, 8, 18, 3, 20, tzinfo=UTC)
_REPORT_FILES = {
    "manifest.json",
    "signals.parquet",
    "contracts.parquet",
    "cash.parquet",
    "summary.txt",
}


def _timestamp(seconds):
    return (_BASE + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _ns(seconds):
    return int((_BASE + timedelta(seconds=seconds)).timestamp()) * 1_000_000_000


def _quote(ordinal, seconds, bid, ask):
    return {
        "timestamp": _timestamp(seconds),
        "ordinal": ordinal,
        "bid": bid,
        "ask": ask,
    }


# Bid and ask are 0.50 apart, so mid sits 0.25 from each.
_QUOTES = [
    _quote(1, 2, "100.00", "100.50"),
    _quote(2, 30, "100.40", "100.90"),
    _quote(3, 60, "101.00", "101.50"),
    _quote(4, 61, "99.00", "99.50"),
    _quote(5, 120, "98.00", "98.50"),
    _quote(6, 122, "98.20", "98.70"),
]
# A Rise win and a Fall loss in the first minute, then a Fall win in the next minute.
_SIGNALS = [
    {"signal_id": "rise-win", "direction": "CALL", "start_time": _timestamp(0)},
    {"signal_id": "fall-loss", "direction": "PUT", "start_time": _timestamp(0)},
    {"signal_id": "fall-win", "direction": "PUT", "start_time": _timestamp(61)},
]


def _run(tmp_path, scenario="bid", signals=_SIGNALS, starting_cash_cents=1_000_000):
    return run_story_1(
        quote_rows=_QUOTES,
        frozen_signals=signals,
        scenario=scenario,
        output_dir=tmp_path / scenario,
        starting_cash_cents=starting_cash_cents,
    )


def _data(run_dir):
    return chart_data(run_dir, parse_quote_rows(_QUOTES)[0])


def test_run_writes_self_contained_chart_next_to_unchanged_report(tmp_path):
    """C-01: a run writes a self-contained chart.html next to its unchanged report files"""
    run_dir = _run(tmp_path)
    names = {path.name for path in run_dir.iterdir()}
    assert names == _REPORT_FILES | {"chart.html"}
    html = (run_dir / "chart.html").read_text(encoding="utf-8")
    assert "<script" in html and "Plotly" in html
    assert '<script src="http' not in html and "<script src='http" not in html
    for title in ("Price and entries", "Equity", "Drawdown"):
        assert title in html
    assert "MT5 simulation" in html
    read_run(run_dir)


@pytest.mark.parametrize(
    ("scenario", "expected_half_cents"),
    [
        ("bid", [20000, 20080, 20200, 19800, 19600, 19640]),
        ("mid", [20050, 20130, 20250, 19850, 19650, 19690]),
        ("ask", [20100, 20180, 20300, 19900, 19700, 19740]),
    ],
)
def test_price_panel_shows_every_quote_at_the_run_price(
    tmp_path, scenario, expected_half_cents
):
    """C-02: the price panel shows every quote at the run's own price"""
    data = _data(_run(tmp_path, scenario))
    assert data["scenario"] == scenario
    assert data["price"]["timestamp_ns"] == [_ns(s) for s in (2, 30, 60, 61, 120, 122)]
    assert data["price"]["half_cents"] == expected_half_cents


def test_each_opened_contract_has_one_entry_and_one_exit_marker(tmp_path):
    """C-03: each opened contract has one entry and one exit marker"""
    run_dir = _run(tmp_path)
    data = _data(run_dir)
    contracts = read_run(run_dir)["contracts"].set_index("signal_id")
    assert [m["signal_id"] for m in data["entries"]] == [
        "rise-win",
        "fall-loss",
        "fall-win",
    ]
    for marker in data["entries"]:
        row = contracts.loc[marker["signal_id"]]
        assert marker["timestamp_ns"] == row["entry_timestamp_ns"]
        assert marker["half_cents"] == row["entry_spot"]
        assert marker["direction"] == row["direction"]
        assert marker["outcome"] == row["outcome"]
    assert [m["outcome"] for m in data["entries"]] == ["win", "loss", "win"]
    for marker in data["exits"]:
        row = contracts.loc[marker["signal_id"]]
        assert marker["timestamp_ns"] == row["exit_timestamp_ns"]
        assert marker["half_cents"] == row["exit_spot"]
    assert len(data["exits"]) == 3


def test_skipped_signal_has_no_marker(tmp_path):
    """C-03: a skipped signal (not enough cash) has no entry or exit marker"""
    data = _data(_run(tmp_path, signals=_SIGNALS[:1], starting_cash_cents=34))
    assert data["entries"] == []
    assert data["exits"] == []


def test_equity_points_follow_the_saved_cash_log(tmp_path):
    """C-04: the equity points follow the saved cash log, timed by entry and expiry"""
    run_dir = _run(tmp_path)
    run = read_run(run_dir)
    data = _data(run_dir)
    cash = run["cash"].to_dict("records")
    contracts = run["contracts"].set_index("signal_id")
    assert len(data["equity"]) == len(cash)
    first = data["equity"][0]
    assert first["timestamp_ns"] == _ns(2)
    assert (first["available_cents"], first["locked_cents"]) == (1_000_000, 0)
    for point, row in zip(data["equity"], cash, strict=True):
        assert point["available_cents"] == row["available_cents"]
        assert point["locked_cents"] == row["locked_cents"]
        assert point["value_cents"] == row["available_cents"] + row["locked_cents"]
        if row["event"] == "entry":
            assert (
                point["timestamp_ns"]
                == contracts.loc[row["signal_id"], "entry_timestamp_ns"]
            )
        elif row["event"] == "settlement":
            assert (
                point["timestamp_ns"]
                == contracts.loc[row["signal_id"], "expiry_timestamp_ns"]
            )
    times = [point["timestamp_ns"] for point in data["equity"]]
    assert times == sorted(times)
    assert (
        data["equity"][-1]["value_cents"]
        == cash[-1]["available_cents"] + cash[-1]["locked_cents"]
    )


def test_equity_times_never_go_backward_when_signals_are_out_of_order(tmp_path):
    """C-04: two contracts settled on one quote, listed latest first, keep the time order"""
    rows = [
        _quote(1, 8, "100.00", "100.50"),
        _quote(2, 58, "100.20", "100.70"),
        _quote(3, 100, "100.40", "100.90"),
    ]
    signals = [
        {"signal_id": "late", "direction": "CALL", "start_time": _timestamp(6)},
        {"signal_id": "early", "direction": "PUT", "start_time": _timestamp(0)},
    ]
    run_dir = run_story_1(
        quote_rows=rows,
        frozen_signals=signals,
        scenario="bid",
        output_dir=tmp_path / "out_of_order",
    )
    data = chart_data(run_dir, parse_quote_rows(rows)[0])
    cash = read_run(run_dir)["cash"].to_dict("records")
    assert [p["signal_id"] for p in data["equity"]] == [r["signal_id"] for r in cash]
    assert [p["event"] for p in data["equity"]][-2:] == ["settlement", "settlement"]
    assert [p["signal_id"] for p in data["equity"]][-2:] == ["late", "early"]
    times = [p["timestamp_ns"] for p in data["equity"]]
    assert times == sorted(times)
    assert times[-2:] == [_ns(66), _ns(66)]


def test_drawdown_is_the_fall_below_the_highest_value_so_far(tmp_path):
    """C-05: drawdown is the fall of account value below its highest value so far"""
    data = _data(_run(tmp_path))
    peak = 0
    for point in data["equity"]:
        peak = max(peak, point["value_cents"])
        assert point["peak_cents"] == peak
        assert point["drawdown_cents"] == point["value_cents"] - peak
        assert point["drawdown_cents"] <= 0
    assert data["equity"][0]["drawdown_cents"] == 0
    assert min(point["drawdown_cents"] for point in data["equity"]) < 0


def test_open_contract_has_an_entry_marker_but_no_exit_marker(tmp_path):
    """C-03: a contract still open when the data ends has an entry marker and no exit"""
    run_dir = run_story_1(
        quote_rows=_QUOTES[:5],
        frozen_signals=_SIGNALS,
        scenario="bid",
        output_dir=tmp_path / "open",
    )
    data = chart_data(run_dir, parse_quote_rows(_QUOTES[:5])[0])
    assert [m["outcome"] for m in data["entries"]] == ["win", "loss", ""]
    assert [m["signal_id"] for m in data["exits"]] == ["rise-win", "fall-loss"]
    assert data["equity"][-1]["locked_cents"] > 0
