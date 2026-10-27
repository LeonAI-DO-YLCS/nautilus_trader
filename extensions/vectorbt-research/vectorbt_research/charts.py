"""
Run charts: one interactive chart.html per saved run.

The chart has three panels: the run's price (bid, mid or ask) with entry and exit markers,
the equity curve (account value and free cash) and the drawdown. `chart_data` returns the
exact integer series; only `write_chart` turns them into display floats for plotly, through
vectorbt's plotting accessors. Charts never feed back into money or results.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import vectorbt  # noqa: F401  (registers the .vbt accessors)
from plotly.subplots import make_subplots

from vectorbt_research.input import ParsedQuote
from vectorbt_research.reports import read_run

_PRICE_FIELDS = {
    "bid": "bid_half_cents",
    "mid": "mid_half_cents",
    "ask": "ask_half_cents",
}
_OPENED_STATUSES = ("settled", "unresolved")
# Marker shape by direction and colour by outcome. An unresolved contract has no outcome.
_SYMBOLS = {"CALL": "triangle-up", "PUT": "triangle-down"}
_COLORS = {"win": "#1a9850", "loss": "#7f7f7f", "": "#e0a63a"}
_DIRECTION_NAMES = {"CALL": "Rise", "PUT": "Fall"}
_OUTCOME_NAMES = {"win": "win", "loss": "loss", "": "open"}


def chart_data(run_dir: str | Path, quotes: Sequence[ParsedQuote]) -> dict[str, Any]:
    """
    Return the exact integer series behind a run's chart.

    Prices are in half-cents (price times 200) and money in cents. The equity points
    follow the saved cash log row by row: the initial row at the first quote, an entry
    row at its contract's entry time and a settlement row at its contract's expiry. Times
    never go backward.
    """
    run = read_run(run_dir)
    scenario = run["manifest"]["scenario"]
    field = _PRICE_FIELDS[scenario]
    contracts = run["contracts"]
    opened = contracts[contracts["status"].isin(_OPENED_STATUSES)].to_dict("records")
    entries = [_marker(row, "entry_timestamp_ns", "entry_spot") for row in opened]
    exits = [
        _marker(row, "exit_timestamp_ns", "exit_spot")
        for row in opened
        if row["status"] == "settled"
    ]
    by_signal = contracts.set_index("signal_id")
    start_ns = quotes[0].timestamp_ns if quotes else 0
    equity = []
    peak = None
    latest_ns = start_ns
    for row in run["cash"].to_dict("records"):
        if row["event"] == "entry":
            time_ns = int(by_signal.loc[row["signal_id"], "entry_timestamp_ns"])
        elif row["event"] == "settlement":
            time_ns = int(by_signal.loc[row["signal_id"], "expiry_timestamp_ns"])
        else:
            time_ns = start_ns
        # Two contracts settled on the same quote can have expiries in the reverse of
        # their cash order; keep the time axis from going backward.
        time_ns = latest_ns = max(time_ns, latest_ns)
        available = int(row["available_cents"])
        locked = int(row["locked_cents"])
        value = available + locked
        peak = value if peak is None else max(peak, value)
        equity.append(
            {
                "event": row["event"],
                "signal_id": row["signal_id"],
                "timestamp_ns": time_ns,
                "available_cents": available,
                "locked_cents": locked,
                "value_cents": value,
                "peak_cents": peak,
                "drawdown_cents": value - peak,
            }
        )
    return {
        "scenario": scenario,
        "label": run["manifest"]["label"],
        "price": {
            "timestamp_ns": [quote.timestamp_ns for quote in quotes],
            "half_cents": [getattr(quote, field) for quote in quotes],
        },
        "entries": entries,
        "exits": exits,
        "equity": equity,
    }


def write_chart(run_dir: str | Path, quotes: Sequence[ParsedQuote]) -> Path:
    """
    Write chart.html into the run folder and return its path.

    The page is self-contained (plotly is inlined), so it opens offline. The file is
    written to a hidden temporary name first and then renamed.
    """
    run_dir = Path(run_dir)
    data = chart_data(run_dir, quotes)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=("Price and entries", "Equity", "Drawdown"),
    )
    price = _series(data["price"]["timestamp_ns"], data["price"]["half_cents"], 200)
    price.vbt.plot(
        fig=fig,
        add_trace_kwargs={"row": 1, "col": 1},
        trace_kwargs={"name": f"{data['scenario']} price", "line": {"width": 1}},
    )
    for direction, symbol in _SYMBOLS.items():
        for outcome, color in _COLORS.items():
            group = [
                m
                for m in data["entries"]
                if m["direction"] == direction and m["outcome"] == outcome
            ]
            if group:
                _markers(
                    fig,
                    group,
                    name=(
                        f"{_DIRECTION_NAMES[direction]} entry, "
                        f"{_OUTCOME_NAMES[outcome]}"
                    ),
                    symbol=symbol,
                    color=color,
                    size=11,
                )
    if data["exits"]:
        _markers(
            fig, data["exits"], name="Exit", symbol="circle", color="#3b6fd8", size=6
        )
    times = [point["timestamp_ns"] for point in data["equity"]]
    _series(times, [p["value_cents"] for p in data["equity"]], 100).vbt.plot(
        fig=fig,
        add_trace_kwargs={"row": 2, "col": 1},
        trace_kwargs={"name": "Account value (USD)", "line": {"shape": "hv"}},
    )
    _series(times, [p["available_cents"] for p in data["equity"]], 100).vbt.plot(
        fig=fig,
        add_trace_kwargs={"row": 2, "col": 1},
        trace_kwargs={
            "name": "Free cash (USD)",
            "line": {"shape": "hv", "dash": "dot"},
        },
    )
    drawdown = [
        p["drawdown_cents"] / p["peak_cents"] * 100 if p["peak_cents"] else 0.0
        for p in data["equity"]
    ]
    pd.Series(drawdown, index=_index(times)).vbt.plot(
        fig=fig,
        add_trace_kwargs={"row": 3, "col": 1},
        trace_kwargs={
            "name": "Drawdown (%)",
            "line": {"shape": "hv"},
            "fill": "tozeroy",
        },
    )
    fig.update_layout(
        title=f"{data['label']} · {data['scenario']} · {run_dir.name}",
        height=900,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.08},
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="USD", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
    path = run_dir / "chart.html"
    partial = run_dir / ".chart.html.partial"
    fig.write_html(partial, include_plotlyjs=True, full_html=True)
    partial.replace(path)
    return path


def _marker(row: dict[str, Any], time_field: str, spot_field: str) -> dict[str, Any]:
    return {
        "signal_id": row["signal_id"],
        "direction": row["direction"],
        "outcome": row["outcome"],
        "timestamp_ns": int(row[time_field]),
        "half_cents": int(row[spot_field]),
    }


def _index(times_ns: Sequence[int]) -> pd.DatetimeIndex:
    return pd.to_datetime(list(times_ns), unit="ns", utc=True)


def _series(times_ns: Sequence[int], values: Sequence[int], scale: int) -> pd.Series:
    """
    Return exact integers divided by their scale, as floats for display only.
    """
    return pd.Series([value / scale for value in values], index=_index(times_ns))


def _markers(fig, markers, *, name, symbol, color, size):
    series = _series(
        [m["timestamp_ns"] for m in markers], [m["half_cents"] for m in markers], 200
    )
    series.vbt.plot(
        fig=fig,
        add_trace_kwargs={"row": 1, "col": 1},
        trace_kwargs={
            "name": name,
            "mode": "markers",
            "marker": {"symbol": symbol, "color": color, "size": size},
            "text": [m["signal_id"] for m in markers],
            "hovertemplate": "%{text}<br>%{y}<extra>" + name + "</extra>",
        },
    )
