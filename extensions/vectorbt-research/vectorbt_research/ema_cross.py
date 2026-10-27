"""
EMA crossover strategy: three exponential moving averages on 1-minute closes.

The close of a 1-minute bar is the last quote of that UTC minute, in the run's price
scenario (bid, mid or ask). A Rise (CALL) signal fires on the bar where the close moves
above all three averages; a Fall (PUT) signal fires on the bar where it moves below all
three. A signal starts when its bar closes, so it never uses a price before it exists.
The averages are floats used only for comparisons; money stays exact in the ledger.
"""

import argparse
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from vectorbt_research.dataset import read_quote_window
from vectorbt_research.input import ParsedQuote, parse_quote_rows
from vectorbt_research.input import _utc_nanoseconds as _utc_ns
from vectorbt_research.reports import read_run
from vectorbt_research.runner import run_story_1_window

DEFAULT_WINDOWS = (9, 21, 200)
_PRICE_FIELDS = {
    "bid": "bid_half_cents",
    "mid": "mid_half_cents",
    "ask": "ask_half_cents",
}
_BAR = pd.Timedelta(minutes=1)


def minute_closes(quotes: Sequence[ParsedQuote], scenario: str) -> pd.Series:
    """
    Return the last price of each UTC minute that has quotes, in half-cents.

    The index is the bar's opening minute. Minutes without quotes have no bar.
    """
    field = _PRICE_FIELDS[scenario]
    prices = pd.Series(
        [getattr(quote, field) for quote in quotes],
        index=pd.to_datetime(
            [quote.timestamp_ns for quote in quotes], unit="ns", utc=True
        ),
        dtype="int64",
    )
    return prices.groupby(prices.index.floor("1min")).last()


def ema_cross_signals(
    quotes: Sequence[ParsedQuote],
    scenario: str,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> list[dict[str, str]]:
    """
    Return the strategy's signals as frozen signals, in time order.

    Each signal holds `signal_id`, `direction` (CALL or PUT) and `start_time` (UTC text,
    the close of its bar). A bar counts only when every average and the previous bar's
    averages exist, so the first bars of a window only warm the averages up.
    """
    if len(windows) == 0 or any(int(w) != w or w < 1 for w in windows):
        raise ValueError("EMA windows must be positive whole numbers")
    close = minute_closes(quotes, scenario).astype("float64")
    if close.empty:
        return []
    emas = vbt.MA.run(close, window=list(windows), ewm=True).ma.to_numpy()
    ready = ~np.isnan(emas).any(axis=1)
    values = close.to_numpy()[:, None]
    above = ready & (values > emas).all(axis=1)
    below = ready & (values < emas).all(axis=1)
    was_ready = np.concatenate(([False], ready[:-1]))
    was_above = np.concatenate(([False], above[:-1]))
    was_below = np.concatenate(([False], below[:-1]))
    rise = was_ready & above & ~was_above
    fall = was_ready & below & ~was_below
    signals = []
    for opened, is_rise, is_fall in zip(close.index, rise, fall, strict=True):
        if not (is_rise or is_fall):
            continue
        direction = "CALL" if is_rise else "PUT"
        start = opened + _BAR
        signals.append(
            {
                "signal_id": f"ema-{start:%Y%m%dT%H%M}-{direction}",
                "direction": direction,
                "start_time": f"{start:%Y-%m-%dT%H:%M:%SZ}",
            }
        )
    return signals


def run_ema_cross_window(
    *,
    dataset_folder: str | Path,
    window_start_ns: int,
    window_end_ns: int,
    scenario: str,
    output_dir: str | Path,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    starting_cash_cents: int = 1_000_000,
) -> Path:
    """
    Backtest the strategy on [window_start_ns, window_end_ns) and return the saved run.

    The signals come from the window's quotes; the saved run, its checks and its chart
    come from the shared runner. The window must lie in one yearly file.
    """
    year = datetime.fromtimestamp(window_start_ns // 1_000_000_000, tz=UTC).year
    parquet_path = Path(dataset_folder) / f"Volatility75Index_{year}.parquet"
    quotes, _ = parse_quote_rows(
        read_quote_window(parquet_path, window_start_ns, window_end_ns)
    )
    signals = [
        signal
        for signal in ema_cross_signals(quotes, scenario, windows)
        if _utc_ns(signal["start_time"]) < window_end_ns
    ]
    return run_story_1_window(
        dataset_folder=dataset_folder,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        frozen_signals=signals,
        scenario=scenario,
        output_dir=output_dir,
        starting_cash_cents=starting_cash_cents,
        purpose="EMA crossover backtest",
        strategy={
            "source": "vectorbt_research.ema_cross",
            "hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parameters": {
                "ema_windows": [int(w) for w in windows],
                "bar": "1 minute, last quote of the minute",
                "rise": "close moves above all averages",
                "fall": "close moves below all averages",
            },
            "seed": "not applicable",
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the 9, 21 and 200 EMA crossover on one V75 window."
    )
    parser.add_argument("--dataset", required=True, help="yearly dataset folder")
    parser.add_argument(
        "--start", required=True, help="UTC start, e.g. 2026-08-17T00:00:00Z"
    )
    parser.add_argument("--end", required=True, help="UTC end, not included")
    parser.add_argument("--scenario", choices=sorted(_PRICE_FIELDS), default="mid")
    parser.add_argument("--output", required=True, help="folder for the saved run")
    args = parser.parse_args(argv)
    run_dir = run_ema_cross_window(
        dataset_folder=args.dataset,
        window_start_ns=_utc_ns(args.start),
        window_end_ns=_utc_ns(args.end),
        scenario=args.scenario,
        output_dir=args.output,
    )
    print(run_dir)
    print(read_run(run_dir)["summary"])


if __name__ == "__main__":
    main()
