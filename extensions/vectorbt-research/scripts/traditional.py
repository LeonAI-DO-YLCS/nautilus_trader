"""
The 3-EMA crossover as traditional trading on Volatility 75, with vectorbt portfolios.

Always in the market after the first signal: long when the 1-minute close crosses above all
three EMAs, short when it crosses below all three (stop and reverse). Orders fill at the next
minute's first mid price and pay half the real bid/ask spread on each side. Swap (overnight
financing) is not included. Settings are searched on the first 70% of a span and the best is
tested once on the last 30%; the best and 9/21/200 are then run on every year.

Example:
    python scripts/traditional.py --start 2026-02-16 --end 2026-08-17 --output /tmp/trad
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import vectorbt as vbt

DATASET = Path("/mnt/c/LeonAI-DO/dev/TRADING/Datasets/Volatility75Index_yearly_deduped")
GRID = {
    "fast": range(5, 21, 3),
    "middle": range(15, 61, 5),
    "slow": range(100, 301, 25),
}
DEFAULT = (9, 21, 200)
BARS: dict = {}


def _clock_part(clock, start, stop):
    return pc.cast(pc.utf8_slice_codeunits(clock, start, stop), pa.int64()).to_numpy()


def load_bars(year, start=None, end=None):
    """1-minute bars of one yearly file: open/close mid price and half spread at the open."""
    times, bids, asks = [], [], []
    for batch in pq.ParquetFile(
        DATASET / f"Volatility75Index_{year}.parquet"
    ).iter_batches(
        columns=["<DATE>", "<TIME>", "<BID>", "<ASK>"], batch_size=2_000_000
    ):
        clock = batch.column(1)
        day = pc.cast(
            pc.strptime(batch.column(0), format="%Y.%m.%d", unit="s"), pa.int64()
        )
        times.append(
            day.to_numpy() * 1000
            + _clock_part(clock, 0, 2) * 3_600_000
            + _clock_part(clock, 3, 5) * 60_000
            + _clock_part(clock, 6, 8) * 1000
            + _clock_part(clock, 9, 12)
        )
        bids.append(batch.column(2).to_numpy(zero_copy_only=False))
        asks.append(batch.column(3).to_numpy(zero_copy_only=False))
    ts, bid, ask = np.concatenate(times), np.concatenate(bids), np.concatenate(asks)
    ok = ~(np.isnan(bid) | np.isnan(ask))
    if start is not None:
        ok &= (ts >= start) & (ts < end)
    order = np.argsort(ts[ok], kind="stable")
    ts, bid, ask = ts[ok][order], bid[ok][order], ask[ok][order]
    minute, first = np.unique(ts // 60_000, return_index=True)
    last = np.append(first[1:] - 1, ts.size - 1)
    mid = (bid + ask) / 2
    index = pd.to_datetime(minute * 60_000, unit="ms", utc=True)
    return pd.DataFrame(
        {
            "open": mid[first],
            "close": mid[last],
            "half_spread": (ask[first] - bid[first]) / 2,
        },
        index=index,
    )


def targets(close, emas, setting):
    """+1 (long) or -1 (short) on the bar after each crossing, NaN when nothing changes."""
    ema = np.column_stack([emas[w] for w in setting])
    ready = ~np.isnan(ema).any(axis=1)
    c = close[:, None]
    above = ready & (c > ema).all(axis=1)
    below = ready & (c < ema).all(axis=1)
    was_ready = np.concatenate(([False], ready[:-1]))
    rise = was_ready & above & ~np.concatenate(([False], above[:-1]))
    fall = was_ready & below & ~np.concatenate(([False], below[:-1]))
    target = np.full(close.size, np.nan)
    target[1:][rise[:-1]] = 1.0
    target[1:][fall[:-1]] = -1.0
    return target


def backtest(bars, settings, part=None, fees=True):
    """Run settings as columns of one vectorbt portfolio; return one metrics row each."""
    close = bars["close"].to_numpy()
    windows = sorted({w for s in settings for w in s})
    ma = vbt.MA.run(bars["close"], window=windows, ewm=True).ma.to_numpy()
    emas = {w: ma[:, i] for i, w in enumerate(windows)}
    target = np.column_stack([targets(close, emas, s) for s in settings])
    if part is not None:
        # EMAs are warmed up on the whole span; trading happens only inside the part.
        target = target[part[0] : part[1]].copy()
        bars = bars.iloc[part[0] : part[1]]
    target[-1] = 0.0  # close any position at the end
    fee = (bars["half_spread"] / bars["open"]).to_numpy()[:, None] if fees else 0.0
    pf = vbt.Portfolio.from_orders(
        close=bars["close"],
        size=pd.DataFrame(target, index=bars.index),
        size_type="targetpercent",
        price=bars["open"],
        fees=fee,
        init_cash=10_000.0,
        freq="1min",
    )
    trades = pf.trades
    out = []
    for i, s in enumerate(settings):
        days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).total_seconds() / 86400
        n = int(trades.count().iloc[i])
        out.append(
            {
                "setting": list(s),
                "return_pct": round(float(pf.total_return().iloc[i]) * 100, 3),
                "max_drawdown_pct": round(float(pf.max_drawdown().iloc[i]) * -100, 3),
                "sharpe": round(float(pf.sharpe_ratio().iloc[i]), 3),
                "trades": n,
                "trades_per_day": round(n / days, 2) if days else 0,
                "win_rate": round(float(trades.win_rate().iloc[i]), 4) if n else None,
                "profit_factor": round(float(trades.profit_factor().iloc[i]), 4)
                if n
                else None,
            }
        )
    return out


def _search_chunk(args):
    settings, part, fees = args
    return backtest(BARS["six"], settings, part, fees)


def _year(args):
    year, settings = args
    bars = load_bars(year)
    return year, backtest(bars, settings), backtest(bars, settings, fees=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    start = int(
        datetime.fromisoformat(args.start).replace(tzinfo=UTC).timestamp() * 1000
    )
    end = int(datetime.fromisoformat(args.end).replace(tzinfo=UTC).timestamp() * 1000)
    t0 = time.time()
    BARS["six"] = load_bars(datetime.fromtimestamp(start / 1000, UTC).year, start, end)
    split = int(len(BARS["six"]) * 0.7)
    settings = [s for s in itertools.product(*GRID.values()) if s[0] < s[1] < s[2]]
    workers = max(1, int(os.cpu_count() * 0.8))
    chunks = [settings[i::workers] for i in range(workers)]
    with mp.get_context("fork").Pool(workers) as pool:
        search = [
            r
            for rows in pool.map(_search_chunk, [(c, (0, split), True) for c in chunks])
            for r in rows
        ]
        gross = [
            r
            for rows in pool.map(
                _search_chunk, [(c, (0, split), False) for c in chunks]
            )
            for r in rows
        ]
        search_s = time.time() - t0
        ranked = sorted(
            (r for r in search if r["trades"] >= 100),
            key=lambda r: r["sharpe"],
            reverse=True,
        )
        best = tuple(ranked[0]["setting"])
        held = backtest(BARS["six"], [best, DEFAULT], (split, len(BARS["six"])))
    years = sorted(
        int(p.stem[-4:]) for p in DATASET.glob("Volatility75Index_20[0-9][0-9].parquet")
    )
    # A full year needs about 3 GB, so memory (32 GB) caps this step at 6 workers.
    with mp.get_context("fork").Pool(min(workers, 6)) as pool:
        per_year = pool.map(_year, [(y, [best, DEFAULT]) for y in years])
    summary = {
        "span": [args.start, args.end],
        "search_bars": split,
        "held_back_bars": len(BARS["six"]) - split,
        "settings": len(settings),
        "workers": workers,
        "search_seconds": round(search_s, 1),
        "costs": "half the bid/ask spread per side; swap not included",
        "search_profitable_after_costs": sum(r["return_pct"] > 0 for r in search),
        "search_profitable_before_costs": sum(r["return_pct"] > 0 for r in gross),
        "search_median_return_pct": float(np.median([r["return_pct"] for r in search])),
        "search_median_return_before_costs_pct": float(
            np.median([r["return_pct"] for r in gross])
        ),
        "best_by_sharpe": ranked[0],
        "top5": ranked[:5],
        "held_back": {"best": held[0], "default_9_21_200": held[1]},
        "by_year": {
            y: {"best": net[0], "default": net[1], "best_before_costs": raw[0]}
            for y, net, raw in per_year
        },
    }
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "traditional.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
