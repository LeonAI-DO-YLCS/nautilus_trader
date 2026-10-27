"""
Return-versus-drawdown frontier of the 3-EMA crossover on gold, against buy and hold.

Same rule and costs as scripts/traditional_slow.py, on 1-hour, 4-hour and daily bars, trading
both ways or buy only (flat instead of short). Every setting is scored on the first 70% of the
span (Aug 2016 to Aug 2023): return, worst drop, yearly return per unit of worst drop, daily
Sharpe, time in market and number of position changes. Settings that no other setting beats
on both return and worst drop form the frontier, as scripts/ema_frontier.py did for
Volatility 75. The frontier is then tested once on the last 30%, next to buy and hold, and the pick (the
best yearly return per unit of worst drop in the search) is also run on every calendar year.

Example:
    python scripts/trend_frontier.py --data ~/data/histdata --output /tmp/gold-frontier
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import traditional as t
import traditional_markets as m
import traditional_slow as s
import vectorbt as vbt

MIN_CHANGES = 20
DATA: dict = {}


def _scores(value, changes, total, drawdown):
    days = (value.index[-1] - value.index[0]).total_seconds() / 86_400
    yearly = (1 + total) ** (365.25 / days) - 1
    daily = value.resample("1D").last().dropna().pct_change().dropna()
    return {
        "return_pct": round(total * 100, 2),
        "max_drawdown_pct": round(drawdown * 100, 2),
        "yearly_return_pct": round(yearly * 100, 2),
        "return_per_drawdown": round(yearly / drawdown, 3) if drawdown else None,
        "daily_sharpe": round(float(daily.mean() / daily.std() * np.sqrt(252)), 3)
        if daily.std() > 0
        else 0.0,
        "position_changes": int(changes),
    }


def backtest(bars, settings, part, buy_only):
    """One vectorbt portfolio, one column per EMA setting, traded only inside part."""
    close = bars["close"].to_numpy()
    windows = sorted({w for st in settings for w in st})
    ma = vbt.MA.run(bars["close"], window=windows, ewm=True).ma.to_numpy()
    emas = {w: ma[:, i] for i, w in enumerate(windows)}
    target = np.column_stack([t.targets(close, emas, st) for st in settings])
    if buy_only:
        target[target < 0] = 0.0
    target = target[part[0] : part[1]].copy()
    bars = bars.iloc[part[0] : part[1]]
    target[-1] = 0.0
    pf = vbt.Portfolio.from_orders(
        close=bars["close"],
        size=pd.DataFrame(target, index=bars.index),
        size_type="targetpercent",
        price=bars["open"],
        fees=(bars["half_spread"] / bars["open"]).to_numpy()[:, None],
        init_cash=10_000.0,
        freq="1h",
    )
    value = pf.value()
    total = pf.total_return().to_numpy()
    drawdown = -pf.max_drawdown().to_numpy()
    state = pd.DataFrame(target).ffill().fillna(0.0)
    changes = state.diff().fillna(state).ne(0).sum().to_numpy()
    in_market = (pf.assets() != 0).mean().to_numpy()
    out = []
    for i, st in enumerate(settings):
        row = _scores(value.iloc[:, i], changes[i], total[i], drawdown[i])
        row["in_market_pct"] = round(float(in_market[i]) * 100, 1)
        out.append({"setting": list(st), **row})
    return out


def hold(bars, part):
    close = bars["close"].iloc[part[0] : part[1]]
    value = close / bars["open"].iloc[part[0]] * 10_000.0
    drawdown = float((1 - value / value.cummax()).max())
    return _scores(value, 1, value.iloc[-1] / 10_000.0 - 1, drawdown)


def frontier(rows):
    """Settings that no other setting beats on both return and worst drop."""
    goals = np.array([[r["return_pct"], -r["max_drawdown_pct"]] for r in rows])
    keep = []
    for i, g in enumerate(goals):
        if not ((goals >= g).all(axis=1) & (goals > g).any(axis=1)).any():
            keep.append(rows[i])
    return sorted(keep, key=lambda r: r["max_drawdown_pct"])


def _search(args):
    frame, buy_only, settings = args
    part = (0, int(len(DATA[frame]) * 0.7))
    rows = backtest(DATA[frame], settings, part, buy_only)
    return [{"frame": frame, "buy_only": buy_only, **r} for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair", default="XAUUSD")
    args = parser.parse_args()
    t0 = time.time()
    minute = m.load(args.data, args.pair)
    for frame, offset in s.FRAMES.items():
        DATA[frame] = s.resample(minute, frame, offset)
    settings = [
        st for st in itertools.product(*t.GRID.values()) if st[0] < st[1] < st[2]
    ]
    workers = max(1, int(os.cpu_count() * 0.8))
    per = max(1, workers // (len(DATA) * 2))
    tasks = [
        (frame, buy_only, settings[i::per])
        for frame in DATA
        for buy_only in (False, True)
        for i in range(per)
    ]
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for chunk in pool.map(_search, tasks) for r in chunk]
    rows = [r for r in rows if r["position_changes"] >= MIN_CHANGES]
    front = frontier(rows)
    pick = max(rows, key=lambda r: r["return_per_drawdown"] or -1e9)
    held = []
    for (frame, buy_only), group in itertools.groupby(
        sorted(front + [pick], key=lambda r: (r["frame"], r["buy_only"])),
        key=lambda r: (r["frame"], r["buy_only"]),
    ):
        bars = DATA[frame]
        sts = sorted({tuple(r["setting"]) for r in group})
        for r in backtest(bars, sts, (int(len(bars) * 0.7), len(bars)), buy_only):
            held.append({"frame": frame, "buy_only": buy_only, **r})
    bars = DATA[pick["frame"]]
    years = np.asarray(bars.index.year)
    by_year = {}
    for year in sorted(set(years)):
        part = (
            int(np.argmax(years == year)),
            int(len(years) - np.argmax(years[::-1] == year)),
        )
        by_year[int(year)] = {
            "pick": backtest(bars, [tuple(pick["setting"])], part, pick["buy_only"])[0],
            "hold": hold(bars, part),
        }
    split = {f: int(len(b) * 0.7) for f, b in DATA.items()}
    summary = {
        "pair": args.pair,
        "costs": "half of a fixed spread per side; overnight fees not included (holding pays them too)",
        "spread": m.SPREAD[args.pair],
        "settings_per_combination": len(settings),
        "backtests": len(settings) * len(DATA) * 2,
        "min_position_changes": MIN_CHANGES,
        "workers": workers,
        "seconds": round(time.time() - t0, 1),
        "held_back_from": {f: str(DATA[f].index[split[f]]) for f in DATA},
        "hold_search": hold(DATA["1h"], (0, split["1h"])),
        "hold_held_back": hold(DATA["1h"], (split["1h"], len(DATA["1h"]))),
        "frontier": front,
        "pick": pick,
        "held_back": held,
        "pick_by_year": by_year,
    }
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "trend_frontier.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (folder / "points.json").write_text(json.dumps(rows), encoding="utf-8")
    print(
        f"done in {summary['seconds']} s; {len(rows)} settings, {len(front)} on the frontier"
    )


if __name__ == "__main__":
    main()
