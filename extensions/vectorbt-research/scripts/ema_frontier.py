"""
Pareto frontier of the 3-EMA crossover: win rate, trades per day, drawdown and daily Sharpe.

Backtests a grid of settings (EMA lengths, expiry, slow-EMA trend filter) on the first 70% of a
span, on 80% of the CPU cores, with the exact cash rules of `ema_genetic`. Every setting gets
win rate, trust score, profit factor, daily Sharpe, drawdown, time in market and weekly
consistency. The frontier settings, and the best by trust score, are then tested once on the
last 30%.

Example:
    python scripts/ema_frontier.py --start 2026-02-16T00:00:00Z --end 2026-08-17T00:00:00Z \
        --output /tmp/ema-frontier
"""

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import os
import time
from functools import partial
from pathlib import Path

import ema_genetic as eg
import numpy as np
from vectorbt_research.input import _utc_nanoseconds as utc_ns

DAY_NS = 86_400 * 10**9
BREAK_EVEN = 1000 / 1952  # win rate where wins at 95.2% just cover losses
GRID = {
    "fast": range(5, 21, 3),
    "middle": range(15, 61, 5),
    "slow": range(100, 301, 25),
    "expiry": eg.EXPIRIES,
    "trend": (False, True),
}


def metrics(result, from_ns, to_ns):
    """Score one backtest. Money is exact cents; the scores are display floats."""
    days = (to_ns - from_ns) / DAY_NS
    n, wins = result["trades"], result["wins"]
    rate = wins / n if n else 0.0
    trust = (
        (rate - BREAK_EVEN) / math.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / n) if n else 0.0
    )
    times = np.array(result["times"], dtype=np.int64)
    values = np.array(result["values"], dtype=np.int64)

    def period_changes(step_ns):
        edges = np.arange(from_ns, to_ns + step_ns, step_ns)
        at = values[np.searchsorted(times, edges, side="right") - 1]
        return np.diff(at)

    daily = period_changes(DAY_NS)
    weekly = period_changes(7 * DAY_NS)
    sharpe = (
        float(daily.mean() / daily.std() * math.sqrt(365)) if daily.std() > 0 else 0.0
    )
    covered, end = 0, from_ns
    for begin, finish in sorted(result["in_market"]):
        begin = max(begin, end)
        if finish > begin:
            covered += finish - begin
            end = finish
    taken = n + result["skipped"] + result["unresolved"]
    return {
        "trades": n,
        "trades_per_day": round(n / days, 2),
        "win_rate": round(rate, 5),
        "trust": round(trust, 2),
        "profit_factor": round(result["profit_cents"] / result["loss_cents"], 4)
        if result["loss_cents"]
        else None,
        "net_pct": round(result["net_cents"] / eg.START_CASH * 100, 3),
        "net_usd": result["net_cents"] / 100,
        "max_drawdown_pct": round(result["drawdown"] * 100, 3),
        "daily_sharpe": round(sharpe, 3),
        "in_market_pct": round(covered / (to_ns - from_ns) * 100, 2),
        "weeks_positive_pct": round(float((weekly > 0).mean() * 100), 1),
        "skipped_pct": round(result["skipped"] / taken * 100, 1) if taken else 0.0,
    }


def evaluate(setting, from_ns, to_ns):
    fast, middle, slow, expiry, trend = setting
    result = eg.simulate(fast, middle, slow, expiry, from_ns, to_ns, trend)
    return setting, metrics(result, from_ns, to_ns)


def pareto(rows):
    """Settings no other setting beats on all of: win rate, trades/day, drawdown, Sharpe."""
    goals = np.array(
        [
            [
                r["win_rate"],
                r["trades_per_day"],
                -r["max_drawdown_pct"],
                r["daily_sharpe"],
            ]
            for r in rows
        ]
    )
    keep = []
    for i, g in enumerate(goals):
        better_or_equal = (goals >= g).all(axis=1)
        strictly = (goals > g).any(axis=1)
        if not (better_or_equal & strictly).any():
            keep.append(i)
    return keep


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default="mid", choices=["bid", "mid", "ask"])
    parser.add_argument("--min-trades", type=int, default=300)
    args = parser.parse_args()
    start_ns, end_ns = utc_ns(args.start), utc_ns(args.end)
    split_ns = start_ns + (end_ns - start_ns) * 7 // 10
    t0 = time.time()
    eg.load(start_ns, end_ns, args.scenario)
    print(
        f"loaded {len(eg.DATA['ts']):,} ticks in {time.time() - t0:.1f} s", flush=True
    )
    settings = [s for s in itertools.product(*GRID.values()) if s[0] < s[1] < s[2]]
    workers = max(1, int(os.cpu_count() * 0.8))
    t1 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        search = dict(
            pool.imap_unordered(
                partial(evaluate, from_ns=start_ns, to_ns=split_ns),
                settings,
                chunksize=8,
            )
        )
        search_s = time.time() - t1
        print(
            f"{len(search)} backtests on {workers} workers in {search_s:.1f} s",
            flush=True,
        )
        rows = [
            {"setting": list(s), **m}
            for s, m in search.items()
            if m["trades"] >= args.min_trades
        ]
        front = [rows[i] for i in pareto(rows)]
        by_trust = sorted(rows, key=lambda r: r["trust"], reverse=True)[:10]
        to_test = {tuple(r["setting"]) for r in front + by_trust}
        held = dict(
            pool.imap_unordered(
                partial(evaluate, from_ns=split_ns, to_ns=end_ns), sorted(to_test)
            )
        )
    for r in rows:
        if tuple(r["setting"]) in held:
            r["held_back"] = held[tuple(r["setting"])]
        r["frontier"] = r in front
    luck = math.sqrt(2 * math.log(len(settings)))
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "span": [args.start, args.end],
        "split_ns": split_ns,
        "scenario": args.scenario,
        "payout": "95.2% fixed for every expiry",
        "break_even_win_rate": round(BREAK_EVEN, 5),
        "settings_tested": len(settings),
        "settings_with_min_trades": len(rows),
        "min_trades": args.min_trades,
        "luck_trust_level": round(luck, 2),
        "workers": workers,
        "search_seconds": round(search_s, 1),
        "frontier_count": len(front),
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (folder / "points.json").write_text(json.dumps(rows), encoding="utf-8")
    with (folder / "all-settings.csv").open("w", newline="", encoding="utf-8") as out:
        fields = ["setting", *search[settings[0]].keys()]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for s, m in search.items():
            writer.writerow({"setting": "/".join(map(str, s)), **m})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
