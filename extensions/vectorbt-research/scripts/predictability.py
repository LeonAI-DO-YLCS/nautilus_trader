"""
Is Volatility 75 predictable? Simple pattern tests on every tick of every yearly file.

For each pattern the script counts how often the next 1-minute Rise/Fall contract (entry: first
tick at or after the minute start; exit: last tick at or before start + 60 s; a tie loses)
would win if traded in the pattern's best direction. A pattern can only make money above the
51.23% break-even of a 95.2% payout. Years run in parallel on up to 80% of the CPU cores.

Example:
    python scripts/predictability.py --output /tmp/predictability
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DATASET = Path("/mnt/c/LeonAI-DO/dev/TRADING/Datasets/Volatility75Index_yearly_deduped")
BREAK_EVEN = 1000 / 1952
MINUTE_MS = 60_000
FRESH_MS = 10_000  # entry and exit ticks must be within 10 s of the minute edges


def _clock_part(clock, start, stop):
    """Digits [start, stop) of HH:MM:SS.fff as integers."""
    return pc.cast(pc.utf8_slice_codeunits(clock, start, stop), pa.int64()).to_numpy()


def load_year(path):
    """Return tick times (UTC ms) and mid prices (half-cents) of one yearly file."""
    times, mids = [], []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        columns=["<DATE>", "<TIME>", "<BID>", "<ASK>"], batch_size=2_000_000
    ):
        date, clock = batch.column(0), batch.column(1)
        day_s = pc.cast(pc.strptime(date, format="%Y.%m.%d", unit="s"), pa.int64())

        ms = (
            pc.multiply(day_s, 1000).to_numpy()
            + _clock_part(clock, 0, 2) * 3_600_000
            + _clock_part(clock, 3, 5) * 60_000
            + _clock_part(clock, 6, 8) * 1000
            + _clock_part(clock, 9, 12)
        )
        bid = np.rint(batch.column(2).to_numpy(zero_copy_only=False) * 100)
        ask = np.rint(batch.column(3).to_numpy(zero_copy_only=False) * 100)
        ok = ~(np.isnan(bid) | np.isnan(ask))
        times.append(ms[ok])
        mids.append((bid[ok] + ask[ok]).astype(np.int64))
    return np.concatenate(times), np.concatenate(mids)


def minute_outcomes(ts, mid):
    """
    Return +1 (Rise wins), -1 (Fall wins) or 0 (tie) for every minute with fresh ticks, the
    minute's index, and its log return.
    """
    first = ts[0] // MINUTE_MS * MINUTE_MS + MINUTE_MS
    starts = np.arange(first, ts[-1] - MINUTE_MS, MINUTE_MS)
    entry = np.searchsorted(ts, starts, side="left")
    exit_ = np.searchsorted(ts, starts + MINUTE_MS, side="right") - 1
    valid = (
        (entry < len(ts))
        & (exit_ >= entry)
        & (ts[np.minimum(entry, len(ts) - 1)] - starts <= FRESH_MS)
        & (starts + MINUTE_MS - ts[exit_] <= FRESH_MS)
    )
    entry, exit_, index = entry[valid], exit_[valid], (starts[valid] // MINUTE_MS)
    sign = np.sign(mid[exit_] - mid[entry])
    ret = np.log(mid[exit_] / mid[entry])
    return sign, index, ret


def count(condition, next_sign):
    """Counts for 'trade Rise' and 'trade Fall' after the condition."""
    nxt = next_sign[condition]
    return {
        "n": int(nxt.size),
        "up": int((nxt > 0).sum()),
        "down": int((nxt < 0).sum()),
    }


def year_stats(path):
    t0 = time.time()
    ts, mid = load_year(path)
    order = np.argsort(ts, kind="stable")
    ts, mid = ts[order], mid[order]
    sign, index, ret = minute_outcomes(ts, mid)
    # Pairs of back-to-back minutes only.
    pair = np.diff(index) == 1
    prev, nxt = sign[:-1][pair], sign[1:][pair]
    prev_ret = ret[:-1][pair]
    sigma = ret.std()
    stats = {
        "ticks": int(ts.size),
        "minutes": int(sign.size),
        "ties": int((sign == 0).sum()),
        "all minutes": count(np.ones_like(prev, dtype=bool), nxt),
        "after a Rise minute": count(prev > 0, nxt),
        "after a Fall minute": count(prev < 0, nxt),
        "after a move over 2 sigma up": count(prev_ret > 2 * sigma, nxt),
        "after a move over 2 sigma down": count(prev_ret < -2 * sigma, nxt),
        "after a move over 3 sigma up": count(prev_ret > 3 * sigma, nxt),
        "after a move over 3 sigma down": count(prev_ret < -3 * sigma, nxt),
        "annual_volatility": float(sigma * math.sqrt(365 * 24 * 60)),
    }
    for k in (2, 3, 5):
        # k Rise (or Fall) minutes in a row, all back to back, then the next minute.
        runs_up = np.ones(sign.size - 1, dtype=bool)
        runs_down = np.ones(sign.size - 1, dtype=bool)
        for lag in range(k):
            s = np.roll(sign[:-1], lag)
            back = np.roll(np.diff(index) == 1, lag)
            runs_up &= (s > 0) & back
            runs_down &= (s < 0) & back
        runs_up[:k], runs_down[:k] = False, False
        stats[f"after {k} Rise minutes in a row"] = count(runs_up[pair], nxt)
        stats[f"after {k} Fall minutes in a row"] = count(runs_down[pair], nxt)
    diff = np.diff(mid)
    moves = np.sign(diff[diff != 0])
    stats["ticks: next move same direction"] = {
        "n": int(moves.size - 1),
        "same": int((moves[1:] == moves[:-1]).sum()),
    }
    stats["seconds"] = round(time.time() - t0, 1)
    return path.stem[-4:], stats


def best_direction(c):
    """Best win rate from trading Rise or Fall after the pattern; ties lose."""
    if not c["n"]:
        return None
    up, down = c["up"] / c["n"], c["down"] / c["n"]
    rate = max(up, down)
    return {
        "trades": c["n"],
        "trade": "Rise" if up >= down else "Fall",
        "win_rate": round(rate, 5),
        "z_vs_break_even": round(
            (rate - BREAK_EVEN) / math.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / c["n"]), 2
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = sorted(DATASET.glob("Volatility75Index_20[0-9][0-9].parquet"))
    workers = max(1, min(len(files), int(os.cpu_count() * 0.8)))
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        years = dict(pool.map(year_stats, files))
    patterns = [
        k
        for k in years[next(iter(years))]
        if k.startswith("after") or k == "all minutes"
    ]
    total = {}
    for name in patterns:
        c = {"n": 0, "up": 0, "down": 0}
        for y in years.values():
            for key in c:
                c[key] += y[name][key]
        total[name] = best_direction(c)
    tick = {"n": 0, "same": 0}
    for y in years.values():
        tick["n"] += y["ticks: next move same direction"]["n"]
        tick["same"] += y["ticks: next move same direction"]["same"]
    by_year = {
        year: {
            "ticks": y["ticks"],
            "minutes": y["minutes"],
            "tie_rate": round(y["ties"] / y["minutes"], 5),
            "annual_volatility_pct": round(y["annual_volatility"] * 100, 2),
            "best_pattern": max(
                ((n, best_direction(y[n])) for n in patterns if best_direction(y[n])),
                key=lambda item: item[1]["win_rate"],
            ),
        }
        for year, y in sorted(years.items())
    }
    summary = {
        "files": [f.name for f in files],
        "workers": workers,
        "seconds": round(time.time() - t0, 1),
        "break_even": round(BREAK_EVEN, 5),
        "patterns_all_years": total,
        "ticks_same_direction_rate": round(tick["same"] / tick["n"], 5),
        "tick_moves": tick["n"],
        "by_year": by_year,
    }
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "predictability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
