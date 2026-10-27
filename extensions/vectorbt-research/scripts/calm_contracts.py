"""
Would calm-paying Deriv contracts have paid off on 10 years of Volatility 75 ticks?

Reads quotes saved by the read-only quote probe (type, length, barrier offset as a fraction
of spot, payout for a 10 USD stake) and counts how often each contract would have won on
back-to-back windows of every yearly file. Entry spot: first tick at or after the start;
exit spot: last tick at or before expiry (both within 10 s). RANGE wins when every tick after
entry stays strictly inside both barriers, EXPIRYRANGE when the exit spot does, NOTOUCH when
no tick after entry reaches the upper barrier. Years run in parallel on up to 80% of cores.

Example:
    python scripts/calm_contracts.py quotes.json results.json
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from predictability import DATASET, load_year

FRESH_MS = 10_000
QUOTES: list = []


def year_counts(path):
    ts, mid = load_year(path)
    order = np.argsort(ts, kind="stable")
    ts, mid = ts[order], mid[order]
    out = []
    for q in QUOTES:
        length = q["minutes"] * 60_000
        starts = np.arange(ts[0] // length * length + length, ts[-1] - length, length)
        entry = np.searchsorted(ts, starts, side="left")
        exit_ = np.searchsorted(ts, starts + length, side="right") - 1
        ok = (
            (exit_ > entry)
            & (ts[entry] - starts <= FRESH_MS)
            & (starts + length - ts[exit_] <= FRESH_MS)
        )
        entry, exit_ = entry[ok], exit_[ok]
        spot = mid[entry].astype(np.float64)
        high = spot * (1 + q["offset_frac"])
        low = spot * (1 - q["offset_frac"])
        if q["type"] == "EXPIRYRANGE":
            end = mid[exit_]
            win = (end > low) & (end < high)
        else:
            # Max and min of the ticks after entry up to the exit tick, per window.
            bounds = np.empty(entry.size * 2, dtype=np.int64)
            bounds[0::2], bounds[1::2] = entry + 1, exit_ + 1
            top = np.maximum.reduceat(
                mid, bounds[:-1] if bounds[-1] == mid.size else bounds
            )[0::2]
            bottom = np.minimum.reduceat(
                mid, bounds[:-1] if bounds[-1] == mid.size else bounds
            )[0::2]
            win = (
                top < high if q["type"] == "NOTOUCH" else (top < high) & (bottom > low)
            )
        out.append({"n": int(win.size), "wins": int(win.sum())})
    return path.stem[-4:], out


def _init(quotes):
    QUOTES.extend(quotes)


def main():
    quotes = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    files = sorted(DATASET.glob("Volatility75Index_20[0-9][0-9].parquet"))
    workers = max(1, min(len(files), int(os.cpu_count() * 0.8)))
    with mp.get_context("spawn").Pool(
        workers, initializer=_init, initargs=(quotes,)
    ) as pool:
        years = dict(pool.map(year_counts, files))
    rows = []
    for i, q in enumerate(quotes):
        price = q["ask_price"]
        per_year = {y: c[i] for y, c in sorted(years.items())}

        def summary(selected, q=q, price=price):
            n = sum(c["n"] for c in selected)
            w = sum(c["wins"] for c in selected)
            rate = w / n
            return {
                "windows": n,
                "win_rate": round(rate, 5),
                "return_per_stake": round(rate * q["payout"] / price - 1, 4),
                "se": round((rate * (1 - rate) / n) ** 0.5 * q["payout"] / price, 4),
            }

        rows.append(
            {
                **{k: q[k] for k in ("type", "minutes", "k", "offset_frac", "payout")},
                "price_implied_win_rate": round(price / q["payout"], 5),
                "all_years": summary(per_year.values()),
                "since_2023": summary([c for y, c in per_year.items() if y >= "2023"]),
                "by_year_return": {
                    y: round(c["wins"] / c["n"] * q["payout"] / price - 1, 4)
                    for y, c in per_year.items()
                },
            }
        )
    Path(sys.argv[2]).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for r in rows:
        a, s = r["all_years"], r["since_2023"]
        print(
            f"{r['type']:12} {r['minutes']:>4}m k={r['k']:<4} payout {r['payout']:>6} "
            f"implied {r['price_implied_win_rate']:.3f} real {a['win_rate']:.3f} "
            f"return {a['return_per_stake']:+.3f}±{a['se']:.3f} "
            f"since2023 {s['return_per_stake']:+.3f}±{s['se']:.3f}"
        )


if __name__ == "__main__":
    main()
