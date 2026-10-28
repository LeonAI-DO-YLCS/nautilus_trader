"""
Does the Volatility 75 price generator have habits? A check of every tick, 2016 to 2025.

A fair generator moves like a fair coin with a fixed step size: no memory, no favourite
hours, no favourite digits. This measures, on mid prices:
  - volatility: the yearly size of moves, against the advertised 75%;
  - big moves: how often a tick moves more than 4 and 5 typical sizes (normal = rare);
  - memory: tick-to-tick correlation, variance ratios, and what follows 1 to 5 ticks in a
    row in the same direction;
  - bounce-back: after the biggest rises or falls over h ticks, the average move over the
    next h ticks, against the round-trip spread a trader pays;
  - hours and weekdays: size of moves by UTC hour and weekday;
  - last digits of the quoted bid price.
Gaps longer than a minute between ticks are left out of every move, and so are the spans
in BAD_TICKS (two interleaved price feeds in the dataset); --keep-bad-ticks keeps them.

Example:
    python scripts/v75_generator_check.py --output /tmp/gen
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import v75_recovery as rec
from scipy import stats

SECONDS_PER_YEAR = 365 * 86_400
# Spans where the file alternates between two price levels every tick (lag-1 correlation
# -0.4 to -0.9, ticks every second instead of about two): a recording fault, not the index.
BAD_TICKS = (
    ("2016-09-09 03:00", "2016-09-09 06:00"),
    ("2024-04-17", "2024-05-01"),
)
HORIZONS = (1, 5, 30, 300, 1800)
LAGS = (1, 2, 3, 5, 10)
RATIOS = (2, 5, 10, 30, 300, 1800)


def variance_ratio(c, q):
    """Variance of q-tick moves over q times the variance of 1-tick moves (1 = no memory)."""
    one = np.diff(c)
    many = c[q::q] - c[:-q:q][: c[q::q].size]
    return float(many.var() / (q * one.var()))


def bounce(c, h, spread):
    """Mean next-h-tick move after the top and bottom tenth of past-h-tick moves."""
    i = np.arange(h, c.size - h, h)
    past = c[i] - c[i - h]
    nxt = c[i + h] - c[i]
    lo, hi = np.quantile(past, [0.1, 0.9])
    after_fall = nxt[past <= lo].mean()
    after_rise = nxt[past >= hi].mean()
    # A bounce-back trader buys after falls and sells after rises.
    gain = (after_fall - after_rise) / 2
    return {
        "ticks": h,
        "after_big_fall_pct": round(float(after_fall) * 100, 5),
        "after_big_rise_pct": round(float(after_rise) * 100, 5),
        "bounce_gain_pct": round(float(gain) * 100, 5),
        "spread_pct": round(spread * 100, 5),
        "gain_after_spread_pct": round(float(gain - spread) * 100, 5),
        "samples": int(i.size),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2026-01-01")
    p.add_argument("--keep-bad-ticks", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    began = time.time()
    ts, bid, ask = rec.load_ticks(args.start, args.end)
    mid = (bid + ask) / 2
    spread = float(np.median((ask - bid) / mid))
    log_mid = np.log(mid)
    r = np.diff(log_mid)
    ok = np.diff(ts) <= 60_000
    bad = np.zeros(ok.size, dtype=bool)
    if not args.keep_bad_ticks:
        for lo, hi in BAD_TICKS:
            lo_ms = pd.Timestamp(lo, tz="UTC").value // 1_000_000
            hi_ms = pd.Timestamp(hi, tz="UTC").value // 1_000_000
            bad |= (ts[1:] >= lo_ms) & (ts[:-1] < hi_ms)
    ok &= ~bad
    r[~ok] = 0.0  # moves across data gaps are left out
    c = np.concatenate(([0.0], np.cumsum(r)))
    rv = r[ok]
    years = (ts[-1] - ts[0]) / 1000 / SECONDS_PER_YEAR
    out = {
        "span": [args.start, args.end],
        "ticks": int(ts.size),
        "gaps_over_1_min": int((np.diff(ts) > 60_000).sum()),
        "bad_tick_moves_left_out": int(bad.sum()),
        "seconds_per_tick": round(float(np.median(np.diff(ts)[ok]) / 1000), 3),
        "median_spread_pct": round(spread * 100, 5),
    }

    # Volatility and big moves.
    sd = rv.std()
    out["yearly_volatility_pct"] = round(float(np.sqrt((rv**2).sum() / years)) * 100, 2)
    out["excess_kurtosis"] = round(float(stats.kurtosis(rv)), 4)
    for k in (4, 5):
        seen = int((np.abs(rv) > k * sd).sum())
        expected = float(2 * stats.norm.sf(k) * rv.size)
        out[f"moves_over_{k}_sd"] = {"seen": seen, "fair_coin_expects": round(expected)}
    out["drift_pct_per_year"] = round(float(rv.mean() / years * rv.size) * 100, 2)

    # Memory.
    out["autocorrelation"] = {
        str(k): round(float(np.dot(r[:-k], r[k:]) / np.dot(r, r)), 5) for k in LAGS
    }
    out["variance_ratio"] = {str(q): round(variance_ratio(c, q), 4) for q in RATIOS}
    up = r > 0
    down = r < 0
    streaks = {}
    for k in range(1, 6):
        run_up = np.ones(r.size - k, dtype=bool)
        run_down = np.ones(r.size - k, dtype=bool)
        for j in range(k):
            run_up &= up[j : r.size - k + j]
            run_down &= down[j : r.size - k + j]
        nxt_up = up[k:]
        nxt_move = r[k:] != 0
        streaks[str(k)] = {
            "next_up_after_ups_pct": round(
                float(nxt_up[run_up & nxt_move].mean()) * 100, 3
            ),
            "next_up_after_downs_pct": round(
                float(nxt_up[run_down & nxt_move].mean()) * 100, 3
            ),
        }
    out["streaks"] = streaks
    out["flat_ticks_pct"] = round(float((r[ok] == 0).mean()) * 100, 3)

    # Bounce-back against the spread.
    out["bounce"] = [bounce(c, h, spread) for h in HORIZONS]

    # Stability by year.
    year = ts[1:].astype("datetime64[ms]").astype("datetime64[Y]").astype(int) + 1970
    by_year = {}
    for y in np.unique(year):
        m = (year == y) & ok
        ry = r[m]
        cy = np.concatenate(([0.0], np.cumsum(ry)))
        by_year[str(y)] = {
            "ticks": int(m.sum()),
            "autocorrelation_1": round(
                float(np.dot(ry[:-1], ry[1:]) / np.dot(ry, ry)), 5
            ),
            "variance_ratio_30": round(variance_ratio(cy, 30), 4),
            "bounce_gain_30_pct": bounce(cy, 30, spread)["bounce_gain_pct"],
        }
    out["by_year"] = by_year

    # Hours and weekdays: average squared move relative to the overall average.
    t = ts[1:][ok]
    r2 = rv**2
    hour = (t // 3_600_000) % 24
    weekday = (t // 86_400_000 + 3) % 7  # 0 = Monday
    for name, key, n in (("by_hour_utc", hour, 24), ("by_weekday", weekday, 7)):
        total = np.bincount(key, weights=r2, minlength=n)
        count = np.bincount(key, minlength=n)
        size = np.sqrt(total / np.maximum(count, 1) / r2.mean())
        out[name] = [round(float(x), 4) for x in size]

    # Last digit of the quoted bid (two decimals).
    digit = np.rint(bid * 100).astype(np.int64) % 10
    counts = np.bincount(digit, minlength=10)
    out["last_digit_share_pct"] = [
        round(float(x) * 100, 3) for x in counts / counts.sum()
    ]
    out["last_digit_chi2_p"] = float(stats.chisquare(counts).pvalue)

    out["seconds"] = round(time.time() - began, 1)
    (args.output / "generator_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
