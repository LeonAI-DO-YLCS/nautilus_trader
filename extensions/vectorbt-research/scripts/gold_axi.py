"""
Gold rules for the Axi Select stages: mine four simple rule families, check them fairly, size.

Bars are 1-hour bars from the Pepperstone MT5 gold export (bid prices, spread of each bar's
first minute, broker time = New York + 7 h, day change at 00:00 broker). Orders fill at the
next bar's open: buys at bid + spread, sells at bid; exits pay the spread the same way. Stops
and targets are checked inside each bar on bid lows/highs (ask for shorts); when both are hit
in one bar the stop counts first. Positions held over the day change pay or earn the
Pepperstone overnight rate read on 2026-10-21 (long -79.38, short +28.65 points per lot at
4,118.65; three times on Wednesday nights). Everything is closed at the last bar of each month
(Axi pays out only when flat on the 1st). One position at a time.

Families (decision at a bar's close, entry at the next open):
  asia_long   buy at broker hour E, sell at broker hour X, stop k x 24-hour ATR (from the
              gold study: most of gold's rise came in the Asian session);
  breakout    range of broker hours 01-08; from 09 to 19 trade the first close outside it,
              stop at the far side or the middle, target m x range, out at 22;
  donchian    close beyond the high/low of the last N hours; out when the close crosses the
              N/2-hour low/high; stop k x ATR;
  dip         daily close above its D-day average and hourly RSI(14) below L: buy; out when
              RSI > 50 or after 24 hours; stop k x ATR.

Fair process: settings are scored on 2017-2022 (t-value of the mean trade in units of risk),
the top 5 of each family are compared on 2023, and the one picked is tested once on 2024 to
the end of the data. Picked rules are compared with random entries (same side, same hold,
same stop distance, same year). The Axi stage check starts a stage on every trading day of the
test years and risks a fixed share of the stage balance per trade: pass = +7% with at least
30 days and 20 trades, fail = equity (with open losses) touching -7%.

Example:
    python scripts/gold_axi.py --output /tmp/gold-axi
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import gold_real_costs as grc
import numpy as np
import pandas as pd

POINT = 0.01
LONG_FEE = 79.38 * POINT / 4118.65
SHORT_FEE = -28.65 * POINT / 4118.65  # negative fee: shorts earn it
TUNE = ("2017-01-01", "2023-01-01")
CHOOSE = ("2023-01-01", "2024-01-01")
TEST = ("2024-01-01", "2027-01-01")
TARGET, LOSS, MIN_DAYS, MIN_TRADES, MAX_DAYS = 0.07, 0.07, 30, 20, 365
RISKS = (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
GRID = {
    "asia_long": list(itertools.product((1, 2), (5, 9, 12), (0.5, 1.0, 2.0))),
    "breakout": list(
        itertools.product(("far", "mid"), (1.0, 2.0, 0.0), ("both", "long"))
    ),
    "donchian": list(
        itertools.product((24, 48, 120, 240), (2.0, 3.0), ("both", "long"))
    ),
    "dip": list(itertools.product((50, 100, 200), (25, 30, 35), (2.0, 3.0))),
}
DATA: dict = {}


def load():
    m1 = pd.read_parquet(grc.EXPORT).set_index("time").loc["2017-01-01":]
    g = m1.resample("1h")
    bars = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "spread": g["spread"].first() * POINT,
        }
    ).dropna()
    prev_close = bars["close"].shift(1)
    tr = np.maximum(bars["high"], prev_close) - np.minimum(bars["low"], prev_close)
    bars["atr"] = tr.rolling(24).mean()
    delta = bars["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    bars["rsi"] = 100 - 100 / (1 + up / down)
    day = bars.index.normalize()
    bars["new_day"] = np.r_[False, day[1:] != day[:-1]]
    bars["nights"] = 0
    wednesday = np.r_[False, day[:-1].day_of_week == 2]
    bars.loc[bars["new_day"], "nights"] = np.where(wednesday[bars["new_day"]], 3, 1)
    month = bars.index.to_period("M")
    bars["month_end"] = np.r_[month[1:] != month[:-1], True]
    daily = bars["close"].groupby(day).last()
    for d in (50, 100, 200):
        above = (daily > daily.rolling(d).mean()).shift(1)  # known at the day's start
        bars[f"trend{d}"] = above.reindex(day).fillna(False).to_numpy().astype(bool)
    return bars


def signals(bars, family, setting):
    """Arrays: entry side, stop distance, target distance, exit flags, hold limit, once a day."""
    n = len(bars)
    hour = bars.index.hour.to_numpy()
    close = bars["close"].to_numpy()
    atr = bars["atr"].to_numpy()
    sig = np.zeros(n)
    stop = atr.copy()
    target = np.full(n, np.nan)
    exit_open = np.zeros(n, dtype=bool)  # leave at this bar's open
    exit_long = np.zeros(n, dtype=bool)  # leave at the next open
    exit_short = np.zeros(n, dtype=bool)
    hold, once = 10_000, False
    if family == "asia_long":
        e, x, k = setting
        sig[:-1] = np.where(hour[1:] == e, 1, 0)
        exit_open[hour == x] = True
        stop = k * atr
    elif family == "breakout":
        where, m, side = setting
        day = bars.index.normalize()
        in_range = (hour >= 1) & (hour <= 8)
        hi = bars["high"].where(in_range).groupby(day).transform("max").to_numpy()
        lo = bars["low"].where(in_range).groupby(day).transform("min").to_numpy()
        active = (hour >= 9) & (hour <= 19) & ~np.isnan(hi)
        sig[active & (close > hi)] = 1
        sig[active & (close < lo)] = -1
        far = np.where(sig > 0, close - lo, hi - close)
        mid = np.abs(close - (hi + lo) / 2)
        stop = far if where == "far" else mid
        if m:
            target = m * (hi - lo)
        exit_open[hour == 22] = True
        once = True
        if side == "long":
            sig[sig < 0] = 0
    elif family == "donchian":
        nbars, k, side = setting
        hh = bars["high"].rolling(nbars).max().shift(1).to_numpy()
        ll = bars["low"].rolling(nbars).min().shift(1).to_numpy()
        half_l = bars["low"].rolling(nbars // 2).min().shift(1).to_numpy()
        half_h = bars["high"].rolling(nbars // 2).max().shift(1).to_numpy()
        sig[close > hh] = 1
        sig[close < ll] = -1
        exit_long = close < half_l
        exit_short = close > half_h
        stop = k * atr
        if side == "long":
            sig[sig < 0] = 0
    else:
        d, level, k = setting
        sig[bars[f"trend{d}"].to_numpy() & (bars["rsi"].to_numpy() < level)] = 1
        exit_long = bars["rsi"].to_numpy() > 50
        stop = k * atr
        hold = 24
    sig[np.isnan(stop) | (stop <= 0)] = 0
    return sig, stop, target, exit_open, exit_long, exit_short, hold, once


def trades(bars, family, setting):
    """Run one setting; one row per trade with its result and worst point in units of risk."""
    o, h, lo = (bars[k].to_numpy() for k in ("open", "high", "low"))
    sp = bars["spread"].to_numpy()
    nights = bars["nights"].to_numpy()
    month_end = bars["month_end"].to_numpy()
    day = bars.index.normalize().to_numpy()
    sig, stop_d, tgt_d, exit_open, exit_long, exit_short, hold, once = signals(
        bars, family, setting
    )
    out = []
    pos = 0
    entry = stop = target = risk = swap = worst = 0.0
    start = held = 0
    last_day = None

    def close_trade(i, price):
        pnl = (price - entry) * pos - swap
        out.append((start, i, pos, pnl / risk, worst / risk, held))

    def inside(i):
        """Stop and target inside bar i; returns the exit price or None."""
        nonlocal worst
        if pos > 0:
            worst = max(worst, entry - lo[i])
            if lo[i] <= stop:
                return min(o[i], stop) if i != start else stop
            if not np.isnan(target) and h[i] >= target:
                return max(o[i], target) if i != start else target
        else:
            ask_h, ask_l = h[i] + sp[i], lo[i] + sp[i]
            worst = max(worst, ask_h - entry)
            if ask_h >= stop:
                return max(o[i] + sp[i], stop) if i != start else stop
            if not np.isnan(target) and ask_l <= target:
                return min(o[i] + sp[i], target) if i != start else target
        return None

    for i in range(1, len(o)):
        if pos:
            held += 1
            if nights[i]:
                swap += (LONG_FEE if pos > 0 else SHORT_FEE) * o[i] * nights[i]
            leave = exit_open[i] or month_end[i] or held > hold
            leave |= exit_long[i - 1] if pos > 0 else exit_short[i - 1]
            if leave:
                close_trade(i, o[i] if pos > 0 else o[i] + sp[i])
                pos = 0
            else:
                price = inside(i)
                if price is not None:
                    close_trade(i, price)
                    pos = 0
        if (
            not pos
            and sig[i - 1]
            and not month_end[i]
            and not exit_open[i]
            and not (once and last_day == day[i])
        ):
            pos = int(sig[i - 1])
            entry = o[i] + sp[i] if pos > 0 else o[i]
            risk = stop_d[i - 1]
            stop = entry - pos * risk
            target = (
                entry + pos * tgt_d[i - 1] if not np.isnan(tgt_d[i - 1]) else np.nan
            )
            start, held, swap, worst, last_day = i, 0, 0.0, 0.0, day[i]
            price = inside(i)
            if price is not None:
                close_trade(i, price)
                pos = 0
    t = pd.DataFrame(out, columns=["start", "end", "side", "r", "worst_r", "hours"])
    t["entry_time"] = bars.index[t["start"]]
    t["exit_time"] = bars.index[t["end"]]
    return t


def score(t, lo, hi):
    x = t[
        (t["entry_time"] >= pd.Timestamp(lo, tz="UTC"))
        & (t["entry_time"] < pd.Timestamp(hi, tz="UTC"))
    ]
    r = x["r"].to_numpy()
    if r.size < 20:
        return {"trades": int(r.size), "mean_r": None, "t": -99.0}
    end = min(pd.Timestamp(hi, tz="UTC"), DATA["bars"].index[-1])
    months = max((end - pd.Timestamp(lo, tz="UTC")).days / 30.44, 1)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(r)
    return {
        "trades": int(r.size),
        "trades_per_month": round(r.size / months, 1),
        "mean_r": round(float(r.mean()), 4),
        "t": round(float(r.mean() / r.std(ddof=1) * np.sqrt(r.size)), 2),
        "win_rate": round(float((r > 0).mean()), 3),
        "profit_factor": round(float(gains / losses), 3) if losses else None,
        "worst_drop_r": round(float((np.maximum.accumulate(eq) - eq).max()), 1),
        "median_hours": float(np.median(x["hours"])),
        "long_share": round(float((x["side"] > 0).mean()), 2),
    }


def _run(job):
    family, setting = job
    t = trades(DATA["bars"], family, setting)
    return (
        family,
        setting,
        t,
        {
            name: score(t, *span)
            for name, span in (("tune", TUNE), ("choose", CHOOSE), ("test", TEST))
        },
    )


def random_entries(bars, t, repeats, seed=0):
    """Mean R of random trades: same side, hold and stop distance, random start in the same year."""
    rng = np.random.default_rng(seed)
    o, h, lo = (bars[k].to_numpy() for k in ("open", "high", "low"))
    sp = bars["spread"].to_numpy()
    years = bars.index.year.to_numpy()
    starts_by_year = {y: np.flatnonzero(years == y) for y in np.unique(years)}
    means = np.empty(repeats)
    hold = np.maximum(t["end"].to_numpy() - t["start"].to_numpy(), 1)
    side = t["side"].to_numpy()
    yr = years[t["start"].to_numpy()]
    stop_d = DATA["stops"]  # stop distance of each real trade, in price
    last = len(o) - 1
    for k in range(repeats):
        r = np.empty(len(t))
        for j in range(len(t)):
            pool = starts_by_year[yr[j]]
            s = pool[rng.integers(0, max(pool.size - hold[j] - 1, 1))]
            e = min(s + hold[j], last)
            entry = o[s] + sp[s] if side[j] > 0 else o[s]
            stop = entry - side[j] * stop_d[j]
            seg_lo, seg_hi = lo[s:e].min(), (h[s:e] + sp[s:e]).max()
            if side[j] > 0 and seg_lo <= stop or side[j] < 0 and seg_hi >= stop:
                r[j] = -1.0
            else:
                exit_ = o[e] if side[j] > 0 else o[e] + sp[e]
                r[j] = (exit_ - entry) * side[j] / stop_d[j]
        means[k] = r.mean()
    return means


def axi_stages(t, lo, hi, risk):
    """Start a stage on every trading day: share passed, failed, unfinished; days to pass."""
    x = t[
        (t["entry_time"] >= pd.Timestamp(lo, tz="UTC"))
        & (t["entry_time"] < pd.Timestamp(hi, tz="UTC"))
    ]
    entry = x["entry_time"].values
    exit_ = x["exit_time"].values
    r = x["r"].to_numpy()
    worst = x["worst_r"].to_numpy()
    days = np.unique(x["entry_time"].dt.normalize().values)
    passed = failed = 0
    to_pass = []
    for d in days:
        j = np.searchsorted(entry, d)
        eq, n = 1.0, 0
        result = None
        while j < len(r):
            if (entry[j] - d) / np.timedelta64(1, "D") > MAX_DAYS:
                break
            if eq * (1 - risk * worst[j]) <= 1 - LOSS:
                result = "fail"
                break
            eq *= 1 + risk * r[j]
            n += 1
            elapsed = (exit_[j] - d) / np.timedelta64(1, "D")
            if eq <= 1 - LOSS:
                result = "fail"
                break
            if eq >= 1 + TARGET and n >= MIN_TRADES and elapsed >= MIN_DAYS:
                result = "pass"
                to_pass.append(elapsed)
                break
            j += 1
        passed += result == "pass"
        failed += result == "fail"
    total = len(days)
    return {
        "risk_per_trade_pct": risk * 100,
        "stages": total,
        "passed_pct": round(passed / total * 100, 1),
        "failed_pct": round(failed / total * 100, 1),
        "unfinished_pct": round((total - passed - failed) / total * 100, 1),
        "median_days_to_pass": float(np.median(to_pass)) if to_pass else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=300)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    began = time.time()
    bars = DATA["bars"] = load()
    jobs = [(f, s) for f, settings in GRID.items() for s in settings]
    with mp.Pool(max(1, int(os.cpu_count() * 0.8))) as pool:
        results = pool.map(_run, jobs)
    rows = [
        {
            "family": f,
            "setting": list(s),
            **{f"{k}_{m}": v for k, sc in scores.items() for m, v in sc.items()},
        }
        for f, s, _, scores in results
    ]
    pd.DataFrame(rows).to_csv(args.output / "settings.csv", index=False)
    report = {
        "bars": len(bars),
        "first": str(bars.index[0]),
        "last": str(bars.index[-1]),
        "families": {},
    }
    for family in GRID:
        mine = [r for r in results if r[0] == family]
        positive = [r for r in mine if (r[3]["tune"]["mean_r"] or 0) > 0]
        top = sorted(positive, key=lambda r: r[3]["tune"]["t"], reverse=True)[:5]
        entry = {
            "settings": len(mine),
            "positive_in_tune": len(positive),
            "positive_in_test": sum((r[3]["test"]["mean_r"] or 0) > 0 for r in mine),
        }
        if top:
            pick = max(top, key=lambda r: r[3]["choose"]["t"])
            _, setting, t, scores = pick
            in_test = (t["entry_time"] >= pd.Timestamp(TEST[0], tz="UTC")).to_numpy()
            stops = signals(bars, family, setting)[1]
            DATA["stops"] = stops[t["start"].to_numpy() - 1][in_test]
            test = t[in_test].reset_index(drop=True)
            means = random_entries(bars, test, args.repeats)
            actual = scores["test"]["mean_r"]
            entry.update(
                {
                    "pick": list(setting),
                    "tune": scores["tune"],
                    "choose": scores["choose"],
                    "test": scores["test"],
                    "random_entries_test_mean_r": round(float(means.mean()), 4),
                    "share_of_random_runs_beaten": round(
                        float((means < actual).mean()), 3
                    )
                    if actual is not None
                    else None,
                    "axi_test": [axi_stages(t, *TEST, k) for k in RISKS],
                    "axi_all_years": [
                        axi_stages(t, TUNE[0], TEST[1], k) for k in RISKS
                    ],
                }
            )
            t.to_csv(args.output / f"trades_{family}.csv", index=False)
        report["families"][family] = entry
        print(family, json.dumps(entry), flush=True)
    report["seconds"] = round(time.time() - began, 1)
    (args.output / "gold_axi.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(json.dumps({"seconds": report["seconds"]}))


if __name__ == "__main__":
    main()
