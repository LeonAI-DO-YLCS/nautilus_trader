"""
Random entries with a recovery system on Volatility 75, tick by tick, at the minimum lot.

Each trade picks buy or sell at random. Four ways to handle it:
  none        fixed minimum lot, take-profit and stop-loss at the same distance;
  martingale  same, but after a loss the next lot is multiplied (reset after a win);
  grid        no stop: add a lot (minimum x mult^level) every `distance` against the last
              entry; close everything when the basket makes one take-profit of the first lot;
  hedge       zone recovery: when price crosses the zone against the open side, open the
              other side with a lot sized so the basket makes one take-profit at the next
              target (target = zone x mult); close everything at that profit.

Buys fill at the ask and sell at the bid, so the real spread is paid on every order; a
"no spread" run uses the mid price to show the effect of the spread alone. The account
starts at a fixed balance. Profit above it is withdrawn whenever the basket is flat; when
equity reaches zero the account is blown and refilled with a new deposit. Swap and margin
stop-out are not modelled. Contract size is 1: 1 lot x 1 point = 1 USD.

Example:
    python scripts/v75_recovery.py --start 2025-12-24 --end 2025-12-31 --output /tmp/rec
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import traditional as t

MIN_LOT = 0.001
SYSTEMS = {"none": 0, "martingale": 1, "grid": 2, "hedge": 3}
MULTS = {
    "none": (1.0,),
    "martingale": (2.0, 2.5),
    "grid": (1.0, 1.5, 2.0),
    "hedge": (1.0, 2.0),
}
DISTANCES = (0.0005, 0.001, 0.002, 0.005, 0.01)
BALANCES = (1_000.0, 10_000.0)


def load_ticks(start, end):
    """Bid/ask ticks between two UTC dates, from the yearly files."""
    lo = pd.Timestamp(start, tz="UTC").value // 1_000_000
    hi = pd.Timestamp(end, tz="UTC").value // 1_000_000
    times, bids, asks = [], [], []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        path = t.DATASET / f"Volatility75Index_{year}.parquet"
        if not path.exists():
            continue
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["<DATE>", "<TIME>", "<BID>", "<ASK>"], batch_size=2_000_000
        ):
            clock = batch.column(1)
            day = pc.cast(
                pc.strptime(batch.column(0), format="%Y.%m.%d", unit="s"), pa.int64()
            )
            times.append(
                day.to_numpy() * 1000
                + t._clock_part(clock, 0, 2) * 3_600_000
                + t._clock_part(clock, 3, 5) * 60_000
                + t._clock_part(clock, 6, 8) * 1000
                + t._clock_part(clock, 9, 12)
            )
            bids.append(batch.column(2).to_numpy(zero_copy_only=False))
            asks.append(batch.column(3).to_numpy(zero_copy_only=False))
    ts, bid, ask = np.concatenate(times), np.concatenate(bids), np.concatenate(asks)
    ok = ~(np.isnan(bid) | np.isnan(ask)) & (ts >= lo) & (ts < hi)
    order = np.argsort(ts[ok], kind="stable")
    return ts[ok][order], bid[ok][order], ask[ok][order]


@nb.njit(cache=True)
def _lot_up(x):
    return np.ceil(x / MIN_LOT - 1e-9) * MIN_LOT


@nb.njit(cache=True)
def run_account(bid, ask, system, dist, mult, balance0, seed):
    """One account over all ticks. Returns net, deposits, cycles, wins, blow-ups, max lot."""
    np.random.seed(seed)
    bal = balance0
    withdrawn = 0.0
    refills = 0
    cycles = 0
    wins = 0
    max_lot = 0.0
    ll = 0.0  # long lots
    lc = 0.0  # long lots x entry ask
    sl = 0.0  # short lots
    sv = 0.0  # short lots x entry bid
    side = 0  # 0 flat, else side of the last order
    lot = MIN_LOT
    level = 0
    anchor = 0.0
    last = 0.0
    lower = 0.0
    upper = 0.0
    target = 0.0
    for i in range(bid.size):
        b = bid[i]
        a = ask[i]
        if side == 0:
            side = 1 if np.random.random() < 0.5 else -1
            if system != 1:
                lot = MIN_LOT
            if side > 0:
                ll += lot
                lc += lot * a
                anchor = a
            else:
                sl += lot
                sv += lot * b
                anchor = b
            max_lot = max(max_lot, lot)
            level = 1
            last = anchor
            target = MIN_LOT * anchor * dist * (mult if system == 3 else 1.0)
            if side > 0:
                upper, lower = anchor, anchor * (1 - dist)
            else:
                upper, lower = anchor * (1 + dist), anchor
            continue
        pnl = ll * b - lc + sv - sl * a
        if bal + pnl <= 0.0:  # blown: everything lost, refill
            refills += 1
            bal = balance0
            ll = lc = sl = sv = 0.0
            side = 0
            lot = MIN_LOT
            continue
        close = False
        win = False
        if system <= 1:
            if side > 0:
                if b >= anchor * (1 + dist):
                    close, win = True, True
                elif b <= anchor * (1 - dist):
                    close = True
            else:
                if a <= anchor * (1 - dist):
                    close, win = True, True
                elif a >= anchor * (1 + dist):
                    close = True
        elif pnl >= target:
            close, win = True, True
        elif system == 2:
            if side > 0 and a <= last * (1 - dist):
                add = _lot_up(MIN_LOT * mult**level)
                ll += add
                lc += add * a
                last = a
                level += 1
                max_lot = max(max_lot, add)
            elif side < 0 and b >= last * (1 + dist):
                add = _lot_up(MIN_LOT * mult**level)
                sl += add
                sv += add * b
                last = b
                level += 1
                max_lot = max(max_lot, add)
        else:
            tp = dist * mult
            if side > 0 and b <= lower:
                x = lower * (1 - tp)  # next target below
                need = target - (ll * x - lc + sv - sl * x)
                add = _lot_up(max(need / (b - x), MIN_LOT))
                sl += add
                sv += add * b
                side = -1
                level += 1
                max_lot = max(max_lot, add)
            elif side < 0 and a >= upper:
                x = upper * (1 + tp)  # next target above
                need = target - (ll * x - lc + sv - sl * x)
                add = _lot_up(max(need / (x - a), MIN_LOT))
                ll += add
                lc += add * a
                side = 1
                level += 1
                max_lot = max(max_lot, add)
        if close:
            bal += pnl
            ll = lc = sl = sv = 0.0
            side = 0
            cycles += 1
            if win:
                wins += 1
                lot = MIN_LOT
            else:
                lot = _lot_up(lot * mult)
            if bal > balance0:
                withdrawn += bal - balance0
                bal = balance0
    final = bal + ll * bid[-1] - lc + sv - sl * ask[-1]
    net = withdrawn + final - balance0 - refills * balance0
    return net, refills, cycles, wins, max_lot


@nb.njit(cache=True, parallel=True)
def run_jobs(bid, ask, systems, dists, mults, balances, seeds):
    out = np.zeros((systems.size, 5))
    for j in nb.prange(systems.size):
        net, refills, cycles, wins, top = run_account(
            bid, ask, systems[j], dists[j], mults[j], balances[j], seeds[j]
        )
        out[j, 0] = net
        out[j, 1] = refills
        out[j, 2] = cycles
        out[j, 3] = wins
        out[j, 4] = top
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-24")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    nb.set_num_threads(max(1, int(os.cpu_count() * 0.8)))
    began = time.time()
    ts, bid, ask = load_ticks(args.start, args.end)
    mid = (bid + ask) / 2
    days = (ts[-1] - ts[0]) / 86_400_000
    print(
        f"{ts.size:,} ticks, {days:.2f} days, price {bid.min():,.0f} to {bid.max():,.0f}"
    )
    settings = [
        (name, m, d, b)
        for name in SYSTEMS
        for m, d, b in itertools.product(MULTS[name], DISTANCES, BALANCES)
    ]
    jobs = [(s, k) for s in settings for k in range(args.seeds)]
    arrays = (
        np.array([SYSTEMS[s[0]] for s, _ in jobs], dtype=np.int64),
        np.array([s[2] for s, _ in jobs]),
        np.array([s[1] for s, _ in jobs]),
        np.array([s[3] for s, _ in jobs]),
        np.array([k for _, k in jobs], dtype=np.int64),
    )
    rows = []
    for label, b_, a_ in (("spread", bid, ask), ("no_spread", mid, mid)):
        out = run_jobs(b_, a_, *arrays).reshape(len(settings), args.seeds, 5)
        for (name, m, d, bal), r in zip(settings, out):
            net, refills, cycles, wins = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
            rows.append(
                {
                    "prices": label,
                    "system": name,
                    "mult": m,
                    "distance_pct": d * 100,
                    "balance": bal,
                    "mean_net_usd": round(float(net.mean()), 2),
                    "median_net_usd": round(float(np.median(net)), 2),
                    "pct_accounts_up": round(float((net > 0).mean() * 100), 1),
                    "pct_blown_once": round(float((refills > 0).mean() * 100), 1),
                    "mean_blowups": round(float(refills.mean()), 2),
                    "cycles_per_day": round(float(cycles.mean() / days), 1),
                    "win_rate_pct": round(
                        float(wins.sum() / max(cycles.sum(), 1) * 100), 2
                    ),
                    "max_lot": round(float(r[:, 4].max()), 3),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "results.csv", index=False)
    summary = {
        "span": [args.start, args.end],
        "ticks": int(ts.size),
        "days": round(days, 2),
        "seeds_per_setting": args.seeds,
        "seconds": round(time.time() - began, 1),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(table.to_string(index=False))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
