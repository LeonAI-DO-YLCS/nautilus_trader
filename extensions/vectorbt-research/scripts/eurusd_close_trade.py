"""
Re-test of the EURUSD "trade near the daily close" binary options from the binary options lab.

The lab found, on 1-minute bid bars, that buying Fall at 21:50 UTC with a 10-minute expiry won
65.7%, 73.1% and 71.7% in three periods (break-even 58.82% at a 70% payout), and Rise at 22:50
UTC won 60% to 63%. This script uses HistData tick files, which carry bid and ask, so each
contract is settled three ways: on bid (as the lab did), on the mid price (bid plus ask over
two, close to the spot price a binary broker settles on), and with the spread at entry and at
expiry reported. Results are split by New York winter time (21:50 UTC = 4:50 PM) and summer
time (21:50 UTC = 5:50 PM, after the 5 PM daily close). The entry price is the last tick at or
before the start; the exit price is the last tick at or before expiry; a tie loses.

Example:
    python scripts/eurusd_close_trade.py --ticks ~/data/histdata-ticks --output /tmp/eurusd-close
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import time
import zipfile
from pathlib import Path

import histdata_m1 as h
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

TICK_PAGE = h.PAGE.replace("1-minute-bar-quotes", "tick-data-quotes")
PAYOUT = 0.70
BREAK_EVEN = 1 / (1 + PAYOUT)
FIRST, LAST = (2019, 10), (2026, 9)
WINDOW_MS = (20 * 3_600_000, 24 * 3_600_000)  # keep ticks from 20:00 to 24:00 UTC
STALE_MS = 5 * 60_000
TRADES = {
    "Fall 21:50 UTC, 10 min": (21 * 60 + 50, 10, -1),
    "Fall 21:45 UTC, 15 min": (21 * 60 + 45, 15, -1),
    "Rise 22:50 UTC, 10 min": (22 * 60 + 50, 10, 1),
    "Rise 22:45 UTC, 15 min": (22 * 60 + 45, 15, 1),
}
PERIODS = {
    "select 2019-10 to 2022-09": ("2019-10-01", "2022-10-01"),
    "test 2022-10 to 2024-09": ("2022-10-01", "2024-10-01"),
    "test 2024-10 to 2026-09": ("2024-10-01", "2026-10-01"),
}


def months():
    y, m = FIRST
    while (y, m) <= LAST:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def read_month(path):
    """Ticks between 20:00 and 24:00 UTC: time (ms, UTC), bid, ask."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        table = pv.read_csv(
            io.BytesIO(archive.read(name)),
            read_options=pv.ReadOptions(column_names=["time", "bid", "ask", "volume"]),
            convert_options=pv.ConvertOptions(column_types={"time": pa.string()}),
        )
    stamp = table["time"]
    seconds = pc.cast(
        pc.strptime(
            pc.utf8_slice_codeunits(stamp, 0, 15), format="%Y%m%d %H%M%S", unit="s"
        ),
        pa.int64(),
    ).to_numpy()
    millis = pc.cast(pc.utf8_slice_codeunits(stamp, 15, 18), pa.int64()).to_numpy()
    ts = seconds * 1000 + millis + h.EST_MS
    clock = ts % 86_400_000
    keep = (clock >= WINDOW_MS[0]) & (clock < WINDOW_MS[1])
    return ts[keep], table["bid"].to_numpy()[keep], table["ask"].to_numpy()[keep]


def settle(ts, bid, ask, days, start_min, expiry_min, side):
    """One contract per day; returns a table of entry and exit prices and outcomes."""
    start = days + start_min * 60_000
    end = start + expiry_min * 60_000
    i = np.searchsorted(ts, start, side="right") - 1
    j = np.searchsorted(ts, end, side="right") - 1
    ok = (i >= 0) & (j > i) & (start - ts[i] <= STALE_MS) & (end - ts[j] <= STALE_MS)
    i, j, start = i[ok], j[ok], start[ok]
    out = pd.DataFrame(
        {
            "bid_in": bid[i],
            "bid_out": bid[j],
            "mid_in": (bid[i] + ask[i]) / 2,
            "mid_out": (bid[j] + ask[j]) / 2,
            "spread_in": ask[i] - bid[i],
            "spread_out": ask[j] - bid[j],
        },
        index=pd.to_datetime(start, unit="ms", utc=True),
    )
    for price in ("bid", "mid"):
        move = side * (out[f"{price}_out"] - out[f"{price}_in"])
        out[f"won_{price}"] = move > 0
    return out


def stats(part):
    n = len(part)
    out = {"contracts": n}
    for price in ("bid", "mid"):
        rate = float(part[f"won_{price}"].mean()) if n else 0.0
        out[f"win_rate_{price}"] = round(rate, 4)
        out[f"return_per_stake_{price}_pct"] = round((rate * (1 + PAYOUT) - 1) * 100, 2)
    out["median_spread_in_pips"] = round(float(part["spread_in"].median()) * 1e4, 2)
    out["median_spread_out_pips"] = round(float(part["spread_out"].median()) * 1e4, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ticks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    t0 = time.time()
    folder = Path(args.ticks).expanduser() / "EURUSD"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for y, m in months():
        dest = folder / f"EURUSD_T_{y}{m:02d}.zip"
        h.fetch("EURUSD", f"{y}/{m}", dest, page=TICK_PAGE)
        paths.append(dest)
    download_s = time.time() - t0
    workers = max(1, min(len(paths), int(os.cpu_count() * 0.8)))
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(read_month, paths)
    ts = np.concatenate([p[0] for p in parts])
    bid = np.concatenate([p[1] for p in parts])
    ask = np.concatenate([p[2] for p in parts])
    order = np.argsort(ts, kind="stable")
    ts, bid, ask = ts[order], bid[order], ask[order]
    days = np.unique(ts // 86_400_000) * 86_400_000
    result = {
        "ticks_kept": int(ts.size),
        "first_tick": str(pd.Timestamp(ts[0], unit="ms", tz="UTC")),
        "last_tick": str(pd.Timestamp(ts[-1], unit="ms", tz="UTC")),
        "payout": PAYOUT,
        "break_even": round(BREAK_EVEN, 4),
        "download_seconds": round(download_s, 1),
        "trades": {},
    }
    for name, (start_min, expiry_min, side) in TRADES.items():
        book = settle(ts, bid, ask, days, start_min, expiry_min, side)
        summer = book.index.tz_convert("America/New_York").map(lambda t: bool(t.dst()))
        rows = {}
        for period, (a, b) in PERIODS.items():
            part = book[(book.index >= a) & (book.index < b)]
            rows[period] = stats(part)
        summer = np.asarray(summer, dtype=bool)
        rows["all, New York winter time"] = stats(book[~summer])
        rows["all, New York summer time"] = stats(book[summer])
        result["trades"][name] = rows
    result["seconds"] = round(time.time() - t0, 1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eurusd_close_trade.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
