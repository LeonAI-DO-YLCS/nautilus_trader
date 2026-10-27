"""
Build 1-minute bid bars from MT5 tick exports (tab-separated <DATE> <TIME> <BID> <ASK> ...).

Reads each file as an Arrow stream, so memory stays small, and writes one Parquet file with
time (UTC as MT5 gives it), open, high, low, close, the mean spread of the minute and its tick
count. Files may overlap; a minute found in several files keeps its first open, last close,
highest high and lowest low.

Example:
    python scripts/ticks_to_m1.py --output ~/data/step/step_m1.parquet "<ticks 1>.csv" "<ticks 2>.csv"
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from traditional import _clock_part


def minutes_of(path):
    reader = pv.open_csv(
        path,
        read_options=pv.ReadOptions(block_size=256 << 20),
        parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(
            include_columns=["<DATE>", "<TIME>", "<BID>", "<ASK>"],
            column_types={"<DATE>": pa.string(), "<TIME>": pa.string()},
        ),
    )
    parts = []
    for batch in reader:
        day = pc.cast(
            pc.strptime(batch.column(0), format="%Y.%m.%d", unit="s"), pa.int64()
        )
        clock = batch.column(1)
        minute = (
            day.to_numpy() // 60
            + _clock_part(clock, 0, 2) * 60
            + _clock_part(clock, 3, 5)
        )
        bid = batch.column(2).to_numpy(zero_copy_only=False)
        ask = batch.column(3).to_numpy(zero_copy_only=False)
        ok = ~(np.isnan(bid) | np.isnan(ask))
        minute, bid, ask = minute[ok], bid[ok], ask[ok]
        keys, first = np.unique(minute, return_index=True)
        if not np.all(np.diff(minute) >= 0):
            raise ValueError(f"{path}: ticks are not in time order")
        last = np.append(first[1:] - 1, minute.size - 1)
        count = np.diff(np.append(first, minute.size))
        parts.append(
            pd.DataFrame(
                {
                    "minute": keys,
                    "open": bid[first],
                    "high": np.maximum.reduceat(bid, first),
                    "low": np.minimum.reduceat(bid, first),
                    "close": bid[last],
                    "spread_sum": np.add.reduceat(ask - bid, first),
                    "ticks": count,
                }
            )
        )
    return pd.concat(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    t0 = time.time()
    rows = pd.concat([minutes_of(f) for f in args.files], ignore_index=True)
    bars = rows.groupby("minute", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        spread_sum=("spread_sum", "sum"),
        ticks=("ticks", "sum"),
    )
    bars["spread"] = bars.pop("spread_sum") / bars["ticks"]
    bars.index = pd.to_datetime(bars.index * 60, unit="s", utc=True).rename("time")
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.reset_index().to_parquet(out)
    print(
        f"{out}: {len(bars):,} bars from {int(bars['ticks'].sum()):,} ticks, "
        f"{bars.index[0]} to {bars.index[-1]}, {time.time() - t0:.0f} s"
    )


if __name__ == "__main__":
    main()
