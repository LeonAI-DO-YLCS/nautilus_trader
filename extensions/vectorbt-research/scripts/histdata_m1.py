"""
Download free 1-minute bid bars from HistData.com and save one Parquet file per pair.

HistData serves one zip per past year and one per month of the current year, each behind a
page that holds a one-time token. File times are EST without daylight saving (UTC-5) and are
moved to UTC here. Prices are bid only; the files carry no spread. Zips already downloaded
are reused.

Example:
    python scripts/histdata_m1.py --pair XAUUSD --first-year 2016 --last-month 2026-08 \
        --output ~/data/histdata
"""

import argparse
import io
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

PAGE = (
    "https://www.histdata.com/download-free-forex-historical-data/"
    "?/ascii/1-minute-bar-quotes/{pair}/{period}"
)
GET = "https://www.histdata.com/get.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (research download)"}
FORM = ("tk", "date", "datemonth", "platform", "timeframe", "fxpair")
EST_MS = 5 * 3_600_000


def fetch(pair, period, dest, page=PAGE):
    """Download one zip through its token page; skip it when it is already on disk."""
    if dest.exists():
        return
    page = page.format(pair=pair, period=period)
    html = urllib.request.urlopen(
        urllib.request.Request(page, headers=HEADERS), timeout=60
    ).read()
    form = dict(re.findall(rb'name="(\w+)" id="\w+" value="([^"]*)"', html))
    body = urllib.parse.urlencode({k: form[k.encode()] for k in FORM}).encode()
    request = urllib.request.Request(
        GET, data=body, headers={**HEADERS, "Referer": page}
    )
    data = urllib.request.urlopen(request, timeout=300).read()
    if data[:2] != b"PK":
        raise RuntimeError(f"{pair} {period}: no zip returned ({len(data)} bytes)")
    dest.write_bytes(data)
    time.sleep(1)


def read_zip(path):
    """Time (ms, UTC) and open/high/low/close of one HistData ASCII M1 zip."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".csv"))
        table = pv.read_csv(
            io.BytesIO(archive.read(name)),
            read_options=pv.ReadOptions(
                column_names=["time", "open", "high", "low", "close", "volume"]
            ),
            parse_options=pv.ParseOptions(delimiter=";"),
            convert_options=pv.ConvertOptions(column_types={"time": pa.string()}),
        )
    seconds = pc.cast(
        pc.strptime(table["time"], format="%Y%m%d %H%M%S", unit="s"), pa.int64()
    )
    columns = {"time": seconds.to_numpy() * 1000 + EST_MS}
    for c in ("open", "high", "low", "close"):
        columns[c] = table[c].to_numpy()
    return columns


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--pair", required=True)
    parser.add_argument("--first-year", type=int, required=True)
    parser.add_argument("--last-month", required=True, help="YYYY-MM, inclusive")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    last_year, last_month = (int(x) for x in args.last_month.split("-"))
    folder = Path(args.output).expanduser() / args.pair
    folder.mkdir(parents=True, exist_ok=True)
    periods = [str(y) for y in range(args.first_year, last_year)]
    periods += [f"{last_year}/{m}" for m in range(1, last_month + 1)]
    zips = []
    for period in periods:
        dest = folder / f"{args.pair}_M1_{period.replace('/', '')}.zip"
        fetch(args.pair, period, dest)
        zips.append(dest)
        print(f"{dest.name}: {dest.stat().st_size:,} bytes", flush=True)
    parts = [read_zip(z) for z in zips]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(merged["time"], kind="stable")
    merged = {k: v[order] for k, v in merged.items()}
    keep = np.append(merged["time"][1:] != merged["time"][:-1], True)
    table = pa.table(
        {
            "time": pa.array(merged["time"][keep], pa.timestamp("ms", tz="UTC")),
            **{c: merged[c][keep] for c in ("open", "high", "low", "close")},
        }
    )
    out = folder.parent / f"{args.pair}_m1.parquet"
    pq.write_table(table, out)
    print(
        f"{out}: {table.num_rows:,} bars, "
        f"{table['time'][0].as_py()} to {table['time'][-1].as_py()}"
    )


if __name__ == "__main__":
    main()
