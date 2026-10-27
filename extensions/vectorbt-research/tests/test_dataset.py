from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vectorbt_research.dataset import check_window_against_csv, read_quote_window

DATASET_DIR = Path(
    "/mnt/c/LeonAI-DO/dev/TRADING/Datasets/Volatility75Index_yearly_deduped"
)
DATE = "2026.08.18"
WINDOW_START_NS = 1787023200000000000  # 2026-08-18T03:20:00Z
WINDOW_END_NS = 1787026800000000000  # 2026-08-18T04:20:00Z


def _write_parquet(path, rows, row_group_size=None):
    """Write (date, time, bid, ask) rows with the yearly file's column names and types."""
    count = len(rows)
    table = pa.table(
        {
            "<DATE>": pa.array([row[0] for row in rows], pa.large_string()),
            "<TIME>": pa.array([row[1] for row in rows], pa.large_string()),
            "<BID>": pa.array([row[2] for row in rows], pa.float64()),
            "<ASK>": pa.array([row[3] for row in rows], pa.float64()),
            "<LAST>": pa.array([""] * count, pa.large_string()),
            "<VOLUME>": pa.array([""] * count, pa.large_string()),
            "<FLAGS>": pa.array([6] * count, pa.int64()),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _write_csv(path, rows):
    """Write tab-separated (date, time, bid, ask) text rows in the exporter's layout."""
    lines = ["<DATE>\t<TIME>\t<BID>\t<ASK>\t<LAST>\t<VOLUME>\t<FLAGS>"]
    lines += [
        f'{date}\t{time}\t{bid}\t{ask}\t""\t""\t6' for date, time, bid, ask in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_window_includes_start_and_excludes_end(tmp_path):
    path = tmp_path / "Volatility75Index_2026.parquet"
    _write_parquet(
        path,
        [
            (DATE, "03:19:59.999", 49191.70, 49208.26),
            (DATE, "03:20:00.000", 49192.02, 49208.58),
            (DATE, "03:20:00.500", 49192.49, 49209.05),
            (DATE, "04:19:59.999", 48664.95, 48681.51),
            (DATE, "04:20:00.000", 48667.25, 48683.81),
        ],
    )

    rows = read_quote_window(path, WINDOW_START_NS, WINDOW_END_NS)

    assert [row["timestamp"] for row in rows] == [
        "2026-08-18T03:20:00.000Z",
        "2026-08-18T03:20:00.500Z",
        "2026-08-18T04:19:59.999Z",
    ]
    assert [row["ordinal"] for row in rows] == [1, 2, 3]
    assert rows[0]["bid"] == "49192.02"
    assert rows[0]["ask"] == "49208.58"


def test_row_ordinals_stay_global_across_row_groups(tmp_path):
    path = tmp_path / "grouped.parquet"
    _write_parquet(
        path,
        [
            (DATE, "03:19:58.000", 49191.70, 49208.26),
            (DATE, "03:19:59.000", 49191.71, 49208.27),
            (DATE, "03:20:00.000", 49192.02, 49208.58),
            (DATE, "03:20:01.000", 49192.03, 49208.59),
            (DATE, "03:20:02.000", 49192.04, 49208.60),
            (DATE, "04:20:00.000", 48667.25, 48683.81),
        ],
        row_group_size=2,
    )

    rows = read_quote_window(path, WINDOW_START_NS, WINDOW_END_NS)

    assert [row["ordinal"] for row in rows] == [2, 3, 4]
    assert [row["bid"] for row in rows] == ["49192.02", "49192.03", "49192.04"]


def test_equal_timestamps_keep_source_order(tmp_path):
    path = tmp_path / "ties.parquet"
    _write_parquet(
        path,
        [
            (DATE, "03:20:00.000", 49192.02, 49208.58),
            (DATE, "03:20:00.000", 49192.03, 49208.59),
            (DATE, "03:20:00.000", 49192.01, 49208.57),
        ],
    )

    rows = read_quote_window(path, WINDOW_START_NS, WINDOW_END_NS)

    assert [row["bid"] for row in rows] == ["49192.02", "49192.03", "49192.01"]
    assert [row["ordinal"] for row in rows] == [0, 1, 2]


def test_off_grid_price_raises_and_names_time_and_column(tmp_path):
    path = tmp_path / "off_grid.parquet"
    _write_parquet(path, [(DATE, "03:20:00.000", 49191.672, 49208.26)])

    with pytest.raises(ValueError, match=r"BID at 2026-08-18T03:20:00\.000Z"):
        read_quote_window(path, WINDOW_START_NS, WINDOW_END_NS)


def test_missing_bid_stays_missing(tmp_path):
    path = tmp_path / "missing_bid.parquet"
    _write_parquet(path, [(DATE, "03:20:00.000", None, 49208.26)])

    (row,) = read_quote_window(path, WINDOW_START_NS, WINDOW_END_NS)

    assert row["bid"] is None
    assert row["ask"] == "49208.26"


def test_trailing_zero_csv_matches_shortest_parquet_text(tmp_path):
    parquet = tmp_path / "Volatility75Index_2025.parquet"
    csv = tmp_path / "Volatility75Index_2025.csv"
    _write_parquet(
        parquet,
        [
            (DATE, "03:20:00.000", 50197.9, 50202.7),
            (DATE, "03:20:00.500", 50196.3, 50197.3),
        ],
    )
    _write_csv(
        csv,
        [
            (DATE, "03:20:00.000", "50197.90", "50202.70"),
            (DATE, "03:20:00.500", "50196.30", "50197.30"),
        ],
    )

    assert check_window_against_csv(parquet, csv, WINDOW_START_NS, WINDOW_END_NS) == 2


def test_changed_price_fails_csv_check(tmp_path):
    parquet = tmp_path / "Volatility75Index_2025.parquet"
    csv = tmp_path / "Volatility75Index_2025.csv"
    _write_parquet(parquet, [(DATE, "03:20:00.000", 50197.9, 50202.7)])
    _write_csv(csv, [(DATE, "03:20:00.000", "50197.95", "50202.70")])

    with pytest.raises(ValueError, match=r"BID differs at 2026-08-18T03:20:00\.000Z"):
        check_window_against_csv(parquet, csv, WINDOW_START_NS, WINDOW_END_NS)


def test_missing_bid_matches_empty_csv_bid(tmp_path):
    parquet = tmp_path / "Volatility75Index_2025.parquet"
    csv = tmp_path / "Volatility75Index_2025.csv"
    _write_parquet(parquet, [(DATE, "03:20:00.000", None, 50202.7)])
    _write_csv(csv, [(DATE, "03:20:00.000", "", "50202.70")])

    assert check_window_against_csv(parquet, csv, WINDOW_START_NS, WINDOW_END_NS) == 1


def test_csv_row_count_mismatch_fails(tmp_path):
    parquet = tmp_path / "Volatility75Index_2025.parquet"
    csv = tmp_path / "Volatility75Index_2025.csv"
    _write_parquet(
        parquet,
        [
            (DATE, "03:20:00.000", 50197.9, 50202.7),
            (DATE, "03:20:00.500", 50196.3, 50197.3),
        ],
    )
    _write_csv(csv, [(DATE, "03:20:00.000", "50197.90", "50202.70")])

    with pytest.raises(ValueError, match="row count differs"):
        check_window_against_csv(parquet, csv, WINDOW_START_NS, WINDOW_END_NS)


def test_csv_timestamp_mismatch_fails(tmp_path):
    parquet = tmp_path / "Volatility75Index_2025.parquet"
    csv = tmp_path / "Volatility75Index_2025.csv"
    _write_parquet(parquet, [(DATE, "03:20:00.000", 50197.9, 50202.7)])
    _write_csv(csv, [(DATE, "03:20:00.001", "50197.90", "50202.70")])

    with pytest.raises(ValueError, match="timestamp differs"):
        check_window_against_csv(parquet, csv, WINDOW_START_NS, WINDOW_END_NS)


@pytest.mark.skipif(
    not DATASET_DIR.is_dir(), reason="Volatility75 dataset folder is absent"
)
def test_real_2026_window_has_1797_rows_and_matches_csv():
    parquet = DATASET_DIR / "Volatility75Index_2026.parquet"
    csv = DATASET_DIR / "Volatility75Index_2026.csv"

    assert len(read_quote_window(parquet, WINDOW_START_NS, WINDOW_END_NS)) == 1797
    assert (
        check_window_against_csv(parquet, csv, WINDOW_START_NS, WINDOW_END_NS) == 1797
    )
