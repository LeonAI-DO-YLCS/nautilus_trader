import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from vectorbt_research.dataset import read_quote_window
from vectorbt_research.reports import read_run
from vectorbt_research.runner import run_story_1_window

DATASET_DIR = Path(
    "/mnt/c/LeonAI-DO/dev/TRADING/Datasets/Volatility75Index_yearly_deduped"
)
_DATE = "2026.08.18"
# 2026-08-18T03:20:00Z and 2026-08-18T03:21:00Z, the small test window.
_SMALL_START_NS = 1787023200000000000
_SMALL_END_NS = 1787023260000000000
# S8: 2026-08-18 03:20-04:20 UTC.
_S8_START_NS = 1787023200000000000
_S8_END_NS = 1787026800000000000
# S7: the five hand-written test trades, in list order. Their starts lie inside S8.
_S7_SIGNALS = [
    {
        "signal_id": "signal-01",
        "direction": "CALL",
        "start_time": "2026-08-18T03:22:00Z",
    },
    {
        "signal_id": "signal-02",
        "direction": "PUT",
        "start_time": "2026-08-18T03:21:00Z",
    },
    {
        "signal_id": "signal-03",
        "direction": "PUT",
        "start_time": "2026-08-18T03:21:00Z",
    },
    {
        "signal_id": "signal-04",
        "direction": "CALL",
        "start_time": "2026-08-18T03:24:00Z",
    },
    {
        "signal_id": "signal-05",
        "direction": "CALL",
        "start_time": "2026-08-18T03:23:00Z",
    },
]


def _write_dataset(folder, quotes, csv_quotes=None):
    """
    Write a tiny 2026 Parquet file and its exporter CSV, and return the folder.

    Each quote is (clock, bid, ask) with exact price text, or None when empty. The CSV
    holds the same rows unless csv_quotes is given.
    """
    folder.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "<DATE>": [_DATE] * len(quotes),
                "<TIME>": [clock for clock, _, _ in quotes],
                "<BID>": pa.array(
                    [None if bid is None else float(bid) for _, bid, _ in quotes],
                    type=pa.float64(),
                ),
                "<ASK>": pa.array(
                    [None if ask is None else float(ask) for _, _, ask in quotes],
                    type=pa.float64(),
                ),
            }
        ),
        folder / "Volatility75Index_2026.parquet",
    )
    lines = ["<DATE>\t<TIME>\t<BID>\t<ASK>"]
    for clock, bid, ask in csv_quotes or quotes:
        lines.append(f"{_DATE}\t{clock}\t{bid or ''}\t{ask or ''}")
    (folder / "Volatility75Index_2026.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return folder


def _read_run(run_dir):
    run_dir = Path(run_dir)
    run = read_run(run_dir)
    return run["manifest"], run["contracts"]


def _continuation_quotes():
    return [
        ("03:20:02.000", "99.00", "99.00"),
        ("03:20:40.000", "100.00", "100.00"),  # entry for signal-open (start 03:20:30)
        ("03:21:29.000", "101.00", "101.00"),  # after the window; expiry is 03:21:30
        ("03:21:31.000", "101.50", "101.50"),  # after expiry; closes the exit check
    ]


def test_entry_after_window_end_is_skipped(tmp_path):
    dataset = _write_dataset(
        tmp_path / "dataset",
        [
            ("03:20:02.000", "100.00", "100.00"),  # entry for signal-in
            ("03:20:55.000", "100.50", "100.50"),  # exit: 5 s before expiry 03:21:00
            ("03:21:05.000", "101.00", "101.00"),  # first tick after the window end
        ],
    )

    run_dir = run_story_1_window(
        dataset_folder=dataset,
        window_start_ns=_SMALL_START_NS,
        window_end_ns=_SMALL_END_NS,
        frozen_signals=[
            {
                "signal_id": "signal-in",
                "direction": "CALL",
                "start_time": "2026-08-18T03:20:00Z",
            },
            {
                "signal_id": "signal-late",
                "direction": "PUT",
                "start_time": "2026-08-18T03:20:56Z",
            },
        ],
        scenario="bid",
        output_dir=tmp_path / "runs",
    )

    manifest, contracts = _read_run(run_dir)
    rows = contracts.set_index("signal_id")
    assert rows.loc["signal-in", "status"] == "settled"
    assert rows.loc["signal-in", "outcome"] == "win"
    assert rows.loc["signal-late", "status"] == "skipped"
    assert str(rows.loc["signal-late", "reason"]).strip()
    assert manifest["counts"]["skipped_signals"] == 1


def test_continuation_rows_settle_a_contract_still_open(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset", _continuation_quotes())

    run_dir = run_story_1_window(
        dataset_folder=dataset,
        window_start_ns=_SMALL_START_NS,
        window_end_ns=_SMALL_END_NS,
        frozen_signals=[
            {
                "signal_id": "signal-open",
                "direction": "CALL",
                "start_time": "2026-08-18T03:20:30Z",
            }
        ],
        scenario="bid",
        output_dir=tmp_path / "runs",
    )

    _, contracts = _read_run(run_dir)
    contract = contracts.set_index("signal_id").loc["signal-open"]
    assert contract["status"] == "settled"
    assert contract["outcome"] == "win"
    assert contract["entry_timestamp_ns"] == 1787023240000000000  # 03:20:40
    assert contract["exit_timestamp_ns"] == 1787023289000000000  # 03:21:29
    assert contract["gross_credit_cents"] == 976


def test_manifest_records_source_rows_and_csv_check(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset", _continuation_quotes())

    run_dir = run_story_1_window(
        dataset_folder=dataset,
        window_start_ns=_SMALL_START_NS,
        window_end_ns=_SMALL_END_NS,
        frozen_signals=[
            {
                "signal_id": "signal-open",
                "direction": "CALL",
                "start_time": "2026-08-18T03:20:30Z",
            }
        ],
        scenario="bid",
        output_dir=tmp_path / "runs",
    )

    manifest, _ = _read_run(run_dir)
    source = manifest["sources"][0]
    assert source["path"] == str(dataset / "Volatility75Index_2026.parquet")
    assert (source["first_row"], source["last_row"]) == (0, 3)
    rows = read_quote_window(
        dataset / "Volatility75Index_2026.parquet", _SMALL_START_NS, 1787023291000000001
    )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert source["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert manifest["windows"][0]["start_timestamp_ns"] == _SMALL_START_NS
    assert manifest["windows"][0]["end_timestamp_ns"] == _SMALL_END_NS
    provenance = manifest["quote_provenance"]
    assert provenance["csv_check"] == "matched"
    assert provenance["window_rows"] == 2
    assert provenance["continuation_rows"] == 2


def test_csv_mismatch_stops_the_run(tmp_path):
    rows = _continuation_quotes()
    dataset = _write_dataset(
        tmp_path / "dataset",
        rows,
        csv_quotes=[rows[0], ("03:20:40.000", "100.10", "100.00"), *rows[2:]],
    )

    with pytest.raises(ValueError, match="BID differs"):
        run_story_1_window(
            dataset_folder=dataset,
            window_start_ns=_SMALL_START_NS,
            window_end_ns=_SMALL_END_NS,
            frozen_signals=_S7_SIGNALS[:1],
            scenario="bid",
            output_dir=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def test_output_inside_dataset_folder_is_refused(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset", _continuation_quotes())

    with pytest.raises(ValueError, match="dataset folder"):
        run_story_1_window(
            dataset_folder=dataset,
            window_start_ns=_SMALL_START_NS,
            window_end_ns=_SMALL_END_NS,
            frozen_signals=_S7_SIGNALS[:1],
            scenario="bid",
            output_dir=dataset / "runs",
        )
    assert not (dataset / "runs").exists()


@pytest.mark.skipif(
    not DATASET_DIR.is_dir(), reason="Volatility75 dataset folder is absent"
)
def test_s8_window_runs_on_real_data(tmp_path):
    run_dir = run_story_1_window(
        dataset_folder=DATASET_DIR,
        window_start_ns=_S8_START_NS,
        window_end_ns=_S8_END_NS,
        frozen_signals=_S7_SIGNALS,
        scenario="bid",
        output_dir=tmp_path / "s8-bid",
    )

    manifest, contracts = _read_run(run_dir)
    assert manifest["windows"][0]["start_timestamp_ns"] == _S8_START_NS
    assert manifest["windows"][0]["end_timestamp_ns"] == _S8_END_NS
    assert manifest["quote_provenance"]["csv_check"] == "matched"
    assert manifest["quote_provenance"]["window_rows"] == 1797
    assert manifest["counts"]["signals"] == 5
    assert len(contracts) == 5


def test_continuation_crosses_a_gap_longer_than_two_minutes(tmp_path):
    rows = _continuation_quotes()[:3] + [("03:25:00.000", "999.00", "999.00")]
    dataset = _write_dataset(tmp_path / "dataset", rows)
    run_dir = run_story_1_window(
        dataset_folder=dataset,
        window_start_ns=_SMALL_START_NS,
        window_end_ns=_SMALL_END_NS,
        frozen_signals=[
            {
                "signal_id": "signal-open",
                "direction": "CALL",
                "start_time": "2026-08-18T03:20:30Z",
            }
        ],
        scenario="bid",
        output_dir=tmp_path / "runs",
    )
    manifest, contracts = _read_run(run_dir)
    contract = contracts.iloc[0]
    assert contract["status"] == "settled"
    assert contract["outcome"] == "win"
    assert contract["exit_timestamp_ns"] == 1787023289000000000
    assert contract["gross_credit_cents"] == 976
    assert manifest["completeness"]["status"] == "complete"
    assert manifest["quote_provenance"]["continuation_rows"] == 2


def test_true_end_of_file_leaves_open_stake_locked(tmp_path):
    dataset = _write_dataset(tmp_path / "dataset", _continuation_quotes()[:3])
    run_dir = run_story_1_window(
        dataset_folder=dataset,
        window_start_ns=_SMALL_START_NS,
        window_end_ns=_SMALL_END_NS,
        frozen_signals=[
            {
                "signal_id": "signal-open",
                "direction": "CALL",
                "start_time": "2026-08-18T03:20:30Z",
            }
        ],
        scenario="bid",
        output_dir=tmp_path / "runs",
    )
    manifest, contracts = _read_run(run_dir)
    assert contracts.iloc[0]["status"] == "unresolved"
    assert contracts.iloc[0]["gross_credit_cents"] == 0
    assert manifest["completeness"]["status"] == "incomplete"
    assert manifest["completeness"]["reason"]


@pytest.mark.parametrize(
    "start,end",
    [
        (1798761540000000000, 1798761660000000000),
    ],
)
def test_cross_year_window_is_a_clear_error(tmp_path, start, end):
    with pytest.raises(ValueError, match="one yearly file"):
        run_story_1_window(
            dataset_folder=tmp_path / "dataset",
            window_start_ns=start,
            window_end_ns=end,
            frozen_signals=[],
            scenario="bid",
            output_dir=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def test_cross_year_expiry_is_a_clear_error(tmp_path):
    dataset = _write_dataset(
        tmp_path / "dataset", [("23:59:31.000", "100.00", "100.00")]
    )
    path = dataset / "Volatility75Index_2026.parquet"
    table = pq.read_table(path)
    table = table.set_column(0, "<DATE>", pa.array(["2026.12.31"]))
    pq.write_table(table, path)
    (dataset / "Volatility75Index_2026.csv").write_text(
        "<DATE>\t<TIME>\t<BID>\t<ASK>\n2026.12.31\t23:59:31.000\t100.00\t100.00\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one yearly file"):
        run_story_1_window(
            dataset_folder=dataset,
            window_start_ns=1798761540000000000,
            window_end_ns=1798761600000000000,
            frozen_signals=[
                {
                    "signal_id": "year-end",
                    "direction": "CALL",
                    "start_time": "2026-12-31T23:59:30Z",
                }
            ],
            scenario="bid",
            output_dir=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()
