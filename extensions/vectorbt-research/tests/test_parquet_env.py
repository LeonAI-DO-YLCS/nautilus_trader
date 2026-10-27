import numpy as np
import pandas as pd


def test_research_environment_writes_and_reads_parquet_with_exact_integer_columns(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "balance_cents": np.array([1000476, 0], dtype=np.int64),
            "ts_ns": np.array(
                [1787023200000000000, 1787023260000000000], dtype=np.int64
            ),
        },
    )
    path = tmp_path / "exact_integers.parquet"

    frame.to_parquet(path, engine="pyarrow")
    loaded = pd.read_parquet(path, engine="pyarrow")

    pd.testing.assert_frame_equal(loaded, frame)
    assert loaded["balance_cents"].dtype == np.int64
    assert loaded["ts_ns"].dtype == np.int64
