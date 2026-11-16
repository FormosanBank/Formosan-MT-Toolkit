from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import columnar_io
import pandas as pd
from columnar_io import read_csv_or_columnar, write_columnar_cache


class ColumnarCacheTests(unittest.TestCase):
    def test_round_trip_is_bound_to_canonical_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "corpus.csv"
            release = pd.DataFrame({"row_id": ["a", "b"], "value": ["x", "y"]})
            cached = release.assign(_normalized=["x", "y"])
            release.to_csv(csv_path, index=False)
            with mock.patch.object(
                columnar_io,
                "sha256_file",
                wraps=columnar_io.sha256_file,
            ) as hasher:
                write_columnar_cache(cached, csv_path)
                hashed_calls = hasher.call_count
                pd.testing.assert_frame_equal(
                    read_csv_or_columnar(csv_path, keep_default_na=False),
                    cached,
                )
                self.assertEqual(hasher.call_count, hashed_calls)

                csv_path.write_text("row_id,value\na,changed\n", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    read_csv_or_columnar(csv_path, keep_default_na=False)
                self.assertGreater(hasher.call_count, hashed_calls)

    def test_v1_manifest_is_verified_and_upgraded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "corpus.csv"
            frame = pd.DataFrame({"row_id": ["a"], "value": ["x"]})
            frame.to_csv(csv_path, index=False)
            write_columnar_cache(frame, csv_path)
            manifest_path = csv_path.with_suffix(".parquet.json")
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "complete": True,
                        "canonical_csv_sha256": current["canonical_csv"]["sha256"],
                        "parquet_sha256": current["parquet"]["sha256"],
                        "rows": current["rows"],
                        "columns": current["columns"],
                    }
                ),
                encoding="utf-8",
            )

            pd.testing.assert_frame_equal(
                read_csv_or_columnar(csv_path, keep_default_na=False),
                frame,
            )
            upgraded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(upgraded["schema_version"], 2)
            self.assertEqual(
                upgraded["canonical_csv"]["sha256"],
                current["canonical_csv"]["sha256"],
            )

    def test_column_projection_reads_only_requested_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "corpus.csv"
            frame = pd.DataFrame(
                {
                    "row_id": ["a", "b"],
                    "value": ["x", "y"],
                    "unused": ["large", "payload"],
                }
            )
            frame.to_csv(csv_path, index=False)
            write_columnar_cache(frame, csv_path)
            projected = read_csv_or_columnar(
                csv_path,
                columns=["row_id", "value"],
            )
            self.assertEqual(list(projected.columns), ["row_id", "value"])
            pd.testing.assert_frame_equal(projected, frame[["row_id", "value"]])
