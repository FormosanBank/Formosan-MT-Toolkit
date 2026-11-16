from __future__ import annotations

import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fetch_xml
import github_snapshot
import pandas as pd
from build_big_corpus import (
    corpus_frame,
    write_target,
)
from fetch_xml import (
    classify_xml,
    download_blob_for_languages,
    get_tree,
    git_blob_sha,
    load_or_create_repository_snapshot,
    repository_selection,
    resolve_default_repository_refs,
    write_blob_cache,
)
from pipeline_common import load_pipeline_config, write_columnar_cache
from support import mt_contract_fields

ROOT = Path(__file__).resolve().parents[1]


class AcquisitionTests(unittest.TestCase):
    def test_multi_language_fetch_parses_and_routes_one_cached_blob(self) -> None:
        xml_bytes = (
            b'<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
            b'xml:lang="ami" dialect="Coastal"><S /></TEXT>'
        )
        blob = git_blob_sha(xml_bytes)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            (cache / f"{blob}.xml").write_bytes(xml_bytes)
            out_dirs = {"ami": root / "ami", "tay": root / "tay"}
            for path in out_dirs.values():
                path.mkdir()
            with mock.patch.object(fetch_xml, "RAW_XML_CACHE_DIR", cache):
                result = download_blob_for_languages(
                    "FormosanBank",
                    "FixtureRepo",
                    {"path": "Final_XML/Atayal/misleading-name.xml", "sha": blob},
                    ("ami", "tay"),
                    None,
                    "a" * 40,
                    out_dirs,
                    None,
                    download_retries=1,
                    retry_base_sleep=0,
                    retry_max_sleep=0,
                )
            self.assertEqual(result["ami"].status, "kept")
            self.assertEqual(result["tay"].status, "source_language_mismatch")
            self.assertTrue((out_dirs["ami"] / result["ami"].destination).is_file())
            self.assertFalse(list(out_dirs["tay"].rglob("*.xml")))

    def test_aggregation_uses_xml_language_metadata_not_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tay_zh_processed.csv"
            frame = pd.DataFrame(
                [
                    {
                        **mt_contract_fields("O maan ko faloco'?"),
                        "lang_code": "ami",
                        "formosan_sentence": "O maan ko faloco'?",
                        "english": "What are you thinking?",
                        "target_lang": "eng",
                        "row_id": "row-1",
                        "source_record_id": "record-1",
                        "source": "Repo/XML/misleading-path.xml",
                        "row_type": "sentence",
                        "source_bucket": "narrative",
                    }
                ]
            )
            frame.to_csv(path, index=False)

            target, loaded, input_type = corpus_frame(path)

            frame.loc[0, "target_lang"] = ""
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(SystemExit, "TRANSL/@xml:lang"):
                corpus_frame(path)

        self.assertEqual(target, "en")
        self.assertEqual(input_type, "pairwise")
        self.assertEqual(loaded.loc[0, "lang_code"], "ami")
        self.assertNotIn("source_bucket", loaded.columns)
        self.assertEqual(
            loaded.loc[0, "english_sentence"],
            "What are you thinking?",
        )

    def test_aggregation_normalizes_columnar_provenance_to_csv_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = []
            for name, row_id, xml_id in (
                ("numeric", "row-1", 0),
                ("text", "row-2", "sample_S1"),
            ):
                path = root / f"{name}.csv"
                frame = pd.DataFrame(
                    [
                        {
                            **mt_contract_fields("malu ku su."),
                            "lang_code": "ami",
                            "formosan_sentence": "malu ku su.",
                            "english": f"Good {name} example.",
                            "target_lang": "eng",
                            "row_id": row_id,
                            "source_record_id": f"record-{row_id}",
                            "source": f"Repo/XML/{name}.xml",
                            "row_type": "sentence",
                            "xml_id": xml_id,
                        }
                    ]
                )
                frame.to_csv(path, index=False)
                write_columnar_cache(frame, path)
                _, cached, _ = corpus_frame(path)
                loaded.append(cached)

            output = root / "big_corpus_en.csv"
            combined = write_target(loaded, output, "english_sentence")

            self.assertEqual(combined["xml_id"].tolist(), ["0", "sample_S1"])
            self.assertTrue(output.with_suffix(".parquet").is_file())

    def test_graphql_repository_resolution_preserves_selection_order(self) -> None:
        response = mock.Mock()
        response.json.return_value = {
            "data": {
                "organization": {
                    "repositories": {
                        "nodes": [
                            {
                                "name": "RepoA",
                                "defaultBranchRef": {
                                    "name": "main",
                                    "target": {"oid": "a" * 40},
                                },
                            },
                            {
                                "name": "RepoB",
                                "defaultBranchRef": {
                                    "name": "release",
                                    "target": {"oid": "b" * 40},
                                },
                            },
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
        with mock.patch.object(
            github_snapshot,
            "api_post",
            return_value=response,
        ):
            refs = resolve_default_repository_refs(
                "FormosanBank",
                ["RepoB", "RepoA"],
            )
        self.assertEqual([ref.name for ref in refs], ["RepoB", "RepoA"])
        self.assertEqual(refs[0].requested_ref, "release")

    def test_malformed_xml_is_not_a_language_mismatch(self) -> None:
        status, language, dialect, error = classify_xml(
            b"<TEXT><S>",
            "ami",
            None,
            None,
        )
        self.assertEqual(status, "parse_error")
        self.assertEqual(language, "")
        self.assertTrue(error)

    def test_git_blob_hash_matches_git_object_format(self) -> None:
        self.assertEqual(
            git_blob_sha(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )

    def test_concurrent_blob_cache_writes_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "blob.xml"
            content = b"<TEXT xml:lang=\"ami\"/>"
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                writes = [
                    executor.submit(write_blob_cache, cache_path, content)
                    for _ in range(100)
                ]
                for write in writes:
                    write.result()
            self.assertEqual(cache_path.read_bytes(), content)
            self.assertEqual(list(cache_path.parent.glob("*.tmp")), [])

    def test_tree_discovery_is_scoped_to_final_xml(self) -> None:
        commit_sha = "a" * 40
        final_xml_sha = "b" * 40

        def response(payload: dict) -> mock.Mock:
            result = mock.Mock()
            result.json.return_value = payload
            return result

        def api_get(url: str, *, params: dict | None = None) -> mock.Mock:
            if url.endswith(commit_sha):
                self.assertIsNone(params)
                return response(
                    {
                        "truncated": False,
                        "tree": [
                            {
                                "path": "Final_XML",
                                "type": "tree",
                                "sha": final_xml_sha,
                            },
                            {
                                "path": "large-unrelated-data",
                                "type": "tree",
                                "sha": "c" * 40,
                            },
                        ],
                    }
                )
            if url.endswith(final_xml_sha):
                self.assertEqual(params, {"recursive": "1"})
                return response(
                    {
                        "truncated": False,
                        "tree": [
                            {
                                "path": "Paiwan/sample.xml",
                                "type": "blob",
                                "sha": "d" * 40,
                            }
                        ],
                    }
                )
            raise AssertionError(f"Unexpected API request: {url}")

        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(
                    github_snapshot,
                    "CACHE_DIR",
                    Path(temporary),
                ),
                mock.patch.object(
                    github_snapshot,
                    "api_get",
                    side_effect=api_get,
                ) as get,
            ):
                tree = get_tree(
                    "FormosanBank",
                    "example",
                    commit_sha,
                    root_path="Final_XML",
                )
        self.assertEqual(tree[0]["path"], "Final_XML/Paiwan/sample.xml")
        self.assertEqual(get.call_count, 2)

    def test_private_xml_discovery_accepts_both_supported_roots(self) -> None:
        self.assertTrue(
            fetch_xml.is_private_release_xml_path(
                "Final_XML/Paiwan/sample.xml"
            )
        )
        self.assertTrue(
            fetch_xml.is_private_release_xml_path("XML/Paiwan/sample.xml")
        )
        self.assertFalse(
            fetch_xml.is_private_release_xml_path(
                "archive/XML/Paiwan/sample.xml"
            )
        )
        self.assertFalse(
            fetch_xml.is_private_release_xml_path("Final_XML/README.md")
        )

    def test_repository_snapshot_is_resolved_once_and_reused(self) -> None:
        selection = repository_selection(
            org="FormosanBank",
            public=False,
            branch=None,
            discovered=["RepoB", "RepoA"],
            selected=["RepoB", "RepoA"],
            excluded=[],
        )
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "source_repository_snapshot.json"
            with (
                mock.patch.object(
                    github_snapshot,
                    "get_default_branch",
                    return_value="main",
                ) as branch,
                mock.patch.object(
                    github_snapshot,
                    "resolve_commit",
                    side_effect=["a" * 40, "b" * 40],
                ) as resolve,
            ):
                first = load_or_create_repository_snapshot(
                    snapshot,
                    selection=selection,
                    refresh_metadata=True,
                )
            self.assertEqual(branch.call_count, 2)
            self.assertEqual(resolve.call_count, 2)

            with (
                mock.patch.object(
                    github_snapshot,
                    "get_default_branch",
                    side_effect=AssertionError("snapshot should be reused"),
                ),
                mock.patch.object(
                    github_snapshot,
                    "resolve_commit",
                    side_effect=AssertionError("snapshot should be reused"),
                ),
            ):
                second = load_or_create_repository_snapshot(
                    snapshot,
                    selection=selection,
                    refresh_metadata=True,
                )
        self.assertEqual(first, second)

    def test_pipeline_pins_full_formosanbank_revision(self) -> None:
        config = load_pipeline_config()
        revision = config["formosanbank"]["qc_revision"]
        self.assertEqual(len(revision), 40)
        self.assertTrue(all(character in "0123456789abcdef" for character in revision))
        json.loads((ROOT / "config/corpus_pipeline.json").read_text(encoding="utf-8"))
