from __future__ import annotations

import unittest

import pandas as pd
from publish_huggingface_dataset import validate_release_frame


class HuggingFaceDatasetPublisherTests(unittest.TestCase):
    def release_frame(self, repository: str, source: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lang_code": ["ami"],
                "formosan_sentence": ["O maan ko faloco'?"],
                "english_sentence": ["What are you thinking?"],
                "dialect": ["Coastal"],
                "source": [source],
                "split": ["test"],
                "repository": [repository],
                "pivot_origin": ["original"],
                "mt_eval_eligible": [True],
                "row_type": ["sentence"],
                **{
                    column: ["value"]
                    for column in (
                        "row_id",
                        "source_record_id",
                        "content_sha256",
                        "repository_commit",
                        "xml_path",
                        "xml_id",
                        "kindOf",
                        "standard_namespace",
                        "standard_origin",
                        "pivot_provider",
                        "pivot_direction",
                        "eval_tier",
                        "document_id",
                    )
                },
            }
        )

    def test_private_release_allows_private_repository_rows(self) -> None:
        frame = self.release_frame("Private-Dev-Repo", "Private-Dev-Repo/Final_XML/a.xml")
        validate_release_frame(frame, "english_sentence", private_release=True)

    def test_public_release_rejects_private_repository_rows(self) -> None:
        frame = self.release_frame("Private-Dev-Repo", "Private-Dev-Repo/Final_XML/a.xml")
        with self.assertRaisesRegex(SystemExit, "non-public repository"):
            validate_release_frame(frame, "english_sentence", private_release=False)
