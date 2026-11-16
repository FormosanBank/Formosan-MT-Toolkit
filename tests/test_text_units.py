from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
sys.path.insert(0, str(ROOT / "scripts/shared"))
sys.path.insert(0, str(ROOT / "formosan_mt_experiments/scripts"))

import corpus_quality  # noqa: E402
import mt_common  # noqa: E402
import text_units  # noqa: E402
from text_units import has_terminal_punctuation, target_units, token_count  # noqa: E402


class TextUnitTests(unittest.TestCase):
    def test_tokens_need_a_letter_number_or_mark(self) -> None:
        self.assertEqual(token_count("Pa'araw cingra ."), 2)
        self.assertEqual(token_count("-- ... !"), 0)
        self.assertEqual(token_count("3 wawa"), 2)
        self.assertEqual(token_count(None), 0)
        self.assertEqual(token_count("   "), 0)

    def test_spacing_accents_do_not_split_words(self) -> None:
        # NFKC turns the spacing acute into a space plus a combining mark,
        # which used to make the splitter count one extra token.
        self.assertEqual(token_count("papicakayen ni Panay ci Kuli tu lami´."), 7)
        self.assertEqual(token_count("βi˘waʔ"), 1)

    def test_chinese_counts_each_han_character(self) -> None:
        self.assertEqual(target_units("他是誰？", "chinese"), 3)
        self.assertEqual(target_units("他說 hello world。", "chinese"), 4)
        self.assertEqual(target_units("他是誰？", "english"), 1)

    def test_radical_block_characters_count_as_han(self) -> None:
        # PDF extraction can emit Kangxi radicals (U+2F00 block) for ordinary
        # characters such as 用 and 石.
        self.assertEqual(target_units("老鼠應該要⽤⽯頭抓", "chinese"), 9)
        self.assertEqual(target_units("老鼠應該要用石頭抓", "chinese"), 9)

    def test_terminal_punctuation_ignores_closing_quotes_and_brackets(self) -> None:
        self.assertTrue(has_terminal_punctuation("He left."))
        self.assertTrue(has_terminal_punctuation("他說：「走吧！」 "))
        self.assertTrue(has_terminal_punctuation('"Where?")'))
        self.assertFalse(has_terminal_punctuation("He left"))
        self.assertFalse(has_terminal_punctuation(""))
        self.assertFalse(has_terminal_punctuation(None))

    def test_filter_and_splitter_share_one_definition(self) -> None:
        self.assertIs(corpus_quality.token_count, text_units.token_count)
        self.assertIs(corpus_quality.target_units, text_units.target_units)
        self.assertIs(mt_common.token_count, text_units.token_count)
        frame = pd.DataFrame(
            {
                "lang_code": ["ami"],
                "formosan_sentence": ["papicakayen ni Panay ci Kuli tu lami´."],
                "chinese_sentence": ["老鼠應該要⽤⽯頭抓。"],
                "source": ["Corpora/Example/XML/a.xml"],
                "row_type": ["sentence"],
            }
        )
        keyed = mt_common.add_normalized_columns(frame, target_col="chinese_sentence")
        self.assertEqual(int(keyed["_formosan_tokens"].iloc[0]), 7)
        self.assertEqual(int(keyed["_target_tokens"].iloc[0]), 9)
        self.assertEqual(
            int(keyed["_target_tokens"].iloc[0]),
            corpus_quality.target_units(frame["chinese_sentence"].iloc[0], "chinese"),
        )
        self.assertTrue(bool(keyed["_formosan_terminal"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
