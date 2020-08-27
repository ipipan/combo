import os
import pathlib
import shutil
import unittest
from unittest import mock

import combo.data as data
import combo.predict as predict


class PredictionTest(unittest.TestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "combo"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = TESTS_ROOT / "fixtures"

    def setUp(self) -> None:
        def _cleanup_archive_dir_without_logging(path: str):
            if os.path.exists(path):
                shutil.rmtree(path)

        self.patcher = mock.patch(
            "allennlp.models.archival._cleanup_archive_dir", _cleanup_archive_dir_without_logging
        )
        self.mock_cleanup_archive_dir = self.patcher.start()

    def test_prediction_are_equal_given_the_same_input_in_different_form(self):
        # given
        raw_sentence = "Test."
        raw_sentence_collection = ["Test."]
        tokenized_sentence_collection = [["Test", "."]]
        wrapped_tokenized_sentence = [data.Sentence(tokens=[
            data.Token(id=1, token="Test"),
            data.Token(id=2, token=".")
        ])]
        api_wrapped_tokenized_sentence = [data.conllu2sentence(data.tokens2conllu(["Test", "."]), [])]
        nlp = predict.SemanticMultitaskPredictor.from_pretrained(os.path.join(self.FIXTURES_ROOT, "model.tar.gz"))

        # when
        results = [
            nlp(raw_sentence),
            nlp(raw_sentence_collection)[0],
            nlp(tokenized_sentence_collection)[0],
            nlp(wrapped_tokenized_sentence)[0],
            nlp(api_wrapped_tokenized_sentence)[0]
        ]

        # then
        self.assertTrue(all(x == results[0] for x in results))
