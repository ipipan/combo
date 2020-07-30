"""Sampler tests."""
import unittest

from allennlp import data
from allennlp.data import fields

from combo.data import TokenCountBatchSampler


class TokenCountBatchSamplerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = []
        self.sentences = ["First sentence makes full batch.", "Short", "This ends first batch"]
        for sentence in self.sentences:
            tokens = [data.Token(t)
                      for t in sentence.split()]
            text_field = fields.TextField(tokens, {})
            self.dataset.append(data.Instance({"sentence": text_field}))

    def test_batches(self):
        # given
        sampler = TokenCountBatchSampler(self.dataset, word_batch_size=2, shuffle_dataset=False)

        # when
        length = len(sampler)
        values = list(sampler)

        # then
        self.assertEqual(2, length)
        # sort by lengths + word_batch_size makes 1, 2 first batch
        self.assertListEqual([[1, 2], [0]], values)
