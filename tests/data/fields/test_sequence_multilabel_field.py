"""Sequence multilabel field tests."""
import unittest
from typing import List

import torch
from allennlp import data as allen_data
from allennlp.data import fields as allen_fields

from combo.data import fields


class IndexingSequenceMultiLabelFieldTest(unittest.TestCase):

    def setUp(self) -> None:
        self.namespace = "test_labels"
        self.vocab = allen_data.Vocabulary()
        self.vocab.add_tokens_to_namespace(
            tokens=["t" + str(idx) for idx in range(3)],
            namespace=self.namespace
        )

        def _indexer(vocab: allen_data.Vocabulary):
            vocab_size = vocab.get_vocab_size(self.namespace)

            def _mapper(multi_label: List[str]) -> List[int]:
                one_hot = [0] * vocab_size
                for label in multi_label:
                    index = vocab.get_token_index(label, self.namespace)
                    one_hot[index] = 1
                return one_hot

            return _mapper

        self.indexer = _indexer
        self.sequence_field = _SequenceFieldTestWrapper(self.vocab.get_vocab_size(self.namespace))

    def test_indexing(self):
        # given
        field = fields.SequenceMultiLabelField(
            multi_labels=[["t1", "t2"], [], ["t0"]],
            multi_label_indexer=self.indexer,
            sequence_field=self.sequence_field,
            label_namespace=self.namespace
        )
        expected = [[0, 1, 1], [0, 0, 0], [1, 0, 0]]

        # when
        field.index(self.vocab)

        # then
        self.assertEqual(expected, field._indexed_multi_labels)

    def test_mapping_to_tensor(self):
        # given
        field = fields.SequenceMultiLabelField(
            multi_labels=[["t1", "t2"], [], ["t0"]],
            multi_label_indexer=self.indexer,
            sequence_field=self.sequence_field,
            label_namespace=self.namespace
        )
        field.index(self.vocab)
        expected = torch.tensor([[0, 1, 1], [0, 0, 0], [1, 0, 0]])

        # when
        actual = field.as_tensor(field.get_padding_lengths())

        # then
        self.assertTrue(torch.all(expected.eq(actual)))

    def test_sequence_method(self):
        # given
        field = fields.SequenceMultiLabelField(
            multi_labels=[["t1", "t2"], [], ["t0"]],
            multi_label_indexer=self.indexer,
            sequence_field=self.sequence_field,
            label_namespace=self.namespace
        )

        # when
        length = len(field)
        iter_length = len(list(iter(field)))
        middle_value = field[1]

        # then
        self.assertEqual(3, length)
        self.assertEqual(3, iter_length)
        self.assertEqual([], middle_value)


class _SequenceFieldTestWrapper(allen_fields.SequenceField):

    def __init__(self, length: int):
        self.length = length

    def sequence_length(self) -> int:
        return self.length
