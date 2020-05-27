"""Sequence multilabel field implementation."""
import logging
import textwrap
from typing import Set, List, Callable, Iterator, Union, Dict

import torch
from allennlp import data
from allennlp.common import checks, util
from allennlp.data import fields
from overrides import overrides

logger = logging.getLogger(__name__)


class SequenceMultiLabelField(data.Field[torch.Tensor]):
    """
    A `SequenceMultiLabelField` is an extension of the :class:`MultiLabelField` that allows for multiple labels
    while keeping sequence dimension.

    This field will get converted into a sequence of vectors of length equal to the vocabulary size with
    M from N encoding for the labels (all zeros, and ones for the labels).

    # Parameters

    multi_labels : `List[List[str]]`
    multi_label_indexer : `Callable[[data.Vocabulary], Callable[[List[str]], List[int]]]`
        Nested callable which based on vocab creates mapper for multilabel field in the sequence from strings
        to indexed, int values.
    sequence_field : `SequenceField`
        A field containing the sequence that this `SequenceMultiLabelField` is labeling.  Most often, this is a
        `TextField`, for tagging individual tokens in a sentence.
    label_namespace : `str`, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the `Vocabulary` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    """
    _already_warned_namespaces: Set[str] = set()

    def __init__(
            self,
            multi_labels: List[List[str]],
            multi_label_indexer: Callable[[data.Vocabulary], Callable[[List[str]], List[int]]],
            sequence_field: fields.SequenceField,
            label_namespace: str = "labels",
    ) -> None:
        self.multi_labels = multi_labels
        self.sequence_field = sequence_field
        self.multi_label_indexer = multi_label_indexer
        self._label_namespace = label_namespace
        self._indexed_multi_labels = None
        self._maybe_warn_for_namespace(label_namespace)
        if len(multi_labels) != sequence_field.sequence_length():
            raise checks.ConfigurationError(
                "Label length and sequence length "
                "don't match: %d and %d" % (len(multi_labels), sequence_field.sequence_length())
            )

        if not all([isinstance(x, str) for multi_label in multi_labels for x in multi_label]):
            raise checks.ConfigurationError(
                "SequenceMultiLabelField must be passed either all "
                "strings or all ints. Found labels {} with "
                "types: {}.".format(multi_labels, [type(x) for multi_label in multi_labels for x in multi_label])
            )

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (self._label_namespace.endswith("labels") or self._label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning(
                    "Your label namespace was '%s'. We recommend you use a namespace "
                    "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                    "default to your vocabulary.  See documentation for "
                    "`non_padded_namespaces` parameter in Vocabulary.",
                    self._label_namespace,
                )
                self._already_warned_namespaces.add(label_namespace)

    # Sequence methods
    def __iter__(self) -> Iterator[Union[List[str], int]]:
        return iter(self.multi_labels)

    def __getitem__(self, idx: int) -> Union[List[str], int]:
        return self.multi_labels[idx]

    def __len__(self) -> int:
        return len(self.multi_labels)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._indexed_multi_labels is None:
            for multi_label in self.multi_labels:
                for label in multi_label:
                    counter[self._label_namespace][label] += 1  # type: ignore

    @overrides
    def index(self, vocab: data.Vocabulary):
        indexer = self.multi_label_indexer(vocab)

        indexed = []
        for multi_label in self.multi_labels:
            indexed.append(indexer(multi_label))
        self._indexed_multi_labels = indexed

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": self.sequence_field.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens = padding_lengths["num_tokens"]
        assert len(self._indexed_multi_labels) > 0
        classes_count = len(self._indexed_multi_labels[0])
        default_value = [0.0] * classes_count
        padded_tags = util.pad_sequence_to_length(self._indexed_multi_labels, desired_num_tokens, lambda: default_value)
        tensor = torch.LongTensor(padded_tags)
        return tensor

    @overrides
    def empty_field(self) -> "SequenceMultiLabelField":
        # The empty_list here is needed for mypy
        empty_list: List[List[str]] = [[]]
        sequence_label_field = SequenceMultiLabelField(empty_list, lambda x: lambda y: y,
                                                       self.sequence_field.empty_field())
        sequence_label_field._indexed_labels = empty_list
        return sequence_label_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.multi_labels), 100)
        )
        return (
            f"SequenceMultiLabelField of length {length} with "
            f"labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."
        )
