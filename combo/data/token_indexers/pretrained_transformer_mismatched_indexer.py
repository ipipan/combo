from typing import Optional, Dict, Any, List, Tuple

from allennlp import data
from allennlp.data import token_indexers, tokenizers


@data.TokenIndexer.register("pretrained_transformer_mismatched_fixed")
class PretrainedTransformerMismatchedIndexer(token_indexers.PretrainedTransformerMismatchedIndexer):
    """TODO(mklimasz) Remove during next allennlp update, fixed on allennlp master."""

    def __init__(self, model_name: str, namespace: str = "tags", max_length: int = None,
                 tokenizer_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        # The matched version v.s. mismatched
        super().__init__(model_name, namespace, max_length, tokenizer_kwargs, **kwargs)
        self._matched_indexer = PretrainedTransformerIndexer(
            model_name,
            namespace=namespace,
            max_length=max_length,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )
        self._allennlp_tokenizer = self._matched_indexer._allennlp_tokenizer
        self._tokenizer = self._matched_indexer._tokenizer
        self._num_added_start_tokens = self._matched_indexer._num_added_start_tokens
        self._num_added_end_tokens = self._matched_indexer._num_added_end_tokens


class PretrainedTransformerIndexer(token_indexers.PretrainedTransformerIndexer):

    def __init__(
            self,
            model_name: str,
            namespace: str = "tags",
            max_length: int = None,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> None:
        super().__init__(model_name, namespace, max_length, tokenizer_kwargs, **kwargs)
        self._namespace = namespace
        self._allennlp_tokenizer = PretrainedTransformerTokenizer(
            model_name, tokenizer_kwargs=tokenizer_kwargs
        )
        self._tokenizer = self._allennlp_tokenizer.tokenizer
        self._added_to_vocabulary = False

        self._num_added_start_tokens = len(self._allennlp_tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(self._allennlp_tokenizer.single_sequence_end_tokens)

        self._max_length = max_length
        if self._max_length is not None:
            num_added_tokens = len(self._allennlp_tokenizer.tokenize("a")) - 1
            self._effective_max_length = (  # we need to take into account special tokens
                    self._max_length - num_added_tokens
            )
            if self._effective_max_length <= 0:
                raise ValueError(
                    "max_length needs to be greater than the number of special tokens inserted."
                )


class PretrainedTransformerTokenizer(tokenizers.PretrainedTransformerTokenizer):

    def _intra_word_tokenize(
            self, string_tokens: List[str]
    ) -> Tuple[List[data.Token], List[Optional[Tuple[int, int]]]]:
        tokens: List[data.Token] = []
        offsets: List[Optional[Tuple[int, int]]] = []
        for token_string in string_tokens:
            wordpieces = self.tokenizer.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            wp_ids = wordpieces["input_ids"]

            if len(wp_ids) > 0:
                offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
                tokens.extend(
                    data.Token(text=wp_text, text_id=wp_id)
                    for wp_id, wp_text in zip(wp_ids, self.tokenizer.convert_ids_to_tokens(wp_ids))
                )
            else:
                offsets.append(None)
        return tokens, offsets
