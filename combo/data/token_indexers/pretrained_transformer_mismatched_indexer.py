from typing import List

from allennlp import data
from allennlp.data import token_indexers
from overrides import overrides


@token_indexers.TokenIndexer.register("pretrained_transformer_mismatched_tmp_fix")
class PretrainedTransformerMismatchedIndexer(token_indexers.PretrainedTransformerMismatchedIndexer):
    """TODO remove when fixed in AllenNLP (fc47bf6ae5c0df6d473103d459b75fa7edbdd979)"""

    @overrides
    def tokens_to_indices(self, tokens: List[data.Token], vocabulary: data.Vocabulary) -> data.IndexedTokenList:
        self._matched_indexer._add_encoding_to_vocabulary_if_needed(vocabulary)

        wordpieces, offsets = self._allennlp_tokenizer.intra_word_tokenize([t.text for t in tokens])

        # For tokens that don't correspond to any word pieces, we put (-1, -1) into the offsets.
        # That results in the embedding for the token to be all zeros.
        offsets = [x if x is not None else (-1, -1) for x in offsets]

        output: data.IndexedTokenList = {
            "token_ids": [t.text_id for t in wordpieces],
            "mask": [True] * len(tokens),  # for original tokens (i.e. word-level)
            "type_ids": [t.type_id for t in wordpieces],
            "offsets": offsets,
            "wordpiece_mask": [True] * len(wordpieces),  # for wordpieces (i.e. subword-level)
        }

        return self._matched_indexer._postprocess_output(output)
