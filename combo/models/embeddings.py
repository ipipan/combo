"""Embeddings."""
from typing import Optional

import torch
import torch.nn as nn
from allennlp import nn as allen_nn, data
from allennlp.modules import token_embedders
from overrides import overrides
from transformers import modeling_auto

from combo.models import base, dilated_cnn


@token_embedders.TokenEmbedder.register('char_embeddings')
@token_embedders.TokenEmbedder.register('char_embeddings_from_config', constructor='from_config')
class CharacterBasedWordEmbeddings(token_embedders.TokenEmbedder):
    """Character-based word embeddings."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 dilated_cnn_encoder: dilated_cnn.DilatedCnnEncoder):
        super().__init__()
        self.char_embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.dilated_cnn_encoder = dilated_cnn_encoder
        self.output_dim = embedding_dim

    def forward(self,
                x: torch.Tensor,
                char_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        if char_mask is None:
            char_mask = x.new_ones(x.size())

        x = self.char_embed(x)
        x = x * char_mask.unsqueeze(-1).float()

        BATCH_SIZE, SENTENCE_LENGTH, MAX_WORD_LENGTH, CHAR_EMB = x.size()

        words = []
        for i in range(SENTENCE_LENGTH):
            word = x[:, i, :, :].reshape(BATCH_SIZE, MAX_WORD_LENGTH, CHAR_EMB).transpose(1, 2)
            word = self.dilated_cnn_encoder(word)
            word, _ = torch.max(word, dim=2)
            words.append(word)
        return torch.stack(words, dim=1)

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @classmethod
    def from_config(cls,
                    embedding_dim: int,
                    vocab: data.Vocabulary,
                    dilated_cnn_encoder: dilated_cnn.DilatedCnnEncoder,
                    vocab_namespace: str = 'token_characters'):
        assert vocab_namespace in vocab.get_namespaces()
        return cls(
            embedding_dim=embedding_dim,
            num_embeddings=vocab.get_vocab_size(vocab_namespace),
            dilated_cnn_encoder=dilated_cnn_encoder
        )


@token_embedders.TokenEmbedder.register('embeddings_projected')
class ProjectedWordEmbedder(token_embedders.Embedding):
    """Word embeddings."""

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 vocab_namespace: str = "tokens",
                 pretrained_file: str = None,
                 vocab: data.Vocabulary = None,
                 projection_layer: Optional[base.Linear] = None):
        super().__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            weight=weight,
            padding_index=padding_index,
            trainable=trainable,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            vocab_namespace=vocab_namespace,
            pretrained_file=pretrained_file,
            vocab=vocab
        )
        self._projection = projection_layer
        self.output_dim = embedding_dim if projection_layer is None else projection_layer.out_features

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim


@token_embedders.TokenEmbedder.register('transformers_word_embeddings')
class TransformersWordEmbedder(token_embedders.PretrainedTransformerMismatchedEmbedder):
    """
    Transformers word embeddings as last hidden state + optional projection layers.

    Tested with Bert (but should work for other models as well).
    """

    def __init__(self,
                 model_name: str,
                 projection_dim: int,
                 projection_activation: Optional[allen_nn.Activation] = lambda x: x,
                 projection_dropout_rate: Optional[float] = 0.0):
        super().__init__(model_name)
        self.transformers_encoder = modeling_auto.AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformers_encoder.config.hidden_size
        if projection_dim:
            self.projection_layer = base.Linear(in_features=self.output_dim,
                                                out_features=projection_dim,
                                                dropout_rate=projection_dropout_rate,
                                                activation=projection_activation)
            self.output_dim = projection_dim
        else:
            self.projection_layer = None

    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = super().forward(token_ids=token_ids, mask=mask, offsets=offsets, wordpiece_mask=wordpiece_mask,
                            type_ids=type_ids, segment_concat_mask=segment_concat_mask)
        if self.projection_layer:
            x = self.projection_layer(x)
        return x

    @overrides
    def get_output_dim(self):
        return self.output_dim
