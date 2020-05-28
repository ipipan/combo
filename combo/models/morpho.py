"""Morphological features models."""
from typing import Dict, List, Optional, Union

import torch
from allennlp import data
from allennlp.common import checks
from allennlp.modules import feedforward
from allennlp.nn import Activation

from combo.data import dataset
from combo.models import base, utils


@base.Predictor.register('combo_morpho_from_vocab', constructor='from_vocab')
class MorphologicalFeatures(base.Predictor):
    """Morphological features predicting model."""

    def __init__(self, feedforward_network: feedforward.FeedForward, slices: Dict[str, List[int]]):
        super().__init__()
        self.feedforward_network = feedforward_network
        self.slices = slices

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                mask: Optional[torch.BoolTensor] = None,
                labels: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                sample_weights: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            mask = x.new_ones(x.size()[:-1])

        x = self.feedforward_network(x)

        prediction = []
        for cat, cat_indices in self.slices.items():
            prediction.append(x[:, :, cat_indices].argmax(dim=-1))

        output = {
            'prediction': torch.stack(prediction, dim=-1),
            'probability': x
        }

        if labels is not None:
            if sample_weights is None:
                sample_weights = labels.new_ones([mask.size(0)])
            output['loss'] = self._loss(x, labels, mask, sample_weights)

        return output

    def _loss(self, pred: torch.Tensor, true: torch.Tensor, mask: torch.BoolTensor,
              sample_weights: torch.Tensor) -> torch.Tensor:
        assert pred.size() == true.size()
        BATCH_SIZE, SENTENCE_LENGTH, MORPHOLOGICAL_FEATURES = pred.size()

        valid_positions = mask.sum()

        pred = pred.reshape(-1, MORPHOLOGICAL_FEATURES)
        true = true.reshape(-1, MORPHOLOGICAL_FEATURES)
        mask = mask.reshape(-1)
        loss = None
        loss_func = utils.masked_cross_entropy
        for cat, cat_indices in self.slices.items():
            if cat not in ['__PAD__', '_']:
                if loss is None:
                    loss = loss_func(pred[:, cat_indices],
                                     true[:, cat_indices].argmax(dim=1),
                                     mask)
                else:
                    loss += loss_func(pred[:, cat_indices],
                                      true[:, cat_indices].argmax(dim=1),
                                      mask)
        loss = loss.reshape(BATCH_SIZE, -1) * sample_weights.unsqueeze(-1)
        return loss.sum() / valid_positions

    @classmethod
    def from_vocab(cls,
                   vocab: data.Vocabulary,
                   vocab_namespace: str,
                   input_dim: int,
                   num_layers: int,
                   hidden_dims: List[int],
                   activations: Union[Activation, List[Activation]],
                   dropout: Union[float, List[float]] = 0.0,
                   ):
        if len(hidden_dims) + 1 != num_layers:
            raise checks.ConfigurationError(
                "len(hidden_dims) (%d) + 1 != num_layers (%d)" % (len(hidden_dims), num_layers)
            )

        assert vocab_namespace in vocab.get_namespaces()
        hidden_dims = hidden_dims + [vocab.get_vocab_size(vocab_namespace)]

        slices = dataset.get_slices_if_not_provided(vocab)

        return cls(
            feedforward_network=feedforward.FeedForward(
                input_dim=input_dim,
                num_layers=num_layers,
                hidden_dims=hidden_dims,
                activations=activations,
                dropout=dropout),
            slices=slices
        )
