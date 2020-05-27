from typing import Dict, Optional, List, Union

import torch
import torch.nn as nn
from allennlp import common, data
from allennlp import nn as allen_nn
from allennlp.common import checks
from allennlp.modules import feedforward
from allennlp.nn import Activation

from combo.models import utils


class Predictor(nn.Module, common.Registrable):

    default_implementation = 'feedforward_predictor_from_vocab'

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                mask: Optional[torch.BoolTensor] = None,
                labels: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                sample_weights: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


class Linear(nn.Linear, common.FromParams):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: Optional[allen_nn.Activation] = lambda x: x,
                 dropout_rate: Optional[float] = 0.0):
        super().__init__(in_features, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else lambda x: x

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = super().forward(x)
        x = self.activation(x)
        return self.dropout(x)

    def get_output_dim(self) -> int:
        return self.out_features


@Predictor.register('feedforward_predictor')
@Predictor.register('feedforward_predictor_from_vocab', constructor='from_vocab')
class FeedForwardPredictor(Predictor):
    """Feedforward predictor. Should be used on top of Seq2Seq encoder."""

    def __init__(self, feedforward_network: feedforward.FeedForward):
        super().__init__()
        self.feedforward_network = feedforward_network

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                mask: Optional[torch.BoolTensor] = None,
                labels: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                sample_weights: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            mask = x.new_ones(x.size()[:-1])

        x = self.feedforward_network(x)
        output = {
            'prediction': x.argmax(-1),
            'probability': x
        }

        if labels is not None:
            if sample_weights is None:
                sample_weights = labels.new_ones([mask.size(0)])
            output['loss'] = self._loss(x, labels, mask, sample_weights)

        return output

    def _loss(self,
              pred: torch.Tensor,
              true: torch.Tensor,
              mask: torch.BoolTensor,
              sample_weights: torch.Tensor) -> torch.Tensor:
        BATCH_SIZE, _, CLASSES = pred.size()
        valid_positions = mask.sum()
        pred = pred.reshape(-1, CLASSES)
        true = true.reshape(-1)
        mask = mask.reshape(-1)
        loss = utils.masked_cross_entropy(pred, true, mask) * mask
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

        return cls(feedforward.FeedForward(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activations=activations,
            dropout=dropout)
        )
