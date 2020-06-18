"""Dependency parsing models."""
from typing import Tuple, Dict, Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from allennlp import data
from allennlp.nn import chu_liu_edmonds

from combo.models import base, utils


class HeadPredictionModel(base.Predictor):
    """Head prediction model."""

    def __init__(self,
                 head_projection_layer: base.Linear,
                 dependency_projection_layer: base.Linear,
                 cycle_loss_n: int = 0):
        super().__init__()
        self.head_projection_layer = head_projection_layer
        self.dependency_projection_layer = dependency_projection_layer
        self.cycle_loss_n = cycle_loss_n

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                mask: Optional[torch.BoolTensor] = None,
                labels: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                sample_weights: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        if mask is None:
            mask = x.new_ones(x.size()[-1])

        head_arc_emb = self.head_projection_layer(x)
        dep_arc_emb = self.dependency_projection_layer(x)
        x = dep_arc_emb.bmm(head_arc_emb.transpose(2, 1))

        if self.training:
            pred = x.argmax(-1)
        else:
            pred = []
            # Adding non existing in mask ROOT to lengths
            lengths = mask.data.sum(dim=1).long().cpu().numpy() + 1
            for idx, length in enumerate(lengths):
                probs = x[idx, :].softmax(dim=-1).cpu().numpy()

                # We do not want any word to be parent of the root node (ROOT, 0).
                # Also setting it to -1 instead of 0 fixes edge case where softmax made all
                # but ROOT prediction to EXACTLY 0.0 and it might cause in many ROOT -> word edges)
                probs[:, 0] = -1
                heads, _ = chu_liu_edmonds.decode_mst(probs.T, length=length, has_labels=False)
                heads[0] = 0
                pred.append(heads)
            pred = torch.from_numpy(np.stack(pred)).to(x.device)

        output = {
            "prediction": pred[:, 1:],
            "probability": x
        }

        if labels is not None:
            if sample_weights is None:
                sample_weights = labels.new_ones([mask.size(0)])
            output["loss"], output["cycle_loss"] = self._loss(x, labels, mask, sample_weights)

        return output

    def _cycle_loss(self, pred: torch.Tensor):
        BATCH_SIZE, _, _ = pred.size()
        loss = pred.new_zeros(BATCH_SIZE)
        # Index from 1: as using non __ROOT__ tokens
        pred = pred.softmax(-1)[:, 1:, 1:]
        x = pred
        for i in range(self.cycle_loss_n):
            loss += self._batch_trace(x)

            # Don't multiple on last iteration
            if i < self.cycle_loss_n - 1:
                x = x.bmm(pred)

        return loss

    @staticmethod
    def _batch_trace(x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 3
        BATCH_SIZE, N, M = x.size()
        assert N == M
        identity = x.new_tensor(torch.eye(N))
        identity = identity.reshape((1, N, N))
        batch_identity = identity.repeat(BATCH_SIZE, 1, 1)
        return (x * batch_identity).sum((-1, -2))

    def _loss(self, pred: torch.Tensor, true: torch.Tensor, mask: torch.BoolTensor,
              sample_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        BATCH_SIZE, N, M = pred.size()
        assert N == M
        SENTENCE_LENGTH = N

        valid_positions = mask.sum()

        result = []
        # Ignore first pred dimension as it is ROOT token prediction
        for i in range(SENTENCE_LENGTH - 1):
            pred_i = pred[:, i + 1, :].reshape(BATCH_SIZE, SENTENCE_LENGTH)
            true_i = true[:, i].reshape(-1)
            mask_i = mask[:, i]
            cross_entropy_loss = utils.masked_cross_entropy(pred_i, true_i, mask_i)
            result.append(cross_entropy_loss)
        cycle_loss = self._cycle_loss(pred)
        loss = torch.stack(result).transpose(1, 0) * sample_weights.unsqueeze(-1)
        return loss.sum() / valid_positions + cycle_loss.mean(), cycle_loss.mean()


@base.Predictor.register("combo_dependency_parsing_from_vocab", constructor="from_vocab")
class DependencyRelationModel(base.Predictor):
    """Dependency relation parsing model."""

    def __init__(self,
                 head_predictor: HeadPredictionModel,
                 head_projection_layer: base.Linear,
                 dependency_projection_layer: base.Linear,
                 relation_prediction_layer: base.Linear):
        super().__init__()
        self.head_predictor = head_predictor
        self.head_projection_layer = head_projection_layer
        self.dependency_projection_layer = dependency_projection_layer
        self.relation_prediction_layer = relation_prediction_layer

    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                mask: Optional[torch.BoolTensor] = None,
                labels: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                sample_weights: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> Dict[str, torch.Tensor]:
        if mask is not None:
            mask = mask[:, 1:]
        relations_labels, head_labels = None, None
        if labels is not None and labels[0] is not None:
            relations_labels, head_labels = labels
            if mask is None:
                mask = head_labels.new_ones(head_labels.size())

        head_output = self.head_predictor(x, mask, head_labels, sample_weights)
        head_pred = head_output["probability"]
        head_pred_soft = F.softmax(head_pred, dim=-1)

        head_rel_emb = self.head_projection_layer(x)

        dep_rel_emb = self.dependency_projection_layer(x)

        dep_rel_pred = head_pred_soft.bmm(head_rel_emb)
        dep_rel_pred = torch.cat((dep_rel_pred, dep_rel_emb), dim=-1)
        relation_prediction = self.relation_prediction_layer(dep_rel_pred)
        output = head_output

        output["prediction"] = (relation_prediction.argmax(-1)[:, 1:], head_output["prediction"])

        if labels is not None and labels[0] is not None:
            if sample_weights is None:
                sample_weights = labels.new_ones([mask.size(0)])
            loss = self._loss(relation_prediction[:, 1:], relations_labels, mask, sample_weights)
            output["loss"] = (loss, head_output["loss"])

        return output

    @staticmethod
    def _loss(pred: torch.Tensor,
              true: torch.Tensor,
              mask: torch.BoolTensor,
              sample_weights: torch.Tensor) -> torch.Tensor:

        valid_positions = mask.sum()

        BATCH_SIZE, _, DEPENDENCY_RELATIONS = pred.size()
        pred = pred.reshape(-1, DEPENDENCY_RELATIONS)
        true = true.reshape(-1)
        mask = mask.reshape(-1)
        loss = utils.masked_cross_entropy(pred, true, mask)
        loss = loss.reshape(BATCH_SIZE, -1) * sample_weights.unsqueeze(-1)
        return loss.sum() / valid_positions

    @classmethod
    def from_vocab(cls,
                   vocab: data.Vocabulary,
                   vocab_namespace: str,
                   head_predictor: HeadPredictionModel,
                   head_projection_layer: base.Linear,
                   dependency_projection_layer: base.Linear
                   ):
        """Creates parser combining model configuration and vocabulary data."""
        assert vocab_namespace in vocab.get_namespaces()
        relation_prediction_layer = base.Linear(
            in_features=head_projection_layer.get_output_dim() + dependency_projection_layer.get_output_dim(),
            out_features=vocab.get_vocab_size(vocab_namespace)
        )
        return cls(
            head_predictor=head_predictor,
            head_projection_layer=head_projection_layer,
            dependency_projection_layer=dependency_projection_layer,
            relation_prediction_layer=relation_prediction_layer
        )
