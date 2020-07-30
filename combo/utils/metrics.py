"""Metrics implementation."""
from typing import Optional, List, Dict

import torch
from allennlp.training import metrics
from overrides import overrides


class SequenceBoolAccuracy(metrics.Metric):
    """BoolAccuracy implementation to handle sequences."""

    def __init__(self, prod_last_dim: bool = False):
        self._correct_count = 0.0
        self._total_count = 0.0
        self.prod_last_dim = prod_last_dim
        self.correct_indices = torch.ones([])

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.BoolTensor] = None):
        if gold_labels is None:
            return
        predictions, gold_labels, mask = self.detach_tensors(predictions,
                                                             gold_labels,
                                                             mask)

        # Some sanity checks.
        if gold_labels.size() != predictions.size():
            raise ValueError(
                f"gold_labels must have shape == predictions.size() but "
                f"found tensor of shape: {gold_labels.size()}"
            )
        if mask is not None and mask.size() not in [predictions.size()[:-1], predictions.size()]:
            raise ValueError(
                f"mask must have shape in one of [predictions.size()[:-1], predictions.size()] but "
                f"found tensor of shape: {mask.size()}"
            )
        if mask is None:
            mask = predictions.new_ones(predictions.size()[:-1]).bool()
        if mask.dim() < predictions.dim():
            mask = mask.unsqueeze(-1)

        correct = predictions.eq(gold_labels) * mask

        if self.prod_last_dim:
            correct = correct.prod(-1).unsqueeze(-1)

        correct = correct.float()

        self.correct_indices = correct.flatten().bool()
        self._correct_count += correct.sum()
        self._total_count += mask.sum()

    @overrides
    def get_metric(self, reset: bool) -> float:
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self) -> None:
        self._correct_count = 0.0
        self._total_count = 0.0
        self.correct_indices = torch.ones([])


class AttachmentScores(metrics.Metric):
    """
    Computes labeled and unlabeled attachment scores for a
    dependency parse, as well as sentence level exact match
    for both labeled and unlabeled trees. Note that the input
    to this metric is the sampled predictions, not the distribution
    itself.

    # Parameters

    ignore_classes : `List[int]`, optional (default = None)
        A list of label ids to ignore when computing metrics.
    """

    def __init__(self, ignore_classes: List[int] = None) -> None:
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
        self.correct_indices = torch.ones([])

        self._ignore_classes: List[int] = ignore_classes or []

    def __call__(  # type: ignore
            self,
            predicted_indices: torch.Tensor,
            predicted_labels: torch.Tensor,
            gold_indices: torch.Tensor,
            gold_labels: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predicted_indices : `torch.Tensor`, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : `torch.Tensor`, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_indices`.
        gold_labels : `torch.Tensor`, required.
            A tensor of the same shape as `predicted_labels`.
        mask : `torch.BoolTensor`, optional (default = None).
            A tensor of the same shape as `predicted_indices`.
        """
        detached = self.detach_tensors(
            predicted_indices, predicted_labels, gold_indices, gold_labels, mask
        )
        predicted_indices, predicted_labels, gold_indices, gold_labels, mask = detached

        if mask is None:
            mask = torch.ones_like(predicted_indices).bool()

        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        # Multiply by a mask denoting locations of
        # gold labels which we should ignore.
        for label in self._ignore_classes:
            label_mask = gold_labels.eq(label)
            mask = mask & ~label_mask

        correct_indices = predicted_indices.eq(gold_indices).long() * mask
        unlabeled_exact_match = (correct_indices + ~mask).prod(dim=-1)
        correct_labels = predicted_labels.eq(gold_labels).long() * mask
        correct_labels_and_indices = correct_indices * correct_labels
        self.correct_indices = correct_labels_and_indices.flatten()
        labeled_exact_match = (correct_labels_and_indices + ~mask).prod(dim=-1)

        self._unlabeled_correct += correct_indices.sum()
        self._exact_unlabeled_correct += unlabeled_exact_match.sum()
        self._labeled_correct += correct_labels_and_indices.sum()
        self._exact_labeled_correct += labeled_exact_match.sum()
        self._total_sentences += correct_indices.size(0)
        self._total_words += correct_indices.numel() - (~mask).sum()

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated metrics as a dictionary.
        """
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_exact_match = 0.0
        labeled_exact_match = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = float(self._unlabeled_correct) / float(self._total_words)
            labeled_attachment_score = float(self._labeled_correct) / float(self._total_words)
        if self._total_sentences > 0:
            unlabeled_exact_match = float(self._exact_unlabeled_correct) / float(
                self._total_sentences
            )
            labeled_exact_match = float(self._exact_labeled_correct) / float(self._total_sentences)
        if reset:
            self.reset()
        return {
            "UAS": unlabeled_attachment_score,
            "LAS": labeled_attachment_score,
            "UEM": unlabeled_exact_match,
            "LEM": labeled_exact_match,
        }

    @overrides
    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._exact_labeled_correct = 0.0
        self._exact_unlabeled_correct = 0.0
        self._total_words = 0.0
        self._total_sentences = 0.0
        self.correct_indices = torch.ones([])


class SemanticMetrics(metrics.Metric):
    """Groups metrics for all predictions."""

    def __init__(self) -> None:
        self.upos_score = SequenceBoolAccuracy()
        self.xpos_score = SequenceBoolAccuracy()
        self.semrel_score = SequenceBoolAccuracy()
        self.feats_score = SequenceBoolAccuracy(prod_last_dim=True)
        self.lemma_score = SequenceBoolAccuracy(prod_last_dim=True)
        self.attachment_scores = AttachmentScores()
        self.em_score = 0.0

    def __call__(  # type: ignore
            self,
            predictions: Dict[str, torch.Tensor],
            gold_labels: Dict[str, torch.Tensor],
            mask: torch.BoolTensor):
        self.upos_score(predictions["upostag"], gold_labels["upostag"], mask)
        self.xpos_score(predictions["xpostag"], gold_labels["xpostag"], mask)
        self.semrel_score(predictions["semrel"], gold_labels["semrel"], mask)
        self.feats_score(predictions["feats"], gold_labels["feats"], mask)
        self.lemma_score(predictions["lemma"], gold_labels["lemma"], mask)
        self.attachment_scores(predictions["head"],
                               predictions["deprel"],
                               gold_labels["head"],
                               gold_labels["deprel"],
                               mask)
        total = mask.sum()
        correct_indices = (self.upos_score.correct_indices *
                           self.xpos_score.correct_indices *
                           self.semrel_score.correct_indices *
                           self.feats_score.correct_indices *
                           self.lemma_score.correct_indices *
                           self.attachment_scores.correct_indices
                           )

        total, correct_indices = self.detach_tensors(total, correct_indices)
        self.em_score = (correct_indices.float().sum() / total).item()

    def get_metric(self, reset: bool) -> Dict[str, float]:
        metrics_dict = {
            "UPOS_ACC": self.upos_score.get_metric(reset),
            "XPOS_ACC": self.xpos_score.get_metric(reset),
            "SEMREL_ACC": self.semrel_score.get_metric(reset),
            "LEMMA_ACC": self.lemma_score.get_metric(reset),
            "FEATS_ACC": self.feats_score.get_metric(reset),
            "EM": self.em_score
        }
        metrics_dict.update(self.attachment_scores.get_metric(reset))
        return metrics_dict

    def reset(self) -> None:
        self.upos_score.reset()
        self.xpos_score.reset()
        self.semrel_score.reset()
        self.lemma_score.reset()
        self.feats_score.reset()
        self.attachment_scores.reset()
        self.em_score = 0.0
