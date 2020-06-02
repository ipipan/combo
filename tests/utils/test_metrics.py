"""Metrics tests."""
import unittest

import torch

from combo.utils import metrics


class SemanticMetricsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.mask: torch.BoolTensor = torch.tensor([
            [True, True, True, True],
            [True, True, True, False],
            [True, True, True, False],
        ])
        pred = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ])
        pred_seq = pred.reshape(3, 4, 1)
        gold = pred.clone()
        gold_seq = pred_seq.clone()
        self.upostag, self.upostag_l = (("upostag", x) for x in [pred, gold])
        self.xpostag, self.xpostag_l = (("xpostag", x) for x in [pred, gold])
        self.semrel, self.semrel_l = (("semrel", x) for x in [pred, gold])
        self.head, self.head_l = (("head", x) for x in [pred, gold])
        self.deprel, self.deprel_l = (("deprel", x) for x in [pred, gold])
        self.feats, self.feats_l = (("feats", x) for x in [pred_seq, gold_seq])
        self.lemma, self.lemma_l = (("lemma", x) for x in [pred_seq, gold_seq])
        self.predictions = dict(
            [self.upostag, self.xpostag, self.semrel, self.feats, self.lemma, self.head, self.deprel])
        self.gold_labels = dict([self.upostag_l, self.xpostag_l, self.semrel_l, self.feats_l, self.lemma_l, self.head_l,
                                 self.deprel_l])
        self.eps = 1e-6

    def test_every_prediction_correct(self):
        # given
        metric = metrics.SemanticMetrics()

        # when
        metric(self.predictions, self.gold_labels, self.mask)

        # then
        self.assertEqual(1.0, metric.em_score)

    def test_missing_predictions_for_one_target(self):
        # given
        metric = metrics.SemanticMetrics()
        self.predictions["upostag"] = None
        self.gold_labels["upostag"] = None

        # when
        metric(self.predictions, self.gold_labels, self.mask)

        # then
        self.assertEqual(1.0, metric.em_score)

    def test_missing_predictions_for_two_targets(self):
        # given
        metric = metrics.SemanticMetrics()
        self.predictions["upostag"] = None
        self.gold_labels["upostag"] = None
        self.predictions["lemma"] = None
        self.gold_labels["lemma"] = None

        # when
        metric(self.predictions, self.gold_labels, self.mask)

        # then
        self.assertEqual(1.0, metric.em_score)

    def test_one_classification_in_one_target_is_wrong(self):
        # given
        metric = metrics.SemanticMetrics()
        self.predictions["upostag"][0][0] = 100

        # when
        metric(self.predictions, self.gold_labels, self.mask)

        # then
        self.assertAlmostEqual(0.9, metric.em_score, delta=self.eps)

    def test_classification_errors_and_target_without_predictions(self):
        # given
        metric = metrics.SemanticMetrics()
        self.predictions["feats"] = None
        self.gold_labels["feats"] = None
        self.predictions["upostag"][0][0] = 100
        self.predictions["upostag"][2][0] = 100
        # should be ignored due to masking
        self.predictions["upostag"][1][3] = 100

        # when
        metric(self.predictions, self.gold_labels, self.mask)

        # then
        self.assertAlmostEqual(0.8, metric.em_score, delta=self.eps)


class SequenceBoolAccuracyTest(unittest.TestCase):

    def setUp(self) -> None:
        self.mask: torch.BoolTensor = torch.tensor([
            [True, True, True, True],
            [True, True, True, False],
            [True, True, True, False],
        ])

    def test_regular_classification_accuracy(self):
        # given
        metric = metrics.SequenceBoolAccuracy()
        predictions = torch.tensor([
            [1, 1, 0, 8],
            [1, 2, 3, 4],
            [9, 4, 3, 9],
        ])
        gold_labels = torch.tensor([
            [11, 1, 0, 8],
            [14, 2, 3, 14],
            [9, 4, 13, 9],
        ])

        # when
        metric(predictions, gold_labels, self.mask)

        # then
        self.assertEqual(metric._correct_count.item(), 7)
        self.assertEqual(metric._total_count.item(), 10)

    def test_feats_classification_accuracy(self):
        # given
        metric = metrics.SequenceBoolAccuracy(prod_last_dim=True)
        # batch_size, sequence_length, classes
        predictions = torch.tensor([
            [[1, 4], [0, 2], [0, 2], [0, 3]],
            [[1, 4], [0, 2], [0, 2], [0, 3]],
            [[1, 4], [0, 2], [0, 2], [0, 3]],
        ])
        gold_labels = torch.tensor([
            [[1, 14], [0, 2], [0, 2], [0, 3]],
            [[11, 4], [0, 2], [0, 2], [10, 3]],
            [[1, 4], [0, 2], [10, 12], [0, 3]],
        ])

        # when
        metric(predictions, gold_labels, self.mask)

        # then
        self.assertEqual(metric._correct_count.item(), 7)
        self.assertEqual(metric._total_count.item(), 10)
