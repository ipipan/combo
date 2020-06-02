"""Main COMBO model."""
from typing import Optional, Dict, Any, List

import torch
from allennlp import data, modules, models as allen_models, nn as allen_nn
from allennlp.modules import text_field_embedders
from allennlp.nn import util
from overrides import overrides

from combo.models import base
from combo.utils import metrics


@allen_models.Model.register("semantic_multitask")
class SemanticMultitaskModel(allen_models.Model):
    """Main COMBO model."""

    def __init__(self,
                 vocab: data.Vocabulary,
                 loss_weights: Dict[str, float],
                 text_field_embedder: text_field_embedders.TextFieldEmbedder,
                 seq_encoder: modules.Seq2SeqEncoder,
                 use_sample_weight: bool = True,
                 lemmatizer: Optional[base.Predictor] = None,
                 upos_tagger: Optional[base.Predictor] = None,
                 xpos_tagger: Optional[base.Predictor] = None,
                 semantic_relation: Optional[base.Predictor] = None,
                 morphological_feat: Optional[base.Predictor] = None,
                 dependency_relation: Optional[base.Predictor] = None,
                 regularizer: allen_nn.RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self.text_field_embedder = text_field_embedder
        self.loss_weights = loss_weights
        self.use_sample_weight = use_sample_weight
        self.seq_encoder = seq_encoder
        self.lemmatizer = lemmatizer
        self.upos_tagger = upos_tagger
        self.xpos_tagger = xpos_tagger
        self.semantic_relation = semantic_relation
        self.morphological_feat = morphological_feat
        self.dependency_relation = dependency_relation
        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, self.seq_encoder.get_output_dim()]))
        self.scores = metrics.SemanticMetrics()
        self._partial_losses = None

    @overrides
    def forward(self,
                sentence: Dict[str, Dict[str, torch.Tensor]],
                metadata: List[Dict[str, Any]],
                upostag: torch.Tensor = None,
                xpostag: torch.Tensor = None,
                lemma: Dict[str, Dict[str, torch.Tensor]] = None,
                feats: torch.Tensor = None,
                head: torch.Tensor = None,
                deprel: torch.Tensor = None,
                semrel: torch.Tensor = None, ) -> Dict[str, torch.Tensor]:

        # Prepare masks
        char_mask: torch.BoolTensor = sentence["char"]["token_characters"] > 0
        word_mask = util.get_text_field_mask(sentence)

        # If enabled weight samples loss by log(sentence_length)
        sample_weights = word_mask.sum(-1).float().log() if self.use_sample_weight else None

        encoder_input = self.text_field_embedder(sentence, char_mask=char_mask)
        encoder_emb = self.seq_encoder(encoder_input, word_mask)

        batch_size, _, encoding_dim = encoder_emb.size()

        # Concatenate the head sentinel (ROOT) onto the sentence representation.
        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        encoder_emb = torch.cat([head_sentinel, encoder_emb], 1)
        word_mask = torch.cat([word_mask.new_ones((batch_size, 1)), word_mask], 1)

        upos_output = self._optional(self.upos_tagger,
                                     encoder_emb[:, 1:],
                                     mask=word_mask[:, 1:],
                                     labels=upostag,
                                     sample_weights=sample_weights)
        xpos_output = self._optional(self.xpos_tagger,
                                     encoder_emb[:, 1:],
                                     mask=word_mask[:, 1:],
                                     labels=xpostag,
                                     sample_weights=sample_weights)
        semrel_output = self._optional(self.semantic_relation,
                                       encoder_emb[:, 1:],
                                       mask=word_mask[:, 1:],
                                       labels=semrel,
                                       sample_weights=sample_weights)
        morpho_output = self._optional(self.morphological_feat,
                                       encoder_emb[:, 1:],
                                       mask=word_mask[:, 1:],
                                       labels=feats,
                                       sample_weights=sample_weights)
        lemma_output = self._optional(self.lemmatizer,
                                      (encoder_emb[:, 1:], sentence.get("char").get("token_characters")
                                      if sentence.get("char") else None),
                                      mask=word_mask[:, 1:],
                                      labels=lemma.get("char").get("token_characters") if lemma else None,
                                      sample_weights=sample_weights)
        parser_output = self._optional(self.dependency_relation,
                                       encoder_emb,
                                       returns_tuple=True,
                                       mask=word_mask,
                                       labels=(deprel, head),
                                       sample_weights=sample_weights)
        relations_pred, head_pred = parser_output["prediction"]
        output = {
            "upostag": upos_output["prediction"],
            "xpostag": xpos_output["prediction"],
            "semrel": semrel_output["prediction"],
            "feats": morpho_output["prediction"],
            "lemma": lemma_output["prediction"],
            "head": head_pred,
            "deprel": relations_pred,
            "sentence_embedding": torch.max(encoder_emb[:, 1:], dim=1)[0],
        }

        if self._has_labels([upostag, xpostag, lemma, feats, head, deprel, semrel]):

            # Feats mapping
            if self.morphological_feat:
                mapped_gold_labels = []
                for _, cat_indices in self.morphological_feat.slices.items():
                    mapped_gold_labels.append(feats[:, :, cat_indices].argmax(dim=-1))

                feats = torch.stack(mapped_gold_labels, dim=-1)

            labels = {
                "upostag": upostag,
                "xpostag": xpostag,
                "semrel": semrel,
                "feats": feats,
                "lemma": lemma.get("char").get("token_characters") if lemma else None,
                "head": head,
                "deprel": deprel,
            }
            self.scores(output, labels, word_mask[:, 1:])
            relations_loss, head_loss = parser_output["loss"]
            losses = {
                "upostag_loss": upos_output["loss"],
                "xpostag_loss": xpos_output["loss"],
                "semrel_loss": semrel_output["loss"],
                "feats_loss": morpho_output["loss"],
                "lemma_loss": lemma_output["loss"],
                "head_loss": head_loss,
                "deprel_loss": relations_loss,
                # Cycle loss is only for the metrics purposes.
                "cycle_loss": parser_output.get("cycle_loss")
            }
            self._partial_losses = losses.copy()
            losses["loss"] = self._calculate_loss(losses)
            output.update(losses)

        return self._clean(output)

    @staticmethod
    def _has_labels(labels):
        return any(x is not None for x in labels)

    def _calculate_loss(self, output):
        losses = []
        for name, value in self.loss_weights.items():
            if output.get(f"{name}_loss"):
                losses.append(output[f"{name}_loss"] * value)
        return torch.stack(losses).sum()

    @staticmethod
    def _optional(callable_model: Optional[torch.nn.Module],
                  *args,
                  returns_tuple: bool = False,
                  **kwargs):
        if callable_model:
            return callable_model(*args, **kwargs)
        if returns_tuple:
            return {"prediction": (None, None), "loss": (None, None)}
        return {"prediction": None, "loss": None}

    @staticmethod
    def _clean(output):
        for k, v in dict(output).items():
            if v is None:
                del output[k]
        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.scores.get_metric(reset)
        if self._partial_losses:
            losses = self._clean(self._partial_losses)
            losses = {f"partial_loss/{k}": v.detach().item() for k, v in losses.items()}
            metrics.update(losses)
        return metrics
