import collections
import logging
import os
from typing import List, Union

import conllu
from allennlp import data as allen_data, common, models
from allennlp.common import util
from allennlp.data import tokenizers
from allennlp.predictors import predictor
from overrides import overrides

from combo import data
from combo.utils import download

logger = logging.getLogger(__name__)


@predictor.Predictor.register("semantic-multitask-predictor")
@predictor.Predictor.register("semantic-multitask-predictor-spacy", constructor="with_spacy_tokenizer")
class SemanticMultitaskPredictor(predictor.Predictor):

    def __init__(self,
                 model: models.Model,
                 dataset_reader: allen_data.DatasetReader,
                 tokenizer: allen_data.Tokenizer = tokenizers.WhitespaceTokenizer(),
                 batch_size: int = 500) -> None:
        super().__init__(model, dataset_reader)
        self.batch_size = batch_size
        self.vocab = model.vocab
        self._dataset_reader.generate_labels = False
        self._tokenizer = tokenizer

    def __call__(self, sentence: Union[str, List[str], List[List[str]], List[data.Sentence]]):
        """Depending on the input uses (or ignores) tokenizer.
        When model isn't only text-based only List[data.Sentence] is possible input.

        * str - tokenizer is used
        * List[str] - tokenizer is used for each string (treated as list of raw sentences)
        * List[List[str]] - tokenizer isn't used (treated as list of tokenized sentences)
        * List[data.Sentence] - tokenizer isn't used (treated as list of tokenized sentences)

        :param sentence: sentence(s) representation
        :return: Sentence or List[Sentence] depending on the input
        """
        return self.predict(sentence)

    def predict(self, sentence: Union[str, List[str], List[List[str]], List[data.Sentence]]):
        if isinstance(sentence, str):
            return data.Sentence.from_json(self.predict_json({"sentence": sentence}))
        elif isinstance(sentence, list):
            if len(sentence) == 0:
                return []
            example = sentence[0]
            if isinstance(example, str) or isinstance(example, list):
                sentences = []
                for sentences_batch in util.lazy_groups_of(sentence, self.batch_size):
                    trees = self.predict_batch_json([self._to_input_json(s) for s in sentences_batch])
                    sentences.extend([data.Sentence.from_json(t) for t in trees])
                return sentences
            elif isinstance(example, data.Sentence):
                sentences = []
                for sentences_batch in util.lazy_groups_of(sentence, self.batch_size):
                    trees = self.predict_batch_instance([self._to_input_instance(s) for s in sentences_batch],
                                                        serialize=False)
                    sentences.extend([data.Sentence.from_json(t) for t in trees])
                return sentences
            else:
                raise ValueError("List must have either sentences as str, List[str] or Sentence object.")
        else:
            raise ValueError("Input must be either string or list of strings.")

    @overrides
    def predict_batch_instance(self, instances: List[allen_data.Instance], serialize: bool = True) -> List[
        common.JsonDict]:
        trees = []
        predictions = super().predict_batch_instance(instances)
        for prediction, instance in zip(predictions, instances):
            tree = self._predictions_as_tree(prediction, instance)
            if serialize:
                tree = tree.serialize()
            tree_json = util.sanitize(tree)
            trees.append(collections.OrderedDict([
                ("tree", tree_json),
            ]))
        return trees

    @overrides
    def predict_instance(self, instance: allen_data.Instance, serialize: bool = True) -> common.JsonDict:
        tree = self.predict_instance_as_tree(instance)
        if serialize:
            tree = tree.serialize()
        tree_json = util.sanitize(tree)
        result = collections.OrderedDict([
            ("tree", tree_json),
        ])
        return result

    @overrides
    def predict_batch_json(self, inputs: List[common.JsonDict]) -> List[common.JsonDict]:
        trees = []
        instances = self._batch_json_to_instances(inputs)
        predictions = self.predict_batch_instance(instances, serialize=False)
        for prediction, instance in zip(predictions, instances):
            tree = self._predictions_as_tree(prediction, instance)
            tree_json = util.sanitize(tree)
            trees.append(collections.OrderedDict([
                ("tree", tree_json),
            ]))
        return trees

    @overrides
    def predict_json(self, inputs: common.JsonDict) -> common.JsonDict:
        instance = self._json_to_instance(inputs)
        tree = self.predict_instance_as_tree(instance)
        tree_json = util.sanitize(tree)
        result = collections.OrderedDict([
            ("tree", tree_json),
        ])
        return result

    def predict_instance_as_tree(self, instance: allen_data.Instance) -> conllu.TokenList:
        predictions = super().predict_instance(instance)
        return self._predictions_as_tree(predictions, instance)

    @overrides
    def _json_to_instance(self, json_dict: common.JsonDict) -> allen_data.Instance:
        sentence = json_dict["sentence"]
        if isinstance(sentence, str):
            tokens = [t.text for t in self._tokenizer.tokenize(json_dict["sentence"])]
        elif isinstance(sentence, list):
            tokens = sentence
        else:
            raise ValueError("Input must be either string or list of strings.")
        tree = self._sentence_to_tree(tokens)
        return self._dataset_reader.text_to_instance(tree)

    @overrides
    def load_line(self, line: str) -> common.JsonDict:
        return self._to_input_json(line.replace("\n", "").strip())

    @overrides
    def dump_line(self, outputs: common.JsonDict) -> str:
        # Check whether serialized (str) tree or token's list
        # Serialized tree has already separators between lines
        if type(outputs["tree"]) == str:
            return str(outputs["tree"])
        else:
            return str(outputs["tree"]) + "\n"

    @staticmethod
    def _sentence_to_tree(sentence: List[str]):
        d = collections.OrderedDict
        return _TokenList(
            [d({"id": idx, "token": token}) for
             idx, token
             in enumerate(sentence)],
            metadata=collections.OrderedDict()
        )

    @staticmethod
    def _to_input_json(sentence: str):
        return {"sentence": sentence}

    def _to_input_instance(self, sentence: data.Sentence) -> allen_data.Instance:
        tree = _TokenList([t.__dict__ for t in sentence.tokens])
        return self._dataset_reader.text_to_instance(tree)

    def _predictions_as_tree(self, predictions, instance):
        tree = instance.fields["metadata"]["input"]
        field_names = instance.fields["metadata"]["field_names"]
        tree_tokens = [t for t in tree if isinstance(t["id"], int)]
        for idx, token in enumerate(tree_tokens):
            for field_name in field_names:
                if field_name in predictions:
                    if field_name in ["xpostag", "upostag", "semrel", "deprel"]:
                        value = self.vocab.get_token_from_index(predictions[field_name][idx], field_name + "_labels")
                        token[field_name] = value
                    elif field_name in ["head"]:
                        token[field_name] = int(predictions[field_name][idx])
                    elif field_name in ["feats"]:
                        slices = self._model.morphological_feat.slices
                        features = []
                        prediction = predictions[field_name][idx]
                        for (cat, cat_indices), pred_idx in zip(slices.items(), prediction):
                            if cat not in ["__PAD__", "_"]:
                                value = self.vocab.get_token_from_index(cat_indices[pred_idx],
                                                                        field_name + "_labels")
                                # Exclude auxiliary values
                                if "=None" not in value:
                                    features.append(value)
                        if len(features) == 0:
                            field_value = "_"
                        else:
                            field_value = "|".join(sorted(features))

                        token[field_name] = field_value
                    elif field_name == "head":
                        pass
                    elif field_name == "lemma":
                        prediction = predictions[field_name][idx]
                        word_chars = []
                        for char_idx in prediction[1:-1]:
                            pred_char = self.vocab.get_token_from_index(char_idx, "lemma_characters")

                            if pred_char == "__END__":
                                break
                            elif pred_char == "__PAD__":
                                continue
                            elif "_" in pred_char:
                                pred_char = "?"

                            word_chars.append(pred_char)
                        token[field_name] = "".join(word_chars)
                    else:
                        raise NotImplementedError(f"Unknown field name {field_name}!")

        if self._dataset_reader and "sent" in self._dataset_reader._targets:
            tree.metadata["sentence_embedding"] = str(predictions["sentence_embedding"])
        return tree

    @classmethod
    def with_spacy_tokenizer(cls, model: models.Model,
                             dataset_reader: allen_data.DatasetReader):
        return cls(model, dataset_reader, tokenizers.SpacyTokenizer())

    @classmethod
    def from_pretrained(cls, path: str, tokenizer=tokenizers.SpacyTokenizer(),
                        batch_size: int = 500,
                        cuda_device: int = -1):
        util.import_module_and_submodules("combo.commands")
        util.import_module_and_submodules("combo.models")
        util.import_module_and_submodules("combo.training")

        if os.path.exists(path):
            model_path = path
        else:
            try:
                logger.debug("Downloading model.")
                model_path = download.download_file(path)
            except Exception as e:
                logger.error(e)
                raise e

        archive = models.load_archive(model_path, cuda_device=cuda_device)
        model = archive.model
        dataset_reader = allen_data.DatasetReader.from_params(
            archive.config["dataset_reader"])
        return cls(model, dataset_reader, tokenizer, batch_size)


class _TokenList(conllu.TokenList):

    @overrides
    def __repr__(self):
        return 'TokenList<' + ', '.join(token['token'] for token in self) + '>'
