"""Main entry point."""
import logging
import os
import tempfile
from typing import Dict

import torch
from absl import app
from absl import flags
from allennlp import common, models, predictors
from allennlp.commands import train, predict as allen_predict
from allennlp.common import checks as allen_checks, util
from allennlp.models import archival

from combo import predict
from combo.data import dataset
from combo.utils import checks

logger = logging.getLogger(__name__)
_FEATURES = ["token", "char", "upostag", "xpostag", "lemma", "feats"]
_TARGETS = ["deprel", "feats", "head", "lemma", "upostag", "xpostag", "semrel", "sent"]

FLAGS = flags.FLAGS
flags.DEFINE_enum(name="mode", default=None, enum_values=["train", "predict"],
                  help="Specify COMBO mode: train or predict")

# Common flags
flags.DEFINE_integer(name="cuda_device", default=-1,
                     help="Cuda device id (default -1 cpu)")
flags.DEFINE_string(name="output_file", default="output.log",
                    help="Predictions result file.")

# Training flags
flags.DEFINE_list(name="training_data_path", default="./tests/fixtures/example.conllu",
                  help="Training data path(s)")
flags.DEFINE_list(name="validation_data_path", default="",
                  help="Validation data path(s)")
flags.DEFINE_string(name="pretrained_tokens", default="",
                    help="Pretrained tokens embeddings path")
flags.DEFINE_integer(name="embedding_dim", default=300,
                     help="Embeddings dim")
flags.DEFINE_integer(name="num_epochs", default=400,
                     help="Epochs num")
flags.DEFINE_integer(name="word_batch_size", default=2500,
                     help="Minimum words in batch")
flags.DEFINE_string(name="pretrained_transformer_name", default="",
                    help="Pretrained transformer model name (see transformers from HuggingFace library for list of "
                         "available models) for transformers based embeddings.")
flags.DEFINE_list(name="features", default=["token", "char"],
                  help=f"Features used to train model (required 'token' and 'char'). Possible values: {_FEATURES}.")
flags.DEFINE_list(name="targets", default=["deprel", "feats", "head", "lemma", "upostag", "xpostag"],
                  help=f"Targets of the model (required `deprel` and `head`). Possible values: {_TARGETS}.")
flags.DEFINE_string(name="serialization_dir", default=None,
                    help="Model serialization directory (default - system temp dir).")
flags.DEFINE_boolean(name="tensorboard", default=False,
                     help="When provided model will log tensorboard metrics.")

# Finetune after training flags
flags.DEFINE_list(name="finetuning_training_data_path", default="",
                  help="Training data path(s)")
flags.DEFINE_list(name="finetuning_validation_data_path", default="",
                  help="Validation data path(s)")
flags.DEFINE_string(name="config_path", default="config.template.jsonnet",
                    help="Config file path.")

# Test after training flags
flags.DEFINE_string(name="test_path", default=None,
                    help="Test path file.")

# Experimental
flags.DEFINE_boolean(name="use_pure_config", default=False,
                     help="Ignore ext flags (experimental).")

# Prediction flags
flags.DEFINE_string(name="model_path", default=None,
                    help="Pretrained model path.")
flags.DEFINE_string(name="input_file", default=None,
                    help="File to predict path")
flags.DEFINE_integer(name="batch_size", default=1,
                     help="Prediction batch size.")
flags.DEFINE_boolean(name="silent", default=True,
                     help="Silent prediction to file (without printing to console).")
flags.DEFINE_enum(name="predictor_name", default="semantic-multitask-predictor-spacy",
                  enum_values=["semantic-multitask-predictor", "semantic-multitask-predictor-spacy"],
                  help="Use predictor with whitespace or spacy tokenizer.")


def run(_):
    """Run model."""
    # Imports are required to make Registrable modules visible without passing parameter
    util.import_module_and_submodules("combo.commands")
    util.import_module_and_submodules("combo.models")
    util.import_module_and_submodules("combo.training")

    if FLAGS.mode == "train":
        checks.file_exists(FLAGS.config_path)
        params = common.Params.from_file(FLAGS.config_path, ext_vars=_get_ext_vars())
        model_params = params.get("model").as_ordered_dict()
        serialization_dir = tempfile.mkdtemp(prefix="allennlp", dir=FLAGS.serialization_dir)
        model = train.train_model(params, serialization_dir=serialization_dir, file_friendly_logging=True)
        logger.info(f"Training model stored in: {serialization_dir}")

        if FLAGS.finetuning_training_data_path:
            for f in FLAGS.finetuning_training_data_path:
                checks.file_exists(f)

            # Loading will be performed from stored model.tar.gz
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            params = common.Params.from_file(FLAGS.config_path, ext_vars=_get_ext_vars(finetuning=True))
            # Replace model definition with pretrained archive
            params["model"] = {
                "type": "from_archive",
                "archive_file": serialization_dir + "/model.tar.gz",
            }
            serialization_dir = tempfile.mkdtemp(prefix="allennlp", suffix="-finetuning", dir=FLAGS.serialization_dir)
            model = train.train_model(params.duplicate(), serialization_dir=serialization_dir,
                                      file_friendly_logging=True)

            # Make finetuning model serialization independent from training serialization
            # Storing model definition instead of archive
            params["model"] = model_params
            params.to_file(os.path.join(serialization_dir, archival.CONFIG_NAME))
            archival.archive_model(serialization_dir)

            logger.info(f"Finetuned model stored in: {serialization_dir}")

        if FLAGS.test_path and FLAGS.output_file:
            checks.file_exists(FLAGS.test_path)
            params = common.Params.from_file(FLAGS.config_path, ext_vars=_get_ext_vars())["dataset_reader"]
            params.pop("type")
            dataset_reader = dataset.UniversalDependenciesDatasetReader.from_params(params)
            predictor = predict.SemanticMultitaskPredictor(
                model=model,
                dataset_reader=dataset_reader
            )
            test_path = FLAGS.test_path
            test_trees = dataset_reader.read(test_path)
            with open(FLAGS.output_file, "w") as file:
                for tree in test_trees:
                    file.writelines(predictor.predict_instance_as_tree(tree).serialize())
    else:
        use_dataset_reader = ".conllu" in FLAGS.input_file.lower()
        manager = allen_predict._PredictManager(
            _get_predictor(),
            FLAGS.input_file,
            FLAGS.output_file,
            FLAGS.batch_size,
            not FLAGS.silent,
            use_dataset_reader,
        )
        manager.run()


def _get_predictor() -> predictors.Predictor:
    allen_checks.check_for_gpu(FLAGS.cuda_device)
    checks.file_exists(FLAGS.model_path)
    archive = models.load_archive(
        FLAGS.model_path,
        cuda_device=FLAGS.cuda_device,
    )

    return predictors.Predictor.from_archive(
        archive, FLAGS.predictor_name
    )


def _get_ext_vars(finetuning: bool = False) -> Dict:
    if FLAGS.use_pure_config:
        return {}
    return {
        "training_data_path": (
            ",".join(FLAGS.training_data_path if not finetuning else FLAGS.finetuning_training_data_path)),
        "validation_data_path": (
            ",".join(FLAGS.validation_data_path if not finetuning else FLAGS.finetuning_validation_data_path)),
        "pretrained_tokens": FLAGS.pretrained_tokens,
        "pretrained_transformer_name": FLAGS.pretrained_transformer_name,
        "features": " ".join(FLAGS.features),
        "targets": " ".join(FLAGS.targets),
        "type": "finetuning" if finetuning else "default",
        "embedding_dim": str(FLAGS.embedding_dim),
        "cuda_device": str(FLAGS.cuda_device),
        "num_epochs": str(FLAGS.num_epochs),
        "word_batch_size": str(FLAGS.word_batch_size),
        "use_tensorboard": str(FLAGS.tensorboard),
    }


def main():
    """Parse flags."""
    flags.register_validator(
        "features",
        lambda values: all(
            value in _FEATURES for value in values),
        message="Flags --features contains unknown value(s)."
    )
    flags.register_validator(
        "mode",
        lambda value: value is not None,
        message="Flag --mode must be set with either `predict` or `train` value")
    flags.register_validator(
        "targets",
        lambda values: all(
            value in _TARGETS for value in values),
        message="Flag --targets contains unknown value(s)."
    )
    app.run(run)


if __name__ == "__main__":
    main()
