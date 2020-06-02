"""Finetuning train model wrapper."""
import os
from typing import List

from allennlp import data, models, common, training
from allennlp.commands import train
from allennlp.common import checks
from allennlp.common import util as common_util
from allennlp.training import util as training_util


@train.TrainModel.register("finetuning", constructor="from_partial_objects_finetuning")
class FinetuningTrainModel(train.TrainModel):
    """Class made only for finetuning, the only difference is saving vocab from concatenated
    (archive and current) datasets."""

    @classmethod
    def from_partial_objects_finetuning(
            cls,
            serialization_dir: str,
            local_rank: int,  # pylint: disable=unused-argument
            batch_weight_key: str,
            dataset_reader: data.DatasetReader,
            train_data_path: str,
            model: common.Lazy[models.Model],
            data_loader: common.Lazy[data.DataLoader],
            trainer: common.Lazy[training.Trainer],
            vocabulary: common.Lazy[data.Vocabulary] = None,
            datasets_for_vocab_creation: List[str] = None,
            validation_dataset_reader: data.DatasetReader = None,
            validation_data_path: str = None,
            validation_data_loader: common.Lazy[data.DataLoader] = None,
            test_data_path: str = None,
            evaluate_on_test: bool = False,
    ) -> "train.TrainModel":
        """
        This method is intended for use with our `FromParams` logic, to construct a `TrainModel`
        object from a config file passed to the `allennlp train` command.  The arguments to this
        method are the allowed top-level keys in a configuration file (except for the first three,
        which are obtained separately).

        You *could* use this outside of our `FromParams` logic if you really want to, but there
        might be easier ways to accomplish your goal than instantiating `Lazy` objects.  If you are
        writing your own training loop, we recommend that you look at the implementation of this
        method for inspiration and possibly some utility functions you can call, but you very likely
        should not use this method directly.

        The `Lazy` type annotations here are a mechanism for building dependencies to an object
        sequentially - the `TrainModel` object needs data, a model, and a trainer, but the model
        needs to see the data before it's constructed (to create a vocabulary) and the trainer needs
        the data and the model before it's constructed.  Objects that have sequential dependencies
        like this are labeled as `Lazy` in their type annotations, and we pass the missing
        dependencies when we call their `construct()` method, which you can see in the code below.

        # Parameters
        serialization_dir: `str`
            The directory where logs and model archives will be saved.
        local_rank: `int`
            The process index that is initialized using the GPU device id.
        batch_weight_key: `str`
            The name of metric used to weight the loss on a per-batch basis.
        dataset_reader: `DatasetReader`
            The `DatasetReader` that will be used for training and (by default) for validation.
        train_data_path: `str`
            The file (or directory) that will be passed to `dataset_reader.read()` to construct the
            training data.
        model: `Lazy[Model]`
            The model that we will train.  This is lazy because it depends on the `Vocabulary`;
            after constructing the vocabulary we call `model.construct(vocab=vocabulary)`.
        data_loader: `Lazy[DataLoader]`
            The data_loader we use to batch instances from the dataset reader at training and (by
            default) validation time. This is lazy because it takes a dataset in it's constructor.
        trainer: `Lazy[Trainer]`
            The `Trainer` that actually implements the training loop.  This is a lazy object because
            it depends on the model that's going to be trained.
        vocabulary: `Lazy[Vocabulary]`, optional (default=None)
            The `Vocabulary` that we will use to convert strings in the data to integer ids (and
            possibly set sizes of embedding matrices in the `Model`).  By default we construct the
            vocabulary from the instances that we read.
        datasets_for_vocab_creation: `List[str]`, optional (default=None)
            If you pass in more than one dataset but don't want to use all of them to construct a
            vocabulary, you can pass in this key to limit it.  Valid entries in the list are
            "train", "validation" and "test".
        validation_dataset_reader: `DatasetReader`, optional (default=None)
            If given, we will use this dataset reader for the validation data instead of
            `dataset_reader`.
        validation_data_path: `str`, optional (default=None)
            If given, we will use this data for computing validation metrics and early stopping.
        validation_data_loader: `Lazy[DataLoader]`, optional (default=None)
            If given, the data_loader we use to batch instances from the dataset reader at
            validation and test time. This is lazy because it takes a dataset in it's constructor.
        test_data_path: `str`, optional (default=None)
            If given, we will use this as test data.  This makes it available for vocab creation by
            default, but nothing else.
        evaluate_on_test: `bool`, optional (default=False)
            If given, we will evaluate the final model on this data at the end of training.  Note
            that we do not recommend using this for actual test data in every-day experimentation;
            you should only very rarely evaluate your model on actual test data.
        """

        datasets = training_util.read_all_datasets(
            train_data_path=train_data_path,
            dataset_reader=dataset_reader,
            validation_dataset_reader=validation_dataset_reader,
            validation_data_path=validation_data_path,
            test_data_path=test_data_path,
        )

        if datasets_for_vocab_creation:
            for key in datasets_for_vocab_creation:
                if key not in datasets:
                    raise checks.ConfigurationError(f"invalid 'dataset_for_vocab_creation' {key}")

        instance_generator = (
            instance
            for key, dataset in datasets.items()
            if not datasets_for_vocab_creation or key in datasets_for_vocab_creation
            for instance in dataset
        )

        vocabulary_ = vocabulary.construct(instances=instance_generator)
        if not vocabulary_:
            vocabulary_ = data.Vocabulary.from_instances(instance_generator)
        model_ = model.construct(vocab=vocabulary_)

        # Initializing the model can have side effect of expanding the vocabulary.
        # Save the vocab only in the master. In the degenerate non-distributed
        # case, we're trivially the master.
        if common_util.is_master():
            vocabulary_path = os.path.join(serialization_dir, "vocabulary")
            # Only difference compared to TrainModel!
            model_.vocab.save_to_files(vocabulary_path)

        for dataset in datasets.values():
            dataset.index_with(model_.vocab)

        data_loader_ = data_loader.construct(dataset=datasets["train"])
        validation_data = datasets.get("validation")
        if validation_data is not None:
            # Because of the way Lazy[T] works, we can't check it's existence
            # _before_ we've tried to construct it. It returns None if it is not
            # present, so we try to construct it first, and then afterward back off
            # to the data_loader configuration used for training if it returns None.
            validation_data_loader_ = validation_data_loader.construct(dataset=validation_data)
            if validation_data_loader_ is None:
                validation_data_loader_ = data_loader.construct(dataset=validation_data)
        else:
            validation_data_loader_ = None

        test_data = datasets.get("test")
        if test_data is not None:
            test_data_loader = validation_data_loader.construct(dataset=test_data)
            if test_data_loader is None:
                test_data_loader = data_loader.construct(dataset=test_data)
        else:
            test_data_loader = None

        # We don't need to pass serialization_dir and local_rank here, because they will have been
        # passed through the trainer by from_params already, because they were keyword arguments to
        # construct this class in the first place.
        trainer_ = trainer.construct(
            model=model_, data_loader=data_loader_, validation_data_loader=validation_data_loader_,
        )

        return cls(
            serialization_dir=serialization_dir,
            model=model_,
            trainer=trainer_,
            evaluation_data_loader=test_data_loader,
            evaluate_on_test=evaluate_on_test,
            batch_weight_key=batch_weight_key,
        )
