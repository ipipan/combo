import logging
import os
import pathlib
import shutil
import tempfile
import unittest
from allennlp.commands import train
from allennlp.common import Params, util


class TrainingEndToEndTest(unittest.TestCase):
    PROJECT_ROOT = (pathlib.Path(__file__).parent / '..').resolve()
    MODULE_ROOT = PROJECT_ROOT / 'combo'
    TESTS_ROOT = PROJECT_ROOT / 'tests'
    FIXTURES_ROOT = TESTS_ROOT / 'fixtures'
    TEST_DIR = pathlib.Path(tempfile.mkdtemp(prefix='allennlp_tests'))

    def setUp(self) -> None:
        logging.getLogger('allennlp.common.util').disabled = True
        logging.getLogger('allennlp.training.tensorboard_writer').disabled = True
        logging.getLogger('allennlp.common.params').disabled = True
        logging.getLogger('allennlp.nn.initializers').disabled = True

    def test_training_produces_model(self):
        # given
        util.import_module_and_submodules('combo.models')
        util.import_module_and_submodules('combo.training')
        ext_vars = {
            'training_data_path': os.path.join(self.FIXTURES_ROOT, 'example.conllu'),
            'validation_data_path': os.path.join(self.FIXTURES_ROOT, 'example.conllu'),
            'features': 'token char',
            'targets': 'deprel head lemma feats upostag xpostag',
            'type': 'default',
            'pretrained_tokens': os.path.join(self.FIXTURES_ROOT, 'example.vec'),
            'pretrained_transformer_name': '',
            'embedding_dim': '300',
            'cuda_device': '-1',
            'num_epochs': '1',
            'word_batch_size': '1',
            'use_tensorboard': 'False'
        }
        params = Params.from_file(os.path.join(self.PROJECT_ROOT, 'config.template.jsonnet'),
                                  ext_vars=ext_vars)

        # when
        model = train.train_model(params, serialization_dir=self.TEST_DIR)

        # then
        self.assertIsNotNone(model)

    def tearDown(self) -> None:
        shutil.rmtree(self.TEST_DIR)
