"""Checks tests."""
import unittest

import torch
from allennlp.common import checks as allen_checks

from combo.utils import checks


class SizeCheckTest(unittest.TestCase):

    def test_equal_sizes(self):
        # given
        size = (10, 2)
        tensor1 = torch.rand(size)
        tensor2 = torch.rand(size)

        # when
        checks.check_size_match(size_1=tensor1.size(),
                                size_2=tensor2.size(),
                                tensor_1_name="", tensor_2_name="")

        # then
        # nothing happens
        self.assertTrue(True)

    def test_different_sizes(self):
        # given
        size1 = (10, 2)
        size2 = (20, 1)
        tensor1 = torch.rand(size1)
        tensor2 = torch.rand(size2)

        # when/then
        with self.assertRaises(allen_checks.ConfigurationError):
            checks.check_size_match(size_1=tensor1.size(),
                                    size_2=tensor2.size(),
                                    tensor_1_name="", tensor_2_name="")
