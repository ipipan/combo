"""Utils to perform configuration checks."""
import os

import torch
from allennlp.common import checks as allen_checks


def file_exists(*paths):
    """Check whether paths exists."""
    for path in paths:
        if path is None:
            raise allen_checks.ConfigurationError("File cannot be None")
        if not os.path.exists(path):
            raise allen_checks.ConfigurationError(f"Could not find the file at path: '{path}'")


def check_size_match(size_1: torch.Size, size_2: torch.Size, tensor_1_name: str, tensor_2_name: str):
    """Check if tensors' sizes are the same."""
    if size_1 != size_2:
        raise allen_checks.ConfigurationError(
            f"{tensor_1_name} must match {tensor_2_name}, but got {size_1} "
            f"and {size_2} instead"
        )
