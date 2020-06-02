from typing import Dict, Optional, List

import torch
from allennlp import models, common
from allennlp.data import dataloader
from allennlp.training import optimizers


class NullTensorboardWriter(common.FromParams):

    def log_batch(
            self,
            model: models.Model,
            optimizer: optimizers.Optimizer,
            batch_grad_norm: Optional[float],
            metrics: Dict[str, float],
            batch_group: List[List[dataloader.TensorDict]],
            param_updates: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        pass

    def reset_epoch(self) -> None:
        pass

    def should_log_this_batch(self) -> bool:
        return False

    def should_log_histograms_this_batch(self) -> bool:
        return False

    def add_train_scalar(self, name: str, value: float, timestep: int = None) -> None:
        pass

    def add_train_histogram(self, name: str, values: torch.Tensor) -> None:
        pass

    def add_validation_scalar(self, name: str, value: float, timestep: int = None) -> None:
        pass

    def log_parameter_and_gradient_statistics(self, model: models.Model, batch_grad_norm: float) -> None:
        pass

    def log_learning_rates(self, model: models.Model, optimizer: torch.optim.Optimizer):
        pass

    def log_histograms(self, model: models.Model) -> None:
        pass

    def log_gradient_updates(self, model: models.Model, param_updates: Dict[str, torch.Tensor]) -> None:
        pass

    def log_metrics(
            self,
            train_metrics: dict,
            val_metrics: dict = None,
            epoch: int = None,
            log_to_console: bool = False,
    ) -> None:
        pass

    def enable_activation_logging(self, model: models.Model) -> None:
        pass

    def log_activation_histogram(self, outputs, log_prefix: str) -> None:
        pass

    def close(self) -> None:
        pass
