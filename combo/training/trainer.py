import datetime
import logging
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data as data
from allennlp import training, common
from allennlp.common import checks
from allennlp.common import util as common_util
from allennlp.models import model
from allennlp.training import checkpointer, optimizers
from allennlp.training import learning_rate_schedulers
from allennlp.training import momentum_schedulers
from allennlp.training import moving_average
from allennlp.training import tensorboard_writer as allen_tensorboard_writer
from allennlp.training import util as training_util
from overrides import overrides

from combo.training import tensorboard_writer as combo_tensorboard_writer

logger = logging.getLogger(__name__)


@training.EpochCallback.register("transfer_patience")
class TransferPatienceEpochCallback(training.EpochCallback):

    def __call__(self, trainer: "training.GradientDescentTrainer", metrics: Dict[str, Any], epoch: int) -> None:
        if trainer._learning_rate_scheduler and trainer._learning_rate_scheduler.patience is not None:
            trainer._metric_tracker._patience = trainer._learning_rate_scheduler.patience
            trainer._metric_tracker._epochs_with_no_improvement = 0
        else:
            raise checks.ConfigurationError("Learning rate scheduler isn't properly setup!")


@training.Trainer.register("gradient_descent_validate_n", constructor="from_partial_objects")
class GradientDescentTrainer(training.GradientDescentTrainer):

    def __init__(self, model: model.Model, optimizer: optim.Optimizer, data_loader: data.DataLoader,
                 patience: Optional[int] = None, validation_metric: str = "-loss",
                 validation_data_loader: data.DataLoader = None, num_epochs: int = 20,
                 serialization_dir: Optional[str] = None, checkpointer: checkpointer.Checkpointer = None,
                 cuda_device: int = -1,
                 grad_norm: Optional[float] = None, grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[learning_rate_schedulers.LearningRateScheduler] = None,
                 momentum_scheduler: Optional[momentum_schedulers.MomentumScheduler] = None,
                 tensorboard_writer: allen_tensorboard_writer.TensorboardWriter = None,
                 moving_average: Optional[moving_average.MovingAverage] = None,
                 batch_callbacks: List[training.BatchCallback] = None,
                 epoch_callbacks: List[training.EpochCallback] = None, distributed: bool = False, local_rank: int = 0,
                 world_size: int = 1, num_gradient_accumulation_steps: int = 1,
                 opt_level: Optional[str] = None) -> None:
        super().__init__(model, optimizer, data_loader, patience, validation_metric, validation_data_loader, num_epochs,
                         serialization_dir, checkpointer, cuda_device, grad_norm, grad_clipping,
                         learning_rate_scheduler, momentum_scheduler, tensorboard_writer, moving_average,
                         batch_callbacks, epoch_callbacks, distributed, local_rank, world_size,
                         num_gradient_accumulation_steps, opt_level)
        # TODO extract param to constructor (+ constructor method?)
        self.validate_every_n = 5

    @overrides
    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise checks.ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for callback in self._epoch_callbacks:
            callback(self, metrics={}, epoch=-1)

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            if self._validation_data_loader is not None:
                val_metrics = {}
                this_epoch_val_metric = None
                if epoch % self.validate_every_n == 0:
                    with torch.no_grad():
                        # We have a validation set, so compute all the metrics on it.
                        val_loss, val_reg_loss, num_batches = self._validation_loss(epoch)

                        # It is safe again to wait till the validation is done. This is
                        # important to get the metrics right.
                        if self._distributed:
                            dist.barrier()

                        val_metrics = training_util.get_metrics(
                            self.model,
                            val_loss,
                            val_reg_loss,
                            num_batches,
                            reset=True,
                            world_size=self._world_size,
                            cuda_device=self.cuda_device,
                        )

                        # Check validation metric for early stopping
                        this_epoch_val_metric = val_metrics[self._validation_metric]
                        # self._metric_tracker.add_metric(this_epoch_val_metric)

                train_metrics["patience"] = self._metric_tracker._patience
                if self._metric_tracker.should_stop_early():
                    logger.info("Ran out of patience.  Stopping training.")
                    break

            if self._master:
                self._tensorboard.log_metrics(
                    train_metrics, val_metrics=val_metrics, log_to_console=True, epoch=epoch + 1
                )  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._master:
                common_util.dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric)

            if self._master:
                self._checkpointer.save_checkpoint(
                    epoch, self, is_best_so_far=self._metric_tracker.is_best_so_far()
                )

            # Wait for the master to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            for callback in self._epoch_callbacks:
                callback(self, metrics=metrics, epoch=epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                        (self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        self._tensorboard.close()

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    @classmethod
    def from_partial_objects(
            cls,
            model: model.Model,
            serialization_dir: str,
            data_loader: data.DataLoader,
            validation_data_loader: data.DataLoader = None,
            local_rank: int = 0,
            patience: int = None,
            validation_metric: str = "-loss",
            num_epochs: int = 20,
            cuda_device: int = -1,
            grad_norm: float = None,
            grad_clipping: float = None,
            distributed: bool = None,
            world_size: int = 1,
            num_gradient_accumulation_steps: int = 1,
            opt_level: Optional[str] = None,
            no_grad: List[str] = None,
            optimizer: common.Lazy[optimizers.Optimizer] = None,
            learning_rate_scheduler: common.Lazy[learning_rate_schedulers.LearningRateScheduler] = None,
            momentum_scheduler: common.Lazy[momentum_schedulers.MomentumScheduler] = None,
            tensorboard_writer: common.Lazy[allen_tensorboard_writer.TensorboardWriter] = None,
            moving_average: common.Lazy[moving_average.MovingAverage] = None,
            checkpointer: common.Lazy[training.Checkpointer] = None,
            batch_callbacks: List[training.BatchCallback] = None,
            epoch_callbacks: List[training.EpochCallback] = None,
    ) -> "training.Trainer":
        if tensorboard_writer.construct() is None:
            tensorboard_writer = common.Lazy(combo_tensorboard_writer.NullTensorboardWriter)
        return super().from_partial_objects(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=data_loader,
            validation_data_loader=validation_data_loader,
            local_rank=local_rank,
            patience=patience,
            validation_metric=validation_metric,
            num_epochs=num_epochs,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            distributed=distributed,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            opt_level=opt_level,
            no_grad=no_grad,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler,
            momentum_scheduler=momentum_scheduler,
            tensorboard_writer=tensorboard_writer,
            moving_average=moving_average,
            checkpointer=checkpointer,
            batch_callbacks=batch_callbacks,
            epoch_callbacks=epoch_callbacks,
        )
