from typing import Union, Any, Dict, Tuple

from allennlp import training
from allennlp.training import trainer as allen_trainer


@training.Checkpointer.register("finishing_only_checkpointer")
class FinishingTrainingCheckpointer(training.Checkpointer):
    """Checkpointer disables restoring interrupted training and saves only weights
       when this is last epoch / learning rate is on the last lr decrease.

       Remove checkpointer configuration from config template to get regular, on best score, saving."""

    def save_checkpoint(
            self,
            epoch: Union[int, str],
            trainer: "allen_trainer.Trainer",
            is_best_so_far: bool = False,
    ) -> None:
        if trainer._learning_rate_scheduler.decreases <= 1 or epoch == trainer._num_epochs - 1:
            super().save_checkpoint(epoch, trainer, is_best_so_far)

    def restore_checkpoint(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return {}, {}

    def maybe_save_checkpoint(
            self, trainer: "allen_trainer.Trainer", epoch: int, batches_this_epoch: int
    ) -> None:
        pass
