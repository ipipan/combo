import torch.optim.lr_scheduler as lr_scheduler
from allennlp.training.learning_rate_schedulers import learning_rate_scheduler
from overrides import overrides


@learning_rate_scheduler.LearningRateScheduler.register("combo_scheduler")
class Scheduler(learning_rate_scheduler._PyTorchLearningRateSchedulerWrapper):

    def __init__(self, optimizer, patience: int = 6, decreases: int = 2, threshold: float = 1e-3):
        super().__init__(lr_scheduler.LambdaLR(optimizer, lr_lambda=[self._lr_lambda]))
        self.threshold = threshold
        self.decreases = decreases
        self.patience = patience
        self.start_patience = patience
        self.best_score = 0.0

    @staticmethod
    def _lr_lambda(idx: int) -> float:
        return 1.0 / (1.0 + idx * 1e-4)

    @overrides
    def step(self, metric: float = None) -> None:
        self.lr_scheduler.step()

        if metric is not None:
            if metric - self.best_score > self.threshold:
                self.best_score = metric if metric > self.best_score else self.best_score
                self.patience = self.start_patience
            else:
                if self.patience <= 1:
                    if self.decreases == 0:
                        # This is condition for Trainer to trigger early stopping
                        self.patience = 0
                    else:
                        self.patience = self.start_patience
                        self.decreases -= 1
                        self.threshold /= 2
                        self.lr_scheduler.base_lrs = [x / 2 for x in self.lr_scheduler.base_lrs]
                else:
                    self.patience -= 1
