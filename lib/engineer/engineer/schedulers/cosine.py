import math
import warnings
from typing import List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        warmup_steps: int = 0,
        decay_steps: int = 0,
        last_step=-1,
        min_lrs=False
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.stable_steps = max_steps - warmup_steps - decay_steps
        self.decay_steps = decay_steps
        self.min_lr=1e-5,
        self.min_lrs=min_lrs
        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        if not self.min_lrs:    
            base_lrs = torch.tensor(self.base_lrs)


            if self.last_epoch < self.warmup_steps:
                s = 0.5 - 0.5 * math.cos(math.pi * self.last_epoch / self.warmup_steps)

            elif self.last_epoch < self.warmup_steps + self.stable_steps:
                s = 1.0

            else:
                s = 0.5 + 0.5 * math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_steps - self.stable_steps)
                    / self.decay_steps
                    )

            return base_lrs * s

        else:
            min_lrs = torch.tensor(self.min_lr)

            if self.last_epoch < self.warmup_steps:
                s = 0.5 - 0.5 * math.cos(math.pi * self.last_epoch / self.warmup_steps)
                return base_lrs * s
            elif self.last_epoch < self.warmup_steps + self.stable_steps:
                s = 1.0
                return base_lrs * s
            else:
                cosine = 0.5 + 0.5 * math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_steps - self.stable_steps)
                    / self.decay_steps
                    )
                return (min_lrs + (base_lrs - min_lrs) * cosine)


class CosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        cycle_steps: int,
        last_step=-1,
        lr_min=1e-7,
    ) -> None:
        self.cycle_steps = cycle_steps
        self.max_steps = max_steps
        self.lr_min = lr_min

        super().__init__(optimizer, last_step)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        base_lrs = torch.tensor(self.base_lrs)

        return 0.5 * (
            (self.lr_min - base_lrs)
            * math.cos(2 * math.pi * self.last_epoch / self.cycle_steps)
            + self.lr_min
            + base_lrs
        )
