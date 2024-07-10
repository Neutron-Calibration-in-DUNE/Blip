"""
Schedulers for blip.
"""
import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt

from blip.utils.logger import BlipError
from blip.utils.utils import profiler
from blip.utils.utils import fig_to_array


class BlipScheduler:
    """
    A standard scheduler for pytorch models.
    """
    def __init__(
        self,
        config: dict = {},
        meta:   dict = {}
    ):
        self.config = config
        self.meta = meta

        self.parse_config()

    @profiler
    def parse_config(self):
        if "optimizer" not in self.meta:
            raise BlipError('no optimizer specified in meta!')

        if self.config['lr_schedule'] == 'cosine':
            if "warmup" not in self.config:
                self.config["warmup"] = 1
            if self.config['warmup'] > 0:
                lr_scale = lambda x: min(
                    (x + 1) / self.config['warmup'],
                    0.5 * (1 + np.cos(np.pi * x / self.meta['num_iterations']))
                )
                self.scheduler = optim.lr_scheduler.LambdaLR(
                    self.meta['optimizer'].optimizer,
                    lr_scale
                )
            else:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.meta['optimizer'].optimizer,
                    self.meta['num_iterations']
                )
        elif self.config['lr_schedule'] == '1cycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.meta['optimizer'].optimizer,
                max_lr=self.config["max_lr"],
                steps_per_epoch=len(self.meta["loader"].train_loader),
                epochs=self.config["num_epochs"],
                anneal_strategy='cos',
                div_factor=25.0,
                final_div_factor=1e4
            )
        else:
            self.scheduler = None

    def step(self):
        return self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def find_optimum_lr(
        self,
        iteration_loss,
        iteration_learning_rate
    ):
        min_loss_idx = np.argmin(iteration_loss)
        optimal_lr = iteration_learning_rate[min_loss_idx]
        self.meta["max_lr"] = optimal_lr

    def report_tensorboard(self, iterations, train_type):
        """Report learning rate to tensorboard"""
        self.meta['tensorboard'].add_scalar(
            f'1cycle learning rate ({train_type})',
            self.scheduler.get_last_lr(),
            iterations
        )

    def report_tensorboard_optimization(
        self, loss, learning_rate
    ):
        iterations = [ii+1 for ii in range(len(loss))]
        fig, axs1 = plt.subplots(figsize=(10, 10))
        axs2 = axs1.twinx()
        axs1.set_title(f"Learning Rate Optimization ({len(loss)} iterations)")
        axs1.plot(iterations, learning_rate, linestyle='--', c='k')
        axs2.plot(iterations, loss, linestyle='--', c='r')
        axs1.set_xlabel('Iteration')
        axs1.set_ylabel('Learning rate')
        axs2.set_ylabel('Loss')
        axs1.set_yscale('log')
        fig_array = fig_to_array(fig)
        self.meta['tensorboard'].add_image(
            'learning rate optimization',
            fig_array,
            0,
            dataformats='HWC'
        )
        plt.close()
