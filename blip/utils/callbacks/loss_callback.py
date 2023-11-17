"""
Callback for recording loss information
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from blip.losses.loss_handler    import LossHandler
from blip.metrics.metric_handler import MetricHandler
from blip.utils.callbacks        import GenericCallback

class LossCallback(GenericCallback):
    """
    """
    def __init__(self,
        name:               str='loss_callback',
        criterion_handler:  LossHandler=None,
        metrics_handler:    MetricHandler=None,
        meta:               dict={}
    ):
        super(LossCallback, self).__init__(
            name, criterion_handler, metrics_handler, meta
        )
        if self.criterion_handler != None:
            self.loss_names = [
                loss.name for name, loss in self.criterion_handler.losses.items()
            ]

            # containers for training loss
            self.training_loss = torch.empty(
                size=(0,len(self.loss_names)), 
                dtype=torch.float, device=self.device
            )
            self.validation_loss = torch.empty(
                size=(0,len(self.loss_names)), 
                dtype=torch.float, device=self.device
            )
            self.test_loss = torch.empty(
                size=(0,len(self.loss_names)), 
                dtype=torch.float, device=self.device
            )
            self.training_target_loss = {
                name: torch.empty(
                    size=(0,len(loss.targets)),
                    dtype=torch.float, device=self.device
                )
                for name, loss in self.criterion_handler.losses.items()
            }
            self.validation_target_loss = {
                name: torch.empty(
                    size=(0,len(loss.targets)),
                    dtype=torch.float, device=self.device
                )
                for name, loss in self.criterion_handler.losses.items()
            }
            self.test_target_loss = {
                name: torch.empty(
                    size=(0,len(loss.targets)),
                    dtype=torch.float, device=self.device
                )
                for name, loss in self.criterion_handler.losses.items()
            }
        
    def save_losses(self):
        for name in self.training_target_loss.keys():
            self.training_target_loss[name].cpu().numpy()
        for name in self.validation_target_loss.keys():
            self.validation_target_loss[name].cpu().numpy()
        for name in self.test_target_loss.keys():
            self.test_target_loss[name].cpu().numpy()
        np.savez(
            f"{self.meta['local_scratch']}/losses.npz",
            loss_names=self.loss_names,
            training_loss=self.training_loss.cpu().numpy(),
            validation_loss=self.validation_loss.cpu().numpy(),
            test_loss=self.test_loss.cpu().numpy(),
            training_target_loss=self.training_target_loss,
            validation_target_loss=self.validation_target_loss,
            test_target_loss=self.test_target_loss
        )
    
    def reset_batch(self):
        self.training_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
        self.validation_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
        self.test_loss = torch.empty(
            size=(0,len(self.loss_names)), 
            dtype=torch.float, device=self.device
        )
        self.training_target_loss = {
            name: torch.empty(
                size=(0,len(loss.targets)),
                dtype=torch.float, device=self.device
            )
            for name, loss in self.criterion_handler.losses.items()
        }
        self.validation_target_loss = {
            name: torch.empty(
                size=(0,len(loss.targets)),
                dtype=torch.float, device=self.device
            )
            for name, loss in self.criterion_handler.losses.items()
        }
        self.test_target_loss = {
            name: torch.empty(
                size=(0,len(loss.targets)),
                dtype=torch.float, device=self.device
            )
            for name, loss in self.criterion_handler.losses.items()
        }

    def evaluate_epoch(self,
        train_type='train'
    ):  
        temp_losses = torch.empty(
            size=(1,0), 
            dtype=torch.float, device=self.device
        )
        # run through criteria
        if train_type == 'train':
            for name, loss in self.criterion_handler.batch_loss.items():
                temp_loss = loss.sum()/self.num_training_batches
                temp_losses = torch.cat(
                    (temp_losses,torch.tensor([[temp_loss]], device=self.device)),
                    dim=1
                )
            self.training_loss = torch.cat(
                (self.training_loss, temp_losses),
                dim=0
            )
            for name, loss in self.criterion_handler.losses.items():
                temp_target_losses = torch.empty(
                    size=(1,0), 
                    dtype=torch.float, device=self.device
                )
                for target in loss.targets:
                    temp_loss = loss.batch_loss[target].sum()/self.num_training_batches
                    temp_target_losses = torch.cat(
                        (temp_target_losses, torch.tensor([[temp_loss]], device=self.device)),
                        dim=1
                    )
                self.training_target_loss[name] = torch.cat(
                    (self.training_target_loss[name], temp_target_losses),
                    dim=0
                )

        elif train_type == 'validation':
            for name, loss in self.criterion_handler.batch_loss.items():
                temp_loss = loss.sum()/self.num_validation_batches
                temp_losses = torch.cat(
                    (temp_losses,torch.tensor([[temp_loss]], device=self.device)),
                    dim=1
                )
            self.validation_loss = torch.cat(
                (self.validation_loss, temp_losses),
                dim=0
            )
            for name, loss in self.criterion_handler.losses.items():
                temp_target_losses = torch.empty(
                    size=(1,0), 
                    dtype=torch.float, device=self.device
                )
                for target in loss.targets:
                    temp_loss = loss.batch_loss[target].sum()/self.num_validation_batches
                    temp_target_losses = torch.cat(
                        (temp_target_losses, torch.tensor([[temp_loss]], device=self.device)),
                        dim=1
                    )
                self.validation_target_loss[name] = torch.cat(
                    (self.validation_target_loss[name], temp_target_losses),
                    dim=0
                )
        else:
            for name, loss in self.criterion_handler.batch_loss.items():
                temp_loss = loss.sum()/self.num_test_batches
                temp_losses = torch.cat(
                    (temp_losses,torch.tensor([[temp_loss]], device=self.device)),
                    dim=1
                )
            self.test_loss = torch.cat(
                (self.test_loss, temp_losses),
                dim=0
            )
            for name, loss in self.criterion_handler.losses.items():
                temp_target_losses = torch.empty(
                    size=(1,0), 
                    dtype=torch.float, device=self.device
                )
                for target in loss.targets:
                    temp_loss = loss.batch_loss[target].sum()/self.num_test_batches
                    temp_target_losses = torch.cat(
                        (temp_target_losses, torch.tensor([[temp_loss]], device=self.device)),
                        dim=1
                    )
                self.test_target_loss[name] = torch.cat(
                    (self.test_target_loss[name], temp_target_losses),
                    dim=0
                )
        self.criterion_handler.reset_batch()
    
    def evaluate_training(self):
        pass

    def evaluate_testing(self):
        epoch_ticks = np.arange(1, self.epochs+1)
        ########### Plots with total from each loss ##########
        if self.num_training_batches != 0:
            fig, axs = plt.subplots(figsize=(15, 10))
            if len(self.loss_names) > 1:
                final_training_value = f"(final={self.training_loss.sum(dim=1)[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    self.training_loss.sum(dim=1).cpu().numpy(),
                    c='k',
                    label=rf"{'(total)':<12} {final_training_value:>16}"
                )
            for ii, loss in enumerate(self.criterion_handler.losses.keys()):
                temp_loss = self.training_loss[:,ii]
                final_training_value = f"(final={temp_loss[-1]:.2e})"
                # plot using specified line colors
                axs.plot(
                    epoch_ticks,
                    temp_loss.cpu().numpy(),
                    c=self.plot_colors[ii],
                    label=rf"{loss:<12} {final_training_value:>16}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"($\alpha=${self.criterion_handler.losses[loss].alpha})"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("loss")
            axs.set_yscale('log')
            plt.title("loss vs. epoch (training)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_training_loss.png")
        # validation plot
        if self.num_validation_batches != 0:
            fig, axs = plt.subplots(figsize=(15, 10))
            if len(self.loss_names) > 1:
                final_validation_value = f"(final={self.validation_loss.sum(dim=1)[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    self.validation_loss.sum(dim=1).cpu().numpy(),
                    c='k',
                    label=rf"{'(total)':<12} {final_validation_value:>16}"
                )
            for ii, loss in enumerate(self.criterion_handler.losses.keys()):
                temp_loss = self.validation_loss[:,ii]
                final_validation_value = f"(final={temp_loss[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_loss.cpu().numpy(),
                    c=self.plot_colors[ii],
                    label=rf"{loss:<12} {final_validation_value:>16}"
                )
                axs.plot([],[],
                    marker='',
                    linestyle='',
                    label=rf"($\alpha=${self.criterion_handler.losses[loss].alpha})"
                )
            axs.set_xlabel("epoch")
            axs.set_ylabel("loss")
            axs.set_yscale('log')
            plt.title("loss vs. epoch (validation)")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_validation_loss.png")
        # plot both
        if self.num_training_batches != 0 and self.num_validation_batches != 0:
            fig, axs = plt.subplots(figsize=(15, 10))
            final_value = f"(final={self.training_loss.sum(dim=1)[-1]:.2e})"
            axs.plot(
                epoch_ticks,
                self.training_loss.sum(dim=1).cpu().numpy(),
                c='k',
                linestyle='-',
                label=rf"{'(train)':<12} {final_value:>16}"
            )
            final_value = f"(final={self.validation_loss.sum(dim=1)[-1]:.2e})"
            axs.plot(
                epoch_ticks,
                self.validation_loss.sum(dim=1).cpu().numpy(),
                c='k',
                linestyle='--',
                label=rf"{'(validation)':<12} {final_value:>16}"
            )
            if len(self.loss_names) > 1:
                for ii, loss in enumerate(self.criterion_handler.losses.keys()):
                    temp_training_loss = self.training_loss[:,ii]
                    temp_validation_loss = self.validation_loss[:,ii]
                    final_training_value = f"(final={temp_training_loss[-1]:.2e})"
                    final_validation_value = f"(final={temp_validation_loss[-1]:.2e})"
                    alpha_value = rf"($\alpha=${self.criterion_handler.losses[loss].alpha})"
                    axs.plot(
                        epoch_ticks,
                        temp_training_loss.cpu().numpy(),
                        c=self.plot_colors[ii],
                        linestyle='-',
                        label=rf"{loss:<12} {final_training_value:>16}"
                    )
                    axs.plot(
                        epoch_ticks,
                        temp_validation_loss.cpu().numpy(),
                        c=self.plot_colors[ii],
                        linestyle='--',
                        label=rf"{alpha_value:<18} {final_validation_value:>16}"
                    )
            # plot test values
            if self.num_test_batches != 0:
                total_test_loss = f"{self.test_loss.sum(dim=1)[0]:.2e}"
                axs.plot([], [],
                    marker='x', 
                    c='k',
                    linestyle='',
                    label=f"{'(test) total:'} {total_test_loss}"
                )
                for ii, loss in enumerate(self.criterion_handler.losses.keys()):
                    temp_loss_name = f"(test) {loss}:"
                    temp_loss_value = f"{self.test_loss[0][ii]:.2e}"
                    axs.plot([], [],
                        marker='x', 
                        c=self.plot_colors[ii],
                        linestyle='',
                        label=f"{temp_loss_name} {temp_loss_value}"
                    )
            axs.set_xlabel("epoch")
            axs.set_ylabel("loss")
            axs.set_yscale('log')
            plt.title("loss vs. epoch")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_loss.png")

        ########### Plots for each loss with target contributions ##########
        for name, loss in self.criterion_handler.losses.items():
            fig, axs = plt.subplots(figsize=(15, 10))
            for ii, target in enumerate(loss.targets):
                temp_training_loss = self.training_target_loss[name][:, ii]
                final_training_value = f"(final={temp_training_loss[-1]:.2e})"
                axs.plot(
                    epoch_ticks,
                    temp_training_loss.cpu().numpy(),
                    c=self.plot_colors[ii],
                    linestyle='-',
                    label=rf"{target:<12} {final_training_value:>16}"
                )
                axs.set_xlabel("epoch")
                axs.set_ylabel("loss")
                axs.set_yscale('log')
                plt.title(f"{name} - loss vs. epoch")
                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.tight_layout()
                plt.savefig(f"{self.meta['local_scratch']}/plots/epoch_loss_{name}.png")
        # save losses to npz
        self.save_losses()

    def evaluate_inference(self):
        pass