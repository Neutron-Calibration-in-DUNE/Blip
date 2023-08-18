"""
Generic metric callback
"""
import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc

from blip.losses.loss_handler import LossHandler
from blip.metrics.metric_handler import MetricHandler
from blip.utils.callbacks import GenericCallback
from blip.utils import utils


def histogram(x, bins, range=[], density=False):
    # Like torch.histogram, but works with cuda
    if len(range) == 0:
        min, max = x.min(), x.max()
    else:
        min, max = range[0], range[1]
    counts = torch.histc(x, bins, min=min, max=max)
    boundaries = torch.linspace(min, max, bins + 1)
    if density:
        counts /= torch.sum(counts)
    return counts, boundaries

class ConfusionMatrixCallback(GenericCallback):
    """
    """
    def __init__(self,
        name:   str='confusion_matrix_callback',
        criterion_handler: LossHandler=None,
        metrics_handler: MetricHandler=None,
        meta:   dict={}
    ):  
        super(ConfusionMatrixCallback, self).__init__(
            name, criterion_handler, metrics_handler, meta
        )
        if "ConfusionMatrixMetric" in self.metrics_handler.metrics.keys():
            self.metric = self.metrics_handler.metrics["ConfusionMatrixMetric"]
        else:   
            self.logger.error("no ConfusionMatrixMetric in the list of metrics!")

        if not os.path.isdir("plots/confusion_matrix/"):
            os.makedirs("plots/confusion_matrix/")

        # if not os.path.isdir("plots/roc/"):
        #     os.makedirs("plots/roc/")
        
        # if not os.path.isdir("plots/summed_adc/"):
        #     os.makedirs("plots/summed_adc/")

        # self.training_probabilities = {}
        # self.validation_probabilities = {}
        # self.test_probabilities = {}

        # self.training_summed_adc = None
        # self.validation_summed_adc = None
        # self.test_summed_adc = None

        self.training_confusion = None
        self.validation_confusion = None
        self.test_confusion = None

    def reset_batch(self):
        # self.training_probabilities = {}
        # self.validation_probabilities = {}
        # self.test_probabilities = {}

        # self.training_summed_adc = None
        # self.validation_summed_adc = None
        # self.test_summed_adc = None

        self.training_confusion = None
        self.validation_confusion = None
        self.test_confusion = None

    def evaluate_epoch(self,
        train_type='train'
    ):  
        if train_type == "train":
            # for kk, input in enumerate(self.metric.inputs):
            #     self.training_probabilities[input] = self.metric.batch_probabilities[input]
            # self.training_summed_adc = self.metric.batch_summed_adc
            self.training_confusion = self.metric.compute()
        elif train_type == "validation":
            # for kk, input in enumerate(self.metric.inputs):
            #     self.validation_probabilities[input] = self.metric.batch_probabilities[input]
            # self.validation_summed_adc = self.metric.batch_summed_adc
            self.validation_confusion = self.metric.compute()
        else:
            # for kk, input in enumerate(self.metric.inputs):
            #     self.test_probabilities[input] = self.metric.batch_probabilities[input]
            # self.test_summed_adc = self.metric.batch_summed_adc
            self.test_confusion = self.metric.compute()
        # self.metric.reset_probabilities()

    def evaluate_training(self):
        for kk, output in enumerate(self.metric.outputs):
            # plot the training confusion matrix
            if self.metric.consolidate_classes:
                training_display = ConfusionMatrixDisplay(
                    self.training_confusion[output].cpu().numpy()
                ) 
                training_display.plot()
            else:
                training_display = ConfusionMatrixDisplay(
                    self.training_confusion[output].cpu().numpy(),
                    display_labels = self.metric.labels[output]
                ) 
                training_display.plot(
                    xticks_rotation="vertical"
                ) 
                training_display.figure_.set_figwidth(len(self.metric.labels[output]))
                training_display.figure_.set_figheight(len(self.metric.labels[output]))
                     
            plt.suptitle(f"Training Confusion Matrix\nClass {output}")
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/training_confusion_matrix_{output}.png")
            plt.close()

            if self.metric.consolidate_classes:
                validation_display = ConfusionMatrixDisplay(
                    self.validation_confusion[output].cpu().numpy()
                ) 
                validation_display.plot()
            else:
                validation_display = ConfusionMatrixDisplay(
                    self.validation_confusion[output].cpu().numpy(),
                    display_labels = self.metric.labels[output]
                ) 
                validation_display.plot(
                    xticks_rotation="vertical"
                ) 
                validation_display.figure_.set_figwidth(len(self.metric.labels[output]))
                validation_display.figure_.set_figheight(len(self.metric.labels[output]))
            
            plt.suptitle(f"Validation Confusion Matrix\nClass {output}")
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/validation_confusion_matrix_{output}.png")
            plt.close()
            
            # # plot statistics on categorical probabilities
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots(figsize=(10,6))
            #     temp_train_probs = self.training_probabilities[input][:,ii]
            #     temp_train_labels = self.training_probabilities[input][:,self.metric.num_classes[kk]]
            #     for jj, inner_label in enumerate(self.metric.labels[input]):
            #         axs.hist(
            #             temp_train_probs[(temp_train_labels == jj)].cpu(),
            #             bins=100,
            #             range=[0,1],
            #             histtype='step',
            #             stacked=True,
            #             density=True,
            #             label=f"{inner_label}"
            #         )
            #     axs.set_xlabel(r"Class probability $p(\theta=$"+f"{outer_label}"+r"$|y)$")
            #     axs.set_ylabel("Truth Counts")
            #     plt.suptitle(f"Training probability predictions for class {outer_label}")
            #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/confusion_matrix/training_probabilities_{outer_label}.png")
            #     plt.close()
            
            # # generate ROC curves for each class
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots()
            #     temp_train_probs = self.training_probabilities[input][:,ii]
            #     temp_train_labels = self.training_probabilities[input][:,self.metric.num_classes[kk]]

            #     signal = temp_train_probs[(temp_train_labels == ii)]
            #     background = temp_train_probs[(temp_train_labels != ii)]

            #     signal_hist, signal_edges = histogram(
            #         signal,
            #         bins=100,
            #         range=[0,1], density=True
            #     )
            #     background_hist, background_edges = histogram(
            #         background,
            #         bins=100,
            #         range=[0,1], density=True
            #     )
            #     # get the cumulative distributions
            #     dx = (signal_edges[1] - signal_edges[0]).to(self.device)
            #     tpr = torch.tensor([1.0]).to(self.device)
            #     fpr = torch.tensor([0.0]).to(self.device)

            #     tpr = torch.cat((tpr, 1.0 - (torch.cumsum(signal_hist, dim=0) * dx)), dim=0)
            #     fpr = torch.cat((fpr, (torch.cumsum(background_hist, dim=0) * dx)), dim=0)

            #     tpr = torch.cat((tpr, torch.tensor([0.0]).to(self.device)), dim=0)
            #     fpr = torch.cat((fpr, torch.tensor([1.0]).to(self.device)), dim=0)
                
            #     class_auc = auc(1.0 - fpr.cpu().numpy(), tpr.cpu().numpy())

            #     axs.plot(
            #         tpr.cpu().numpy(), fpr.cpu().numpy(),
            #         linestyle='--',
            #         label=f"AUC: {class_auc:.2f}"
            #     )
            #     axs.set_xlabel("Signal Acceptance (TPR)")
            #     axs.set_ylabel("Background Rejection (TNR)")
            #     plt.suptitle(f"Training ROC curve (TNR vs TPR) for class {outer_label}")
            #     plt.grid(True)
            #     plt.legend(loc='best')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/roc/training_roc_{outer_label}.png")
            #     plt.close()
            
            # # generate summed ADC plots for each signal acceptance value
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots()
            #     temp_train_probs = self.training_probabilities[input][:,ii]
            #     temp_train_labels = self.training_probabilities[input][:,self.metric.num_classes[kk]]
            #     class_summed_adc = self.training_summed_adc[(temp_train_labels == ii)]
            #     num_backgrounds = len(temp_train_labels[(temp_train_labels != ii)])

            #     axs.hist(
            #         class_summed_adc.squeeze(1).cpu().numpy(),
            #         bins=100,
            #         histtype='step',
            #         stacked=True,
            #         label=f'{outer_label}'
            #     )
                    
            #     for acceptance in self.sig_acceptance:
            #         signal_summed_adc = self.training_summed_adc[(temp_train_probs > acceptance)].squeeze(1).cpu().numpy()
            #         accepted_labels = temp_train_labels[(temp_train_probs > acceptance)]
            #         accepted_backgrounds = len(accepted_labels[(accepted_labels != ii)])
            #         background_leakage = accepted_backgrounds / num_backgrounds

            #         axs.hist(
            #             signal_summed_adc,
            #             bins=100,
            #             histtype='step',
            #             stacked=True,
            #             label=r'p($\theta$='+f'{outer_label}) > {acceptance:.2f}\nleakage = {background_leakage:.4f}'
            #         )
                
            #     axs.set_xlabel(r"$\Sigma $" + "ADC")
            #     axs.set_ylabel("Counts")
            #     plt.suptitle(r"Training $\Sigma $" + f"ADC for class {outer_label}")
            #     plt.grid(True)
            #     plt.legend(loc='best')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/summed_adc/training_summed_adc_{outer_label}.png")
            #     plt.close()
            
            # # plot statistics on categorical probabilities
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots(figsize=(10,6))
            #     temp_train_probs = self.validation_probabilities[input][:,ii]
            #     temp_train_labels = self.validation_probabilities[input][:,self.metric.num_classes[kk]]
            #     for jj, inner_label in enumerate(self.metric.labels[input]):
            #         axs.hist(
            #             temp_train_probs[(temp_train_labels == jj)].cpu(),
            #             bins=100,
            #             range=[0,1],
            #             histtype='step',
            #             stacked=True,
            #             density=True,
            #             label=f"{inner_label}"
            #         )
            #     axs.set_xlabel(r"Class probability $p(\theta=$"+f"{outer_label}"+r"$|y)$")
            #     axs.set_ylabel("Truth Counts")
            #     plt.suptitle(f"Validation probability predictions for class {outer_label}")
            #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/confusion_matrix/validation_probabilities_{outer_label}.png")
            #     plt.close()
            
            # # generate ROC curves for each class
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots()
            #     temp_train_probs = self.validation_probabilities[input][:,ii]
            #     temp_train_labels = self.validation_probabilities[input][:,self.metric.num_classes[kk]]

            #     signal = temp_train_probs[(temp_train_labels == ii)]
            #     background = temp_train_probs[(temp_train_labels != ii)]

            #     signal_hist, signal_edges = histogram(
            #         signal,
            #         bins=100,
            #         range=[0,1], density=True
            #     )
            #     background_hist, background_edges = histogram(
            #         background,
            #         bins=100,
            #         range=[0,1], density=True
            #     )
            #     # get the cumulative distributions
            #     dx = (signal_edges[1] - signal_edges[0]).to(self.device)
            #     tpr = torch.tensor([1.0]).to(self.device)
            #     fpr = torch.tensor([0.0]).to(self.device)

            #     tpr = torch.cat((tpr, 1.0 - (torch.cumsum(signal_hist, dim=0) * dx)), dim=0)
            #     fpr = torch.cat((fpr, (torch.cumsum(background_hist, dim=0) * dx)), dim=0)

            #     tpr = torch.cat((tpr, torch.tensor([0.0]).to(self.device)),dim=0)
            #     fpr = torch.cat((fpr, torch.tensor([1.0]).to(self.device)),dim=0)
                
            #     class_auc = auc(1.0 - fpr.cpu().numpy(), tpr.cpu().numpy())

            #     axs.plot(
            #         tpr.cpu().numpy(), fpr.cpu().numpy(),
            #         linestyle='--',
            #         label=f"AUC: {class_auc:.2f}"
            #     )
            #     axs.set_xlabel("Signal Acceptance (TPR)")
            #     axs.set_ylabel("Background Rejection (TNR)")
            #     plt.suptitle(f"Validation ROC curve (TNR vs TPR) for class {outer_label}")
            #     plt.grid(True)
            #     plt.legend(loc='best')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/roc/validation_roc_{outer_label}.png")
            #     plt.close()
            
            # # generate summed ADC plots for each signal acceptance value
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots()
            #     temp_train_probs = self.validation_probabilities[input][:,ii]
            #     temp_train_labels = self.validation_probabilities[input][:,self.metric.num_classes[kk]]
            #     class_summed_adc = self.validation_summed_adc[(temp_train_labels == ii)]
            #     num_backgrounds = len(temp_train_labels[(temp_train_labels != ii)])

            #     axs.hist(
            #         class_summed_adc.squeeze(1).cpu().numpy(),
            #         bins=100,
            #         histtype='step',
            #         stacked=True,
            #         label=f'{outer_label}'
            #     )
                    
            #     for acceptance in self.sig_acceptance:
            #         signal_summed_adc = self.validation_summed_adc[(temp_train_probs > acceptance)].squeeze(1).cpu().numpy()
            #         accepted_labels = temp_train_labels[(temp_train_probs > acceptance)]
            #         accepted_backgrounds = len(accepted_labels[(accepted_labels != ii)])
            #         background_leakage = accepted_backgrounds / num_backgrounds

            #         axs.hist(
            #             signal_summed_adc,
            #             bins=100,
            #             histtype='step',
            #             stacked=True,
            #             label=r'p($\theta$='+f'{outer_label}) > {acceptance:.2f}\nleakage = {background_leakage:.4f}'
            #         )
                
            #     axs.set_xlabel(r"$\Sigma $" + "ADC")
            #     axs.set_ylabel("Counts")
            #     plt.suptitle(r"Validation $\Sigma $" + f"ADC for class {outer_label}")
            #     plt.grid(True)
            #     plt.legend(loc='best')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/summed_adc/validation_summed_adc_{outer_label}.png")
            #     plt.close()

            
    def evaluate_testing(self):  
        for kk, output in enumerate(self.metric.outputs):
            # plot the training confusion matrix
            if self.metric.consolidate_classes:
                test_display = ConfusionMatrixDisplay(
                    self.test_confusion[output].cpu().numpy()
                ) 
                test_display.plot()
            else:
                test_display = ConfusionMatrixDisplay(
                    self.test_confusion[output].cpu().numpy(),
                    display_labels = self.metric.labels[output]
                ) 
                test_display.plot(
                    xticks_rotation="vertical"
                )
                test_display.figure_.set_figwidth(len(self.metric.labels[output]))
                test_display.figure_.set_figheight(len(self.metric.labels[output]))
                      
            plt.suptitle(f"Test Confusion Matrix\nClass {output}")
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/test_confusion_matrix_{output}.png")
            plt.close()
    
            # # plot statistics on categorical probabilities
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots(figsize=(10,6))
            #     temp_train_probs = self.test_probabilities[input][:,ii]
            #     temp_train_labels = self.test_probabilities[input][:,self.metric.num_classes[kk]]
            #     for jj, inner_label in enumerate(self.metric.labels[input]):
            #         axs.hist(
            #             temp_train_probs[(temp_train_labels == jj)].cpu(),
            #             bins=100,
            #             range=[0,1],
            #             histtype='step',
            #             stacked=True,
            #             density=True,
            #             label=f"{inner_label}"
            #         )
            #     axs.set_xlabel(r"Class probability $p(\theta=$"+f"{outer_label}"+r"$|y)$")
            #     axs.set_ylabel("Truth Counts")
            #     plt.suptitle(f"Test probability predictions for class {outer_label}")
            #     plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/confusion_matrix/test_probabilities_{outer_label}.png")
            #     plt.close()
            
            # # generate ROC curves for each class
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots()
            #     temp_train_probs = self.test_probabilities[input][:,ii]
            #     temp_train_labels = self.test_probabilities[input][:,self.metric.num_classes[kk]]

            #     signal = temp_train_probs[(temp_train_labels == ii)]
            #     background = temp_train_probs[(temp_train_labels != ii)]
                
            #     signal_hist, signal_edges = histogram(
            #         signal,
            #         bins=100,
            #         range=[0,1], density=True
            #     )
            #     background_hist, background_edges = histogram(
            #         background,
            #         bins=100,
            #         range=[0,1], density=True
            #     )
            #     # get the cumulative distributions
            #     dx = (signal_edges[1] - signal_edges[0]).to(self.device)
            #     tpr = torch.tensor([1.0]).to(self.device)
            #     fpr = torch.tensor([0.0]).to(self.device)

            #     tpr = torch.cat((tpr, 1.0 - (torch.cumsum(signal_hist, dim=0) * dx)), dim=0)
            #     fpr = torch.cat((fpr, (torch.cumsum(background_hist, dim=0) * dx)), dim=0)

            #     tpr = torch.cat((tpr, torch.tensor([0.0]).to(self.device)),dim=0)
            #     fpr = torch.cat((fpr, torch.tensor([1.0]).to(self.device)),dim=0)
                
            #     class_auc = auc(1.0 - fpr.cpu().numpy(), tpr.cpu().numpy())

            #     axs.plot(
            #         tpr.cpu().numpy(), fpr.cpu().numpy(),
            #         linestyle='--',
            #         label=f"AUC: {class_auc:.2f}"
            #     )
            #     axs.set_xlabel("Signal Acceptance (TPR)")
            #     axs.set_ylabel("Background Rejection (TNR)")
            #     plt.suptitle(f"Test ROC curve (TNR vs TPR) for class {outer_label}")
            #     plt.grid(True)
            #     plt.legend(loc='best')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/roc/test_roc_{outer_label}.png")
            #     plt.close()
            
            # # generate summed ADC plots for each signal acceptance value
            # for ii, outer_label in enumerate(self.metric.labels[input]):
            #     fig, axs = plt.subplots()
            #     temp_train_probs = self.test_probabilities[input][:,ii]
            #     temp_train_labels = self.test_probabilities[input][:,self.metric.num_classes[kk]]
            #     class_summed_adc = self.test_summed_adc[(temp_train_labels == ii)]
            #     num_backgrounds = len(temp_train_labels[(temp_train_labels != ii)])

            #     axs.hist(
            #         class_summed_adc.squeeze(1).cpu().numpy(),
            #         bins=100,
            #         histtype='step',
            #         stacked=True,
            #         label=f'{outer_label}'
            #     )
                    
            #     for acceptance in self.sig_acceptance:
            #         signal_summed_adc = self.test_summed_adc[(temp_train_probs > acceptance)].squeeze(1).cpu().numpy()
            #         accepted_labels = temp_train_labels[(temp_train_probs > acceptance)]
            #         accepted_backgrounds = len(accepted_labels[(accepted_labels != ii)])
            #         background_leakage = accepted_backgrounds / num_backgrounds

            #         axs.hist(
            #             signal_summed_adc,
            #             bins=100,
            #             histtype='step',
            #             stacked=True,
            #             label=r'p($\theta$='+f'{outer_label}) > {acceptance:.2f}\nleakage = {background_leakage:.4f}'
            #         )
                
            #     axs.set_xlabel(r"$\Sigma $" + "ADC")
            #     axs.set_ylabel("Counts")
            #     plt.suptitle(r"Test $\Sigma $" + f"ADC for class {outer_label}")
            #     plt.grid(True)
            #     plt.legend(loc='best')
            #     plt.tight_layout()
            #     plt.savefig(f"plots/summed_adc/test_summed_adc_{outer_label}.png")
            #     plt.close()

    def evaluate_inference(self):
        return
        confusion = self.metric.compute()
        probabilities = self.metric.batch_predictions
        summed_adc = self.metric.batch_summed_adc

        # plot the training confusion matrix
        display = ConfusionMatrixDisplay(
            confusion.cpu().numpy(),
            display_labels = self.metrics_handler.labels
        ) 
        display.plot()       
        plt.suptitle("Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix/inference_confusion_matrix.png")
        plt.close()

        # plot statistics on categorical probabilities
        for ii, outer_label in enumerate(self.metric.labels[input]):
            fig, axs = plt.subplots(figsize=(10,6))
            temp_train_probs = probabilities[:,ii]
            temp_train_labels = probabilities[:,self.metric.num_classes[kk]]
            for jj, inner_label in enumerate(self.metric.labels[input]):
                axs.hist(
                    temp_train_probs[(temp_train_labels == jj)],
                    bins=100,
                    range=[0,1],
                    histtype='step',
                    stacked=True,
                    density=True,
                    label=f"{inner_label}"
                )
            axs.set_xlabel(r"Class probability $p(\theta=$"+f"{outer_label}"+r"$|y)$")
            axs.set_ylabel("Truth Counts")
            plt.suptitle(f"Inference probability predictions for class {outer_label}")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/inference_probabilities_{outer_label}.png")
            plt.close()
        
        # generate ROC curves for each class
        for ii, outer_label in enumerate(self.metric.labels[input]):
            fig, axs = plt.subplots()
            temp_train_probs = probabilities[:,ii]
            temp_train_labels = probabilities[:,self.metric.num_classes[kk]]

            signal = temp_train_probs[(temp_train_labels == ii)]
            background = temp_train_probs[(temp_train_labels != ii)]

            signal_hist, signal_edges = torch.histogram(
                signal,
                bins=100,
                range=[0,1], density=True
            )
            background_hist, background_edges = torch.histogram(
                background,
                bins=100,
                range=[0,1], density=True
            )
            # get the cumulative distributions
            dx = signal_edges[1] - signal_edges[0]
            tpr = torch.tensor([1.0])
            fpr = torch.tensor([0.0])

            tpr = torch.cat((tpr, 1.0 - (torch.cumsum(signal_hist, dim=0) * dx)), dim=0)
            fpr = torch.cat((fpr, (torch.cumsum(background_hist, dim=0) * dx)), dim=0)

            tpr = torch.cat((tpr, torch.tensor([0.0])),dim=0)
            fpr = torch.cat((fpr, torch.tensor([1.0])),dim=0)
            
            class_auc = auc(1.0 - fpr.cpu().numpy(), tpr.cpu().numpy())

            axs.plot(
                tpr.cpu().numpy(), fpr.cpu().numpy(),
                linestyle='--',
                label=f"AUC: {class_auc:.2f}"
            )
            axs.set_xlabel("Signal Acceptance (TPR)")
            axs.set_ylabel("Background Rejection (TNR)")
            plt.suptitle(f"Inference ROC curve (TNR vs TPR) for class {outer_label}")
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f"plots/roc/inference_roc_{outer_label}.png")
            plt.close()
        
        # generate summed ADC plots for each signal acceptance value
        for ii, outer_label in enumerate(self.metric.labels[input]):
            fig, axs = plt.subplots()
            temp_train_probs = probabilities[:,ii]
            temp_train_labels = probabilities[:,self.metric.num_classes[kk]]
            class_summed_adc = summed_adc[(temp_train_labels == ii)]
            num_backgrounds = len(temp_train_labels[(temp_train_labels != ii)])

            axs.hist(
                class_summed_adc.squeeze(1).cpu().numpy(),
                bins=100,
                histtype='step',
                stacked=True,
                label=f'{outer_label}'
            )
                
            for acceptance in self.sig_acceptance:
                signal_summed_adc = summed_adc[(temp_train_probs > acceptance)].squeeze(1).cpu().numpy()
                accepted_labels = temp_train_labels[(temp_train_probs > acceptance)]
                accepted_backgrounds = len(accepted_labels[(accepted_labels != ii)])
                background_leakage = accepted_backgrounds / num_backgrounds

                axs.hist(
                    signal_summed_adc,
                    bins=100,
                    histtype='step',
                    stacked=True,
                    label=r'p($\theta$='+f'{outer_label}) > {acceptance:.2f}\nleakage = {background_leakage:.4f}'
                )
            
            axs.set_xlabel(r"$\Sigma $" + "ADC")
            axs.set_ylabel("Counts")
            plt.suptitle(r"Inference $\Sigma $" + f"ADC for class {outer_label}")
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f"plots/summed_adc/inferece_summed_adc_{outer_label}.png")
            plt.close()


