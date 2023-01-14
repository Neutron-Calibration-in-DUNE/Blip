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

from blip.metrics.savers import *
from blip.utils.callbacks import GenericCallback
from blip.utils import utils

class ConfusionMatrixCallback(GenericCallback):
    """
    """
    def __init__(self,
        metrics_list,
    ):  
        super(ConfusionMatrixCallback, self).__init__()
        self.metrics_list = metrics_list
        if "confusion_matrix" in self.metrics_list.metrics.keys():
            self.metric = self.metrics_list.metrics["confusion_matrix"]
        if not os.path.isdir("plots/confusion_matrix/"):
            os.makedirs("plots/confusion_matrix/")
        
        self.training_probabilities = None
        self.validation_probabilities = None
        self.test_probabilities = None

        self.training_confusion = None
        self.validation_confusion = None
        self.test_confusion = None

    def reset_batch(self):
        self.training_probabilities = None
        self.validation_probabilities = None
        self.test_probabilities = None

        self.training_confusion = None
        self.validation_confusion = None
        self.test_confusion = None

    def evaluate_epoch(self,
        train_type='training'
    ):  
        if train_type == "training":
            self.training_probabilities = self.metric.batch_probabilities
            self.training_confusion = self.metric.compute()
        elif train_type == "validation":
            self.validation_probabilities = self.metric.batch_probabilities
            self.validation_confusion = self.metric.compute()
        else:
            self.test_probabilities = self.metric.batch_probabilities
            self.test_confusion = self.metric.compute()
        self.metric.reset_probabilities()

    def evaluate_training(self):
        # plot the training confusion matrix
        training_display = ConfusionMatrixDisplay(
            self.training_confusion.cpu().numpy(),
            display_labels = self.metrics_list.labels
        ) 
        training_display.plot()       
        plt.suptitle("Training Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix/training_confusion_matrix.png")
        plt.close()

        validation_display = ConfusionMatrixDisplay(
            self.validation_confusion.cpu().numpy(),
            display_labels = self.metrics_list.labels
        ) 
        validation_display.plot()       
        plt.suptitle("Validation Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix/validation_confusion_matrix.png")
        plt.close()

        # plot statistics on categorical probabilities
        for outer_label, ii in self.metrics_list.labels.items():
            fig, axs = plt.subplots(figsize=(10,6))
            temp_train_probs = self.training_probabilities[:,ii]
            temp_train_labels = self.training_probabilities[:,self.metric.num_classes]
            for inner_label, jj in self.metrics_list.labels.items():
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
            plt.suptitle(f"Training probability predictions for class {outer_label}")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/training_probabilities_{outer_label}.png")
            plt.close()
        
        # plot statistics on categorical probabilities
        for outer_label, ii in self.metrics_list.labels.items():
            fig, axs = plt.subplots(figsize=(10,6))
            temp_train_probs = self.validation_probabilities[:,ii]
            temp_train_labels = self.validation_probabilities[:,self.metric.num_classes]
            for inner_label, jj in self.metrics_list.labels.items():
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
            plt.suptitle(f"Validation probability predictions for class {outer_label}")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/validation_probabilities_{outer_label}.png")
            plt.close()

            
    def evaluate_testing(self):  
        # plot the training confusion matrix
        test_display = ConfusionMatrixDisplay(
            self.test_confusion.cpu().numpy(),
            display_labels = self.metrics_list.labels
        ) 
        test_display.plot()       
        plt.suptitle("Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix/test_confusion_matrix.png")
        plt.close()

        # plot statistics on categorical probabilities
        for outer_label, ii in self.metrics_list.labels.items():
            fig, axs = plt.subplots(figsize=(10,6))
            temp_train_probs = self.test_probabilities[:,ii]
            temp_train_labels = self.test_probabilities[:,self.metric.num_classes]
            for inner_label, jj in self.metrics_list.labels.items():
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
            plt.suptitle(f"Test probability predictions for class {outer_label}")
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/test_probabilities_{outer_label}.png")
            plt.close()

    def evaluate_inference(self):
        confusion = self.metric.compute()
        probabilities = self.metric.batch_probabilities

        # plot the training confusion matrix
        display = ConfusionMatrixDisplay(
            confusion.cpu().numpy(),
            display_labels = self.metrics_list.labels
        ) 
        display.plot()       
        plt.suptitle("Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"plots/confusion_matrix/inference_confusion_matrix.png")
        plt.close()

        # plot statistics on categorical probabilities
        for outer_label, ii in self.metrics_list.labels.items():
            fig, axs = plt.subplots(figsize=(10,6))
            temp_train_probs = probabilities[:,ii]
            temp_train_labels = probabilities[:,self.metric.num_classes]
            for inner_label, jj in self.metrics_list.labels.items():
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
        for outer_label, ii in self.metrics_list.labels.items():
            fig, axs = plt.subplots(figsize=(10,6))
            temp_train_probs = probabilities[:,ii]
            temp_train_labels = probabilities[:,self.metric.num_classes]

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
            fpr = torch.tensor([1.0])
            tpr = torch.cat((tpr, 1.0 - (torch.cumsum(signal_hist, dim=0) * dx)), dim=0)
            fpr = torch.cat((fpr, 1.0 - (torch.cumsum(background_hist, dim=0) * dx)), dim=0)
            tpr = torch.cat((tpr, torch.tensor([0.0])),dim=0)
            fpr = torch.cat((fpr, torch.tensor([0.0])),dim=0)
            print(tpr)
            print(fpr)
            
            class_auc = auc(fpr.cpu().numpy(), tpr.cpu().numpy())

            axs.plot(
                fpr.cpu().numpy(), tpr.cpu().numpy(),
                linestyle='--',
                label=f"AUC: {class_auc:.2f}"
            )
            axs.set_xlabel("False positive rate")
            axs.set_ylabel("True positive rate")
            plt.suptitle(f"Inference ROC curve for class {outer_label}")
            plt.grid(True)
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix/inference_roc_{outer_label}.png")
            plt.close()


