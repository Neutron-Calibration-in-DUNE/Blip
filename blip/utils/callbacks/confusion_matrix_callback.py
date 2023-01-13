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
        
        self.training_confusion = None
        self.validation_confusion = None
        self.test_confusion = None

    def reset_batch(self):
        self.training_confusion = None
        self.validation_confusion = None
        self.test_confusion = None

    def evaluate_epoch(self,
        train_type='training'
    ):  
        if train_type == "training":
            self.training_confusion = self.metric.compute()
        elif train_type == "validation":
            self.validation_confusion = self.metric.compute()
        else:
            self.test_confusion = self.metric.compute()

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

    def evaluate_inference(self):
        pass