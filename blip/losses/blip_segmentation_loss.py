"""
Wrapper for BlipSegmentation loss
"""
import numpy as np
import torch
import torch.nn as nn

from blip.losses.generic_loss import GenericLoss


class BlipSegmentationLoss(GenericLoss):
    """
    """
    def __init__(
        self,
        name:           str = 'cross_entropy_loss',
        alpha:          float = 1.0,
        meta:           dict = {}
    ):
        super(BlipSegmentationLoss, self).__init__(
            name, alpha, meta
        )
        self.reduction = 'mean'
        self.cross_entropy_loss = {
            'topology': nn.CrossEntropyLoss(
                reduction=self.reduction,
                weight=self.meta['dataset'].class_weights['topology'].to(self.device)
            ),
            'physics': nn.CrossEntropyLoss(
                reduction=self.reduction,
                weight=self.meta['dataset'].class_weights['physics'].to(self.device)
            ),
            'vertex': nn.BCEWithLogitsLoss(
                reduction=self.reduction,
                pos_weight=(
                    self.meta['dataset'].class_weights['vertex'][1]
                    / self.meta['dataset'].class_weights['vertex'][0]
                ).to(self.device)
            ),
            'tracklette_begin': nn.BCEWithLogitsLoss(
                reduction=self.reduction,
                pos_weight=(
                    self.meta['dataset'].class_weights['tracklette_begin'][1]
                    / self.meta['dataset'].class_weights['tracklette_begin'][0]
                ).to(self.device)
            ),
            'tracklette_end': nn.BCEWithLogitsLoss(
                reduction=self.reduction,
                pos_weight=(
                    self.meta['dataset'].class_weights['tracklette_end'][1]
                    / self.meta['dataset'].class_weights['tracklette_end'][0]
                ).to(self.device)
            ),
            'fragment_begin': nn.BCEWithLogitsLoss(
                reduction=self.reduction,
                pos_weight=(
                    self.meta['dataset'].class_weights['fragment_begin'][1]
                    / self.meta['dataset'].class_weights['fragment_begin'][0]
                ).to(self.device)
            ),
            'fragment_end': nn.BCEWithLogitsLoss(
                reduction=self.reduction,
                pos_weight=(
                    self.meta['dataset'].class_weights['fragment_end'][1]
                    / self.meta['dataset'].class_weights['fragment_end'][0]
                ).to(self.device)
            ),
            'shower_begin': nn.BCEWithLogitsLoss(
                reduction=self.reduction,
                pos_weight=(
                    self.meta['dataset'].class_weights['shower_begin'][1]
                    / self.meta['dataset'].class_weights['shower_begin'][0]
                ).to(self.device)
            ),
        }

    def loss(
        self,
        data
    ):
        """Computes and returns/saves loss information"""
        loss = 0
        for ii, output in enumerate(self.meta['dataset'].labels):
            if output in ['topology', 'physics']:
                """We have to convert cross entropy labels to type long"""
                temp_loss = self.alpha * self.cross_entropy_loss[output](
                    data['outputs'][output].to(self.device),
                    data['labels'].squeeze(0)[:, ii].long().to(self.device)
                )
            else:
                """But for binary with logits we want both to be floats (weird)"""
                temp_loss = self.alpha * self.cross_entropy_loss[output](
                    data['outputs'][output].squeeze(1).to(self.device),
                    data['labels'].squeeze(0)[:, ii].float().to(self.device)
                )
            loss += temp_loss
            self.batch_loss[output] = torch.cat(
                (self.batch_loss[output], torch.tensor([[temp_loss]], device=self.device)), dim=0
            )
        return loss
