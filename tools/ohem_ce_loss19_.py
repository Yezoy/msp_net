#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    def __init__(self, thresh, *args, **kwargs, ):
        super(OhemCELoss, self).__init__()
        self.n_min = None
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()

        self.criteria = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    def forward(self, logits, labels):

        self.n_min = labels[labels != 255].numel() // 16
        # self.n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels.long())
        loss = loss.view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

        #return torch.mean(loss).item()


if __name__ == '__main__':
    pass
