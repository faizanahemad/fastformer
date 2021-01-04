import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class AdMSoftmaxLoss(nn.Module):

    def __init__(self, ignore_index=-100, s=10.0, m=0.3):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.ignore_index = ignore_index

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        # x = F.normalize(x, dim=1)

        wf = x
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        correct_labels = (labels != self.ignore_index)
        len_correct_labels = torch.sum(correct_labels)
        L = correct_labels * L * (1 / len_correct_labels)
        return -torch.sum(L)
"""


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, ignore_index=-100, s=1.0, m=0.3):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.ignore_index = ignore_index
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        # x = F.normalize(x, dim=1)
        x = x.clone()
        x[list(range(len(labels))), labels] -= self.m
        x *= self.s
        return self.loss_ce(x, labels)
