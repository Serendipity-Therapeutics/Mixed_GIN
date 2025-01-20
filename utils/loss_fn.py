import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class BalancedBCELoss(nn.Module):
    def __init__(self):
        super(BalancedBCELoss, self).__init__()

    def forward(self, inputs, targets):
        positive_count = (targets == 1).sum().float()
        negative_count = (targets == 0).sum().float()
        
        pos_weight = negative_count / (positive_count + 1e-6) 

        weight_tensor = torch.where(targets == 1, pos_weight, torch.tensor(1.0, device=targets.device))

        loss = F.binary_cross_entropy(inputs, targets, weight=weight_tensor, reduction='mean')
        return loss