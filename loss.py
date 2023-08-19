import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_Loss(nn.Module):
    def __init__(self, class_weight):
        super(L1_Loss, self).__init__()
        self.class_weight = torch.tensor(class_weight)

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        diff = torch.abs(probs - target)
        weight = self.class_weight.to(target.device).repeat((target.shape[0], 1))
        return torch.mean(torch.sum(diff * weight, dim=1))  # lay mean theo tung dong => mean theo example
    

class MyBCELoss(nn.Module):
    def __init__(self, class_weight):
        super(MyBCELoss, self).__init__()
        self.class_weight = class_weight


    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1).to(torch.float32)
        target = target.to(torch.float32)
        return F.binary_cross_entropy(probs, target, weight=torch.tensor(self.class_weight, device=target.device, dtype=torch.float32))
    

class MyCrossEntropyLoss(nn.Module):
    def __init__(self, class_weight):
        super(MyCrossEntropyLoss, self).__init__()
        self.class_weight = class_weight


    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=torch.tensor(self.class_weight, device=target.device))