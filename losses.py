import torch
import torch.nn as nn
from focal_loss.focal_loss import FocalLoss

class ORGFocalLoss(nn.Module):
    def __init__(self, gamma=4, alpha=2):
        super(ORGFocalLoss, self).__init__()
        from focal_loss.focal_loss import FocalLoss
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction='mean')
    def forward(self, pred, gt):
        out = self.focal_loss(pred, gt)
        return out

class L1MaskLoss(nn.Module):
    def __init__(self):
        super(L1MaskLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='none')
    def forward(self, gt, pred, mask):
        out = mask*self.l1(pred, gt)
        return out.mean(axis=-1).mean(axis=-1).mean(axis=-1, keepdim=True)
class CenternetLoss(nn.Module):
    def __init__(self, size_factor=0.1, offset_factor=1):
        super(CenternetLoss, self).__init__()
        self.size_fac = size_factor
        self.offset_fac = offset_factor
        self.focal_loss = ORGFocalLoss(gamma=4, alpha=2)
        self.l1maskloss = L1MaskLoss()

    def forward(self, gt_list, pred_list, mask):
        offset_loss = self.l1maskloss(gt_list[1], pred_list[1], mask)
        size_loss = self.l1maskloss(gt_list[2], pred_list[2], mask)
        heat_loss = self.focal_loss(pred_list[0], gt_list[0])
        return heat_loss + self.offset_fac*offset_loss.mean() + self.size_fac*size_loss.mean(), heat_loss, offset_loss, size_loss
