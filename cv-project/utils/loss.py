import torch
import torch.nn as nn

class DirectionLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.cosine_sim = nn.CosineSimilarity(eps=1e-6)

  def forward(self, eye_position, gt_position, pred_dir):
    gt_dir = gt_position - eye_position
    return (1 - self.cosine_sim(gt_dir, pred_dir)).mean()

class BiasedBCELoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.loss = nn.BCELoss()

  def forward(self, pred, gt):
    l = self.loss(pred, gt)
    l = l * (gt + 0.1)
    return l.mean()