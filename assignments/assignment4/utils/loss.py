import torch
import torch.nn as nn
import torch.nn.functional as F

class GradLoss(nn.Module):
    def __init__(self):
      super(GradLoss, self).__init__()
    def forward(self, pred, gt):
      grad_out = imgrad(pred)
      grad_gt = imgrad(gt)
      return torch.mean(torch.abs(grad_out - grad_gt))

class RMSELoss(nn.Module):
  def __init__(self, mode = "sum"):
    super(RMSELoss, self).__init__()
    if mode == "sum":
      self.reduce = torch.sum
    else:
      self.reduce = torch.mean
  def forward(self, pred, gt):
    r = (pred - gt) ** 2
    return self.reduce(torch.sqrt(torch.mean(r, (1, 2, 3))))

class MixedLoss(nn.Module):
  def __init__(self, grad = True, L1 = True):
    super(MixedLoss, self).__init__()
    self.grad = grad
    self.L1 = L1
    self.gradfunc = GradLoss()
    self.L1func = nn.L1Loss()

  def forward(self, output, ground_truth):
    loss = 0
    if self.grad:
      loss += self.gradfunc(output, ground_truth)
    if self.L1:
      loss += 0.1 * self.L1func(output, ground_truth)
    return loss

def imgrad(img):
  kernel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).float().view([1,1,3,3]).cuda()
  if img.is_cuda:
    kernel = kernel.cuda()
  G_x = F.conv2d(img, kernel)
  kernel = kernel.transpose(2, 3)
  G_y = F.conv2d(img, kernel)
  N,C,_,_ = img.size()
  return torch.cat((G_x.view(N,C,-1), G_y.view(N,C,-1)), dim=1)

