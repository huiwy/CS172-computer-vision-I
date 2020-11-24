import torch
import torch.nn as nn
import torch.nn.functional as F

class GradLoss(nn.Module):
    def __init__(self):
      super(GradLoss, self).__init__()
    def forward(self, output, ground_truth):
      grad_out = imgrad(output)
      grad_gt = imgrad(ground_truth)
      return torch.mean(torch.abs(grad_out - grad_gt))

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