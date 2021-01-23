import torch
import torch.nn as nn
import torch.nn.functional as F

from .heatmap import Deeplabv3p
from .directionnet import DirectionNet

class GazeNet(nn.Module):
  def __init__(self, heatmap_model, gamma_regression = True, gammas = [5, 2, 1], pretrained = True):
    super().__init__()
    self.directionnet = DirectionNet(pretrained)
    
    if gamma_regression:
      h_in = 4
    else:
      h_in = 6
      self.gamma1 = gammas[0]
      self.gamma2 = gammas[1]
      self.gamma3 = gammas[2]

    self.heatmap = Deeplabv3p(h_in, 1, 16, middle_blocks = 4)

    self.gamma_regression = gamma_regression

  def forward(self, head_image, image, head_pos, gaze_field):
    direction, gamma = self.directionnet(head_image, head_pos)

    direction = direction / torch.norm(direction, 2, dim=1).view([-1, 1])
    b, c, h, w = gaze_field.shape

    field = gaze_field.permute([0, 2, 3, 1]).contiguous()
    field = field.view([b, -1, 2])
    field = torch.matmul(field, direction.view([b, 2, 1]))
    field = field.view([b, h, w, 1])
    field = field.permute([0, 3, 1, 2]).contiguous()
    field = F.relu(field)
    
    if self.gamma_regression:
      gamma = 5 * torch.sigmoid(gamma)
      gamma = gamma.reshape((b, 1, 1, 1)).repeat(1, 1, w, h)
      field = torch.pow(field, gamma)
      heatmap = self.heatmap(torch.cat((image, field), dim = 1))
    else:
      field1 = field ** self.gamma1
      field2 = field ** self.gamma2
      field3 = field ** self.gamma3

      heatmap = self.heatmap(torch.cat((image, field1, field2, field3), dim = 1))

    return heatmap, direction