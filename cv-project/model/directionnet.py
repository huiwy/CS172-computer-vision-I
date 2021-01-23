import torch
import torch.nn as nn

from .resnet50 import ResNet50BackBone

class DirectionNet(nn.Module):
  def __init__(self, pretrained = True):
    super().__init__()
    self.headpos = nn.Sequential(nn.Linear(2, 256),
                                 nn.ReLU(inplace = True))
    self.restnetbackbone = ResNet50BackBone(pretrained)
    self.resnet50_fc = nn.Sequential(nn.Linear(2048, 512),
                                     nn.ReLU(inplace=True))

    self.fc_last = nn.Sequential(nn.Linear(768, 256), nn.ReLU(inplace=True))
    self.dir_regressor = nn.Linear(256, 2)
    self.gamma_regressor = nn.Linear(256, 1)


  def forward(self, head_image, pos):
    x = self.restnetbackbone(head_image).reshape(-1, 2048)

    c1 = self.resnet50_fc(x)
    c2 = self.headpos(pos)
    
    f = torch.cat((c1, c2), dim=1)
    f = self.fc_last(f)
    return self.dir_regressor(f), self.gamma_regressor(f)