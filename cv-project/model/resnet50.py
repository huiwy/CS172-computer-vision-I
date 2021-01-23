import torch
import torch.nn as nn
import torchvision

class ResNet50BackBone(nn.Module):
  def __init__(self, pretrained = True):
    super().__init__()
    
    full_resnet50 = torchvision.models.resnet50(pretrained)
    self.backbone = nn.Sequential(*list(full_resnet50.children())[:-1])

  def forward(self, x):
    return self.backbone(x)