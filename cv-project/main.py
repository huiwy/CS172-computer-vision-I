from utils import *
from model import *

import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.utils.data import DataLoader

if __name__ == "__main__":
  device = "cuda"

  model = GazeNet(1, gamma_regression=False).to(device)

  model = nn.DataParallel(model.to(device), 
          device_ids = list(range(torch.cuda.device_count())))

  trainset = GazeFollowDataset(mode = "train")
  valset = GazeFollowDataset(mode = "validation")
  trainloader = DataLoader(trainset, batch_size=48, num_workers=32, drop_last=True)
  valloader = DataLoader(valset, batch_size=64, num_workers=32, drop_last=True)
  try:
    train(model, trainloader, valloader, 1000, device)
  except KeyboardInterrupt:
    torch.save(model, "gaze_.torch")

  torch.save(model, "gaze_.torch")