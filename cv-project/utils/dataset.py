import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms 

from skimage import io
from random import random

import numpy as np
from scipy import signal


def gkern(kernlen=51, std=9):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

KERNAL_SIZE = 15
GAUSSIAN_KERNEL = torch.Tensor(gkern(KERNAL_SIZE, 3))

color_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def default_transform(image, eye, gaze, head_image):
  # random horizontal flip
  # if random() > 0.5:
  #   head_image = head_image[:,::-1,:].copy()
  #   image = image[:,::-1,:].copy()
  #   eye = torch.Tensor((1-eye[0], eye[1]))
  #   gaze = torch.Tensor((1-gaze[0], gaze[1]))
  

  image = TF.to_tensor(image)
  head_image = TF.to_tensor(head_image)
  head_image = TF.resize(head_image, (224, 224))
  image = TF.resize(image, (224, 224))
  return image, eye, gaze, head_image


class GazeFollowDataset(Dataset):
  def __init__(self, mode = "train", transform = default_transform) -> None:
    super().__init__()
    if mode == "train":
      f = open("data_new/train_annotations.txt", "r")
      self.start = 0
      self.len = 120000
    elif mode == "validation":
      f = open("data_new/train_annotations.txt", "r")
      self.start = 120000
      # self.len = 5557
      self.len = 5557
    elif mode == "test":
      f = open("data_new/test_annotations.txt", "r")
      self.start = 0
      self.len = 15000
    else:
      raise NotImplementedError

    self.data = f.read().splitlines()
    self.transfrom = transform

  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    index = self.start + index
    img_dir, img_id, _, _, _, _, eye_x, eye_y, gaze_x, gaze_y, _, _ = \
        self.data[index].split(',')
    
    img_id = int(img_id.split('-')[0]) - 1

    original_image = io.imread('data_new/' + img_dir)
    eye = torch.Tensor((float(eye_x), float(eye_y)))
    gaze = torch.Tensor((float(gaze_x), float(gaze_y)))    
    image = io.imread('data_new/' + img_dir)
    if len(image.shape) == 2:
      image = np.repeat(image.reshape((*(image.shape), 1)), 3, axis = 2)
    head_image = generate_head_image(image, eye)

    if self.transfrom:
      image, eye, gaze, head_image = self.transfrom(image, eye, gaze, head_image)
    
    image = color_normalize(image)
    gaze_field = generate_pos_field(eye, (224, 224))

    heatmap = generate_heatmap(gaze, (224//4, 224//4))

    return head_image, image, eye, gaze_field, heatmap, gaze, img_id, original_image

def generate_pos_field(pos, size):
  h = torch.arange(size[0]).reshape((1, 1, -1)).repeat(1, size[1], 1) / size[0]
  w = torch.arange(size[1]).reshape((1, -1, 1)).repeat(1, 1, size[0]) / size[1]

  h -= pos[0]
  w -= pos[1]

  field = torch.vstack((h, w))
  field = field / torch.clip(torch.norm(field, dim=0), min = 0.01)
  return field


def generate_head_image(image, eye):
  x_c, y_c = eye
  x_0 = x_c - 0.15
  y_0 = y_c - 0.15
  x_1 = x_c + 0.15
  y_1 = y_c + 0.15
  x_0 = max(0, x_0)
  y_0 = max(0, y_0)
  x_1 = min(0.99, x_1)
  y_1 = min(0.99, y_1)
  h, w = image.shape[:2]
  head_image = image[int(y_0 * h):int(y_1 * h), int(x_0 * w):int(x_1 * w)]
  return head_image

def generate_heatmap(gaze, size):
  r = (KERNAL_SIZE - 1) // 2
  w, h = size
  x_c = int(gaze[1] * w) 
  y_c = int(gaze[0] * h)

  x_0 = x_c 
  y_0 = y_c
  x_1 = x_c + KERNAL_SIZE
  y_1 = y_c + KERNAL_SIZE

  heatmap = torch.zeros([w + KERNAL_SIZE, h + KERNAL_SIZE])
  heatmap[x_0:x_1, y_0:y_1] = GAUSSIAN_KERNEL
  return heatmap[r:-(1+r), r:-(1+r)].clone()
