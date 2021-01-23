import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms 

from skimage import io
from random import random

import re
import os
import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

color_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class VideoData(Dataset):
  def __init__(self, log = "log.txt", resolution = (320, 568)) -> None:
    f = open(log, "r")
    lines = f.readlines()

    l = len(lines)

    self.eye_pos = np.zeros([l, 2])
    self.image_names = []
    for i, line in enumerate(lines):
      splited = re.split(r':',line)
      self.image_names.append(splited[0].strip())
      splited = splited[1].replace('(','')
      splited = splited.replace(')','')
      splited = re.split(r',',splited)
      (x1,y1,x2,y2) = [int(i) for i in splited]
      y = (x1 + x2)/2 / resolution[0]
      x = (y1 + y2)/2 / resolution[1]

      self.eye_pos[i] = [x, y]
    f.close()

  def __len__(self):
    return len(self.image_names)
  
  def __getitem__(self, index):
    
    original_image = io.imread(self.image_names[index])
    eye_pos = self.eye_pos[index]
    head_image = generate_head_image(original_image, eye_pos)

    image, eye, head_image = default_transform(original_image.copy(), eye_pos, head_image)

    image = color_normalize(image)
    gaze_field = generate_pos_field(eye, (224, 224))

    return original_image, image, eye, head_image, gaze_field

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

def default_transform(image, eye, head_image):  
  image = TF.to_tensor(image)
  head_image = TF.to_tensor(head_image)
  head_image = TF.resize(head_image, (224, 224))
  image = TF.resize(image, (224, 224))
  return image, eye, head_image

def get_gaze_pos(net, head, image, eye, f):
  with torch.no_grad():
    f, d = net(head.unsqueeze(0).cuda(), image.unsqueeze(0).cuda(), torch.Tensor(eye).unsqueeze(0).cuda(), f.unsqueeze(0).cuda())
  max_idx = torch.argmax(f[0].view([-1]))
  x = max_idx % 56 / 56
  y = max_idx // 56 / 56

  return (x.cpu(), y.cpu())

def get_marked_frame(video, net, index):
  net.eval()

  original_image, image, eye, head, f = video[index]
  x, y = get_gaze_pos(net, head, image, eye, f)

  xx, yy = original_image.shape[:2]

  gaze = (int(x * yy), int(y * xx))
  start = int(eye[0] * yy), int(eye[1] * xx)

  thickness = 9 * yy // 1280
  color = (255, 191, 0)
  original_image = cv2.arrowedLine(original_image, start, gaze, color, thickness, tipLength=0.05)
  

  new_image = np.zeros_like(original_image)
  new_image[:,:,0] = original_image[:,:,2]
  new_image[:,:,1] = original_image[:,:,1]
  new_image[:,:,2] = original_image[:,:,0]
  cv2.imwrite("test_draw_%06d.png"%index, new_image)

  return original_image

# source code from mpl_dataset_tools
class ImageWriter:
    def __init__(self):
        self.save_id = 0

    def save(self, image, dir_path=None):
        name = format("%06d" % self.save_id)
        name += ".png"
        if dir_path is not None:
            name = os.path.join(dir_path, name)
        cv2.imwrite(name, image)
        self.save_id += 1
        
    def savePlt(self, image, dir_path=None):
        name = format("%06d" % self.save_id)
        name += ".png"
        if dir_path is not None:
            name = os.path.join(dir_path, name)
        plt.imsave(name, image)
        self.save_id += 1

def video2frame(video, image_dst_directory):
    cap = cv2.VideoCapture(video)  # load the video
    image_writer = ImageWriter()
    flag, frame = cap.read()  # read a image
    while flag:
        image_writer.save(frame, image_dst_directory)
        flag, frame = cap.read()
    cap.release()

class ImageDir:
    def __init__(self, path=None):
        self.dir_path = None
        self.image_names = None
        self.image_full_paths = None
        if path is not None:
            self.loadDir(path)

    def loadDir(self, path):
        self.dir_path = path
        self.image_names = sorted(os.listdir(path))
        self.image_full_paths = [
            os.path.join(self.dir_path, name) for name in self.image_names]

    def renameById(self, suffix="png"):
        num_images = len(self.image_names)
        name = []
        if suffix == "png":
            name = [format("%06d.png" % i) for i in range(num_images)]
        elif suffix == "jpg":
            name = [format("%06d.jpg" % i) for i in range(num_images)]
        for i in range(num_images):
            os.rename(self.image_full_paths[i], os.path.join(
                self.dir_path, name[i]))

    def renameByTimestamp(self, timestamps, suffix="png"):
        pass

    def frame2Video(self, output_file_path, rate):
        video_form = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # store as .mp4 form
        path_list = self.image_full_paths
        img_first = cv2.imread(path_list[0])
        height = img_first.shape[0]
        width = img_first.shape[1]
        size = (width, height)
        video = cv2.VideoWriter(output_file_path, video_form, rate, size)
        for current_path in path_list:
            img = cv2.imread(current_path)
            video.write(img)
        video.release()

def video2frame(video, image_dst_directory):
    cap = cv2.VideoCapture(video)  # load the video
    image_writer = ImageWriter()
    flag, frame = cap.read()  # read a image
    while flag:
        image_writer.save(frame, image_dst_directory)
        flag, frame = cap.read()
    cap.release()