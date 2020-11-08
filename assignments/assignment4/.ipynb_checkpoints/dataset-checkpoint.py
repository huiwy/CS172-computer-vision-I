import numpy as np
from torch.utils.data import Dataset

class NYU_Depth(Dataset):
  def __init__(self, mode = 'train', transform = None):
    if mode in ['train', 'test', 'validation']:
      prefix = mode
    else:
      raise NotImplementedError('No such data', mode)
    self.images = np.load('dataset/' + prefix + '_image.npy').transpose([0, 3, 1, 2])
    self.depths = np.load('dataset/' + prefix + '_depth.npy')
    self.transform = transform

  def __len__(self):
    return self.images.shape[0]

  def __getitem__(self, index):
    image = self.images[index].astype('float') / 256
    depth = self.depths[index].astype('float')

    if self.transform:
      image, depth = self.transform([image, depth])

    return [image, depth]