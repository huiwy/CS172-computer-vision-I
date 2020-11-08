import numpy as np
from torch.utils.data import Dataset

class NYU_Depth(Dataset):
  def __init__(self, mode = 'train', repo = 'dataset/', transform = None):
    if mode in ['train', 'test', 'validation']:
      prefix = mode
    else:
      raise NotImplementedError('No such data', mode)
    self.images = np.load(repo + prefix + '_image.npy')
    self.depths = np.load(repo + prefix + '_depth.npy')
    self.transform = transform

  def __len__(self):
    return self.images.shape[0]

  def __getitem__(self, index):
    image = self.images[index].astype('float') / 256
    depth = self.depths[index].astype('float')

    if self.transform:
      image = self.transform(image)
      depth = self.transform(depth)

    return [image.transpose([2, 0, 1]), depth]


class Rescale(object):
  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, image):
    import skimage.transform
    img = skimage.transform.resize(image, self.output_size)

    return img