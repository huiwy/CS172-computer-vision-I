import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 
import os
import random
import scipy.sparse
import sklearn.svm as svm

# the width of the rescaled image
w_size = 256

def read_image(f):
  """
  Read the image at a certain directory regardless of the format and convert it into grey scale.
  
  Parameters
  ----------
  f : str
    The directory of the image.
  
  Returns
  -------
  numpy.array
    The array holding the greyscale image at f.
  """
  img = mpimg.imread(f)
  t = img.shape
  factor = w_size / t[0]
  img = cv.resize(img, (int(factor*t[1]), int(factor*t[0])))
  if len(img.shape) == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  return img
            
def show_image(img, label):
  """
  Print the image and show its label
  
  Parameters 
  ----------
  img : numpy.array
    An array contains a single image.
    
  label : str
    The label of img.
  """
  plt.imshow(img, interpolation='nearest')
  plt.title(label)
  plt.show()

class Dataset_old:
  def __init__(self, repo, drop_rate):
    # all training and testing data stored in data_X and data_y
    self.data_X = []
    self.data_y = []
    self.train = []
    self.test = []
    self.labels = []
    
    folders = os.listdir(repo)
    labels = [i.split(".")[1]  for i in folders]
    for i in range(len(folders)):
      current_folder = repo + folders[i] + '/'
      image_names = os.listdir(current_folder)
      self.labels.append(folders[i])
      for j in image_names:
        if(random.random() > drop_rate):
          image = read_image(current_folder + j)
          self.data_X.append(image)
          self.data_y.append(i)

  def show_image(self, index = None, num = 5):
    if index == None:
      shown = np.random.choice(len(self.data_X), num)
      for i in shown:
        show_image(self.data_X[i], self.data_y[i])
    else:
      show_image(self.data_X[index], self.data_y[index])
    
  def get_data_X(self):
    return self.data_X
  
  def seperate_data(self, num_test = 300):
    self.test = list(np.random.choice(len(self.data_X), num_test, replace = False))
    self.train = list(set(range(len(self.data_X)))-set(self.test))
  
class Dataset:
  """
  A class hold the dataset.  
  """
  def __init__(self, repo, samples = 70, no_clutter = True, feature_function = None, pyramid = 0, verbose = False):
    """
    Parameters
    ----------
    repo : str
      The directory of the raw dataset.
      
    samples : int
      The number of samples to read for each class. (default 70)
      
    no_clutter : boolean
      Whether not to read the clutter class. (default True)
      
    feature_function: function(image : numpy.array) -> feature : numpy.array
      The function used to extract the features of an image. (default None)
      None means directly read the image.
    
    pyramid : int
      The depth of the spatial pyramid. default by 1.    
    """
    # all training and testing data stored in data_X for better flexibility
    self.data_X = {}
    self.train_X = []
    self.train_y = []    
    self.validation_X = []
    self.validation_y = []
    self.test_X = []
    self.test_y = []
    self.labels = []
    
    
    folders = os.listdir(repo)
    labels = [i.split(".")[1]  for i in folders]
    
    for i in range(len(folders)):
      current_folder = repo + folders[i] + '/'
      image_names = os.listdir(current_folder)
      self.labels.append(folders[i])
      tmp = np.array([])
      if verbose:
        print(folders[i], 'is loaded.')
      # exclude the clutter class
      if i == 256 and no_clutter:
        continue
        
      t = 0
      for j in image_names:
        image = read_image(current_folder + j)
        if feature_function == None:
          tmp = np.vstack([tmp, image]) if tmp.size else image
        else:
          try:
            added = spatial_pyramid(feature_function,
                              pyramid, image)
          except ValueError as e:
            continue
          tmp = np.vstack([tmp, added]) if tmp.size else added
          
          t += 1
          
          if t >= samples:
            break
      self.data_X[i] = tmp
          
  def show_image(self, index = None, num = 5):
    if index == None:
      # randomly show some images in the test set
      shown = np.random.choice(len(self.test_X), num)
      for i in shown:
        show_image(self.test_X[i], self.test_y[i])
    else:
      show_image(self.test_X[index], self.test_y[index])
  
  def generate_train_test_samples(self, train_number = 60):
    """
    Separate the samples into train set, validation set and test set.
    The size of validation set and test set for each class is 5 by default.
    
    Parameters
    ----------
    train_number : int
      Number of samples contained in the train set each class.
    """
    for i, j in self.data_X.items():
      train_choices = np.random.choice(j.shape[0], train_number, replace = False)
      self.train_X.append(j[train_choices])
      self.train_y.append(np.ones([train_number]) * i)

      validation_choices = list(set(range(j.shape[0])) - set(train_choices))[:5]
      self.validation_X.append(j[validation_choices])
      self.validation_y.append(np.ones([len(validation_choices)]) * i)

      not_choices = list(set(range(j.shape[0])) - set(train_choices) - \
                         set(validation_choices))[:5]
      self.test_X.append(j[not_choices])
      self.test_y.append(np.ones([len(not_choices)]) * i)
  
  def get_train_X(self):
    """
    Reture the train set.
    """
    return self.train_X
  
  def train(self, model, scaler, train_number = 15):
    """
    Train on the train set using a model and a scaler.
    
    Parameters
    ----------
    model : sklearn.estimator
      The estimator model to be trained
      
    scaler : sklearn.scaler
      The scaler to be trained
      
    train_number : int
      The size of train set for each class. (default by 15)
      
    Return
    ------
    float
      The accuracy
    """
    X_t = self.train_X[0][:train_number, :]
    y_t = self.train_y[0][:train_number]
    for i in range(1, len(self.train_X)):
      X_t = np.vstack((X_t, self.train_X[i][:train_number, :]))
      y_t = np.concatenate((y_t, self.train_y[i][:train_number]))
    if scaler:
      scaler.fit(X_t)
      X_t = scaler.transform(X_t)
      
    model.fit(X_t, y_t)
    return sum(model.predict(X_t) == y_t)/X_t.shape[0]
  
  def test(self, model, scaler, validation = True):
    """
    Test the performance of a given model on a validation set or test set.
    
    Parameters
    ----------
    model : sklearn.estimator
      A trained model to be tested.
    
    scaler : sklearn.scaler
      A accompanying scaler.
      
    validation : boolean
      Whether to test on the validation set or test set. (default True)
    
    Returns
    -------
    float
      The accuracy
    """
    if validation: 
      X_tmp = self.validation_X
      y_tmp = self.validation_y
    else:
      X_tmp = self.test_X
      y_tmp = self.test_y
      
    X_t = X_tmp[0][:5, :]
    y_t = y_tmp[0][:5]
    for i in range(1, len(X_tmp)):
      X_t = np.vstack((X_t, X_tmp[i][:5, :]))
      y_t = np.concatenate((y_t, y_tmp[i][:5]))
      
    if scaler:
      X_t = scaler.transform(X_t)
    return sum(model.predict(X_t) == y_t)/X_t.shape[0]
    
def spatial_pyramid(feature_function, layer, image):
  ret = feature_function(image)
  for k in range(1, layer):
    t = 2 ** k
    xx = image.shape[0] // t
    yy = image.shape[1] // t
    for i in range(t):
      for j in range(t):
        ret = np.hstack((ret, feature_function(image[i*xx:(i+1)*xx, j*yy:(j+1)*yy]))) 
  return ret

class HistIntersectionModel:
  def __init__(self, C = 1, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False,
                 random_state=None, **params):
    
    self.model = svm.SVC(kernel = 'precomputed', C = C , shrinking = shrinking, 
                         probability = probability, tol = tol , cache_size = \
                         cache_size,  class_weight = class_weight, 
                         verbose = verbose,  max_iter = max_iter, 
                         decision_function_shape = decision_function_shape,
                         break_ties = break_ties,
                         random_state = random_state)
    self.intersected = None
    
  def fit(self, X, y):
    self.intersected = X
    K = hist_intersection(X, X)
    self.model.fit(K, y)
  
  def predict(self, X):
    K = hist_intersection(X, self.intersected)
    return self.model.predict(K)
  
  def get_params(self, deep=True):
    return self.model.get_params(deep)
  
  def decision_function(self, X):
   K = hist_intersection(X, self.intersected)
   return self.model.decision_function(K)
  
  def predict_proba(self, X):
    K = hist_intersection(X, self.intersected)
    return self.model.predict_proba(K)
    
    
def hist_intersection(X, Y):
  kernel = np.zeros((X.shape[0], Y.shape[0]))

  for i in range(Y.shape[1]):
      c1 = Y[:, i].reshape(-1, 1)
      c2 = Y[:, i].reshape(-1, 1)
      kernel += np.minimum(c1, c2.T)
  return kernel
