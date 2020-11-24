import numpy as np
import torch

def evaluate(model, testdata, device):
  model.eval()
  predictions = np.array([])
  gt = np.array([])
  with torch.no_grad():
    for image, depth in testdata:
      output = model(torch.tensor(image).view([1, 3, 480, 640]).float().cuda()).cpu().numpy()
      predictions = np.vstack((predictions, output.reshape((1, -1))))\
                    if predictions.size else output.reshape((1, -1))
      gt = np.vstack((gt, depth.reshape((1, -1))))\
          if gt.size else depth.reshape((1, -1))

  thresh = np.maximum((gt / predictions), (predictions / gt))
  a1 = (thresh < 1.25   ).mean()
  a2 = (thresh < 1.25 ** 2).mean()
  a3 = (thresh < 1.25 ** 3).mean()
  abs_rel = np.mean(np.abs(gt - predictions) / gt)
  rmse = (gt - predictions) ** 2
  rmse = np.sqrt(rmse.mean())
  log_10 = (np.abs(np.log10(gt)-np.log10(predictions))).mean()
  return a1, a2, a3, abs_rel, rmse, log_10