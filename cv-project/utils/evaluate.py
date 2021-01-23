import torch

from .metrics import *
from .loss import *

from collections import defaultdict

lam = 0.5

def evaluate(model, loader, device):
  # model.eval()

  dir_loss = DirectionLoss()
  bce_loss = nn.BCELoss()
  v_len = len(loader)
  val_loss = 0

  dist_per_image = defaultdict(list)
  angle_per_image = defaultdict(list)
  m_angle_per_image = defaultdict(list)


  with torch.no_grad():
    for _, data in enumerate(loader):
      head_image, image, head_pos, gaze_field, heatmap, gt_gaze, img_id, _ = data
          
      b,w,h = heatmap.shape

      head_image = head_image.float().to(device)
      image = image.float().to(device)
      head_pos = head_pos.float().to(device)
      gaze_field = gaze_field.float().to(device)
      gt_gaze = gt_gaze.float().to(device)
      heatmap = heatmap.view((b, 1, w, h)).float().to(device)

      pred_heatmap, pred_dir = model(head_image, image, head_pos, gaze_field)

      loss = lam * bce_loss(pred_heatmap, heatmap)
      
      val_loss += loss.item()

      dist, angle, m_angle = metrics(head_pos, pred_dir, gt_gaze, pred_heatmap)

      for i in range(img_id.shape[0]):
        idx = img_id[i].item()
        dist_per_image[idx].append(dist[i].item())
        angle_per_image[idx].append(angle[i].item())
        m_angle_per_image[idx].append(m_angle[i].item())

    dist, angle, m_angle = 0, 0, 0

    for id in dist_per_image:
      dist += min(dist_per_image[id])
      angle += min(angle_per_image[id])
      m_angle += min(m_angle_per_image[id])

    return val_loss/v_len, dist/len(dist_per_image), m_angle/len(dist_per_image), angle/len(dist_per_image)
