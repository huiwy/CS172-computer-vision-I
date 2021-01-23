
import torch
import torch.nn as nn


def metrics(head_position, m_dir, gt_position, pred_heatmap):
  b,c,w,h = pred_heatmap.shape

  gaze_position = torch.zeros_like(gt_position)
  for i in range(b):
    max_idx = torch.argmax(pred_heatmap[i].view([-1]))
    gaze_position[i,0] = max_idx // w
    gaze_position[i,1] = max_idx % h
  gaze_position /= w

  dir = gaze_position - head_position
  gt_dir = gt_position - head_position

  cos = torch.zeros(b, device=dir.device)
  m_cos = torch.zeros(b, device=dir.device)

  for i in range(b):
    
    norm = (dir[i][0] **2 + dir[i][1] ** 2 ) ** 0.5
    norm_m = (m_dir[i][0] **2 + m_dir[i][1] ** 2 ) ** 0.5
    norm_gt = (gt_dir[i][0] **2 + gt_dir[i][1] ** 2 ) ** 0.5

    cos[i] = dir[i,0] * gt_dir[i, 0] + dir[i,1] * gt_dir[i, 1]
    cos[i] /= (norm * norm_gt + 1e-6)
    cos[i] = torch.clamp(cos[i], min = -1, max = 1)

    m_cos[i] = (m_dir[i][0]*gt_dir[i][0] + m_dir[i][1]*gt_dir[i][1]) / \
                        (norm_gt * norm_m + 1e-6)
    m_cos[i] = torch.clamp(m_cos[i], min = -1, max = 1)

  dist = (gaze_position - gt_position).norm(p=2, dim=1)

  # inner_product = (dir * gt_dir).sum(dim=1)
  # print(torch.isnan(inner_product).any())
  # dir_norm = torch.clamp(dir, min=1e-4).norm(p=2, dim=1) + 0.0001
  # gt_norm = torch.clamp(gt_dir, min=1e-4).norm(p=2, dim=1) + 0.0001
  # cos = inner_product / dir_norm / gt_norm
  # print(cos.max())
  
  angle = torch.acos(cos) / 3.1415 * 180
  m_angle = torch.acos(m_cos) / 3.1415 * 180

  # print(torch.isnan(angle).any(),"a")

  return dist, angle, m_angle
