import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from .loss import *
from .evaluate import evaluate

import numpy as np
import logging
import os

lam = 0.5

def train(model, trainloader, validationloader, 
  eval_every, device):

  log_dir = 'log/'
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  log_file = log_dir + 'train.log'

  logging.basicConfig(level=logging.INFO,
                      format='%(levelname)s: %(message)s',
                      filename=log_file,
                      filemode='w')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logging.getLogger('').addHandler(console)
    
  dir_loss = DirectionLoss()
  bce_loss = nn.BCELoss()
  counter = 0

  learning_rate = 0.0001
  optimizer_s1 = optim.Adam([{'params': model.module.directionnet.parameters(), 
                                  'initial_lr': learning_rate}],
                                lr=learning_rate, weight_decay=0.0001)
  optimizer_s2 = optim.Adam([{'params': model.module.heatmap.parameters(),
                              'initial_lr': learning_rate}],
                            lr=learning_rate, weight_decay=0.0001)

  optimizer_s3 = optim.Adam([{'params': model.parameters(), 'initial_lr': learning_rate}],
                        lr=learning_rate*0.1, weight_decay=0.0001)

  lr_scheduler_s1 = optim.lr_scheduler.StepLR(optimizer_s1, step_size=5, gamma=0.1, last_epoch=-1)
  lr_scheduler_s2 = optim.lr_scheduler.StepLR(optimizer_s2, step_size=5, gamma=0.1, last_epoch=-1)
  lr_scheduler_s3 = optim.lr_scheduler.StepLR(optimizer_s3, step_size=5, gamma=0.1, last_epoch=-1)


  for epoch in range(30):
    logging.info('epoch: %s'%(str(epoch)))
    if epoch == 0:
      lr_scheduler = lr_scheduler_s1
      optimizer = optimizer_s1
      criterion = lambda pred_heatmap, heatmap, pred_dir, head_pos, gt_gaze: \
                    dir_loss(head_pos, gt_gaze, pred_dir)
    elif epoch == 8:
      lr_scheduler = lr_scheduler_s2
      optimizer = optimizer_s2
      criterion = lambda pred_heatmap, heatmap, pred_dir, head_pos, gt_gaze: \
                    bce_loss(pred_heatmap, heatmap)
    elif epoch == 19:
      lr_scheduler = lr_scheduler_s3
      optimizer = optimizer_s3
      criterion = lambda pred_heatmap, heatmap, pred_dir, head_pos, gt_gaze: \
                    lam * bce_loss(pred_heatmap, heatmap) + dir_loss(head_pos, gt_gaze, pred_dir)

    model.train()
    # reset the stats
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader)):
    
      head_image, image, head_pos, gaze_field, heatmap, gt_gaze, _, _ = data
    
      optimizer.zero_grad()

      b,w,h = heatmap.shape

      head_image = head_image.float().to(device)
      image = image.float().to(device)
      head_pos = head_pos.float().to(device)
      gaze_field = gaze_field.float().to(device)
      gt_gaze = gt_gaze.float().to(device)
      heatmap = heatmap.view((b, 1, w, h)).float().to(device)

      pred_heatmap, pred_dir = model(head_image, image, head_pos, gaze_field)
      
      loss = criterion(pred_heatmap, heatmap, pred_dir, head_pos, gt_gaze)
      loss.backward()

      optimizer.step()

      running_loss += loss.item()

      if i % eval_every == eval_every - 1:
        logging.info('average train loss at %s : %s'%(str(counter), str(running_loss/eval_every)))

        test(model, validationloader, device, counter)

        running_loss = 0
        counter += 1

    save_path = '../model/checkpoint'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.module.state_dict(), save_path+'/model_epoch{}.pkl'.format(epoch))

    test(model, validationloader, device, counter)

    lr_scheduler.step()

  print('Finished Training')

def test(model, validationloader, device, counter):
  val_loss, dist, m_angle, angle = evaluate(model, validationloader, device)
  logging.info('average val loss at %s : %s'%(str(counter), str(val_loss)))
  logging.info('average val dist at %s : %s'%(str(counter), str(dist)))
  logging.info('average val m_angle at %s : %s'%(str(counter), str(m_angle)))
  logging.info('average val angle at %s : %s'%(str(counter), str(angle)))

