import torch

def train(model, trainloader, validationloader, epoches, optimizer, criterion, device):
  best_val = 0
  val_l = len(validationloader)
  train_l = len(trainloader)

  for epoch in range(epoches):
    model.train()
    train_loss = 0.0
    for i, data in enumerate(trainloader):
      image, depth = data

      optimizer.zero_grad()
      
      size = [depth.size(0), 1, depth.size(1), depth.size(2)]
      output = model(image.float().to(device))
      loss = criterion(output, depth.view(size).float().to(device))
      loss.backward()
      optimizer.step()

      train_loss += loss.item()

      if i % 10 == 9:
        print(i)
    
    val_loss = 0.0
    model.eval()
    for _, data in enumerate(validationloader):
      with torch.no_grad():
        image, depth = data
        size = [depth.size(0), 1, depth.size(1), depth.size(2)]
        output = model(image.float().to(device))
        loss = criterion(output, depth.view(size).float().to(device))
        val_loss += loss.item()
        
    print('[%d, %5d] train loss: %.3f  validation loss: %.3f' %
        (epoch + 1, train_l, train_loss / train_l, val_loss / val_l))
  print("Train Finished.")