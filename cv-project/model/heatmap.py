import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import l1_loss

inplace = False
class Deeplabv3pDecoder(nn.Module):
  def __init__(self, low_level_channels, num_classes):
    super(Deeplabv3pDecoder, self).__init__()
    self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
    self.bn1 = nn.BatchNorm2d(48)
    self.relu = nn.ReLU(inplace) 

    self.conv2 = nn.Conv2d(48+256, 256, 3, padding=1)
    self.bn2 = nn.BatchNorm2d(256)
    self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(256)
    self.dropout = nn.Dropout(0.1)
    self.conv4 = nn.Conv2d(256, num_classes, 1)

  def forward(self, x, l):
    x = F.interpolate(x, (l.size(2), l.size(3)), mode='bilinear', align_corners=True)
    l = self.relu(self.bn1(self.conv1(l)))
    x = torch.cat((x, l), dim=1)
    x = self.relu(self.bn2(self.conv2(x)))
    x = self.relu(self.bn3(self.conv3(x)))
    x = self.dropout(x)
    return self.conv4(x)

class Deeplabv3p(nn.Module):
  def __init__(self, in_channels = 3, num_classes = 1, output_stride = 16, backbone = 'Xception', middle_blocks = 4):
    super(Deeplabv3p, self).__init__()
    if backbone == 'Xception':
      self.backbone = Xception(in_channels, output_stride, middle_blocks)
      low_level_channels = 128
    else: 
      raise NotImplementedError("A")
    self.aspp = ASPP(2048, 256)
    self.decoder = Deeplabv3pDecoder(low_level_channels, num_classes)

  def forward(self, x):
    x, l = self.backbone(x)
    x = self.aspp(x)
    x = self.decoder(x, l)
                        
    return torch.sigmoid(x)

class ASPP_branch(nn.Module):
  def __init__(self, in_channels, kernel_size, dilation):
    super(ASPP_branch, self).__init__()
    padding = ((kernel_size-1)//2)*dilation
    self.conv = nn.Conv2d(in_channels, 256, kernel_size, padding=padding,dilation=dilation)
    self.bn = nn.BatchNorm2d(256)
    self.relu = nn.ReLU(inplace)
  
  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))

class ASPP(nn.Module):
  def __init__(self, in_channels, out_channels, base_rate = 1):
    super(ASPP, self).__init__()
    
    self.b1 = ASPP_branch(in_channels, 1, 1)
    self.b2 = ASPP_branch(in_channels, 3, base_rate*6)
    self.b3 = ASPP_branch(in_channels, 3, base_rate*12)
    self.b4 = ASPP_branch(in_channels, 3, base_rate*18)
    
    self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(in_channels, 256, 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(inplace=True))

    self.conv = nn.Conv2d(256*5, out_channels, 1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace)

  def forward(self, x):
    out = self.b1(x)
    out = torch.cat((out, self.b2(x)), dim=1)
    out = torch.cat((out, self.b3(x)), dim=1)
    out = torch.cat((out, self.b4(x)), dim=1)
    out = torch.cat((out, F.interpolate(self.gap(x), size=(x.size(2), x.size(3)),
                     mode="bilinear", align_corners=True)), dim=1)
    return self.relu(self.bn(self.conv(out)))

class Xception(nn.Module):
  def __init__(self, in_channels = 3, output_stride = 16, middle_blocks = 4):
    super(Xception, self).__init__()
    if output_stride == 16:
      b3_s, mf_d, ef_d = 2, 1, 1
    else:
      b3_s, mf_d, ef_d = 1, 2, 2
    self.entry = Entry(in_channels, block3_stride=b3_s)
    self.middle = nn.Sequential(*[Middle(dilation=mf_d) for _ in range(middle_blocks)])
    self.exit = Exit(dilation=ef_d)

  def forward(self, x):
    x, low_level_feature = self.entry(x)
    x = self.middle(x)
    return self.exit(x), low_level_feature

class Entry(nn.Module):
  def __init__(self, in_channels = 3, block3_stride = 1):
    super(Entry, self).__init__()
    self.relu = nn.ReLU(inplace)
    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.conv3 = SepConv2D(64, 128, kernel_size=3)
    self.bn3 = nn.BatchNorm2d(128)
    self.conv4 = SepConv2D(128, 128, kernel_size=3)
    self.bn4 = nn.BatchNorm2d(128)
    self.conv5 = SepConv2D(128, 128, kernel_size=3, stride=2)
    self.bn5 = nn.BatchNorm2d(128)
    self.convb1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
    self.bnb1 = nn.BatchNorm2d(128)
    
    self.conv6 = SepConv2D(128, 256, kernel_size=3)
    self.bn6 = nn.BatchNorm2d(256)
    self.conv7 = SepConv2D(256, 256, kernel_size=3)
    self.bn7 = nn.BatchNorm2d(256)
    self.conv8 = SepConv2D(256, 256, kernel_size=3, stride=2)
    self.bn8 = nn.BatchNorm2d(256)
    self.convb2 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
    self.bnb2 = nn.BatchNorm2d(256)

    self.conv9 = SepConv2D(256, 728, kernel_size=3)
    self.bn9 = nn.BatchNorm2d(728)
    self.conv10 = SepConv2D(728, 728, kernel_size=3)
    self.bn10 = nn.BatchNorm2d(728)
    self.conv11 = SepConv2D(728, 728, kernel_size=3, stride=block3_stride)
    self.bn11 = nn.BatchNorm2d(728)
    self.convb3 = nn.Conv2d(256, 728, kernel_size=1, stride=block3_stride)
    self.bnb3 = nn.BatchNorm2d(728)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.bn2(self.conv2(x))
    c = self.relu(self.bn3(self.conv3(x)))
    c = self.relu(self.bn4(self.conv4(c)))
    c = self.bn5(self.conv5(c))
    x = c + self.bnb1(self.convb1(x))

    low_level_feature = x

    x = F.relu(x)
    c = self.relu(self.bn6(self.conv6(x)))
    c = self.relu(self.bn7(self.conv7(c)))
    c = self.bn8(self.conv8(c))
    x = self.relu(c + self.bnb2(self.convb2(x)))
    
    c = self.relu(self.bn9(self.conv9(x)))
    c = self.relu(self.bn10(self.conv10(c)))
    c = self.bn11(self.conv11(c))
    return c + self.bnb3(self.convb3(x)), low_level_feature

class Middle(nn.Module):
  def __init__(self, dilation = 1):
    super(Middle, self).__init__()
    self.relu = nn.ReLU(inplace)
    self.conv1 = SepConv2D(728, 728, kernel_size=3, stride=1, dilation=dilation)
    self.bn1 = nn.BatchNorm2d(728)

    self.conv2 = SepConv2D(728, 728, kernel_size=3, stride=1, dilation=dilation)
    self.bn2 = nn.BatchNorm2d(728)

    self.conv3 = SepConv2D(728, 728, kernel_size=3, stride=1, dilation=dilation)
    self.bn3 = nn.BatchNorm2d(728)
  
  def forward(self, x):
    c = self.relu(x)
    c = self.conv1(c)
    c = self.bn1(c)

    c = self.relu(c)
    c = self.conv2(c)
    c = self.bn2(c)

    c = self.relu(c)
    c = self.conv3(c)
    c = self.bn3(c)

    return x + c

class Exit(nn.Module):
  def __init__(self, dilation):
    super(Exit, self).__init__()
    self.relu = nn.ReLU(inplace)
    self.conv1 = SepConv2D(728, 728, kernel_size=3, stride=1, dilation=dilation)
    self.bn1 = nn.BatchNorm2d(728)
    self.conv2 = SepConv2D(728, 1024, kernel_size=3, stride=1, dilation=dilation)
    self.bn2 = nn.BatchNorm2d(1024)
    self.conv3 = SepConv2D(1024, 1024, kernel_size=3, stride=1, dilation=dilation)
    self.bn3 = nn.BatchNorm2d(1024)
    
    self.convb = nn.Conv2d(728, 1024, kernel_size=1, stride=1, dilation=dilation)
    self.bnb = nn.BatchNorm2d(1024)

    self.conv4 = SepConv2D(1024, 1536, kernel_size=3, stride=1, dilation=2*dilation)
    self.bn4 = nn.BatchNorm2d(1536)
    self.conv5 = SepConv2D(1536, 1536, kernel_size=3, stride=1, dilation=2*dilation)
    self.bn5 = nn.BatchNorm2d(1536)
    self.conv6 = SepConv2D(1536, 2048, kernel_size=3, stride=1, dilation=2*dilation)
    self.bn6 = nn.BatchNorm2d(2048)

  def forward(self, x):
    c = self.bn1(self.conv1(self.relu(x)))
    c = self.bn2(self.conv2(self.relu(c)))
    c = self.bn3(self.conv3(self.relu(c)))
    x = c + self.bnb(self.convb(self.relu(x)))
    x = self.bn4(self.conv4(self.relu(x)))
    x = self.bn5(self.conv5(self.relu(x)))
    return self.relu(self.bn6(self.conv6(self.relu(x))))
  
class SepConv2D(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
    super(SepConv2D, self).__init__()
    self.a = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                dilation=dilation, padding=dilation, groups=in_channels)
    self.bn = nn.BatchNorm2d(in_channels)
    self.b = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, x):
    return self.b(self.bn(self.a(x)))

class _EncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, dropout=False):
    super(_EncoderBlock, self).__init__()
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    self.encode = nn.Sequential(*layers)

  def forward(self, x):
      return self.encode(x)


class _DecoderBlock(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(_DecoderBlock, self).__init__()
    self.decode = nn.Sequential(
        nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(middle_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(middle_channels),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
    )

  def forward(self, x):
    return self.decode(x)


class UNet(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(UNet, self).__init__()
    self.enc1 = _EncoderBlock(in_channels, 64)
    self.enc2 = _EncoderBlock(64, 128)
    self.enc3 = _EncoderBlock(128, 256)
    self.enc4 = _EncoderBlock(256, 512, dropout=True)
    self.center = _DecoderBlock(512, 1024, 512)
    self.dec4 = _DecoderBlock(1024, 512, 256)
    self.dec3 = _DecoderBlock(512, 256, 128)
    self.dec2 = _DecoderBlock(256, 128, 64)
    self.dec1 = nn.Sequential(
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )
    self.final = nn.Conv2d(64, num_classes, kernel_size=1)

  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(enc1)
    enc3 = self.enc3(enc2)
    enc4 = self.enc4(enc3)
    center = self.center(enc4)
    dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
    dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
    dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
    dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
    final = self.final(dec1)
    return torch.sigmoid(final),

class UNet(nn.Module):
  def __init__(self, in_channels, num_classes = 1):
    super().__init__()
    self.enc1 = _EncoderBlock(in_channels, 64)
    self.enc2 = _EncoderBlock(64, 128)
    self.enc3 = _EncoderBlock(128, 256)
    self.enc4 = _EncoderBlock(256, 512, dropout=True)
    self.center = _DecoderBlock(512, 1024, 512)
    self.dec4 = _DecoderBlock(1024, 512, 256)
    self.dec3 = _DecoderBlock(512, 256, 128)
    self.dec2 = _DecoderBlock(256, 128, 64)
    self.dec1 = nn.Sequential(
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )

    self.final = nn.Conv2d(64, num_classes, kernel_size=1)
  
  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(enc1)
    enc3 = self.enc3(enc2)
    enc4 = self.enc4(enc3)
    center = self.center(enc4)

    dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], \
        mode='bilinear', align_corners=True)], 1))
    dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], \
        mode='bilinear', align_corners=True)], 1))
    dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], \
        mode='bilinear', align_corners=True)], 1))
    dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], \
        mode='bilinear', align_corners=True)], 1))

    final = self.final(dec1)

    return torch.sigmoid(final)

