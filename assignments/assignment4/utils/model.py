import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, bias = False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias = bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias = bias),
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
    def __init__(self, in_channels, middle_channels, out_channels, bias = False):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels),
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias = bias),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, groups=middle_channels),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=1, bias = bias),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, bias = False):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64, bias = bias)
        self.enc2 = _EncoderBlock(64, 128, bias = bias)
        self.enc3 = _EncoderBlock(128, 256, bias = bias)
        self.enc4 = _EncoderBlock(256, 512, dropout=True, bias = bias)
        self.center = _DecoderBlock(512, 1024, 512, bias = bias)
        self.dec4 = _DecoderBlock(1024, 512, 256, bias = bias)
        self.dec3 = _DecoderBlock(512, 256, 128, bias = bias)
        self.dec2 = _DecoderBlock(256, 128, 64, bias = bias)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, bias = bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, bias = bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1, bias = bias)

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
        return F.upsample(final, x.size()[2:], mode='bilinear', align_corners=True)

# High Quality Monocular Depth Estimation via Transfer Learning [https://arxiv.org/pdf/1812.11941v2.pdf]
class UpSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.2)
    def forward(self, x, concat):
        x = self.upsample(x)
        x = torch.cat([x, concat], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x
    
class DepthNet(nn.Module):
    def __init__(self, pretrained = True):
        from torchvision import models
        super(DepthNet, self).__init__()
        densenet = models.densenet169(pretrained)
        # Densenet backbone
        self.conv1 = densenet.features[0]
        self.block1 = nn.Sequential(*(densenet.features[1:4]))
        self.block2 = nn.Sequential(*(densenet.features[4:6]))
        self.block3 = nn.Sequential(*(densenet.features[6:8]))
        self.block4 = nn.Sequential(*(densenet.features[8:-1]))
        self.conv2 = nn.Conv2d(1664, 1664, 1)
        self.upsample1 = UpSample(1920, 832)
        self.upsample2 = UpSample(960, 416)
        self.upsample3 = UpSample(480, 208)
        self.upsample4 = UpSample(272, 104)
        self.conv3 = nn.Conv2d(104, 1, 3, padding=1)
        self.upsampleout = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.block1(c1)
        c3 = self.block2(c2)
        c4 = self.block3(c3)
        x = self.block4(c4)
        x = self.conv2(x)
        x = self.upsample1(x, c4)
        x = self.upsample2(x, c3)
        x = self.upsample3(x, c2)
        x = self.upsample4(x, c1)
        return self.upsampleout(self.conv3(x))