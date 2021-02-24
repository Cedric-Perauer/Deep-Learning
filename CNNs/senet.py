from torch import nn
from torchvision.models import ResNet
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = nn.Conv2d(in_channels=inplanes,out_channels=planes,kernel_size=1)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEModel(nn.Module):
    def __init__(self,in_planes=8):
     super(SEModel,self).__init__()
     self.conv = conv3x3(3,8,stride=1)
     self.relu = nn.ReLU()
     self.bn = nn.BatchNorm2d(8)
     self.se1 = SEBasicBlock(inplanes=8,planes=8,reduction=8)
     self.se2 = SEBasicBlock(inplanes=8,planes=16)
     self.se3 = SEBasicBlock(inplanes=16,planes=32)
     self.se4 = SEBasicBlock(inplanes=32,planes=64)


    def forward(self,x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.se1(x)
        x = self.se2(x)
        x = self.se3(x)
        x = self.se4(x)

        return x
     
