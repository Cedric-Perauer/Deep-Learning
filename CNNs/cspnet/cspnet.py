import torch 
import torch.nn as nn 


#Standard ResNet block 
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        #main path 
        c1 = self.conv1(x)
        b1 = self.bn1(c1)
        act1 = self.relu1(b1)

        c2 = self.conv2(act1) 
        b2 = self.bn2(c2)

        #shortcut path 
        sc_c = self.shortcut_conv(x)
        sc_b = self.shortcut_bn(sc_c)

        out = self.relu2(b2+sc_b)
        return out


def _downsample(inplanes,outplanes,stride):
    return torch.nn.Sequential(
        conv1x1(inplanes, outplanes, stride),
        torch.nn.BatchNorm2d(outplanes),
    )


class Csp_ResNet(nn.Module):
    #csp resnet block 
    def __init__(self,in_channels,out_channels):
        super(Csp_ResNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.resnet_layer = ResNet(in_channels, out_channels)

        
        #half the feature maps are simply downsampled for concatenation  
        self.shortcut_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1,stride=1)
        self.shortcut_convbn = nn.BatchNorm2d(out_channels)


    def forward(self,x):
         
         res_out = self.resnet_layer(x[:,:self.out_channels//2,...])
         short_out = self.shortcut_conv(x[:,self.out_channels//2:,...]) 
    
         out = torch.cat([res_out,short_out],1)
         
         return out 
        
csp = Csp_ResNet(64,128)
res = ResNet(64,128)
x = torch.ones([1,64,80,80])

out = res(x)
print(out.shape)





