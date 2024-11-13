import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .TAU_transform import *
from .Resfft import *
def Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            # nn.BatchNorm2d(out_channels)
        )


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3x3BNReLU(in_channels, out_channels,stride=1),
            Conv3x3BNReLU(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=stride)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        double_conv_out = self.double_conv(x)
        out = self.pool(double_conv_out)
        return double_conv_out,out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels,bilinear=True):
        super().__init__()
        self.reduce = Conv1x1BNReLU(in_channels, in_channels//2)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(self.reduce(x1))
        _, channel1, height1, width1 = x1.size()
        _, channel2, height2, width2 = x2.size()

        # input is CHW
        diffY = height2 - height1
        diffX = width2 - width1

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetBlock(nn.Module):
    def __init__(self, input_channels,output_channels,n_channels,params):
        super(UNetBlock, self).__init__()
        bilinear = True
        self.params = params


        self.down1 = DownConv(input_channels, n_channels )
        self.down2 = DownConv(n_channels , n_channels * 2)
        self.down3 = DownConv(n_channels * 2, n_channels * 4)
        self.down4 = DownConv(n_channels * 4, n_channels * 8)

        self.conv = DoubleConv(n_channels *8, n_channels *16)

        self.up1 = UpConv(n_channels * 16, n_channels * 8, bilinear)
        self.up2 = UpConv(n_channels * 8, n_channels * 4, bilinear)
        self.up3 = UpConv(n_channels * 4, n_channels * 2, bilinear)
        # self.up4 = UpConv(n_channels * 2, n_channels, bilinear)
        self.up4 = UpConv(n_channels * 2, self.params['Filter_channel'], bilinear)


        self.eca_1 = TAU_transform_block(n_channels)
        self.eca_2 = TAU_transform_block(n_channels * 2)
        self.eca_3 = TAU_transform_block(n_channels * 4)
        self.eca_4 = TAU_transform_block(n_channels * 8)
        
        # self.outconv = nn.Conv2d(n_channels, output_channels, kernel_size=1)
        self.outconv = nn.Conv2d(self.params['Filter_channel'], output_channels, kernel_size=1)
        # self.conv_attention = conv_attention(input_channels, output_channels)
        
        # self.Channel_grad = C_grad(input_channels)
        
        self._initialize_weights()
    def forward(self, x):

        skip_1,x1 = self.down1(x)
        skip_2,x2 = self.down2(x1)
        skip_3,x3 = self.down3(x2)
        skip_4,x4 = self.down4(x3)
        x5 = self.conv(x4)
        x6 = self.up1(x5, self.eca_4(skip_4))
        x7 = self.up2(x6, self.eca_3(skip_3))
        x8 = self.up3(x7, self.eca_2(skip_2))
        x9 = self.up4(x8, self.eca_1(skip_1))
        outputs = self.outconv(x9)
        # ca = self.conv_attention(x)
        return outputs,x9
        # return outputs,ca
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print(f'init weight {m}')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
