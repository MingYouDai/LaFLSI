import torch
import torch.nn as nn
import math


class TAU_transform_block(nn.Module):
    def __init__(self,input_channel,gamma=2,b=1):
        super(TAU_transform_block,self).__init__()


        kernel_size = int(abs((math.log(input_channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        dim = 1
        y = torch.fft.rfft(x,dim=1,norm='backward')
        y_imag = y.imag
        y_real = y.real
        x = torch.cat([y_real, y_imag], dim=dim)

        avg = self.avg_pool(x).view([b,1,c+2])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b,c+2,1,1])

        ECA_out = out * x

        y_real, y_imag = torch.chunk(ECA_out, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft(y,dim=1 ,norm='backward')




        return y
