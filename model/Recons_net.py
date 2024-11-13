import torch
from .CBAM import *
from .ConvNext import *
from .Resfft import *
from .Unet import *
from .SE_net import *
from .TAU_transform import *
from .Decay_block import *
class convnext_Res_Block(nn.Module):
    def __init__(self, channel, num_res=4):
        super(convnext_Res_Block, self).__init__()

        layers = [ConvNextBlock(channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class fft_Res_Block(nn.Module):
    def __init__(self, channel, num_res=4):
        super(fft_Res_Block, self).__init__()

        layers = [ResfftBlock(channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class SCMBlock(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(SCMBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel // 4, out_channel // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel // 2, out_channel // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel // 2, out_channel-in_channel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
        self._initialize_weights()
    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print(f'init weight {m}')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class reverseSCMBBlock(nn.Module):
    def __init__(self, in_channel,out_channel):
        super(reverseSCMBBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, in_channel//2, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channel // 2, in_channel // 4, kernel_size=1, stride=1, relu=True),
            BasicConv(in_channel // 4, in_channel // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channel // 4, out_channel , kernel_size=1, stride=1, relu=False)
            
        )
        self._initialize_weights()
    def forward(self, x):
        return self.main(x)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print(f'init weight {m}')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class fuseBlock(nn.Module):
    def __init__(self, channel):
        super(fuseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=True),
            # BasicConv(channel, channel, kernel_size=3, stride=1, relu=True),
            BasicConv(channel , channel, kernel_size=1, stride=1, relu=False),
            
        )
        # self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()
    def forward(self, x1,x2,x3):
        y1 = self.conv1(torch.cat([x2, x3], dim=1))
        # y1 = self.conv1(x2+x3)
        return (y1 + x1), y1

        # return (x1 + x2 + x3),x1


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print(f'init weight {m}')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class TAU_Net(nn.Module):
    def __init__(self,in_channel,out_channel,params):
        super(TAU_Net,self).__init__()
        self.n_channel = 32

        self.Unet_1 = UNetBlock(in_channel,out_channel,self.n_channel,params)

        # self.decay = Decay(params)
        self.SCM_1_1 = SCMBlock(in_channel,self.n_channel)
        self.SCM_1_2 = SCMBlock(in_channel,self.n_channel)
        self.se_1 = se_block(self.n_channel)
        self.convnext_1 = convnext_Res_Block(self.n_channel)
        self.CBAM_1 = CBAMBlock(self.n_channel)
        self.fft_1 = fft_Res_Block(self.n_channel)
        self.reverse_1_1 = reverseSCMBBlock(self.n_channel,out_channel)
        self.reverse_1_2 = reverseSCMBBlock(self.n_channel,out_channel)
        self.fuse_1 = fuseBlock(out_channel)





    def forward(self,x):
        x_1 = x
        U1 ,Filter= self.Unet_1(x_1)
        C1 = self.reverse_1_1(self.convnext_1(self.se_1(self.SCM_1_1(x_1))))
        R1 = self.reverse_1_2(self.fft_1(self.CBAM_1(self.SCM_1_2(x_1))))
        S1 ,debug = self.fuse_1(U1,C1,R1)

        

        return S1 ,Filter,debug,U1,C1,R1
    


class TAU_model(nn.Module):
    def __init__(self,mode,params):
        super(TAU_model,self).__init__()
        self.decay = Decay(params)
        self.mode = mode
        self.params = params

        if params['mode'] == 'global':
            if params['channel'] == 'multi':
                self.devide_model = TAU_Net(params['number_of_patch'],params['num_of_channel'],params)
        
        
    def forward(self,x):

        signal ,Filter,debug0,debug1,debug2,debug3= self.devide_model(x)
        return signal,Filter,debug0,debug1,debug2,debug3
