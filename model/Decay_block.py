import torch
import torch.nn as nn
import numpy as np


    


def bg(input,bg_file,device,params):
    b,c,h,w = input.shape
    bg_file = bg_file.unsqueeze(0).repeat(b,1,1,1)

    bg_1 = torch.rand((b,c,1,1),device=device)*0.5 + 0.5

    bg_2 = torch.rand((b,c,1,1),device=device)*0.5 + 0.5

    bg_noise = torch.randn((b,params['number_of_patch'],2764,3856),device=device)
    GRR_bg = (bg_file*bg_1 + bg_noise*bg_2)*params['bg_ratio']
    row_idx = [203,459,715,971,1227,1483,1739,1995,2251]
    idx = torch.randint(len(row_idx),[1])

    output = input +  GRR_bg[:,:,idx[0]:(idx[0]+256),:256]

    return output



def noise(input,params):



    N_intensity = torch.distributions.Normal(input/(input.max()+0.0001)*params['noise_Normal_mu'], params['noise_Normal_sigma']).sample()

    P_intensity = torch.distributions.Poisson(input/(input.max()+0.0001)).sample()
 
    poisson_noise = torch.distributions.Poisson(torch.ones_like(input)*params['noise_Poisson']).sample()

    return N_intensity  + P_intensity + poisson_noise + input
    

    


class Decay(nn.Module):
    def __init__(self,params):
        super(Decay, self).__init__()

        # 定义模型的可优化参数
        with open('utils/device.txt', 'r') as f:   
            self.line = f.readline().replace('\n', '')
        self.device = torch.device(f'cuda:{str(self.line)}' if torch.cuda.is_available() else 'cpu')
        self.params = params


        if params['channel'] == 'multi':
            self.t = nn.Parameter(torch.tensor(params['Time_init'],dtype=torch.float32),requires_grad=True)
            self.ratio = nn.Parameter(torch.ones_like(self.t,dtype=torch.float32),requires_grad=True)

            self.life = torch.tensor(self.params['flo_life'],dtype=torch.float32).to(self.device)
            self.flo_intensity_ratio = torch.tensor(params['flo_intensity_ratio'],dtype=torch.float32).to(self.device)
            # self.bg = torch.tensor(np.load('data/bg_2d.npy'),dtype=torch.float32).unsqueeze(0).repeat(self.params['number_of_patch'],1,1).to(self.device)



    def forward(self, x):

        b,p,c,h,w = x.size()
        if self.params['channel'] == 'multi':
            Life = self.life.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(b,p,1,h,w)
            Time = (self.t*self.ratio).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(b,1,c,h,w)*1000
            Time[Time<0]=0


            Life_rand = (Life + (torch.rand(Life.shape,device=self.device)-0.5)*2*6) 
            

            sigma = (torch.ones_like(x)*6).to(self.device) 

            sigma_rand = sigma + (torch.rand(sigma.shape,device=self.device)-0.5)*2*sigma*0.05
           

            tao_i = []
            for i in range(len(self.params['flo_life'])):
                tao_i.append(torch.linspace(self.life[i]-40,self.life[i]+40,9))
                
            tao_i = torch.concatenate(tao_i,dim=0)
            tao_i = tao_i.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).repeat(1,b,p,c,h,w).to(self.device)

            w_i = torch.exp(-1*((tao_i-Life_rand)**2)/(2*sigma_rand**2))/((2*3.14)**(1/2)*sigma_rand)
            w_i = w_i/w_i.sum(axis=0)


            tao_gt = (torch.mean(torch.sum(w_i,dim=3,keepdim=False),dim=2,keepdim=False)).permute(1,0,2,3).clone().detach()
            decay_metrics = (w_i * torch.exp(-(Time)/tao_i)).sum(axis=0)
            intensity_ratio = self.flo_intensity_ratio.unsqueeze(0).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(b,p,1,h,w)




            x = x * decay_metrics * intensity_ratio
            x = torch.sum(x,dim=2,keepdim=False)
            
            x = noise(x,self.params)
            x = bg(x,self.bg,self.device,self.params)
            x = torch.clamp(x,min=0,max=255)
            x = x/255
            return x,tao_gt,(self.t*self.ratio),self.ratio
        



        
    
