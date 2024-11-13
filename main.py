import re
import os, glob, datetime, time
import math
from math import exp
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision import transforms
from PIL import Image
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from utils import *
from model import *


if __name__=='__main__':



    params = system_params()

    remove_File(params)

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

    if params['mode'] == 'global':
        print(f'mode:{params["mode"]}')
        Time = time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())
        Temp_params = {}
        Temp_params['Time'] = Time
        generate_global(params,Temp_params)
        temp_para = np.load('utils/temp_params_global_experiment.npy',allow_pickle=True).item()
        print(temp_para)


    with open('utils/device.txt', 'r') as f:   
        line = f.readline().replace('\n', '')
    device = torch.device(f'cuda:{str(line)}' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load('checkpoint/model_state_dict.pth',map_location='cuda:0')
    model = TAU_model('experiment',params).to(device)
    model.load_state_dict(state_dict,strict=True)
    


    print(f'mode:{params["mode"]}')
    print(f'channel:{params["channel"]}')

    output , Filter_Patch,_, _ ,_,_= experiment_rocons_global(model,device,params)

    analysis_img(output,Filter_Patch,params)



    


