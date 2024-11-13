import numpy as np
import torch

def calculate_PSNR(img1,img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1 / torch.sqrt(mse))



