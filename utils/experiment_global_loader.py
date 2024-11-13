import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class experiment_global_dataset(Dataset):
    def __init__(self,root,params):
        self.root = root
        self.params = params
        self.img_dir = sorted(os.listdir(self.root))
        self.path = self.combine_path()

    def combine_path(self):
        path = []
        for i in range(len(self.img_dir)):
            path.append(os.path.join(self.root,self.img_dir[i]))
        return path


    def __getitem__(self,index):
        
        img = np.float32(np.load(self.path[index]))/255
        # img = np.where(img<0.00,0,img)
        img = torch.tensor(img,dtype=torch.float32)
        


        return img
    def __len__(self):
        return len(self.path)