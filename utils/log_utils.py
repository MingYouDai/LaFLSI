import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random


def log_info(log_dir,log):
    with open(log_dir,'a',encoding='utf-8')as f:
        f.write(f'{log}\n')


def save_fig(fig,dir,name):
    if ((fig.ndim == 3) and (len(fig)>1)):
        col = int(np.ceil(len(fig)/5))
        plt.figure(figsize=(10,col*2))
        for i in range(0,len(fig) ):
            plt.subplot(col ,5 , i+1 )
            plt.imshow(fig[i])
            plt.colorbar()
        plt.savefig(f'./{dir}/output_img/{name}.jpg', dpi=300)
        plt.close()
    elif ((fig.ndim == 3) and (len(fig)==1)):
        plt.figure(figsize=(10,10))
        plt.imshow(fig[0])
        plt.colorbar()
        plt.savefig(f'./{dir}/output_img/{name}.jpg', dpi=300)
        plt.close()
    else:
        plt.figure(figsize=(10,10))
        plt.imshow(fig)
        plt.colorbar()
        plt.savefig(f'./{dir}/output_img/{name}.jpg', dpi=300)
        plt.close()

def draw_life(t,dir,params):
    Time = np.tile(np.expand_dims(t,axis=0),(params['num_of_channel'],1))*1000
    Old_Time = np.tile(np.expand_dims(params['Time_init'],axis=0),(params['num_of_channel'],1))*1000
    Life = np.tile(np.expand_dims(params['flo_life'],axis=1),(1,params['number_of_patch']))
    flo_intensity_ratio = np.tile(np.expand_dims(params['flo_intensity_ratio'],axis=1),(1,params['number_of_patch']))
    decay_metrics = np.exp(-(Time)/Life)*flo_intensity_ratio
    old_decay_metrics = np.exp(-(Old_Time)/Life)*flo_intensity_ratio
    np.save(f"{params['Time']}/decay_metrics.npy",decay_metrics)
    for i in range(len(decay_metrics)):
        plt.plot(t*1000,decay_metrics[i],'-o',color=params['colors'][i+1],label = f'{params["flo_life"][i]}')
        plt.plot(params['Time_init']*1000,old_decay_metrics[i],linestyle='dashed',marker='o',color=params['colors'][i+1],markerfacecolor='white',alpha=0.5,label = f'init_{params["flo_life"][i]}')
    
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./{dir}/output_img/life.png', dpi=300)
    plt.close()


def set_seed(seed):
    # seed = 43  # 宇宙答案
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

