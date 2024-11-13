
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
from model import *
from .img_utils import *
from .cfig import *
from .experiment_global_loader import *
import shutil





def experiment_rocons_global(model,device,params):
    debug = True
    temp_para = np.load('utils/temp_params_global_experiment.npy',allow_pickle=True).item()
    test_data_batch_cut_dir = params['global_experiment_BatchCut_path']
    test_data_batch_cut_dir_path = []
    for i in range(len(os.listdir(test_data_batch_cut_dir))):
        test_data_batch_cut_dir_path.append(os.path.join(test_data_batch_cut_dir,sorted(os.listdir(test_data_batch_cut_dir))[i]))
    print(test_data_batch_cut_dir_path)


    Test_data = experiment_global_dataset(test_data_batch_cut_dir_path[i],params=params)
    if params['channel'] == 'multi':
        output = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),params['num_of_channel'],256,256))
        filtered_img = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),params['num_of_channel'],256,256))
        Filter_Patch = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),params['Filter_channel'],256,256))
        debug_0 = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),params['num_of_channel'],256,256))
        debug_1 = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),params['num_of_channel'],256,256))
        debug_2 = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),params['num_of_channel'],256,256))
        debug_3 = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),params['num_of_channel'],256,256))

    input_patch = np.zeros((len(test_data_batch_cut_dir_path),len(Test_data),256,256))
    # print(output.shape)
    print('Rebuilding image, please wait...')


    for i in range(len(test_data_batch_cut_dir_path)):
        Test_data = experiment_global_dataset(test_data_batch_cut_dir_path[i],params=params)
        Test_loader = DataLoader(dataset=Test_data,batch_size=1)

    
        with torch.no_grad():
            
            model.eval()
            for idx , img in enumerate(Test_loader):
                
                
                img= img.to(device)
                
                if debug == True:

                    output_1 ,Filter,d_0,d_1,d_2,d_3= model(img)
                    Filter_Patch[i,idx,:,:,:] = Filter.cpu().numpy()
                    filtered_img[i,idx,0,:,:] = Filter[:,0:8,:,:].cpu().numpy().sum(axis=1)
                    filtered_img[i,idx,1,:,:] = Filter[:,9:17,:,:].cpu().numpy().sum(axis=1)
                    output[i,idx,:,:,:] = output_1.cpu().numpy()
                    debug_0[i,idx,:,:,:] = d_0.cpu().numpy()
                    debug_1[i,idx,:,:,:] = d_1.cpu().numpy()
                    debug_2[i,idx,:,:,:] = d_2.cpu().numpy()
                    debug_3[i,idx,:,:,:] = d_3.cpu().numpy()
                else :
                    output_1 = model(img)
                    output[i,idx,:,:,:] = output_1.cpu().numpy()

                input_patch[i,idx,:,:] = img[0,0].cpu().numpy()


    sh = 128
    sw = 128
    mask = np.ones((sh+2,sw+2))
    mask = np.pad(mask,((sh//2-1,sh//2-1),(sw//2-1,sw//2-1)),'linear_ramp', end_values=((0, 0),(0,0)))



    def combine_img(img,temp_para):
        sh = 128
        sw = 128
        if img.ndim == 5:
            n,p,c,h,w = img.shape
            img_reshape = img.reshape((n,temp_para['h_n'],temp_para['w_n'],c,h,w))
            out_img = np.zeros((n,c,temp_para['h_padded'], temp_para['w_padded']))
            for N in range(n):
                for i in range (int(temp_para['h_n'])):
                    # print(i)
                    for j in range (int(temp_para['w_n'])):
                        temp = np.zeros((n,c,temp_para['h_padded'], temp_para['w_padded']))
                        temp[N,:,sh//2 * 3 *i:sh//2 * (3 *(i+1)) + sh//2,sh//2 * 3 *j:sh//2 * (3 *(j+1)) + sh//2] = img_reshape[N,i,j]*mask
                        out_img = out_img + temp
        else:
            n,p,h,w = img.shape
            img_reshape = img.reshape((n,temp_para['h_n'],temp_para['w_n'],h,w))
            out_img = np.zeros((n,temp_para['h_padded'], temp_para['w_padded']))
            for N in range(n):
                for i in range (int(temp_para['h_n'])):
                    # print(i)
                    for j in range (int(temp_para['w_n'])):
                        temp = np.zeros((n,temp_para['h_padded'], temp_para['w_padded']))
                        temp[N,sh//2 * 3 *i:sh//2 * (3 *(i+1)) + sh//2,sh//2 * 3 *j:sh//2 * (3 *(j+1)) + sh//2] = img_reshape[N,i,j]*mask
                        out_img = out_img + temp
        return out_img
    temp_para = np.load('utils/temp_params_global_experiment.npy',allow_pickle=True).item()
    print('Rebuilding image complete...')
    print('Combining image, please wait...')
    img_recons = combine_img(output,temp_para)
    img_recons = img_recons[:,:,64:-temp_para['h_pad_2'],64:-temp_para['w_pad_2']]
    print(f'img_recons_shape:{img_recons.shape}')
    if os.path.exists(params['global_experiment_BatchCut_path']):
        shutil.rmtree(params['global_experiment_BatchCut_path'])



    for i in range(temp_para['len_img_batch']):
        # show_fig(img_recons[i])
        
        if params['channel'] == 'multi':
            
            temp = np.where((img_recons[i])<0,0,img_recons[i])
            temp_recons = temp
            temp = np.sum(temp,axis=0,keepdims=False)

            one_hot = torch.nn.functional.one_hot(torch.tensor(np.argmax(img_recons[i],axis=0))).permute(2,0,1)

            if params['num_of_channel'] == 2:
                c1 = one_hot[0] * temp
                c2 = one_hot[1] * temp


 
                c1 = c1/c1.max()
                c2 = c2/c2.max()
                c3 = np.zeros_like(c1)

                f = []
                f.append(np.expand_dims(c1,axis=2))
                f.append(np.expand_dims(c3,axis=2))
                f.append(np.expand_dims(c2,axis=2))
                f = np.uint8(np.concatenate(f,axis=2)*255)
                Image.fromarray(f).convert("RGB").save(f'./restored_images/Reconstructed_full-size.png')


    return output,Filter_Patch,debug_0,debug_1,debug_2,debug_3


def analysis_img(output,Filter_Patch,params):
    tao_i = []
    for i in range(len(params['flo_life'])):
        tao_i.append(np.linspace(params['flo_life'][i]-40,params['flo_life'][i]+40,9))
    x = np.concatenate(tao_i,axis=0)

    x_add = np.arange((params['flo_life'][0]+40)+8,(params['flo_life'][1]-40),8)
    y_add = np.zeros_like(x_add)




    save_fig(output[0,437,:,80:120,25:65],'restored_images/ROI_1_Dual_Channel')
    plt.subplots(figsize=(7,4.5))
    x1 , y1 = interp_1d(x,Filter_Patch[0,437,:,95,47],10)
    x2 , y2 = interp_1d(x,Filter_Patch[0,437,:,104,47],10)
    x3 , y3 = interp_1d(x,Filter_Patch[0,437,:,86,53],10)
    plt.plot(x1,y1,linewidth=5,color='#FF5575',alpha=0.7)
    plt.scatter(x,Filter_Patch[0,437,:,95,47],marker='o',s=95,color='#FF5575',label='point 1',alpha=0.7)
    plt.plot(x2,y2,linewidth=5,color='#FFD36A',alpha=0.7)
    plt.scatter(x,Filter_Patch[0,437,:,104,47],marker='o',s=95,color='#FFD36A',label='point 2')
    plt.plot(x3,y3,linewidth=5,color='#6299FF',alpha=0.7)
    plt.scatter(x,Filter_Patch[0,437,:,86,53],marker='o',s=95,color='#6299FF',label='point 3')

    plt.scatter(x_add,y_add,marker='o',s=95,color='#FF5575',alpha=0.7)
    plt.scatter(x_add,y_add,marker='o',s=95,color='#FFD36A',alpha=0.7)
    plt.scatter(x_add,y_add,marker='o',s=95,color='#6299FF',alpha=0.7)

    plt.xlabel('Fluorescence lifetime (μs)',size=16)
    plt.ylabel('Intensity',size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend(prop={'size': 16}, bbox_to_anchor=(1.02, 1.15),handletextpad=1,labelspacing=0.55,ncol=3)
    plt.savefig(f"./restored_images/ROI_1_Point.png", dpi=300,bbox_inches='tight', transparent=True,format="png")
    plt.close()



    save_fig(output[0,440,:,101:151,199:249],'restored_images/ROI_2_Dual_Channel')
    plt.subplots(figsize=(7,4.5))
    x1 , y1 = interp_1d(x,Filter_Patch[0,440,:,123,216],5)
    x2 , y2 = interp_1d(x,Filter_Patch[0,440,:,125,218],5)
    x3 , y3 = interp_1d(x,Filter_Patch[0,440,:,127,220],5)
    plt.plot(x1,y1,linewidth=5,color='#FF5575',alpha=0.7)
    plt.scatter(x,Filter_Patch[0,440,:,123,216],marker='o',s=95,color='#FF5575',label='point 1')
    plt.plot(x2,y2,linewidth=5,color='#FFD36A',alpha=0.7)
    plt.scatter(x,Filter_Patch[0,440,:,125,218],marker='o',s=95,color='#FFD36A',label='point 2')
    plt.plot(x3,y3,linewidth=5,color='#6299FF',alpha=0.7)
    plt.scatter(x,Filter_Patch[0,440,:,127,220],marker='o',s=95,color='#6299FF',label='point 3')

    plt.scatter(x_add,y_add,marker='o',s=95,color='#FF5575',alpha=0.7)
    plt.scatter(x_add,y_add,marker='o',s=95,color='#FFD36A',alpha=0.7)
    plt.scatter(x_add,y_add,marker='o',s=95,color='#6299FF',alpha=0.7)

    plt.xlabel('Fluorescence lifetime (μs)',size=16)
    plt.ylabel('Intensity',size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend(prop={'size': 16}, bbox_to_anchor=(1.02, 1.15),handletextpad=1,labelspacing=0.55,ncol=3)
    plt.savefig(f"restored_images/ROI_2_Point.png", dpi=300,bbox_inches='tight', transparent=True,format="png")
    plt.close()