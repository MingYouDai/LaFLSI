import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from skimage import exposure
from scipy.signal import convolve2d
from utils import *
# from pack import *
import shutil
import time


def generate_global(params,Temp_params):
    if os.path.exists(params['global_experiment_BatchCut_path']):
            shutil.rmtree(params['global_experiment_BatchCut_path'])
            print(f'{params["global_experiment_BatchCut_path"]} is removed!!!')
    else :
            print(f'{params["global_experiment_BatchCut_path"]} is not exist!!!')

    def divide_img_real(img,params):
        def calculate_pad(input):
            if input % 192 <=128:
                pad = (128 - input % 192) + 64
            else:
                pad = input % 192 - 128
            return pad
        h_pad_1 = 64
        w_pad_1 = 64
        sh = 128
        sw = 128
        c1 ,h, w = img.shape

        h_pad_2 = calculate_pad(h)
        w_pad_2 = calculate_pad(w)
        pad_img = np.pad(img,((0,0),(h_pad_1,h_pad_2),(w_pad_1,w_pad_2)), 'constant', constant_values=0)
        c1,h_padded , w_padded = pad_img.shape
        h_n = ((h_padded-64)/192)
        w_n = ((w_padded-64)/192)
        batch_img = np.zeros((int(h_n),int(w_n),c1,256,256))
        for i in range(int(h_n)):
            for j in range(int(w_n)):
                batch_img[i,j] = pad_img[:,sh//2 * 3 *i:sh//2 * (3 *(i+1)) + sh//2,sw//2 * 3 *j:sw//2 * (3 *(j+1)) + sw//2]
        batch = batch_img.reshape((int(h_n)*int(w_n),c1,256,256))
        Temp_params['c'] ,Temp_params['h_pad_2'],Temp_params['w_pad_2'],Temp_params['h_padded'],Temp_params['w_padded'],Temp_params['h_n'],Temp_params['w_n']= (c1,h_pad_2,w_pad_2,h_padded , w_padded,int(h_n),int(w_n))
        np.save(f'utils/temp_params_global_experiment.npy',Temp_params)
        return batch



    test_data_batch_dir =params['global_experiment_root_path']
    if not os.path.exists(test_data_batch_dir):
        os.makedirs(test_data_batch_dir)
    test_data_batch_cut_dir = params['global_experiment_BatchCut_path']
    if not os.path.exists(test_data_batch_cut_dir):
        os.makedirs(test_data_batch_cut_dir)
    test_data_batch = sorted(os.listdir(test_data_batch_dir))

    test_data_batch_path = []
    for i in range(len(test_data_batch)):
        test_data_batch_path.append(os.path.join(test_data_batch_dir,test_data_batch[i]))

    print(f'path:{test_data_batch_path}')
    Temp_params['len_img_batch'] = len(test_data_batch_path)


    img_list = []
    for i in range(len(test_data_batch_path)):
        img_list.append([])
        for j in range(len(os.listdir(test_data_batch_path[i]))):
            img_list[i].append(sorted(os.listdir(test_data_batch_path[i]),reverse=False)[j])
            # img_list[i].append(sorted(os.listdir(test_data_batch_path[i]),reverse=True)[j])
    print(f'img_list_len:{len(img_list[0])}')
    print(f'img_list:{img_list}')



    img_batch = []
    for i in range(len(img_list)):
        img_batch.append([])
        for j in range(len(img_list[0])):
            img_batch[i].append(np.asarray(Image.open(os.path.join(test_data_batch_path[i],img_list[i][j])))[np.newaxis,np.newaxis,:,:])
        img_batch[i] = np.concatenate(img_batch[i],axis=1)
    img_batch = np.concatenate(img_batch,axis=0)
    print(f'img_batch_shape:{img_batch.shape}')



    batch = []
    for i in range(len(img_batch)):
        batch.append(divide_img_real(img_batch[i],params)[np.newaxis])
    batch = np.concatenate(batch,axis=0)
    print(f'devided_img_batch_shape:{batch.shape}')


    for i in range(len(test_data_batch)):
            if not os.path.exists(os.path.join(test_data_batch_cut_dir,test_data_batch[i])):
                os.mkdir(os.path.join(test_data_batch_cut_dir,test_data_batch[i]))
            for j in range(len(batch[0])):
                    path = os.path.join(f'{test_data_batch_cut_dir}/{test_data_batch[i]}',f'block{j//1000}{j%1000//100}{j//10%10}{j%10}.npy')
                    np.save(path,batch[i,j])
            print(f'{os.path.join(test_data_batch_cut_dir,test_data_batch[i])},done!')
