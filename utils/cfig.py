import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

def show_fig(fig):
    if (fig.ndim == 3):
        col = int(np.ceil(len(fig)/5))
        plt.figure(figsize=(50,col*10))
        for i in range(0,len(fig) ):
            plt.subplot(col ,5 , i+1 )
            plt.imshow(fig[i])
            plt.colorbar()
        plt.show()
    else:
        plt.imshow(fig)
        plt.colorbar()
        plt.show()

def save_fig(fig,label):
    if (fig.ndim == 3):
        col = int(np.ceil(len(fig)/5))
        plt.figure(figsize=(50,col*10))
        for i in range(0,len(fig) ):
            plt.subplot(col ,5 , i+1 )
            plt.imshow(fig[i])
            plt.colorbar()
        plt.savefig(f"./{label}.png", dpi=300,bbox_inches='tight', transparent=True,format="png")
        plt.close()
    


def to_cpu_numpy(fig):
    if (fig.ndim == 3):
        x_size = len(fig[0,:,0])
        y_size = len(fig[0,0,:])
        b_size = len(fig[:,0,0])
        fig_cpu_numpy = np.zeros([b_size, x_size, y_size])

        for i in range(0,b_size ):
            fig_cpu_numpy[i] = fig[i].cpu().detach().numpy()
    else:
        fig_cpu_numpy = fig.cpu().detach().numpy()
    return(fig_cpu_numpy)


def to_real(fig):
    
    for i in range(0,len(fig) ):
        fig_real = np.real(fig)
    return(fig_real)


def show_3D(object):
    Main_lens_size_x = len(object[0,:])
    Main_lens_size_y = len(object[:,0])
    # Main_lens_size_x = 3000
    # Main_lens_size_y = 1000
    Dx = np.linspace(-1 * Main_lens_size_x / 2, Main_lens_size_x / 2, Main_lens_size_x)
    Dy = np.linspace(-1 * Main_lens_size_y / 2, Main_lens_size_y / 2, Main_lens_size_y)
    DX, DY = np.meshgrid(Dx, Dy)
    
    
    fig=plt.figure()
    ax3=plt.axes(projection='3d')

    #定义三维数据


    #作图
    ax3.plot_surface(DX,DY,object,cmap='rainbow')
    # ax3.contour(DX,DY,object,zdim='z',offset=-2,cmap='rainbow') #等高线图，要设置offset，为Z的最小值
    plt.show()


def mulchannel_2_RGB(img,mask,mode='CHW'):#0-1
    if mode == 'CHW':
        img = img.transpose(1,2,0)+0.00001
        mask = mask.transpose(1,2,0)
        # print(mask.shape)
    H = np.ones_like(mask)
    for i in range(img.shape[2]):
        base = 255 // (img.shape[2]+1)
        H[:,:,i] = base*(i+1)

    H = np.sum(H * mask,axis=2,keepdims=False)

    # ratio = img / np.sum(img,axis=2,keepdims=True)
    # H = np.sum(H*ratio,axis=2,keepdims=False)

    S = np.ones_like(H)*210

    # V = np.sum(mask*img,axis=2,keepdims=False)*255
    V = np.sum(img,axis=2,keepdims=False)*255

    HSV = np.stack([H,S,V],axis=2).astype(np.uint8)
    RGB = cv2.cvtColor(HSV,cv2.COLOR_HSV2BGR)
    if mode == 'CHW':
        HSV = HSV.transpose(2,0,1)
        RGB = RGB.transpose(2,0,1)
    return HSV,RGB

def rend_RGB(img):
    max = np.argmax(img,axis=0)
    one_hot = np.asarray(torch.nn.functional.one_hot(torch.tensor(max))).transpose(2,0,1)
    HSV,RGB = mulchannel_2_RGB(img,one_hot,mode='CHW')
    return RGB