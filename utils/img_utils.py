from PIL import Image
import numpy as np
from scipy import interpolate
from scipy.interpolate import make_interp_spline

def pil_loader(path):
    img = Image.open(path)
    return img

def save_rgb(img,name):
    Image.fromarray(img[:,:,:3]).convert('RGB').save(f'{name}.png')
def save_bmp(img,name):
    Image.fromarray(img).convert('L').save(f'{name}.bmp')

def gray_to_rgb(img_path):
    img_rgb = []
    for i in img_path:
        img_rgb.append(np.expand_dims(np.asarray(Image.open(i)),axis=2))
    if len(img_path)==1:
        img_black_1 = np.zeros_like(img_rgb[0])
        img_black_2 = np.zeros_like(img_rgb[0])
        img_rgb.append(img_black_1)
        img_rgb.append(img_black_2)
    if len(img_path)==2:
        img_black = np.zeros_like(img_rgb[0])
        img_rgb.append(img_black)
    img_rgb = np.concatenate(img_rgb,axis=2)
    return img_rgb

def gray_to_array(img_path):
    img_array = []
    for i in img_path:
        img_array.append(np.expand_dims(np.asarray(Image.open(i)),axis=2))
    img_array = np.concatenate(img_array,axis=2)
    return img_array
def save_numpy(img,name):
    np.save(f'{name}.npy',img)

def interp_1d(x,y,ratio):
    func = interpolate.interp1d(x,y,kind='cubic')
    x_new = np.linspace(x[0],x[-1],len(x)*ratio)
    y_new = func(x_new)
    y_new[y_new<0.015] = 0
    return x_new , y_new