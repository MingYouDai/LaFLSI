# Ultracompact multilabel high-resolution wide-field fluorescence microscopy based on the lifetime-spectrum transform

Mingyou Dai1, Tao Yue1, Tao Yu, Jimeng Liao, Xun Cao, Xuemei Hu.

This code implements a reconstruction algorithm based on TAU-Net.


## Testing
The function of this example code is to reconstruct a dual channel fluorescence intensity map and corresponding lifetime spectrum by inputting two collected fluorescence images.

In ./test_data/raw, 0001.BMP and 0001.BMP are the original images collected by the system.

./checkpoint/model_state_dict.pth is the parameter of the TAU-Net.

The structure of TAU-Net is in ./model.

Running main.py can obtain the reconstruction result of TAU-Net in ./restored_images.


## Requirements

The code has been tested with Python 3.8.12 using PyTorch1.13.0 running on Linux and the following library packages are installed to run this code:

```
PyTorch >= 1.13.0
PIL == 9.2.0
opencv-python == 4.7.0.68
Numpy
Scipy
matplotlib
```
