import torch
import numpy as np
import skimage
from torchvision import transforms

mul_rgb2xyz = torch.FloatTensor([[0.412453,0.212671,0.019334],[0.357580,0.715160,0.119193],[0.180423,0.072169,0.950227]]).T

def lab2rgb_skimage(img_lab:torch.Tensor):
    '''
        Input: (3,256,256) LAB tensor
        Output: (3,256,256) RGB tensor
        use for displaying results
    '''
    imlab = img_lab.permute(1,2,0).numpy()
    img_rgb = skimage.color.lab2rgb(imlab)
    return torch.tensor(img_rgb).permute(2,0,1)
def rgb2lab_skimage(img_rgb:torch.Tensor):
    '''
        Input: (3,256,256) RGB tensor
        Output: (3,256,256) LAB tensor
        via Skimage; does not support GPU
    '''
    imrgb = img_rgb.permute(1,2,0).numpy()
    img_lab = skimage.color.rgb2lab(imrgb)
    return torch.tensor(img_lab).permute(2,0,1)
def rgb2lab_torch(in_rgb:torch.Tensor):
    '''
        Input: (3,256,256) RGB tensor
        Output: (3,256,256) LAB tensor
        Naive rgb->lab transformation
    '''
    img_rgb = in_rgb.view(3, -1)
    img_lab = torch.zeros_like(img_rgb)
    img_xyz = torch.mm(mul_rgb2xyz, img_rgb)
    img_xyz[0] /= 0.950456
    img_xyz[2] /= 1.088754
    thres = 0.008856
    x = img_xyz[0]
    y = img_xyz[1]
    z = img_xyz[2]
    img_lab[0] = torch.where(y > thres, 116*y.pow(1/3)-16, 903.3*y)
    fx = torch.where(x > thres, x.pow(1/3), 7.787*x+16/116)
    fy = torch.where(y > thres, y.pow(1/3), 7.787*y+16/116)
    fz = torch.where(z > thres, z.pow(1/3), 7.787*z+16/116)
    d1 = 0
    d2 = 0
    #img_lab[0] *= 255/100
    img_lab[1] = 500*(fx-fy)+d1+d2
    img_lab[2] = 200*(fy-fz)+d1+d2
    out_lab = img_lab.view(3, 256, 256)
    return out_lab

def rgb2lab_batch(batch_rgb:torch.Tensor, customTransform):
    '''
        Convert [n, 3, 256, 256] RGB to LAB
        Use rgb2lab_torch if customTransform=True, and rgb2lab_skimage otherwise
    '''
    labs = []
    for img in batch_rgb:
        if customTransform:
            newimg = rgb2lab_torch(img)
        else:
            newimg = rgb2lab_skimage(img)
        labs.append(newimg)
    return torch.stack(labs)

def rgb_to_output(net, img):
    img = img.unsqueeze(0)
    labinput = rgb2lab_batch(img, False)
    L = labinput[:,:1,:,:]
    
    return net(L)