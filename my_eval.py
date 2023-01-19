from PIL import Image

orig_img = Image.open('my_image.png')
noisy_img = Image.open('sp_noise.png')
width,height=orig_img.size

import torchvision.transforms as T
import torchvision.transforms.functional as Fun
import matplotlib.pyplot as plt

from numpy import asarray
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

import scipy.io as sio
from dataset.dataset_denoise import *
import utils
import math
from model import UNet,Uformer

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

save_file = os.path.join('./results/', 'myresult')
utils.mkdir(save_file)

patch_size=512
patch_colnum=width//patch_size
patch_rownum=height//patch_size

def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask

Inoisy = np.zeros((patch_rownum*patch_colnum,patch_size,patch_size,3),dtype='float32')
for i in range(patch_rownum):
    for j in range(patch_colnum):
        crop=asarray(Fun.crop(noisy_img,patch_size*i,patch_size*j,patch_size,patch_size))
        crop=np.float32(np.copy(crop))
        crop /=255.
        Inoisy[patch_colnum*i+j]=crop
restored = np.zeros_like(Inoisy)

gnd_truth=asarray(Fun.crop(orig_img,0,0,patch_size*(i+1),patch_size*(j+1)))
gnd_truth=np.float32(np.copy(gnd_truth))
gnd_truth /=255.
temp_save_file = os.path.join(save_file, 'ground_truth.png')
utils.save_img(temp_save_file, img_as_ubyte(gnd_truth))

ipt_img=asarray(Fun.crop(noisy_img,0,0,patch_size*(i+1),patch_size*(j+1)))
ipt_img=np.float32(np.copy(ipt_img))
ipt_img /=255.
temp_save_file = os.path.join(save_file, 'noise_image.png')
utils.save_img(temp_save_file, img_as_ubyte(ipt_img))

# save cropped region for pre
gndtruth_crop = Image.open('./results/myresult/ground_truth.png')
gndtruth_crop=asarray(Fun.crop(gndtruth_crop,patch_size*0.5,patch_size*4.5,patch_size,patch_size))
gndtruth_crop=np.float32(np.copy(gndtruth_crop))
gndtruth_crop /=255.
temp_save_file = os.path.join(save_file, 'crop_gndtruth.png')
utils.save_img(temp_save_file, img_as_ubyte(gndtruth_crop))

# save cropped region for pre
noisy_crop = Image.open('./results/myresult/noise_image.png')
noisy_crop=asarray(Fun.crop(noisy_crop,patch_size*0.5,patch_size*4.5,patch_size,patch_size))
noisy_crop=np.float32(np.copy(noisy_crop))
noisy_crop /=255.
temp_save_file = os.path.join(save_file, 'crop_noisy.png')
utils.save_img(temp_save_file, img_as_ubyte(noisy_crop))


model_restoration = Uformer(img_size=128,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
            depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=3)  

utils.load_checkpoint(model_restoration,'./logs/denoising/SIDD/Uformer_B/models/model_best.pth')
print("===>Testing using weights: ", './logs/denoising/SIDD/Uformer_B/models/model_best.pth')

model_restoration.cuda()
model_restoration.eval()

# restore using model
with torch.no_grad():
    for i in range(patch_rownum):
        for j in range(patch_colnum):
            noisy_patch = torch.from_numpy(Inoisy[patch_colnum*i+j,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            _, _, h, w = noisy_patch.shape
            noisy_patch, mask = expand2square(noisy_patch, factor=128) 
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.masked_select(restored_patch,mask.bool()).reshape(1,3,h,w)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            if j==0:
                full_restored_row=restored_patch.clone()
            else:
                full_restored_row = torch.cat((full_restored_row, restored_patch), 1)
            
        if i==0:
            full_restored=full_restored_row.clone()
        else:
            full_restored=torch.cat((full_restored, full_restored_row), 0)

# save restored image       
temp_save_file = os.path.join(save_file, 'restored.png')
utils.save_img(temp_save_file, img_as_ubyte(full_restored))

# calculate PSNR
ipt_img=torch.tensor(ipt_img).cuda()
gnd_truth=torch.tensor(gnd_truth).cuda()

psnr_raw_rgb=utils.myPSNR(ipt_img, gnd_truth)
psnr_val_rgb=utils.myPSNR(full_restored.cuda(), gnd_truth)
print("raw PSNR is: %.4f\t" % psnr_raw_rgb)
print("restored PSNR is: %.4f\t" % psnr_val_rgb)

# save cropped region for pre
restored_img = Image.open('./results/myresult/restored.png')
restore_crop=asarray(Fun.crop(restored_img,patch_size*0.5,patch_size*4.5,patch_size,patch_size))
restore_crop=np.float32(np.copy(restore_crop))
restore_crop /=255.
temp_save_file = os.path.join(save_file, 'crop_restore.png')
utils.save_img(temp_save_file, img_as_ubyte(restore_crop))