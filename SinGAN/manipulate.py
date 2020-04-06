from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments

def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=50, output_image = False):
    #if torch.is_tensor(in_s) == False:
    if in_s is None:
        # make in_s a 0 tensor with reals[0] shape
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    #for each layers
    for G,Z_opt,noise_amp in zip(Gs,Zs,NoiseAmp):
        #generate a pad class with width ((ker_size-1)*num_layer)/2
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))

        #the shape inside padding * scale
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h

        #get all the previsous image
        images_prev = images_cur
        images_cur = []
        output_list = []
        #for the number of samples
        for i in range(0,num_samples,1):
            if n == 0:
                #generate the noise
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                #broadcast to the correct shape
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                #padding it
                z_curr = m(z_curr)
            else:
                #generate noise with defined shape
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                #padding
                z_curr = m(z_curr)
            #if it's the first scale
            if images_prev == []:
                #use in_s as the first one
                I_prev = m(in_s)
                #I_prev = m(I_prev)
                #I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                #get the last image
                I_prev = images_prev[i]
                #resize it by 1/scale_factor
                I_prev = imresize(I_prev,1/opt.scale_factor, opt)
                # cut a piece of shape (round(scale_v * reals[n].shape[2] * round(scale_h * reals[n].shape[3]))
                I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                #padding
                I_prev = m(I_prev)
                #cut a piece of shape (z_curr.shape[2], z_curr.shape[3])
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #upsample this piece to original shape, with bilinear policy
                I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            # amplify the z by the param, add the previous graph
            z_in = noise_amp*(z_curr)+I_prev

            # pass this value and previous graph to generator, get the value
            I_curr = G(z_in.detach(),I_prev)

            #for the last loop
            if n == len(reals)-1:
                #generate the directory
                dir2save = functions.generate_dir2save(opt)#modified
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                # new variable
                if (output_image):
                    #save the new generated image
                    plt.imsave(f'{dir2save}/{i}.png', functions.convert_image_np(I_curr.detach()), vmin=0,vmax=1)
                # have the generated image into the list
                output_list.append(functions.convert_image_np(I_curr.detach()))
            images_cur.append(I_curr)
        n+=1
    return I_curr.detach(), output_list#newly added
