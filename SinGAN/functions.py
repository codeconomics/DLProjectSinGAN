import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
#from skimage import morphology
#from skimage import filters
from SinGAN.imresize import imresize
import os
import random
from sklearn.cluster import KMeans


# custom weights initialization called on netG and netD

def read_image(opt):
    #read the image defined in opt then return to a tensor
    x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
    return np2torch(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def convert_image_np(inp):
    if inp.shape[1]==3:#for torch it's channel
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))

    inp = np.clip(inp,0,1) # bound the value in 0-1, np version
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        #reshape the real_cpu image into correct shape and graysclae
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        #ax.imshow(convert_image_np(real_cpu[0,:,:,:].cpu()))
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
    # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])
    # inp = std*
    return inp

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        #return a bilinearly upsampled gaussian
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        #return a sum of two gaussian, fixed gap = 5
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        # uniform variable
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    #plt.title('loss')
    #plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    #bilinearlly upscale the data
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    #only used for disable the param's updating, for tf ver just give model untrainable is fine
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    #a N(0,1) variable in correct shape
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    #linear combination of these two stuff
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)#.cuda()
    #make it a trainable tensor
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    #pass it to discriminator
    disc_interpolates = netD(interpolates)

    #generate teh autograd instance
    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    #The “vector” in the Jacobian-vector product. Usually gradients w.r.t. each output.
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    # vector norm - 1 squared, the mean

    return gradient_penalty

def read_image(opt):
    x = img.imread(f'{opt.input_dir}/{opt.input_name}')
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]#only get the first three channel
    return x

def read_image_dir(dir,opt):
    x = img.imread(str(dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]#only get the first three channel
    return x

def np2torch(x,opt):
    #same to the one in other file
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    #same to the one in other file
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image2np(opt):
    x = img.imread(f'{opt.input_dir}/{opt.input_name}')
    x = x[:, :, 0:3]
    return x

def save_networks(netG,netD,z,opt):
    torch.save(netG.state_dict(), f'{opt.outf}/netG.pth')
    torch.save(netD.state_dict(), f'{opt.outf}/netD.pth')
    torch.save(z, f'{opt.outf}/z_opt.pth')

def adjust_scales2image(real_,opt):
    #opt.num_scales = int((math.log(math.pow(opt.min_size / (real_.shape[2]), 1), opt.scale_factor_init))) + 1
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1 * opt.scale_plus1 + 1 * opt.additional_scale # newly added here

    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))

    opt.stop_scale = opt.num_scales - scale2stop

    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)  # min(250/max([real_.shape[0],real_.shape[1]]),1)

    real = imresize(real_, opt.scale1, opt)
    #opt.scale_factor = math.pow(opt.min_size / (real.shape[2]), 1 / (opt.stop_scale))
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))

    scale2stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return real

def create_reals_pyramid(real,reals,opt):
    real = real[:,0:3,:,:]
    for i in range(0,opt.stop_scale+1,1):
        scale = math.pow(opt.scale_factor,opt.stop_scale-i)
        #resize the real image to correct scale
        curr_real = imresize(real,scale,opt)
        reals.append(curr_real)
    return reals# the image list


def load_trained_pyramid(opt, mode_='train'):
    #get the direction and load every model trained
    dir = generate_dir2save(opt)
    if(os.path.exists(dir)):
        Gs = torch.load(f'%s/Gs.pth' % dir)
        Zs = torch.load('%s/Zs.pth' % dir)
        reals = torch.load('%s/reals.pth' % dir)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
    else:
        print('no appropriate trained model is exist, please train first')
    #opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def generate_in2coarsest(reals,scale_v,scale_h,opt):
    #pick the coarest scale image
    real = reals[opt.gen_start_scale]
    #upsample it back bilinearlly
    real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
    #for fresh start
    #if opt.gen_start_scale == 0:
        #generate from 0
    in_s = torch.full(real_down.shape, 0, device=opt.device)
    #else: #if n!=0
        #otherwise start from real_down
        #in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
    return in_s

def generate_dir2save(opt):
    #some manually defined position
    dir2save = f'{opt.out}/{opt.input_name}/layer={opt.num_layer}, additional_scale={bool(opt.additional_scale)}, iteration={opt.niter}, scale_factor={opt.scale_factor_init}, alpha={opt.alpha}'
    return dir2save

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (opt.input_name[:-4], opt.scale_factor)
    #if opt.mode == 'SR':
    #    opt.alpha = 100

    #if opt.manualSeed is None: seed will be spcified
    #    opt.manualSeed = random.randint(1, 10000)
    #print("Random Seed: ", opt.manualSeed)# stop the output
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    #if torch.cuda.is_available() and opt.not_cuda:
    #    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def calc_init_scale(opt):
    in_scale = math.pow(1/2,1/3)
    iter_num = round(math.log(1 / opt.sr_factor, in_scale))
    in_scale = pow(opt.sr_factor, 1 / iter_num)
    return in_scale,iter_num
