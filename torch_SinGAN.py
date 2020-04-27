from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from skimage import io as img
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm.notebook import tqdm

def img_resize_tensor(img,scale_factor,opt):
    """wrapper for resize function, img_resize_main is for w*h*c format"""
    img = tensor_to_np_format_tensor(img)
    img = img_resize_main(img, scale_factor, opt.device)
    img = np_format_tensor_to_tensor(img)
    return img

def tensor_to_np_format_tensor(x):
    """convert a [-1,1] s*c*w*h tensor to a [0,255] w*h*c tensor"""
    x = denorm(x[0].permute((1,2,0)))*255
    return x

def np_format_tensor_to_tensor(x):
    """convert a [0,255] w*h*c tensor to a [-1,1] s*c*w*h tensor"""
    x = x[:,:,:,None].permute(3, 2, 0, 1)/255
    return norm(x)

def img_resize_main(img, scale_factor, device):
    #for normal image, rescale the first 2 dim, with channel remain 1
    scale_factor = torch.tensor([scale_factor, scale_factor, 1], device = device)
    output_shape = torch.ceil(torch.cuda.FloatTensor(list(img.shape)) * scale_factor)
    #define the interpolation method
    method = cubic
    #define the kernel width
    kernel_width = torch.tensor(4.0, device = device)
    #if we are scaling down use antialising
    antialiasing = (scale_factor[0] < 1)
    
    sorted_dims = torch.argsort(scale_factor)
    out_image = img.clone()
    for dim in sorted_dims:
        #for non-scaled dim, nothing happened
        if scale_factor[dim] == 1:
            continue
        # for each coordinate (along dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        weights, field_of_view = contributions(in_length = img.shape[dim.item()], 
                                                 out_length = output_shape[dim.item()], 
                                                 scale = scale_factor[dim.item()],
                                                 kernel = method, 
                                                 kernel_width = kernel_width, 
                                                 antialiasing = antialiasing,
                                                 device = device)
        #resize the image by one dimension
        out_image = resize_along_dim(image = out_image, 
                                       dim = dim, 
                                       weights = weights, 
                                       field_of_view = field_of_view,
                                       device = device)
        
    return out_image

def cubic(x):
    """cubic interpolation, for tensor"""
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))

def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing, device):
    """get the weight and view"""
    # When anti-aliasing is activated, the receptive field is stretched to size of
    # 1/sf. this means filtering is more 'low-pass filter'.
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0
    
    # coordinates of the output image
    out_coordinates = torch.arange(1, out_length.item()+1, device = device)
    
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)
    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    left_boundary = torch.floor(match_coordinates - kernel_width / 2)
    
    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    expanded_kernel_width = torch.ceil(kernel_width) + 2
    
    #Determine a set of field_of_view for each each output position, 
    #these are the pixels in the input image that the pixel in the output image 'sees'.
    field_of_view = torch.squeeze((torch.unsqueeze(left_boundary, 1) + \
                                   torch.arange(expanded_kernel_width.item(),device = device) - 1).byte())
    
    # weight to each pixel in the field of view.
    weights = fixed_kernel(1.0 * torch.unsqueeze(match_coordinates, 1) - field_of_view - 1)
    
    # standardize the weights
    sum_weights = torch.sum(weights, 1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / torch.unsqueeze(sum_weights, 1)
    
    #(0,1,2,...in_length-1,in_length-1,in_length-2,...1,0),reflection padding at the boundaries
    mirror = torch.cat((torch.arange(in_length, device = device), 
                        torch.arange(in_length - 1, -1, step=-1,device = device)))
    index = torch.fmod(field_of_view, mirror.shape[0]).type(torch.cuda.LongTensor)
    
    # Get rid of  weights and pixel positions that are of zero weight
    field_of_view = torch.take(mirror,index)
    non_zero_out_pixels = torch.squeeze(torch.nonzero(weights.type(torch.bool).any(dim=0)))
    weights = torch.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = torch.squeeze(field_of_view[:, non_zero_out_pixels])
    
    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    return weights, field_of_view

def resize_along_dim(image, dim, weights, field_of_view, device):
    """scale along one dimension"""
    #use dim 0 as the dim be scaled
    tmp_im = torch.transpose(image, dim, 0)
    
    weights = torch.reshape(weights.T, list(weights.T.shape) + (image.dim() - 1) * [1])
    temp = tmp_im[field_of_view.T]
    tmp_out_im = torch.sum(temp * weights.expand(temp.shape).cuda(), dim=0)
    # swap back the axes to the original order
    return torch.transpose(tmp_out_im, dim, 0)

def get_arguments():
    """create a namespace"""
    parser = argparse.ArgumentParser()

    #workspace:
    parser.add_argument('--input_name')
    parser.add_argument('--input_dir', default = os.path.join(os.getcwd(), "input"))
    parser.add_argument('--out', default=os.path.join(os.getcwd(), "output"))
    parser.add_argument('--manualSeed', type = int, default=1)

    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=0)#math.floor(opt.ker_size/2)
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image maximal size at the coarser scale', default=250)
    parser.add_argument('--gen_start_scale', type=int, default = 0)
    parser.add_argument('--additional_scale', type=int, default = 0)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)

    return parser

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cuda")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    return opt

def norm(x):
    """convert [0,1] to [-1,1] scale"""
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def denorm(x):
    """convert [-1,1] to [0,1] scale"""
    out = (x + 1) / 2
    return out.clamp(0, 1)

def np_to_tensor(x):
    """convert [0,255] h*w*c np.array to [-1,1] s*c*h*w tensor"""
    x = x[:,:,:,None].transpose((3, 2, 0, 1))/255
    x = torch.from_numpy(x).to(torch.device('cuda'))
    x = x.type(torch.cuda.FloatTensor)
    x = norm(x)
    return x

def tensor_to_np(x):
    """convert [-1,1] s*c*h*w tensor to [0,255] np.array form, never used, just in case needed"""
    x = x[0]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def read_image(opt):
    """read a image by name"""
    x = img.imread(f'{opt.input_dir}/{opt.input_name}')
    return np_to_tensor(x)

def read_image_np_array(x):
    """wrapper of np_to_tensor, read from a [0,255] h*w*c array directly, never used, just in case needed"""
    return np_to_tensor(x)

def convert_image_np(img):
    """convert the [-1,1] tensor to [0,1] np.array form"""
    img = denorm(img)
    img = img[-1,:,:,:].to(torch.device('cpu'))
    img = img.numpy().transpose((1,2,0))
    img = np.clip(img,0,1) # bound the value in 0-1
    return img

def upsampling(img,size_x,size_y):
    """the wrapper for bilinear upsampling"""
    upsample = nn.Upsample(size=[round(size_x),round(size_y)],
                           mode='bilinear',
                           align_corners=True)
    return upsample(img)

def get_gradient_penalty(netD, real_data, fake_data, opt):
    """
    The gradient penalty for WGAN, from article https://arxiv.org/abs/1704.00028, named WGAN-GP sometimes
    """
    #generate a random interpolation between real and fake
    mix_constant = torch.rand(1, 1, device = opt.device)
    mix_constant = mix_constant.expand(real_data.size())
    interpolates = mix_constant * real_data + (1 - mix_constant) * fake_data

    #test the calculate the gradient of interpolation generated in discriminator
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=interpolates,
                                    #The “vector” in the Jacobian-vector product. Usually gradients w.r.t. each output.
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(opt.device),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
    
    # by the definition of Gradient penalty
    gradient_penalty = opt.lambda_grad * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generate_scale_coef(real_,opt):
    """set all the scale related coefficient"""
    #the number of scales is the log(min size/actual size),base scale_factor specified + 1, added two param to control
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), opt.scale_factor_init))) + 1 + opt.additional_scale # newly added here
    
    #similarly, scale_to_stop is defined by the max size of lowest scale, without +1 this time
    scale_to_stop = math.ceil(math.log(min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale_to_stop
    
    #defined the first scale be 1 if upcasting, 
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)

    real = img_resize_tensor(real_, opt.scale1, opt)

    #scale factor is defined last in case there's any rounding or other issue
    #the ratio of size to the root of opt.stop_scale
    opt.scale_factor = math.pow(opt.min_size/(min(real.shape[2],real.shape[3])),1/(opt.stop_scale))
    
def generate_noise(size,device):
    """generate a normal noise of s*c*h*w size"""
    return torch.randn(1, size[0], size[1], size[2], device=device)

def generate_reals(real,opt):
    """generate the list of difference scales of real picture"""
    #resize the real image to correct scale
    return [img_resize_tensor(real,math.pow(opt.scale_factor,i),opt) for i in range(opt.stop_scale, -1, -1)]
    
def generate_directory(opt):
    """the directory specified in opt"""
    return f'{opt.out}/{opt.input_name}/layer={opt.num_layer}, additional_scale={opt.additional_scale}, iteration={opt.niter}, scale_factor={opt.scale_factor_init}, alpha={opt.alpha}'

def generate_coarsest(reals,opt):
    """pick the coarest scale real image"""
    real = reals[opt.gen_start_scale]
    #for fresh start
    if opt.gen_start_scale == 0:
        #generate from 0
        return torch.full(real.shape, 0., device=opt.device)
    else:
        #otherwise start from real
        return real
    
def inititialize_weight(layer):
    """for a layer of model, initialize it's weight"""
    if isinstance(layer,nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.02)
    elif isinstance(layer,nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
        
def freeze(model):
    """change a model from training mode to evaluating mode"""
    #only used for disable the param's updating, for tf ver just give model untrainable is fine
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model

def init_NN(opt):
    """initialize a pair of generator and discriminator"""
    #generator initialization:
    G = Generator(opt).cuda()
    G.apply(inititialize_weight)

    #discriminator initialization:
    D = Discriminator(opt).cuda()
    D.apply(inititialize_weight)
    return G, D
    
def save_in_scale(netG,netD,z,directory):
    """save the model and the cumulated noise, called in the sclae"""
    torch.save(netG.state_dict(), f'{directory}/netG.pth')
    torch.save(netD.state_dict(), f'{directory}/netD.pth')
    torch.save(z, f'{directory}/z_opt.pth')
    
def save_checkpoint(Zs,Gs,NoiseAmp, directory):
    """save the checkpoints of training"""
    torch.save(Zs, f'{directory}/Zs.pth')
    torch.save(Gs, f'{directory}/Gs.pth')
    torch.save(NoiseAmp, f'{directory}/NoiseAmp.pth')
    
def load_trained_model(directory):
    """get the direction and load every model trained"""    
    if(os.path.exists(directory)):
        Gs = torch.load(f'{directory}/Gs.pth')
        Zs = torch.load(f'{directory}/Zs.pth')
        reals = torch.load(f'{directory}/reals.pth')
        NoiseAmp = torch.load(f'{directory}/NoiseAmp.pth')
        return Gs,Zs,reals,NoiseAmp
    else:
        raise RuntimeError('no specified trained model exist')
        
# the model is from the original version
class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel,
                                         out_channel,
                                         kernel_size=ker_size,
                                         stride=stride,
                                         padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))
        
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(3,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),
                              max(N,opt.min_nfc),
                              opt.ker_size,
                              opt.padd_size,
                              1)
            self.body.add_module(f'block{i+1}',block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),3,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
        
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        #have y in the center of x with correct shape of x
        h_width = int((y.shape[2]-x.shape[2])/2)
        w_width = int((y.shape[3]-x.shape[3])/2)
        y = y[:,:,h_width:(y.shape[2]-h_width),w_width:(y.shape[3]-w_width)]
        return x+y
    
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(3,N,opt.ker_size,opt.padd_size,1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),
                              max(N,opt.min_nfc),
                              opt.ker_size,
                              opt.padd_size,
                              1)
            self.body.add_module(f'block{i+1}',block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),
                              1,
                              kernel_size=opt.ker_size,
                              stride=1,
                              padding=opt.padd_size)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
    
def forward_pass(Gs,Zs,reals,NoiseAmp,in_s,mode,padding,opt):
    """generate a image scale by scale from the bottom
       when in "random_noise" mode, everytime use a newly generated noise z
       when in "Z_opt" mode, use Z_opt recorded
    """
    # if it's the first scale, do nothing, return G_z = in_s
    G_z = in_s
    if len(Gs) > 0:
        # if in random mode
        if mode == 'random_noise':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            #from each scale
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                # for the first loop
                if count == 0:
                    #generate the 1 channel noise, broadcast it to correct shape
                    z = generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    #direct generate the noise in 3 channel
                    z = generate_noise([3,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                #have a noise surrouded by 0, in shape of Z_opt
                z = padding(z)
    #------------------------------------------------------------
                G_z = padding(G_z)
                #amplify the generated noise, then add with the G_z
                z_in = noise_amp*z+G_z
                #generate a new output use noise-lized G_z with G_z
                G_z = G(z_in.detach(),G_z)
                #resize the graph up with 1/opt.scale_factor
                G_z = img_resize_tensor(G_z,1/opt.scale_factor,opt)
                #in case there's rounding issue
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'Z_opt':
            #the only difference is using Z_opt rather than random z
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = padding(G_z)
                z_in = noise_amp*Z_opt+G_z # for here we use Z_opt instead of generated noise
                G_z = G(z_in.detach(),G_z)
                G_z = img_resize_tensor(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
    return G_z

def train_single_scale(reals, Gs, Zs, in_s, NoiseAmp, opt, scale_num):
    """train every single scale of WGAN, with additional gradient penalty and LSGAN loss"""
    # get the current resized real picture
    real = reals[len(Gs)] 
    opt.real_x, opt.real_y = real.shape[2],real.shape[3]

    #padding width
    pad = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    padding = nn.ZeroPad2d(pad)

    #get alpha from opt
    alpha = opt.alpha
    
    #in the start of each scale z_opt is a tensor of size fixed_noise.shape filled with 0.
    #actually it's always 0 after the first scale.
    z_opt = torch.full([1,3,opt.real_x + 2*pad,opt.real_y + 2*pad],0, device=opt.device)
    
#-------------------------model and optimizer setting-------------------

    #model and optimizer setting
    netG,netD = init_NN(opt)

    #if the number of channel of previous layer = current nfc
    #i.e. the scale is not in the same bin of 4
    if (opt.nfc_prev==opt.nfc):
        #get a warm start from last model
        netG.load_state_dict(torch.load(f'{opt.out_}/{scale_num-1}/netG.pth'))
        netD.load_state_dict(torch.load(f'{opt.out_}/{scale_num-1}/netD.pth'))

    # setup optimizer and learning rate, following the original setting
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)
    
#-------------------------training-------------------

    
    #for number of loop specified
    for epoch in tqdm(range(opt.niter), desc = f"scale {len(Gs)}", leave = False):
        #if it's the first scale, for G need an additional imput
        if (Gs == []):
            #generate a 1 channel noise of size [1,opt.real_x,opt.real_y],give it correct shape and zero pad
            z_opt = generate_noise([1,opt.real_x,opt.real_y], device=opt.device)
            z_opt = padding(z_opt.expand(1,3,opt.real_x,opt.real_y))
            #generate another 1 channel noise of size [1,opt.real_x,opt.real_y],give it correct shape and zero pad
            noise_ = generate_noise([1,opt.real_x,opt.real_y], device=opt.device)
            noise_ = padding(noise_.expand(1,3,opt.real_x,opt.real_y))
        # when it's not the first scale
        else:
            #generate 3-channel noise is fine
            noise_ = generate_noise([3,opt.real_x,opt.real_y], device=opt.device)
            noise_ = padding(noise_)

        #-------------dicriminator-----------------

        for j in range(3):
            
            netD.zero_grad()
            # train with real, generate a result, to minimize -D(True) + D(G(noise)), the mean should be negative
            output = netD(real).to(opt.device)
            errD_real = -output.mean()#mean of a matrix
            errD_real.backward(retain_graph=True)

            # generate the noise
            # for the first loop in the first epoch
            if (j==0) & (epoch == 0):
                #if it's the first scale(very first loop)
                if (Gs == []):
                    #set prev to all 0, zero padding
                    prev = torch.full([1,3,opt.real_x,opt.real_y], 0., device=opt.device)
                    in_s = prev
                    prev = padding(prev)

                    #set z_prev to all 0,padding
                    z_prev = torch.full([1,3,opt.real_x,opt.real_y], 0., device=opt.device)
                    z_prev = padding(z_prev)

                    opt.noise_amp = 1
                #if it's the first loop of other scales
                else:
                    # generate a result from bottom to current-1 scale, all from random noise
                    prev = forward_pass(Gs,Zs,reals,NoiseAmp,in_s,'random_noise',padding,opt)
                    prev = padding(prev)
                    
                    # generate a result from bottom to current-1 scale, all by Z_opt
                    z_prev = forward_pass(Gs,Zs,reals,NoiseAmp,in_s,'Z_opt',padding,opt)
                    
                    #update the noise amplifier by the RMSE between actual and best result z_prev
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    
                    # add a padding after the correct shape is used for RMSE
                    z_prev = padding(z_prev)

            #for non-first loop
            else:
                # generate a result from bottom to current-1 scale, all from random noise
                prev = forward_pass(Gs,Zs,reals,NoiseAmp,in_s,'random_noise',padding,opt)
                prev = padding(prev)

            #if it's the first scale
            if (Gs == []):
                #use the noise_ as "formal" one
                noise = noise_
            else:
                #amplify the padded noise then add prev
                noise = opt.noise_amp*noise_+prev

            # train with fake generated from noise, to minimize -D(true) + D(G(noise)), 
            # the output should be positive to be minimized
            fake = netG(noise.detach(),prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)

            #calculate the penalty,calculate gradient penalty on to D(the ref see doc of this function)
            gradient_penalty = get_gradient_penalty(netD, real, fake, opt)
            gradient_penalty.backward()

            optimizerD.step()
            
        schedulerD.step()

        #-----------generator------------------

        for j in range(3):
            # init to 0, generate the output from the discrimator, minimize -D(G(z))
            netG.zero_grad()
            output = netD(fake)
            errG = -output.mean()
            errG.backward(retain_graph=True)

            #use some idea of LSGAN(https://arxiv.org/pdf/1611.04076.pdf), have alpha*MSE as a part of loss
            #amplify the z and add them into a "cumulative weighted sum" of Z
            Z_opt = opt.noise_amp*z_opt+z_prev
            loss = nn.MSELoss()
            rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),real)
            rec_loss.backward(retain_graph=True)

            optimizerG.step()
            
        schedulerG.step()

        #at the end of the iteration save the result
        if epoch == (opt.niter-1): 
            plt.imsave(f'{opt.outf}/fake_sample.png', convert_image_np(fake.detach()), vmin=0, vmax=1)
            plt.imsave(f'{opt.outf}/G(z_opt).png',  convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            torch.save(z_opt, f'{opt.outf}/z_opt.pth')

    # save the models and z_opt
    save_in_scale(netG,netD,z_opt,opt.outf)
    
    return freeze(netG), z_opt.detach(),in_s.detach()

def train(opt,Gs,Zs,NoiseAmp):
    """the main training procedure"""
    #from the name get the picture
    real_ = read_image(opt)
    opt.out_ = generate_directory(opt)
    #scale1 is defined from adjust2scale, saved in opt
    real = img_resize_tensor(real_,opt.scale1,opt)
    #generate a list of resized images and save
    reals = generate_reals(real,opt)
    #print([i.shape for i in reals])
    torch.save(reals, f'{opt.out_}/reals.pth')
    opt.nfc_prev = 0
    in_s = 0

    #for scale 0 to stop scale
    for scale_num in tqdm(range(opt.stop_scale+1), desc = opt.input_name, leave = True):

        #define the number of channels and the minimum number of channels in this scale
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        #the output main directory and the output sub directory for each scale, need create the directory for the scale
        opt.outf = f'{opt.out_}/{scale_num}'
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass
        #save the resized original image for this scale
        plt.imsave(f'{opt.outf}/real_scale.png', convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        #train a single scale, get the current z, in_s, generator
        G_curr, z_curr,in_s = train_single_scale(reals = reals,
                                                 Gs = Gs,
                                                 Zs = Zs,
                                                 in_s = in_s,
                                                 NoiseAmp = NoiseAmp,
                                                 opt = opt,
                                                 scale_num = scale_num
                                                )
        
        # save them into the list and save the record as a "checkpoint"
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)
        save_checkpoint(Zs,Gs,NoiseAmp,opt.out_)

        opt.nfc_prev = opt.nfc
        
        del G_curr
        
def generate_augmentation(Gs, Zs, reals, NoiseAmp, opt, in_s=None, gen_start_scale=0, num_generated=50, save_image = False):
    """main generating procedure, generate num_samples augmentation images from a model"""
    images_cur = []
    if (save_image):
        directory = os.path.join(generate_directory(opt),"image_generated")
        try:
            os.makedirs(directory)
        except OSError:
            pass
    output_list = []
    #for each layers
    for G,Z_opt,noise_amp,n in zip(Gs,Zs,NoiseAmp,range(len(Gs))):
        #generate a pad class with width ((ker_size-1)*num_layer)/2
        pad = ((opt.ker_size-1)*opt.num_layer)/2
        padding = nn.ZeroPad2d(int(pad))

        #the shape inside padding * scale
        real_x = int(Z_opt.shape[2]-pad*2)
        real_y = int(Z_opt.shape[3]-pad*2)
        noise_channel = 3

        #get all the previsous image
        images_prev = images_cur
        images_cur = []
        
        #for the number of samples
        for i in range(0,num_generated,1):
            if n == 0:
                #generate the single channel noise, broadcast to the correct shape
                z_curr = generate_noise([1,real_x,real_y], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                #padding it
                z_curr = padding(z_curr)
            else:
                #generate noise with defined shape
                z_curr = generate_noise([int(3),real_x,real_y], device=opt.device)
                #padding
                z_curr = padding(z_curr)
            #if it's the first scale
            if images_prev == []:
                #use in_s as the first one
                I_prev = padding(in_s)
            else:
                #get the last image
                I_prev = images_prev[i]
                #resize it by 1/scale_factor
                
                I_prev = img_resize_tensor(I_prev,1/opt.scale_factor,opt)
                # in case there's rounding issue
                I_prev = I_prev[:, :, 0:reals[n].shape[2], 0:reals[n].shape[3]]
                #padding
                I_prev = padding(I_prev)
                I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                #upsample this piece to original shape, with bilinear policy
                I_prev = upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])

            # amplify the z by the param, add the previous graph
            z_in = noise_amp*(z_curr)+I_prev

            # pass this value and previous graph to generator, get the value
            I_curr = G(z_in.detach(),I_prev)

            #for the last loop
            if n == len(reals)-1:
                img = I_curr.detach()
                # have the generated image into the list
                output_list.append(convert_image_np(img))
                if (save_image):
                    #save the new generated image
                    plt.imsave(f'{directory}/{i}.png', convert_image_np(img), vmin=0,vmax=1)
            images_cur.append(I_curr)
    return output_list

def train_model(input_name, layer_number = 5, epochs = 2000, additional_scale = 0):
    """wrapper of whole training procedure"""
    #configure the option
    parser_train = get_arguments()
    opt_train = parser_train.parse_args(["--input_name", input_name, 
                                         "--gen_start_scale", str(0),
                                         "--num_layer", str(layer_number),
                                         "--niter", str(epochs),
                                         "--additional_scale", str(additional_scale),
                                        ])
    opt_train = post_config(opt_train)
    # follows the SinGan operation process, slightly simplified
    Gs = []
    Zs = []
    NoiseAmp = []
    #save path(note this function is modified)
    directory = generate_directory(opt_train)
    #if there's existed direction, stop it
    if (os.path.exists(directory)):
        print(f'layer={opt_train.num_layer}, scale = {opt_train.additional_scale},iteration={opt_train.niter}, scale_factor={opt_train.scale_factor_init}, alpha={opt_train.alpha} model for {opt_train.input_name} already exist')
    #else run the training
    else:
        try:
            os.makedirs(directory)
        except OSError:
            pass
        #read the image
        real = read_image(opt_train)
        #decide scales
        generate_scale_coef(real, opt_train)
        #train
        train(opt_train, Gs, Zs, NoiseAmp)
        
def generate_data(dataset, 
                  class_label = 3, 
                  layer_number = 5, 
                  additional_scale = 0, 
                  generate_size = 50, 
                  model_size = 5, 
                  generate_start_scale = 0):
    """wrapper of whole augmentation data generating procedure, 
       generate ceil(generate_size/model_size) form from each model, shuffle, return generate_size of them
       the data returned is in s*h*w*c [0,1] images np.array form
    """
    #Configures
    INPUT_NAME = [f"{dataset}_{class_label}_input_{i}.png" for i in range(model_size)]
    # the list takes all the result
    output = []
    parser_generate = get_arguments()
    # for all the trained samples
    for i in tqdm(range(model_size), desc = "loading", leave = False):
        #run standard procedure, create a namespace
        opt_generate = parser_generate.parse_args(["--input_name", INPUT_NAME[i], 
                                                   "--gen_start_scale", str(generate_start_scale),
                                                   "--num_layer", str(layer_number),
                                                   "--additional_scale", str(additional_scale),
                                                  ])
        opt_generate = post_config(opt_generate)
        
        directory = generate_directory(opt_generate)
        #read the file
        real = read_image(opt_generate)
        #adjust scales, write into opt_generate
        generate_scale_coef(real, opt_generate)
        Gs, Zs, reals, NoiseAmp = load_trained_model(directory)
        #generate coarest graph
        in_s = generate_coarsest(reals, opt_generate)
        #generate the output(the function is modified to generate list_output)
        list_output = generate_augmentation(Gs, Zs, reals, NoiseAmp, opt_generate, in_s,
                                            gen_start_scale=opt_generate.gen_start_scale, 
                                            num_generated = int(np.ceil(generate_size/model_size)),
                                            save_image = False)
        output += list_output
        
    output = np.array(output)
    np.random.shuffle(output)
    return output[:generate_size]