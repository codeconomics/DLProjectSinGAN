import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

def train(opt,Gs,Zs,reals,NoiseAmp):
    #from the name get the picture
    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    #scale1 is defined from adjust2scale, saved in opt
    real = imresize(real_,opt.scale1,opt)
    # a list of resized images
    reals = functions.creat_reals_pyramid(real,reals,opt)
    nfc_prev = 0

    #for scale 0 to stop scale
    for scale_num in tqdm_notebook(range(opt.stop_scale+1), desc = opt.input_name, leave = True):

        #define the number of channels in this scale
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        #define the minimum number of channels in this scale
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        #the output main directory
        opt.out_ = functions.generate_dir2save(opt)

        #the output sub directory for each scale
        opt.outf = f'{opt.out_}/{scale_num}'

        #if need create the directory
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #save the resized original image for this scale
        plt.imsave(f'{opt.outf}/real_scale.png', functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        #create the generator and discriminator
        D_curr,G_curr = init_models(opt)

         #if the number of channel of previous layer = current nfc
        if (nfc_prev==opt.nfc):

            #direct load the weightfrom last model
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

         #train a single scale, get the current z, in_s, generator
        z_curr,in_s,G_curr = train_single_scale(D_curr,#current discriminator
                                                G_curr,#current generator
                                                reals,#the list of all resized data
                                                Gs,#generator list
                                                Zs,# a list initialized as []
                                                in_s,#
                                                NoiseAmp,#
                                                opt #parameters
                                               )

        #make current G and D untrainable,set it into eval mode
        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        # save them into the list
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        #save the checkpoints
        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        nfc_prev = opt.nfc
        #delete the D and G for memory
        del D_curr,G_curr
    return



def train_single_scale(netD,#current discriminator
                       netG,#current generator
                       reals,#the list of all resized data
                       Gs,#generator list
                       Zs,#
                       in_s,#
                       NoiseAmp,#
                       opt,#parameters
                       centers=None):


    real = reals[len(Gs)] # get the current resized real picture

    #get the x and y
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]

    #receptive field
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride

    #padding width
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    #this stuff create a torch.nn class adding 0 pads, tf is slightly harder
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    #get alpha from opt
    alpha = opt.alpha

    #generate a noise in the following size
    fixed_noise = functions.generate_noise([opt.nc_z,#noise # channels
                                            opt.nzx,
                                            opt.nzy],
                                           device=opt.device)

    z_opt = torch.full(fixed_noise.shape,
                       0,
                       device=opt.device)
    #generate a tensor of size fixed_noise.shape filled with 0.

    z_opt = m_noise(z_opt)
    #give it a zero pad with width int(pad_noise)

    # setup optimizer and learning rate
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    #some plot list
    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    #for iteration number' loop
    for epoch in tqdm_notebook(range(opt.niter), desc = f"scale {len(Gs)}", leave = False):
        #if it's the first graph, for G need an additional imput
        if (Gs == []):
            #generate a noise of size [1,opt.nzx,opt.nzy]
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            #give it a zero pad with width int(pad_noise)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            #generate another noise
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            #give it additional dimention with size 3, in all these dimension all 3 layers are the same
            #the padding it in all the dimension
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        # when it's not the first graph
        else:
            #nc_z is 'noise # channels'
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            #the padding it in all the dimension
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        # for Discriminator inner steps' loop
        for j in range(opt.Dsteps):

            # train with real
            #before training reset grad, torch operation
            netD.zero_grad()
            #generate a result
            output = netD(real).to(opt.device)
            #error for D, to minimize -(D(x) + D(G(z))), the mean should be -1
            errD_real = -output.mean()#-a
            # have all the gradients computed
            errD_real.backward(retain_graph=True)
            #return the list with all dictionary keys with negative values
            D_x = -errD_real.item()

            # train with fake
            # for the first loop in the first epoch
            if (j==0) & (epoch == 0):
                #if it's the first scale
                if (Gs == []):
                    #set prev to all 0
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)

                    in_s = prev
                    #zero padding with width int(pad_image)
                    prev = m_image(prev)

                    #set z_prev to all 0
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)

                    #padding with noise
                    z_prev = m_noise(z_prev)

                    #set amp = 1
                    opt.noise_amp = 1
                else:
                    # generate the prev from rand mode
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    #zero padding with width int(pad_image)
                    prev = m_image(prev)
                    # generate the z_prev from rec mode
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)

                    #use opt.noise_amp_init*RMSE as the loss
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    # add a padding of width (int(pad_image))
                    z_prev = m_image(z_prev)
            else:
                #generate the prev form rand mode
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                # add a padding of width (int(pad_image))
                prev = m_image(prev)

            #if it's the first scale
            if (Gs == []):
                #a noise added additional dimention with size 3, in all these dimension all 3 layers are the same
                #the padding it in all the dimension
                noise = noise_
            else:
                #amplify the padded noise + prev
                noise = opt.noise_amp*noise_+prev

            # generate the fake graph with noise
            # detach() detaches the output from the computationnal graph.
            # So no gradient will be backproped along this variable
            # in the very first loop G is RAW now
            fake = netG(noise.detach(),prev)
            # generate the output
            output = netD(fake.detach())

            # generate the error from fake, to minimize -(D(x) + D(G(z))), the mean should be positive
            errD_fake = output.mean()
            # have all the gradients computed
            errD_fake.backward(retain_graph=True)
            #get the discriminator
            D_G_z = output.mean().item()

            #calculate the penalty
            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            #calculate gradient
            gradient_penalty.backward()

            #calculate penal D
            errD = errD_real + errD_fake + gradient_penalty

            #updates the parameters.
            optimizerD.step()

        #add the stuff into a record
        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            # init to 0
            netG.zero_grad()
            # generate the output from the discrimator
            output = netD(fake)
            #the the loss of G is negative to result of D for competition
            errG = -output.mean()
            #calculate the backward
            errG.backward(retain_graph=True)

            if alpha!=0:
                #define MSE loss
                loss = nn.MSELoss()
                #amplify the z
                Z_opt = opt.noise_amp*z_opt+z_prev
                #use the result generate from Z_opt.detach(),z_prev, calculate the MSE with real, scale with alpha
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),real)
                #backward
                rec_loss.backward(retain_graph=True)
                #get a number loss
                rec_loss = rec_loss.detach()
            else: #alpha = 0
                #else get Z as z
                Z_opt = z_opt
                #set the rec_loss = o
                rec_loss = 0

            #update the result
            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        #if epoch % 25 == 0 or epoch == (opt.niter-1): #replaced by tqdm
            #print(f'scale {len(Gs)}:[{epoch}/{opt.niter}]'

        #if epoch % 500 == 0 or epoch == (opt.niter-1):
        if epoch == (opt.niter-1): #only saved once (for small graph)
            #save the fake sample
            plt.imsave(f'{opt.outf}/fake_sample.png', functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            #save the z_opt
            plt.imsave(f'{opt.outf}/G(z_opt).png',  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            #save the model
            torch.save(z_opt, f'{opt.outf}/z_opt.pth')
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
        #update learning rate
        schedulerD.step()
        schedulerG.step()

    # save the model
    functions.save_networks(netG,netD,z_opt,opt)

    # return the z, in_s(what's this), and generator G
    return z_opt,in_s,netG

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    # if it's not the first scale, else do nothign
    if len(Gs) > 0:
        # if in random mode
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            #from each scale
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                # for the first loop
                if count == 0:
                    #generate the noise
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    #broadcast it to correct shape
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    #direct generate the noise
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                #padding the noise
                z = m_noise(z)
    #------------------------------------------------------------
                #generate a shape of current real image's [width,height] from G_z(in_s)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                #padding it with images
                G_z = m_image(G_z)
                #amplify the generated noise, then add with the G_z
                z_in = noise_amp*z+G_z
                #generate a new output from generator
                G_z = G(z_in.detach(),G_z)
                #resize the graph with 1/opt.scale_factor
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                #generate a shape of current real image's [width,height] from G_z(in_s)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            #from each scale
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                # do same thing except
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z # for here we use Z_opt instead of generated noise
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
