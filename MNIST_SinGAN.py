#This file is the final version of MNIST data processing
#Note it's used for the modified version of SinGAN under directory SinGAN/.. not the official version --Alex
import numpy as np
import os
import torch
from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
torch.manual_seed(1)

def train_model(input_name, layer_number = 6, random_seed = 1, epochs = 2000, Scale_plus1 = True, additional_scale = False):
    #configure the option
    INPUT_DIR = os.path.join(os.getcwd(), "input")
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")
    RANDOM_SEED = random_seed
    CHANNEL = 3 #Channel = 1 for SinGan still expect 3 channel, so use this
    INPUT_NAME = input_name
    LAYER_NUMBER = layer_number #5 layer * 3 scale is not quite okay for number 9
    EPOCHS = epochs
    SCALE_PLUS1 = Scale_plus1
    ADDITIONAL_SCALE = additional_scale
    GENERATION_START_SCALE = 0 # for generating 50 examples in the model document
    parser_train = get_arguments()
    parser_train.add_argument('--input_dir')
    parser_train.add_argument('--input_name')
    parser_train.add_argument('--mode')
    
    #newly added, have change in functions.adjust_scales2image --yihao
    parser_train.add_argument('--scale_plus1', type=int, default = 1)
    parser_train.add_argument('--additional_scale', type=int, default = 0)
    parser_train.add_argument("--gen_start_scale", type=int)
    
    opt_train = parser_train.parse_args(["--input_dir", INPUT_DIR, 
                                         "--input_name", INPUT_NAME, 
                                         "--mode", "train",
                                         "--manualSeed", str(RANDOM_SEED),
                                         "--out", OUTPUT_DIR,
                                         "--gen_start_scale", str(GENERATION_START_SCALE),
                                         "--num_layer", str(LAYER_NUMBER),
                                         "--nc_z", str(CHANNEL),
                                         "--nc_im", str(CHANNEL),
                                         "--niter", str(EPOCHS),
                                         "--scale_plus1", str(int(SCALE_PLUS1)),
                                         "--additional_scale", str(int(ADDITIONAL_SCALE))
                                        ])
    opt_train = functions.post_config(opt_train)
    
    # follows the SinGan operation process, slightly simplified
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    
    #save path(note this function is modified)
    dir2save = functions.generate_dir2save(opt_train)
    #if there's existed direction, stop it
    if (os.path.exists(dir2save)):
        print('layer={opt_train.num_layer}, iteration={opt_train.num_niter}, scale_factor={opt_train.scale_factor_init}, alpha={opt_train.alpha} model for {opt.input_name} already exist')
    #else run the training
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        #read the image
        real = functions.read_image(opt_train)
        #decide scales
        functions.adjust_scales2image(real, opt_train)
        #time the training
        start = time.time()
        #train
        train(opt_train, Gs, Zs, reals, NoiseAmp)
        #stop timing
        end = time.time()
        print(f"traing time {end - start}s, for n_iter = {opt_train.niter}, layer_number = {opt_train.num_layer}")
        # generate the example 50 graphs
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, 
                        opt_train, 
                        gen_start_scale=opt_train.gen_start_scale, 
                        output_image = True)
    
    
def generate_data(class_label = 9, generate_size = 50, random_seed = 1, sample_size = 5):
    '''return generated image in np.array form'''
    #Configures
    INPUT_DIR = os.path.join(os.getcwd(), "input")
    RANDOM_SEED = random_seed
    GENERATION_START_SCALE = 0
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")
    CHANNEL = 3 
    INPUT_NAME = [f"MNIST_{class_label}_input_{i}.png" for i in range(sample_size)]
    LAYER_NUMBER = 6
    SCALE_PLUS1 = True
    ADDITIONAL_SCALE = False
    # the list takes all the result
    output = []
    parser_generate = get_arguments()
    parser_generate.add_argument('--input_dir')
    parser_generate.add_argument('--input_name')
    parser_generate.add_argument('--mode')
    parser_generate.add_argument('--gen_start_scale', type=int)
    parser_generate.add_argument('--scale_plus1', type=int)
    parser_generate.add_argument('--additional_scale', type=int)
    # for all the trained samples
    for i in range(sample_size):
        #run standard procedure, create a namespace
        opt_generate = parser_generate.parse_args(["--input_dir", INPUT_DIR, 
                                                   "--input_name", INPUT_NAME[i], 
                                                   "--mode", "random_samples",
                                                   "--manualSeed", str(RANDOM_SEED),
                                                   "--gen_start_scale", str(GENERATION_START_SCALE),
                                                   "--out", OUTPUT_DIR,
                                                   "--num_layer", str(LAYER_NUMBER),
                                                   "--nc_z", str(CHANNEL),
                                                   "--nc_im", str(CHANNEL),
                                                   "--scale_plus1", str(int(SCALE_PLUS1)),
                                                   "--additional_scale", str(int(ADDITIONAL_SCALE))
                                                  ])
        opt_generate = functions.post_config(opt_generate)
        print(opt_generate.input_name)
        Gs = []
        Zs = []
        reals = []
        NoiseAmp = []
        #save the path(note this function is modified)
        dir2save = functions.generate_dir2save(opt_generate)
        print(dir2save)
        #read the file
        real = functions.read_image(opt_generate)
        #adjust scales, write into opt_generate
        functions.adjust_scales2image(real, opt_generate)
        #load the modesl
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt_generate)
        #generate coarest graph
        in_s = functions.generate_in2coarsest(reals,1,1,opt_generate)
        #generate the output(the function is modified to generate list_output)
        current_output, list_output = SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt_generate, 
                                                      gen_start_scale=opt_generate.gen_start_scale, 
                                                      output_image = False, 
                                                      num_samples = int(np.ceil(generate_size/sample_size)))
        output += list_output
    #change list into np.array
    output = np.array(output)
    #shuffle it by the first axis
    np.random.shuffle(output)
    return output[:generate_size]