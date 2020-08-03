import math 
import os 
from copy import deepcopy
import time 

import torch 
import torch.backends.cudnn as cudnn 
import torch.nn as nn 
import torch.nn.functional as F 

def init_seeds(seed=0): 
    torch.manual_seed(seed)

    #Reduce randomness 
    if seed = 0: 
        cudnn.deterministic = False
        cudnn.benchmark = False


def select_device(device='',apex=False,batch_size=None): 
    if cuda : 
        c = 1024**2 #bytes in MB 
        ng = torch.cuda.device_count() 
        if ng > 1 and batch_size : #check if batch size is compatible with the num of GPUs
            assert batch_size % ng == 0, 'batch size %g not compatible with GPU count' %(batch_size,ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using Cuda ' + ('Apex' if apex else '') #apex for mixed precision training 
        for i in range(0,ng): #device count 
            if i == 1 : 
                s = ' ' * len(s)

            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
            else : 
                print('Using CPU, very slow now')

            print('') #skip a line
            return torch.device('cuda:0' if cuda else 'cpu')

def find_modules(model,mclass=nn.Conv2d): 
    #finds layer indices matching module class named mclass, in this case Conv2d
    return [i for i,m in enumerate(model.module_list) if isinstance(m,mclass)]

def fuse_conv_and_bn(conv,bn):
    with torch.no_grad(): 
        #init with conv parameters  s
        fusedconv = torch.nn.Conv2d(conv.in_channels,conv.out_channels,kernel_size=conv.kernel_size,stride=conv.stride
                                    padding=conv.padding,bias=True)

        #prepare filters 
        w_conv = conv.weight.clone().view(conv.out_channels,-1)  #reshape conv and save 
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn,w_conv).view(fusedconv.weight.size()))

        #perpare spatial bias 
        if conv.bias is not None: 
            b_conv = conv.bias 
        else : 
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn))

        return fusedconv 

def model_info(model,verbose=False): 
    #plots a line by line description of a PyTorch Model 
    n_p = sum(x.numel() for  x in model.parameters()) #number of params per layer
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad) #number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    try : #FLOPS 
        from thop import profile 
        macs, _ = profile(model,inputs=(torch.zeros()))
        fs = ', %.1f GLFOPS' % (macs/1E9 * 2)
    except : 
        fs = ''
    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))

def load_classifier(): 
    #Load a pretrained model 
    import pretrainedmodels 
    model = pretrainedmodels.__dict__[name](num_classes=1000,pretrained='imagenet')

    #Display model properties 
    for x in ['model.input_size','model.input_space','model.input_range','model.mean','model.std']: 
        print(x + ' =', eval(x))

    #reshape output to n classes 
    filters = model.last_linear_weights.shape[1]
    model.last_linear.bias = torch.nn.Parameters(torch.zeros(n))
    model.last_linear.weight = torch.nn.Parameters(torch.zeros(n,filters))
    model.last_linear.out_features = n 
    return model 

def scale_image(img,ratio = 1.0, same_shape=True): 
    #scales img(bs,3,x,y) by ratio 
    h,w = img.shape[2:]
    s = (int(h*ratio),int(w*ratio)) #new size 
    img = F.interpolate(img,size=s,mode='bilinear',align_corners=False) #resize 
    if not same_shape : #pad/crop image  
        gs = 64 #pixels grid size 
        h,w = [math.ceil(x * ratio /gs) * gs for x in (h,w)]
    return F.pad(img,[0,w-s[1],0,h-s[0]],value=0.447) #value = imagenet mean (0.447)







