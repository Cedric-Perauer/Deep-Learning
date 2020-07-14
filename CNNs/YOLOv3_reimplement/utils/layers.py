#reimplementation of YOLO by ultralytics for learning purposes 


import torch.nn.functional as F 

from utils.utils import *

def make_visible_by8(channel,divisor) : 
    #this function is based on Mobilenet and makes sure that the channel number is divisible by 8
    #having multiple of 8 makes sure that full  addresses are addressed, which allows for faster implementation 
    return math.ceil(channel/divisor) * divisor #divisor is multiple of 8 which allows for channels being mutiples of 8 

class Flatten(nn.Module) : 
    #This function is used to flatten after applying nn.AdaptiveAvgPoold2(1) to remove last 2 dimensions 
    def forward(self,x): 
        return x.view(x.size(0),-1)  


class Concat(nn.Module): 
    #Conactenate multiple tensors along dimensions 
    def __init__(self,dimension=1): 
        super(Concat,self).__init()__
        self.d = dimension
    
    def forward(self,x):
        return torch.cat(x,self.d) 



class FeatureConcat(nn.Module): 
    def __init__(self,layers): 
        super(FeatureConcat,self).__init__()
        self.layers = layers #layer indices 
        self.multi = len(layers) > 1 #check for multi layers

    def forward(self,x,output): 
        return torch.cat([outputs[i] for i in self.layers],1) if self.multi else outputs[self.layers[0]]

class WeightedFeatureFusion(nn.Module): #weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070 
    
    def __init(self,layers,weight=False) : 
        super(WeightedFeatureFusion,self).__init__()
        self.layers = layers #layer indices 
        self.weight = weight #weight booleans 
        self.n = len(layers) + 1 #number of layers
        
        if self.weight : 
            self.w = nn.Parameter(torch.zeros(self.n),requires_grad=True) #actual layer weights 
       
    def forward(self,x,outputs): 
        if self.weight : 
            w = torch.sigmoid(self.w) * (2/self.n) #sigmoid weights, between 0 and 1 since self.n is 2 at max 
            x = x * w[0] #compute output 

        #Fusion 
        nx = x.shape[0] #input channels 
        for i in range(self.n - 1) : 
            a = ouputs[self.layers[i]] * w[i+1] if self.weight else ouputs[self.layers[i]] #feature to add 
            na = a.shape[0] #feature channels 

            #Adjust channels 
            if nx == na : #same shape case 
                x = x + a #we can add without problems 
            elif nx > na : #slice input 
                x[:,:na] = x[:,:na] + a #or a = nn.ZeroPad2D((0,0,0,0,0,dv))(a); x = x + a 
            else : #slice feature 
                x = x + a[:,:nx] #case where na < nx 
        return x 


class MixConv2d(nn.Module):     
    # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self,in_ch,out_ch,k=(3,5,7), stride = 1 , dilation = 1 ,bias = True, method='equal_params'): 
        super(MixConv2d,self).__init__()
        
        groups = len(k) 
        if method == 'equal_ch' : #equal channels per group 
            i = torch.linspace(0,groups  - 1E-6, out_ch).floor() #out channel indices 
            ch = [(i == g).sum() for g in range(groups)]
        else : 
            #'equal_params' : equal parameter count per group 
            b = [out_ch] + [0] * groups
            a = np.eye(groups+1,groups,k=-1) 
            a -= np.roll(a,1,axis=1) #roll shifts elements in the array 
            a *= np.array(k) ** 2 
            a[0] = 1 
            ch = np.linalg.lstsq(a,b,rcond=None)[0].round().astype(int) #solve for equal weight indices, ax = b 
        
        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,out_channels=ch[g],kernel_size = k[g], stride = stride, padding = k[g]//2, 
                          dilation = dilation, bias=bias) for g in range(groups)]) 
                        #note that same padding is being applied 
    def forward(self,x): 
        return torch.cat([m[x] for m in self.m],1) 
    
            

        

   
