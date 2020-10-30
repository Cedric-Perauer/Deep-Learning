import torch 
import torch.nn as nn 
import torch.nn.functional as F 

#Swish activation 
class Swish(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x) 

class Swish_Mem(nn.Module): #Memory Efficient version of Swish activation 
    class F(torch.autograd.Function):
        """
        Torch autograd : We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """
        @staticmethod
        def forward(ctx,x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid()

        @staticmethod
        def backward(ctx,output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1- sx)))

    def forward(self,x):
        return self.F.apply(x) #F is the class F here, not nn.functional 

