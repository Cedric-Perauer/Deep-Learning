import torch.nn.Functional as F 
from utils.utils import * 

class SwishImplementation(torch.autograd.Function): 
    @staticmethod 
    def forward(ctx,x) : 
        ctx.save_for_backward(x) 
        return x * torch.sigmoid(x) 

    @staticmethod 
    def backward(ctx,grad_output): 
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x) #sigmoid(ctx) 
        return grad_output * (sx * (1 + x * (1 - sx))) 

class MishImplementation(torch.autograd.Function): 
    @staticmethod
    def forward(ctx,x): 
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)) # x * tanh(ln(1 + exp(x)))
   @staticmethod 
   def backward(ctx,grad_output): 
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x) 
        fx = F.sofplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


                                                                                                                                        
class MemoryEfficientSwish(nn.Module) : 
        def forward(self,x) : 
            return SwishImplementation.apply(x) 

class MemoryEfficientMish(nn.Module): 
        def forward(self,x) : 
            return MishImplementation.apply(x) 

class Swish(nn.Module): 
        def forward(self,x): 
            return x * torch.sigmoid(x)

class HardSwish(nn.Module):             # https://arxiv.org/pdf/1905.02244.pdf
        def forward(self,x): 
            return x * F.hardtanh(x + 3,0.,6.,True)/ 6. 

class Mish(nn.Module):             
        def forward(self,x): 
            return x * F.softplus(x).tanh()
