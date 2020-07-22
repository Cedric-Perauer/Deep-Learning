import math 

import torch 
import torch.optim.optimizer import Optimizer 

class AdaBound(Optimizer): 
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """ 
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),final_lr = 0.1,gamma = 1e-3,eps=1e-8,weight_decay=0,amsbound=False): 
        #just catch some errors 
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr,beta=betas,final_lr=final_lr,gamma=gamma,eps=eps,weight_decay=weight_decay,amsbound=amsbound) 
        super(AdaBound,self).__init__(params,defaults) 
        
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups)) 
    
    def __setstate__(self,state): 
        super(AdaBound,self).__setstat__(state)
        for group in self.param_groups : 
           group.setdefault('amsbound',False) 

    def step(self,closure=None): 
        """Single optim step 
           Arguments : Closure : reevaluates the model and returns the loss 

        """
        loss = None 
        if closure isi not  None : 
            loss = closure() 

        for group, base_lr in zip(self.param_groups, self.base_lrs): 
             for p in group['params']: 
                 if p.grad is None : 
                     continue 
                 grad = p.grad.data 
                 if grad.is_sparse : 
                     raise RuntimeError("Adam does not support sparse Gradients, please use sparse Adam instead") 
                 amsbound = group['amsbound']

                 state = self.state[p]

                  #Init the state 
                 if len(state) == 0 : 
                     state['step'] = 0 
                     #Exponential moving average of gradients 
                     state['exp_avg'] = torch.zeros_like(p.data)
                     #Exp moving average of squared gradients 
                     state['exp_avg_sq'] = torch.zeros_like(p.data)
                     if amsbound : 
                         #Maintains max of all exp. moving average of squared gradient values 
                         state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                 
                 #get current averages of squared and normal gradient 
                 exp_avg, exp_avg_sq = state['exp_avg'],state['exp_avg_sq'] 
                 if amsbound : 
                     max_exp_avg_sq = state['max_exp_avg_sq'] 
                 beta1, beta2 = group['betas'] 

                 state['step'] += 1

                 if group['weight_decay'] != 0 : 
                     grad = grad.add(group['weight_decay'],p.data) 
                 
                 #Decay the first and second moment running average coefficients  
                 exp_avg.mul_(beta1).add_(1 - beta1,grad) #update biased first moment estimate  
                 exp_avg_sq.mul_(beta2).addcmul_(1-beta2,grad,grad) #update biased second moment estimate 

                 if amsbound : 
                     #Maintains the maximum of all 2nd moment running avg. till now  
                     torch.max(max_exp_avg_sq,exp_avg_sq,out=max_exp_avg_sq) 
                     #Use the maximum for normalizing running avg of gradient, add eps to avoid 0 division  
                     denom = max_exp_avg_sq.sqrt().add(group['eps']) 
                 else : 
                     denom = exp_avg_sq.sqrt().add(group['eps']) 
                 
                 #do bias correction for both moment terms 
                 bias_correct_1 = 1 - beta1 ** state['step']
                 bias_correct_2 = 1 - beta2 ** state['step']
                 step_size = group['lr'] * math.sqrt(bias_correct_2)/bias_correct_1

                 #Bound the learning rate 
                 #lr_schedule can not affect the final_lr apparently so a workaround is applied : 
                 final_lr = group['final_lr'] * group['lr'] / base_lr
                 lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1)) 
                 upper_bound = final_lr * (1 + 1 /(group['gamma'] * state['step']))
                 step_size = torch.full_like(denom,step_size) 
                 step_size.div_(denom).clamp_(lower_bound,upper_bound).mul_(exp_avg)

                 p.data.add_(-step_size) 

        return loss                  




                    


            



   

