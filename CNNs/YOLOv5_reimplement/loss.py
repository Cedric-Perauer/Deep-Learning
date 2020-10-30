import torch 
import torch.nn as nn
from utils import  intersection_over_union

class YoloLoss(nn.Module): 
    def __init__(self,S=7,B=2,C=20):
        super(YoloLoss,self).__init__()

        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_obj = 0.5
        self.lambda_coord = 5

    def forward(self,predictions,target):
        preds = predictions.reshape(-1,self.S,self.S,self.C + self.B*5)
        
        #these are the two boxes per cell  
        iou_b1 = intersection_over_union(preds[...,21:25],target[...,21:25]) #0-19 class probs, 21-24 4 bb vals
        iou_b2 = intersection_over_union(preds[...,26:30],target[...,21:25]) #26-29 bb vals
        ious = torch.cat([iou_b1.unsqueeze(0),iou_b2.unsqueeze(0)],dim=0)

        iou_maxes, bestbox = torch.max(ious,dim=0)
        exists_box = target[...,20].unsqueeze(3) #if box is in object i 

        ##Box Coordinate Loss, coordinates are 2 and 3 
        box_preds = exists_box * ((bestbox * preds[...,26:30] + (1-bestbox) * preds[...,21:25]))
        print(target[:,21:25])
        box_targets = exists_box * target[...,21:25]
        box_preds[...,2:4] = torch.sign(box_preds[...,2:4]) * torch.sqrt(torch.abs(box_preds[...,2:4] + 1e-6))  
        
        box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4]) 
        
        box_loss = self.mse(torch.flatten(box_preds,end_dim=-2),torch.flatten(box_targets,end_dim=-2))

        ##Object Loss
        pred_box = (bestbox * preds[...,25:26] + (1-bestbox) * preds[...,20:21]) 

        object_loss = self.mse(torch.flatten(exists_box * pred_box),torch.flatten(exists_box * target[...,20:21]))
        
        #no object loss, take loss for both
        no_object_loss = self.mse(torch.flatten((1-exists_box) * preds[...,20:21],start_dim=1),
                torch.flatten((1-exists_box) * preds[...,20:21],start_dim=1))
        
        no_object_loss += self.mse(torch.flatten((1-exists_box) * preds[...,25:26],start_dim=1),
                torch.flatten((1-exists_box)  * preds[...,25:26],start_dim=1))
         
        
        ##Class Loss
        class_loss = self.mse(torch.flatten(exists_box * preds[...,:20],end_dim=-2,),torch.flatten(exists_box * target[...,:20],end_dim=-2,))
        #total loss
        loss = self.lambda_coord * box_loss + object_loss + self.lambda_obj * no_object_loss + class_loss

        return loss
