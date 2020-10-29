import  torch 
import torchvision.transforms as transforms
import torch.optim as optim
import torch.transforms.functional as FT 
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils import (
        intersection_over_union,
        non_max_suppression,
        cellboxes_to_boxes,
        get_bboxes,
        plot_image,
        save_checkpoint,
        load_checkpoint,
        )

from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

#Hyperparams 
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
bs = 16 
wd = 0 
epochs = 100
num_workers = 4 
pin_mem = True
load_model = False
load_model_file = "overfit.pth.tar" 
img_dir = "data/images"
label_dir = "data/labels" 

class Compose(object): 
    def __init__(self,transform):
            self.transforms = transforms

    def __call__(self,img,bboxes):
        for t in self.transforms:
            img,bboxes = t(img),bboxes

transform = Compose([transforms.Resize((448,448)),transforms.ToTensor()])

def train_fn(train_loader,model,optim,loss_fn):
    loop = tqdm(train_loader,leave=True)
    mean_loss = []

    for batch_idx,(x,y) in enumerate(loop):
        x,y = x.to(device),y.to(device)
        out=model(x)
        loss=loss_fn(x,y)
        mean_loss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update the tqdm bar 
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean)/len(mean_loss)}") 
    
def main():
    model = YOLOv1(split_size=7,num_boxes=2,num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=wd)
    loss_fn = YoloLoss()
    if load_model :
        load_checkpoint(torch.load(load_model_file),model,optimizer)

    train_dataset = VOCDataset("data/8examples.csv",
                                transform=transform,
                                img_dir=img_dir,
                                label_dir=label_dir)

    test_dataset = VOCDataset("data/test.csv",
                                transform=transform,
                                img_dir=img_dir,
                                label_dir=label_dir)

    train_loader = DataLoader(dataset=train_dataset,batch_size=bs,num_workers=num_workers,shuffle=True,drop_last=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=bs,num_workers=num_workers,shuffle=True,drop_last=True)

    for epoch in range(epochs):
        pred_boxes,target_boxes = get_bboxes(train_loader,model,iou_threshold=0.5,threshol=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes,target_boxes,iou_threshold=0.5,box_format="midpoint")
        print(f"Train mAP: {mean_avg_prec}")

        train_fn(train_loader,model,optimizer,loss_fn)

if __name__ == "__main__"
    main()
