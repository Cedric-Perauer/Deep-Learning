import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_dataset, check_file, check_img_size, compute_loss, non_max_suppression, scale_coords,
    xyxy2xywh, clip_coords,plot_single_images_with_gt,plot_images_with_gt,plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class, set_logging)
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import LoadStreams, LoadImages
import random 

"""
Implements a testing function that can be used to sort examples by loss and visualize the data 
with the ground truth
"""
class Metrics():
    def __init__(self):
        self.mAP50 = 0.
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.mp = 0.
        self.mr = 0.
        self.t_model = 0. #measure runtime of the model 
        self.t_nms = 0. #measure runtime of the nms algo 
        self.stats = []
        self.jdict = []
        self.ap = []
        self.ap_class = []
        self.loss = [] #only used in train mode
        self.niou = None    
        self.seen = 0 
        #image stats
        self.h = 0 
        self.w = 0 
        self.iouv = None
        self.targets = None
        self.names = None
        self.batch_size = 16 
        self.num_classes = 1
        self.seen = 0
        self.img_size = 0 
        self.save_dir = None 

    def metrics_compute(self,plots,*args):
        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(self.stats) and self.stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*self.stats, plot=plots, fname=self.save_dir / 'precision-recall_curve.png')
            p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(self.stats[3].astype(np.int64), minlength=self.num_classes)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12.3g' * 6  # print format
        print(pf % ('all', self.seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if self.verbose and self.num_classes > 1 and len(self.stats):
            for i, c in enumerate(ap_class):
                print(pf % (self.names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds
        t = tuple(x / self.seen * 1E3 for x in (self.t_model,self.t_nms, self.t_model + self.t_nms)) + (self.img_size, self.img_size, self.batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    
    def sort_loss(self):
        pass 



class Test(Metrics):
    def __init__(self,data,weights=None,batch_size=16,
            img_size=640,conf_thresh=0.001,iou_thresh=0.6,single_cls=False,
            augment=False,verbose=False,model=None,dataloader=None,
            save_dir=Path(''),save_txt=False,save_conf=False,plots=True):
        
        super(Test,self).__init__()
        #store important data as class objects 
        self.data = data
        self.weights = weights 
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.single_cls = single_cls
        self.losses = {} #dict which keeps track of iamges to sort them for loss 
        self.model = model
        self.device = select_device(opt.device,batch_size=self.batch_size) #should be CUDA
        self.save_dir = save_dir
        self.half = True if self.device.type != "cpu" else False
        self.augment = augment
        self.plots = plots
        self.verbose = verbose  
        self.batch_idx = None 
        self.paths = None 
        self.im = None 

    def track_data(self):     
        set_logging()
        self.model_load()
        self.config()
        dataloader = self.Dataloader() #calls compute loop to start training 
        self.compute_loop(dataloader)
         

    def model_load(self):
        self.model = attempt_load(self.weights,map_location=self.device) #FP32 model 
        self.img_size = check_img_size(self.img_size,s=self.model.stride.max()) #image size check
        if self.half:
            self.model.half() #Floating Point 16 on GPU 
    
    def config(self):
        self.model.eval()
        with open(self.data) as f :
            self.data = yaml.load(f,Loader=yaml.FullLoader) #dictionary of the model 
        check_dataset(self.data)
        self.num_classes = 1 if self.single_cls else int(self.data['nc']) #iou vector for mAP @0.5 with confidence of >= 0.95 
        self.iouv = torch.linspace(0.5,0.95,10).to(self.device) 
        self.niou = self.iouv.numel()
    
    def Dataloader(self):
        img = torch.zeros((1,3,self.img_size,self.img_size),device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != "cpu" else None
        path = self.data["test"] if opt.task == "test" else self.data["val"] #path to val/test images
        dataloader = create_dataloader(path,self.img_size,self.batch_size,self.model.stride.max(),opt,
                hyp=None,augment=False,cache=False,pad=0.5,rect=True)[0]
        return dataloader
    
        
        
    def compute_loop(self,dataloader): 
        self.names = self.model.names if hasattr(self.model, 'names') else self.model.module.names
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        self.loss = torch.zeros(3,device=self.device) 
        for self.batch_idx, (data) in enumerate(tqdm(dataloader,desc=s)):
            self.batch_compute(data)
        
        self.metrics_compute(self.plots)

    def img_preprocess(self,img):
        img = img.to(self.device,non_blocking=True)
        img = img.half() if self.half  else img.float()
        img /= 255.0 
        return img 

    def batch_compute(self,*args):
        img,targets,self.paths,shapes = args[0] #data
        self.img = self.img_preprocess(img)
        self.targets = targets.to(self.device) 
        _ , _, self.h,self.w = img.shape #order is batch size,channels (already known from init and image), height,width 
        

        with torch.no_grad(): 
            #model 
            t = time_synchronized()
            inf_out, train_out = self.model(self.img,augment=self.augment) 
            self.t_model += time_synchronized() - t
            
            #loss compute is ommitted here, only a testing script => nms compute
            t = time_synchronized()
            output = non_max_suppression(inf_out,conf_thres=self.conf_thresh,iou_thres=self.iou_thresh)
            self.t_nms += time_synchronized() - t
            self.compute_stats(output) 

    def plot_images(self):
        f = self.save_dir / f'test_batch_compare_{self.batch_idx}.jpg'
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        for p in self.paths :
            dataset = LoadImages(p,img_size=self.img_size)
            img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
            out = [] #use this to store all detections for BB
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = self.model(img, augment=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)


                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0 = path, '', im0s

                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            if det is None : 
                return None,None
            out = torch.cat((det[:,:4],det[:,-1].unsqueeze(1)),dim=1)
            return out, im0.shape #bounding boxes output 

    def compute_stats(self,output):
        whwh = torch.Tensor([self.w,self.h,self.w,self.h]).to(device = self.device) 
        for si, pred in enumerate(output):
            labels = self.targets[self.targets[:,0] == si,1:] #label for corresponding index 
            nl = len(labels) 
            tcls = labels[:,0].tolist() if nl else []
            
            self.seen += 1 
            
            if pred is None : 
                if nl : 
                    self.stats.append((torch.zeros(0,self.niou,dtype=torch.bool),torch.Tensor(),torch.Tensor(),tcls))
                continue

            #clip boxes to image bounds 
            clip_coords(pred,(self.h,self.w))
            

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=self.device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            
            # Append statistics (correct, conf, pcls, tcls)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            if self.plots : 
                self.plot_images()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--sort_loss',type=bool, help='sort images by loss')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='runs/test', help='directory to save results')


    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file

    if opt.task in ['val', 'test']:  # run normally
        test = Test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_dir=Path(opt.save_dir),
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             )
        test.track_data()
        print('Results saved to %s' % opt.save_dir)
