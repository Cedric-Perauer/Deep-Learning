import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import os 
import yaml
import math
import numpy as np
from warnings import warn

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS, autoShape
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
import pytorch_lightning as pl
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_loss, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging, init_seeds)
from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, select_device, intersect_dicts
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

                

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(pl.LightningModule):
    def __init__(self, opt,hyp,cfg='yolov5s.yaml', ch=3, nc=5,tb_writer=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        
        self.nc = nc
        self.hyp = hyp  
        self.opt = opt
        self.tb_writer = tb_writer
        #init seeds

        init_seeds(2 + self.opt.global_rank)
        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])
        
        

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        initialize_weights(self)
        self.info()
        self.config()
        self.run_save()
        self.load_model()
        #self.wandb_logging()
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.optimizer, self.lr_scheduler = self.optimizer[0], self.lr_scheduler[0]
        self.resume()
    

    def config(self):
        logger.info(f'Hyperparameters {hyp}')
        self.log_dir = Path(self.tb_writer.log_dir) if self.tb_writer else Path(self.opt.logdir) / 'evolve'  # logging directory
        wdir = self.log_dir / 'weights'  # weights directory
        os.makedirs(wdir, exist_ok=True)
        self.last = wdir / 'last.pt'
        self.best = wdir / 'best.pt'
        self.results_file = str(self.log_dir / 'results.txt')
        self.epochs, self.batch_size, self.total_batch_size, self.weights, self.rank = \
            self.opt.epochs, self.opt.batch_size, self.opt.total_batch_size, self.opt.weights, self.opt.global_rank
        with open(self.opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        with torch_distributed_zero_first(self.rank):
            check_dataset(data_dict)  # check
        self.train_path = data_dict['train']
        self.test_path = data_dict['val']
        self.nc, self.names = (1, ['item']) if self.opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, self.opt.data)  # check


    def run_save(self): 
        # Save run settings
        with open(self.log_dir / 'hyp.yaml', 'w') as f:
            yaml.dump(hyp, f, sort_keys=False)
        with open(self.log_dir / 'opt.yaml', 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)
    
    def resume(self):
        # Resume
        self.start_epoch, self.best_fitness = 0, 0.0
        # Optimizer
        if self.ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            best_fitness = self.ckpt['best_fitness']

        # Results
        if self.ckpt.get('training_results') is not None:
            with open(self.results_file, 'w') as file:
                file.write(self.ckpt['training_results'])  # write results.txt
        # Epochs
        start_epoch = self.ckpt['epoch'] + 1
        self.start_epoch = start_epoch 
        if self.opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (self.weights, self.epochs)
            shutil.copytree(wdir, wdir.parent / f'weights_backup_epoch{start_epoch - 1}')  # save previous weights
        if self.epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (self.weights, self.ckpt['epoch'], self.epochs))
            self.epochs += self.ckpt['epoch']  # finetune additional epochs

        # Image sizes
        self.gs = int(max(self.stride))  # grid size (max stride)
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in self.opt.img_size]  # verify imgsz are gs-multiples
        


    def train_inits(self):     
        self.batch_count = 0 
        # Epochs
        # Exponential moving average
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None
        
        # DDP mode
        if self.opt.local_rank != -1:
            self.model = DDP(self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank)


        # Process 0
        if self.rank in [-1, 0]:
            self.ema.updates = self.start_epoch * self.nb // self.accumulate  # set EMA updates

            if not self.opt.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                self.c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                plot_labels(labels, save_dir=log_dir)
                if self.tb_writer:
                    # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                    self.tb_writer.add_histogram('classes', c, 0)


        # Model parameters
        self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = self.hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc)
        self.model.names = self.names

        self.nw = max(round(self.hyp["warmup_epochs"] * self.nb),1e3) #number of warmup iterations ma
        self.maps = np.zeros(self.nc) #mAP per Class
        #self.lr_scheduler.last_epoch = self.start_epoch - 1  # do not move
        logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (self.imgsz, self.imgsz_test, self.train_dataloader.num_workers, self.log_dir, self.epochs))
     
        self.cuda = device.type != 'cpu' 
        #self.scaler = amp.GradScaler(enabled=self.cuda)


    def intersect_dicts(self,da, db, exclude=()):
        # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
        return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
        


    def load_model(self): 
        # Model
        print("Loading model") 
        self.ckpt = torch.load(self.weights)  # load checkpoint
        if self.hyp.get('anchors'):
            self.ckpt['model'].yaml['anchors'] = round(self.hyp['anchors'])  # force autoanchor
        exclude = ['anchor'] if self.opt.cfg or self.hyp.get('anchors') else []  # exclude keys
        state_dict = self.ckpt['model'].model.float().state_dict()  # to FP32
        #state_dict = self.intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
        self.model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), self.weights))  # report
        del state_dict 
        #freeeze paramaters 
        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False


    def wandb_logging(self): 
        if wandb and wandb.run is None:
            id = self.ckpt.get('wandb_id') if 'ckpt' in locals() else None
            wandb_run = wandb.init(config=self.opt, resume="allow", project="YOLOv5", name=os.path.basename(self.log_dir), id=id)
        

    def forward(self, x, augment=False, profile=False):
        
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x) # save output

        if profile:
            print('%.1fms total' % sum(dt))
        
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))


    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)
    
    def hyp(self): 
        # Hyperparameters
        hyp_path = "data/hyp.scratch.yaml"
        with open(opt.hyp) as f:
            self.hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
            if 'box' not in hyp:
                warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                     (hyp_path, 'https://github.com/ultralytics/yolov5/pull/1120'))
         


    def train_dataloader(self):
        
        self.train_dataloader, self.dataset = create_dataloader(self.train_path, self.opt.img_size[0], self.batch_size, self.gs, self.opt,
                hyp=self.hyp, augment=True,cache=self.opt.cache_images,rect=self.opt.rect,rank=self.rank,world_size=self.opt.world_size, workers=self.opt.workers)
        mlc = np.concatenate(self.dataset.labels,0)[:,0].max()
        self.nb = len(self.train_dataloader)  # number of batches

        assert mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)
        self.train_inits()
        return self.train_dataloader
       
    def val_dataloader(self):
        pass
        
    def test_dataloader(self):
        self.test_dataloader = create_dataloader(self.test_path, self.imgsz_test, self.total_batch_size, self.gs, self.opt,
                                       hyp=self.hyp, augment=False, cache=self.opt.cache_images and not self.opt.notest, rect=True,
                                       rank=-1, world_size=self.opt.world_size, workers=self.opt.workers)[0]  # testloader

        return self.test_dataloader
    
    def configure_optimizers(self):
        # Optimizer
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.total_batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.total_batch_size * self.accumulate / self.nbs  # scale weight_decay

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # no decay
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        if self.opt.adam:
            optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        optimizer.add_param_group({'params': pg1, 'weight_decay': self.hyp['weight_decay']})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2
        self.lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)

        return [optimizer],[scheduler]
    
    


    def training_step(self,batch,batch_idx):
            
            import pdb; pdb.set_trace()  
            print("Training Step")
            self.batch_count = 0 
            imgs, targets, paths, _ = batch
            ni = batch_idx + self.nb * self.current_epoch  # number integrated batches (since train start)
            imgs = imgs.float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            self.mloss = torch.zeros(4,device=imgs.device) 

            # Warmup
            if ni <= self.nw:
                xi = [0,self.nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.total_batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * self.lf(self.current_epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])
            
            # Multi-scale
            if self.opt.multi_scale:
                sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                sf = self.sz / max(self.imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / self.gs) * self.gs for x in self.imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(self.imgs, size=ns, mode='bilinear', align_corners=False)
            
            # Forward
            with amp.autocast(enabled=self.cuda):
                
                pred = self.forward(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets, self.model)  # loss scaled by batch_size
                import pdb; pdb.set_trace()  
                if self.rank != -1:
                    loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
            
            # Backward
            #loss.backward()

            # Optimize
            if ni % self.accumulate == 0:
                #self.scaler.step(self.optimizer)  # optimizer.step
                self.optimizer.step()
                #self.optimizer.zero_grad()
                #self.scaler.update()
                self.optimizer.zero_grad()
                if self.ema:
                    # Optimize
                    self.ema.update(self.model)

            # Print
            if self.rank in [-1, 0]:
                self.mloss = (self.mloss * self.batch_count + loss_items) / (self.batch_count + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (self.current_epoch, self.epochs - 1), mem, *self.mloss, targets.shape[0], imgs.shape[-1])
                print("----------------------------") 
                print("Mloss is : ",*self.mloss)
                # Plot
                if ni < 3:
                    f = str(self.log_dir / f'train_batch{ni}.jpg')  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    # if tb_writer and result is not None:
                    # tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard
            # end batch ------------------------------------------------------------------------------------------------
            
            import pdb; pdb.set_trace()  
            del imgs, targets, paths, loss, loss_items, pred, result 

    def on_epoch_end(self): 
                print("epoch end") 
                # Scheduler
                lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
                self.lr_scheduler.step()
                
                results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                # DDP process 0 or single-GPU
                if self.rank in [-1, 0]:
                    # mAP
                    if self.ema:
                        ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
                    final_epoch = self.current_epoch + 1 == self.epochs
                    if not opt.notest or final_epoch:  # Calculate mAP
                     results, maps, times = test.test(self.opt.data,
                                                 batch_size=self.total_batch_size,
                                                 imgsz=self.imgsz_test,
                                                 model=self.ema.ema,
                                                 single_cls=self.opt.single_cls,
                                                 dataloader=self.test_dataloader,
                                                 save_dir=self.log_dir,
                                                 plots=self.current_epoch == 0 or self.final_epoch)

                          # Write
                    with open(self.results_file, 'a') as f:
                        f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                    if len(opt.name) and opt.bucket:
                        os.system('gsutil cp %s gs://%s/results/results%s.txt' % (self.results_file, self.opt.bucket, self.opt.name))

                    # Log
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/giou_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                            'x/lr0', 'x/lr1', 'x/lr2']  # params
                    for x, tag in zip(list(self.mloss[:-1]) + list(results) + lr, tags):
                        if tb_writer:
                            tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                        if wandb:
                            wandb.log({tag: x})  # W&B

                    # Update best mAP
                    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                    if fi > best_fitness:
                        best_fitness = fi

                    # Save model
                    save = (not opt.nosave) or (final_epoch and not opt.evolve)
                    if save:
                        with open(results_file, 'r') as f:  # create checkpoint
                            ckpt = {'epoch': epoch,
                                    'best_fitness': best_fitness,
                                    'training_results': f.read(),
                                    'model': ema.ema,
                                    'optimizer': None if final_epoch else optimizer.state_dict(),
                                    'wandb_id': wandb_run.id if wandb else None}

                        # Save last, best and delete
                        torch.save(ckpt, last)
                        if best_fitness == fi:
                            torch.save(ckpt, best)
                        del ckpt



def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            c1, c2 = ch[f], args[0]

        
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--name', default='', help='renames experiment folder exp{N} to exp{N}_{name} if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--log-imgs', type=int, default=10, help='number of images for W&B logging, max 100')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')

   
    opt = parser.parse_args()

    # Set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Resume
    if opt.resume:  # resume an interrupted run
        self.ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(self.ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            self.opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        self.opt.cfg, self.opt.weights, self.opt.resume = '', self.ckpt, True
        logger.info('Resuming training from %s' % self.ckpt)

    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps
        if 'box' not in hyp:
            warn('Compatibility: %s missing "box" which was renamed from "giou" in %s' %
                 (opt.hyp, 'https://github.com/ultralytics/yolov5/pull/1120'))
            hyp['box'] = hyp.pop('giou')
    
   
    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer, wandb = None, None  # init loggers
        if opt.global_rank in [-1, 0]:
            # Tensorboard
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.logdir}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

            # W&B
            try:
                import wandb

                assert os.environ.get('WANDB_DISABLED') != 'true'
                logger.info("Weights & Biases logging enabled, to disable set os.environ['WANDB_DISABLED'] = 'true'")
            except (ImportError, AssertionError):
                opt.log_imgs = 0
                logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")
     

    model = Model(opt,hyp,opt.cfg)
    trainer = pl.Trainer(gpus=1) 
    trainer.fit(model)
