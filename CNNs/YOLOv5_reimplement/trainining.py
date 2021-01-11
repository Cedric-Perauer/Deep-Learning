import argparse
import logging
import os
import random
import shutil
import time
from pathlib import Path
from warnings import warn

import math
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import pytorch_lightning as pl
import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import (
    torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors, labels_to_image_weights,
    compute_losst, plot_images, fitness, strip_optimizer, plot_results, get_latest_run, check_dataset, check_file,
    check_git_status, check_img_size, increment_dir, print_mutation, plot_evolution, set_logging, init_seeds)
from utils.google_utils import attempt_download
from utils.torch_utils import ModelEMA, select_device, intersect_dicts

logger = logging.getLogger(__name__)

class Training(): 

   def  __init__(self,hpy,opt,device,tb_writer=None,wandb=None): 
        logger.info(f'Hyperparameters {hyp}')
        self.tb_writer = tb_writer
        self.wandb = wandb
        self.hyp = hyp 
        self.log_dir = Path(self.tb_writer.log_dir) if self.tb_writer else Path(opt.logdir) / 'evolve'  # logging directory
        self.wdir = self.log_dir / 'weights'  # weights directory
        os.makedirs(self.wdir, exist_ok=True)
        self.last = self.wdir / 'last.pt'
        self.best = self.wdir / 'best.pt'
        self.results_file = str(self.log_dir / 'results.txt')
        self.epochs, self.batch_size, self.total_batch_size, self.weights, self.rank = \
            opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank
        self.opt = opt
        self.device = device
        

        self.save_run_settings()
        self.configure()
        self.configure_optimizers()
        self.logging_init()
        self.scheduler()
        self.resume()
        self.train_prep()
        self.dataloaders()
        self.train_inits()

   def dataloaders(self): 
        # Trainloader
        self.train_dataloader, self.dataset = create_dataloader(self.train_path, self.imgsz, self.batch_size, self.gs, self.opt,
                                                hyp=self.hyp, augment=True, cache=opt.cache_images, rect=self.opt.rect,
                                                rank=self.rank, world_size=self.opt.world_size, workers=self.opt.workers)
        self.mlc = np.concatenate(self.dataset.labels, 0)[:, 0].max()  # max label class
        self.nb = len(self.train_dataloader)  # number of batches
        assert self.mlc < self.nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (self.mlc, self.nc, self.opt.data, self.nc - 1)

        self.testloader = create_dataloader(self.test_path, self.imgsz_test, self.total_batch_size, self.gs, self.opt,
                                       hyp=self.hyp, augment=False, cache=self.opt.cache_images and not self.opt.notest, rect=True,
                                       rank=-1, world_size=self.opt.world_size, workers=self.opt.workers)[0]  # testloader



   
       
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
            self.optimizer = optim.Adam(pg0, lr=self.hyp['lr0'], betas=(self.hyp['momentum'], 0.999))  # adjust beta1 to momentum
        else:
            self.optimizer = optim.SGD(pg0, lr=self.hyp['lr0'], momentum=self.hyp['momentum'], nesterov=True)

        self.optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
        self.optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
        del pg0, pg1, pg2

   def scheduler(self):
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        self.lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - self.hyp['lrf']) + self.hyp['lrf']  # cosine
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # plot_lr_scheduler(optimizer, scheduler, epochs)


       
   def save_run_settings(self): 
       # Save run settings
       with open(self.log_dir / 'hyp.yaml', 'w') as f:
         yaml.dump(hyp, f, sort_keys=False)
       with open(self.log_dir / 'opt.yaml', 'w') as f:
         yaml.dump(vars(self.opt), f, sort_keys=False)
   
   def logging_init(self): 
       # Logging
       if self.wandb and self.wandb.run is None:
        id = self.ckpt.get('wandb_id') if 'ckpt' in locals() else None
        self.wandb_run = wandb.init(config=self.opt, resume="allow", project="YOLOv5", name=os.path.basename(self.log_dir), id=id)



   def configure(self): 
        self.cuda = self.device.type != 'cpu'
        init_seeds(2 + self.rank)
        with open(self.opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        with torch_distributed_zero_first(self.rank):
            check_dataset(data_dict)  # check
        self.train_path = data_dict['train']
        self.test_path = data_dict['val']
        self.nc, self.names = (1, ['item']) if self.opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
        assert len(self.names) == self.nc, '%g names found for nc=%g dataset in %s' % (len(self.names), self.nc, self.opt.data)  # check

        # Model
        self.pretrained = self.weights.endswith('.pt')
        print("model") 
        if self.pretrained:
            with torch_distributed_zero_first(self.rank):
                attempt_download(self.weights)  # download if not found locally
            self.ckpt = torch.load(self.weights, map_location=self.device)  # load checkpoint
            if self.hyp.get('anchors'):
                self.ckpt['model'].yaml['anchors'] = round(self.hyp['anchors'])  # force autoanchor
            self.model = Model(self.opt.cfg or self.ckpt['model'].yaml, ch=3, nc=self.nc).to(self.device)  # create
            exclude = ['anchor'] if self.opt.cfg or self.hyp.get('anchors') else []  # exclude keys
            state_dict = self.ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(state_dict, self.model.state_dict(), exclude=exclude)  # intersect
            
            self.model.load_state_dict(state_dict, strict=False)  # load
            logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(self.model.state_dict()), self.weights))  # report
        else:
            model = Model(self.opt.cfg, ch=3, nc=self.nc).to(self.device)  # create

        # Freeze
        freeze = []  # parameter names to freeze (full or partial)
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False
        
   def resume(self): 
            self.start_epoch, self.best_fitness = 0, 0.0
            if self.pretrained:
                # Optimizer
                if self.ckpt['optimizer'] is not None:
                    self.optimizer.load_state_dict(self.ckpt['optimizer'])
                    self.best_fitness = self.ckpt['best_fitness']

                # Results
                if self.ckpt.get('training_results') is not None:
                    with open(self.results_file, 'w') as file:
                        file.write(self.ckpt['training_results'])  # write results.txt

                # Epochs
                self.start_epoch = self.ckpt['epoch'] + 1
                if self.opt.resume:
                    assert self.start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (self.weights, self.epochs)
                    shutil.copytree(self.wdir, self.wdir.parent / f'weights_backup_epoch{start_epoch - 1}')  # save previous weights
                if self.epochs < self.start_epoch:
                    logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                                (self.weights, self.ckpt['epoch'], self.epochs))
                    self.epochs += self.ckpt['epoch']  # finetune additional epochs


   def train_prep(self):
        # Image sizes
        self.gs = int(max(self.model.stride))  # grid size (max stride)
        self.imgsz, self.imgsz_test = [check_img_size(x, self.gs) for x in opt.img_size]  # verify imgsz are gs-multiples

        # DP mode
        if self.cuda and self.rank == -1 and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # SyncBatchNorm
        if self.opt.sync_bn and self.cuda and self.rank != -1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
            logger.info('Using SyncBatchNorm()')

        # Exponential moving average
        self.ema = ModelEMA(self.model) if self.rank in [-1, 0] else None
        # DDP modei
        if self.cuda and self.rank != -1:
            self.model = DDP(self.model, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank)

   def train_inits(self): 
        # Process 0
        if self.rank in [-1, 0]:
            if self.ema : 
                   self.ema.updates = self.start_epoch *self.nb // self.accumulate  # set EMA updates

            if not self.opt.resume:
                labels = np.concatenate(self.dataset.labels, 0)
                c = torch.tensor(labels[:, 0])  # classes
                # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
                # model._initialize_biases(cf.to(device))
                plot_labels(labels, save_dir=log_dir)
                if self.tb_writer:
                    # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
                    self.tb_writer.add_histogram('classes', c, 0)

                # Anchors
                #if not opt.noautoanchor:
                #check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        # Model parameters
        self.hyp['cls'] *= self.nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
        self.model.nc = self.nc  # attach number of classes to model
        self.model.hyp = hyp  # attach hyperparameters to model
        self.model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.model.class_weights = labels_to_class_weights(self.dataset.labels, self.nc).to(self.device)  # attach class weights
        self.model.names = self.names

        # Start training
        t0 = time.time()
        self.nw = max(round(self.hyp['warmup_epochs'] * self.nb), 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        self.maps = np.zeros(self.nc)  # mAP per class
        self.results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.scaler = amp.GradScaler(enabled=self.cuda)
        logger.info('Image sizes %g train, %g test\n'
                    'Using %g dataloader workers\nLogging results to %s\n'
                    'Starting training for %g epochs...' % (self.imgsz, self.imgsz_test, self.train_dataloader.num_workers, self.log_dir, self.epochs))
             
   def on_epoch_start(self):
        self. model.train()
        # Update image weights (optional)
        if self.opt.image_weights:
            # Generate indices
            if self.rank in [-1, 0]:
                cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2  # clzzass weights
                iw = labels_to_image_weights(self.dataset.labels, nc=self.nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if self.rank != -1:
                indices = (torch.tensor(self.dataset.indices) if self.rank == 0 else torch.zeros(self.dataset.n)).int()
                dist.broadcast(indices, 0)
                if self.rank != 0:
                    self.dataset.indices = indices.cpu().numpy()


        self.mloss = torch.zeros(4, device=self.device)  # mean losses
        if self.rank != -1:
            self.train_dataloader.sampler.set_epoch(self.epoch)
        self.pbar = enumerate(self.train_dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if self.rank in [-1, 0]:
            self.pbar = tqdm(self.pbar, total=self.nb)  # progress bar
        self.optimizer.zero_grad()
        
   def training_step(self,i):
            ni = i + self.nb * self.epoch  # number integrated batches (since train start)
            self.imgs = self.imgs.to(self.device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            # Warmup
            if ni <= self.nw:
                xi = [0, self.nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.total_batch_size]).round())
                for j, x in enumerate(self.optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * self.lf(self.epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])
            # Multi-scale
            if self.opt.multi_scale:
                sz = random.randrange(self.imgsz * 0.5, self.mgsz * 1.5 + self.gs) // self.gs * self.gs  # size
                sf = sz / max(self.imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / self.gs) * self.gs for x in self.imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(self.imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=self.cuda):
                pred = self.model(self.imgs)  # forward
                loss, loss_items = compute_losst(pred, self.targets.to(self.device), self.model)  # loss scaled by batch_size
                if self.rank != -1:
                    loss *= self.opt.world_size  # gradient averaged between devices in DDP mode
            
            # Backward
            self.scaler.scale(loss).backward()

            # Optimize
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)

            # Print
            if self.rank in [-1, 0]:
                self.mloss = (self.mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                self.s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (self.epoch, self.epochs - 1), mem, *self.mloss, self.targets.shape[0], self.imgs.shape[-1])
                self.pbar.set_description(self.s)
                 
                # Plot
                if ni < 3:
                    f = str(self.log_dir / f'train_batch{ni}.jpg')  # filename
                    result = plot_images(images=self.imgs, targets=self.targets, paths=self.paths, fname=f)

   def on_epoch_end(self): 
        # Scheduler
        lr = [x['lr'] for x in self.optimizer.param_groups]  # for tensorboard
        self.scheduler.step()

        # DDP process 0 or single-GPU
        if self.rank in [-1, 0]:
            # mAP
            if self.ema:
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = self.epoch + 1 == self.epochs

            if not self.opt.notest or final_epoch:  # Calculate mAP
                    results, self.maps, times = test.test(self.opt.data,
                                                     batch_size=self.total_batch_size,
                                                     imgsz=self.imgsz_test,
                                                     model=self.ema.ema,
                                                     single_cls=self.opt.single_cls,
                                                     dataloader=self.testloader,
                                                     save_dir=self.log_dir,
                                                     plots=self.epoch == 0 or final_epoch)
            with open(self.results_file, 'a') as f:
                f.write(self.s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(self.opt.name) and self.opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (self.results_file, self.opt.bucket, self.opt.name))

            # Log
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(self.mloss[:-1]) + list(results) + lr, tags):
                if self.tb_writer:
                    self.tb_writer.add_scalar(tag, x, self.epoch)  # tensorboard
                if self.wandb:
                    wandb.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > self.best_fitness:
                self.best_fitness = fi

            # Save model
            save = (not self.opt.nosave) or (final_epoch and not self.opt.evolve)
            if save:
                with open(self.results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': self.epoch,
                            'best_fitness': self.best_fitness,
                            'training_results': f.read(),
                            'model': self.ema.ema,
                            'optimizer': None if final_epoch else self.optimizer.state_dict(),
                            'wandb_id': self.wandb_run.id if self.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, self.last)
                if self.best_fitness == fi:
                    torch.save(ckpt, self.best)
                del ckpt
   
   def train_end(self): 
        if self.rank in [-1, 0]:
            # Strip optimizers
            n = self.opt.name if self.opt.name.isnumeric() else ''
            fresults, flast, fbest = self.log_dir / f'results{n}.txt', self.wdir / f'last{n}.pt', self.wdir / f'best{n}.pt'
            for f1, f2 in zip([self.wdir / 'last.pt', self.wdir / 'best.pt', self.results_file], [flast, fbest, fresults]):
                if os.path.exists(f1):
                    os.rename(f1, f2)  # rename
                    if str(f2).endswith('.pt'):  # is *.pt
                        strip_optimizer(f2)  # strip optimizer
                        os.system('gsutil cp %s gs://%s/weights' % (f2, self.opt.bucket)) if self.opt.bucket else None  # upload
            # Finish
            if not self.opt.evolve:
                plot_results(save_dir=self.log_dir)  # save as results.png
            logger.info('%g epochs completed in %.3f hours.\n' % (self.epoch - self.start_epoch + 1, (time.time() - t0) / 3600))

        dist.destroy_process_group() if self.rank not in [-1, 0] else None
        torch.cuda.empty_cache()

       


   def train(self):
        for self.epoch in range(self.start_epoch, self.epochs):  # epoch ------------------------------------------------------------------
            self.on_epoch_start()
            
            for i, (self.imgs, self.targets, self.paths, _) in self.pbar:  # batch -------------------------------------------------------------
                self.training_step(i)
            # end epoch ----------------------------------------------------------------------------------------------------
            self.on_epoch_end()
        # end training
        self.train_end()

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
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume
    if opt.resume:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)

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

    Train_Model = Training(hyp,opt, device, tb_writer, wandb) 
    Train_Model.train()  
    # Evolve hyperparameters (optional)
    """
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.logdir) / 'evolve' / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')  
        """
