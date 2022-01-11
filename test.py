from datasets.dtu import DTUDataset
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os
from torchvision.utils import make_grid, save_image
from opt import get_opts
torch.backends.cudnn.benchmark = True

from models.NovelDepthNet import NovelDepthNet
from utils import load_ckpt
from inplace_abn import ABN
from utils import *

hparams = get_opts()
model = NovelDepthNet(n_depths=[8,32,48],
                      interval_ratios=[1.0,2.0,4.0],
                      norm_act=ABN,hparams=hparams).cuda()
load_ckpt(model,'ckpts/sup/epoch=27.ckpt')
model.eval()

dataset = DTUDataset(hparams.root_dir, 'test', n_views=3, depth_interval=2.65)

def decode_batch(batch):
    imgs = batch['imgs']
    proj_mats = batch['proj_mats']
    depths = batch['depths']
    masks = batch['masks']
    init_depth_min = batch['init_depth_min'].item()
    depth_interval = batch['depth_interval'].item()
    return imgs, proj_mats, depths, masks, init_depth_min, depth_interval

for i in range(len(dataset)):
    imgs, proj_mats, depths, masks, init_depth_min, depth_interval = decode_batch(dataset[i])
    unpreprocess = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])

    img = unpreprocess(imgs[0]).unsqueeze(0)
    depth = visualize_depth(depths['level_0']*masks['level_0']).unsqueeze(0)
    save_image(img,'results/img_{}.png'.format(i))
    save_image(depth,'results/depth_{}.png'.format(i))

    list_img = []
    for j in range(hparams.n_views):
        img_j = unpreprocess(imgs[j]).cpu()
        list_img.append(img_j)
    stack_imgs = torch.stack([list_img[k] for k in range(hparams.n_views)])
    save_image(stack_imgs,'results/input_{}.png'.format(i))

    t = time.time()
    with torch.no_grad():
        results = model(imgs.unsqueeze(0).cuda(), proj_mats.unsqueeze(0).cuda(), init_depth_min, depth_interval)
        torch.cuda.synchronize()
    print('inference time', time.time()-t)
    pred_img = unpreprocess(results['warp_view_0'][0]).unsqueeze(0)
    pred_depth = visualize_depth(results["depth_0"][0] * masks['level_0'].cuda()).unsqueeze(0)
    save_image(pred_img,'results/pred_img_{}.png'.format(i))
    save_image(pred_depth,'results/pred_depth_{}.png'.format(i))
    print(str(i)+'/{}'.format(len(dataset)))