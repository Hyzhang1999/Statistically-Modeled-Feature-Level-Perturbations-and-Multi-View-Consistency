import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import math


from collections import OrderedDict
import pandas as pd 

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *

sys.path.append(os.path.join(os.path.abspath('.'), 'networks'))

from networks.net_factory import net_factory


def test_all_figures(names,iters,data_name = 'Pancreas',labeled = 12):
    writer = SummaryWriter('/root/tf-logs/GetScaleInfm_in_{}'.format(data_name))
    writers = [SummaryWriter('/root/tf-logs/GetScaleInfm_in_{}/{}'.format(data_name,name)) for name in names]
    nets = [net_factory(net_type='mine_v4_v4', in_chns=1, class_num=2,input_size =(96, 96, 96),mode="test",
                        device='cuda') for name in names]
    for idx, net in enumerate(nets):
        load_path = '../model/{}_Mine_{}_labeled/mine_v4_v3/iter_{}_in_{}.pth'.format(data_name,labeled,iters[idx],data_name)
        load_dir =os.path.join(load_path)   #load_model
        net.load_state_dict(torch.load(load_dir), strict=False)
        net.eval()
        print("init weight from {}".format(load_path))
        
    with open('../data/{}/train.list'.format(data_name), 'r') as f:
        image_list = f.readlines()
    if data_name == 'Pancreas':
        image_list = ["../data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    else:
        image_list = ["../data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
        
    transform=transforms.Compose([
        RandomCrop((96, 96, 96)) if data_name == 'Pancreas' else RandomCrop((128, 128, 80)),
        ToTensor(),])
    
    image_list = image_list[-5:-1]
    loader = tqdm(image_list)
    index = 0
    
    for image_path in loader:
        index+=1
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]  
        sample = {'image': image, 'label': label}
        sample = transform(sample)
        image,label = sample['image'],sample['label'] #label:(96,96,96)
        image = image.reshape((1,)+image.size())
        
        with torch.no_grad():
            image = image.cuda()
            if data_name == 'Pancreas':
                label = label.permute(2,0,1).reshape((96,1,96,96))
            else:
                label = label.permute(2,0,1).reshape((80,1,128,128))
            snapshot_label = label.expand(-1,3,-1,-1)
            writer.add_images('{}_Labels'.format(data_name),snapshot_label,index)
            
            for which,net in enumerate(nets):
                sub_model = net.gbdl
                sub_model.change_N(1)
                #out = net(image)
                out = []
                for i in range(2):
                    out.append(net(image)[-1])
                _,_,H,W,D = out[0].size()
                y1 = F.softmax(out[0], dim=1)[:,0] #[1,H,W,D]
                y2 = F.softmax(out[1], dim=1)[:,0] #[1,H,W,D]
                eps = 1e-4
                ds_mps = torch.abs(y1 - y2)
                ds_mps = (ds_mps - torch.min(ds_mps)) / \
                         (torch.max(ds_mps) - torch.min(ds_mps) + eps)  # [H,W,D]
                ds_mps = ds_mps.permute(3, 0, 1,2) #[D,1,H,W]
                snapshot_img = ds_mps.expand(-1,3,-1,-1)
                writers[which].add_images('{}_{}_Images'.format(data_name,iters),snapshot_img,index)
            
    print('done!')
    for w in writers:w.close()
    
    
def test_all_figures_2(names,iters,data_name = 'LA',labeled = 16):
    #writer = SummaryWriter('/root/tf-logs/GetScaleInfm_in_{}'.format(data_name))
    #writers = [SummaryWriter('/root/tf-logs/GetScaleInfm_in_{}/{}'.format(data_name,name)) for name in names]
    nets = [net_factory(net_type='mine_v4_v4', in_chns=1, class_num=2,input_size =(128, 128, 80),mode="test",
                        device='cuda') for name in names]
    for idx, net in enumerate(nets):
        load_path = '../model/{}_Mine_{}_labeled/mine_v4_v4/iter_{}_in_{}.pth'.format(data_name,labeled,iters[idx],data_name)
        load_dir =os.path.join(load_path)   #load_model
        net.load_state_dict(torch.load(load_dir), strict=False)
        net.eval()
        print("init weight from {}".format(load_path))
        
    with open('../data/{}/train.list'.format(data_name), 'r') as f:
        image_list = f.readlines()
    if data_name == 'Pancreas':
        image_list = ["../data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    else:
        image_list = ["../data/LA/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
        
    transform=transforms.Compose([
        RandomCrop((96, 96, 96)) if data_name == 'Pancreas' else RandomCrop((128, 128, 80)),
        ToTensor(),])
    
    image_list = image_list[-5:-1]
    loader = tqdm(image_list)
    index = 0
    
    for image_path in loader:
        index+=1
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]  
        sample = {'image': image, 'label': label}
        sample = transform(sample)
        image,label = sample['image'],sample['label'] #label:(96,96,96)
        image = image.reshape((1,)+image.size())
        
        with torch.no_grad():
            image = image.cuda()
            if data_name == 'Pancreas':
                label = label.permute(2,0,1).reshape((96,1,96,96))
            else:
                label = label.permute(2,0,1).reshape((80,1,128,128))
            snapshot_label = label.expand(-1,3,-1,-1)
            writer.add_images('{}_Labels'.format(data_name),snapshot_label,index)
            
            for which,net in enumerate(nets):
                sub_model = net.gbdl
                sub_model.change_N(1)
                #out = net(image)
                out = []
                for i in range(2):
                    out.append(net(image)[-1])
                _,_,H,W,D = out[0].size()
                y1 = F.softmax(out[0], dim=1)[:,0] #[1,H,W,D]
                y2 = F.softmax(out[1], dim=1)[:,0] #[1,H,W,D]
                eps = 1e-4
                ds_mps = torch.abs(y1 - y2)
                ds_mps = (ds_mps - torch.min(ds_mps)) / \
                         (torch.max(ds_mps) - torch.min(ds_mps) + eps)  # [H,W,D]
                ds_mps = ds_mps.permute(3, 0, 1,2) #[D,1,H,W]
                snapshot_img = ds_mps.expand(-1,3,-1,-1)
                writers[which].add_images('{}_{}_Images'.format(data_name,iters),snapshot_img,index)
            
    print('done!')
    for w in writers:w.close()
    

    
if __name__ == '__main__':
    names = ['A2','B2','C2','D2','E2']
    test_all_figures(names,iters=[12000,13000,14000,15000,11000])