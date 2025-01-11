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


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Pancreas', help='dataset_name')
parser.add_argument('--device', type=str, default='cuda', help='device: cpu or cuda')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Mine', help='exp_name')
parser.add_argument('--model', type=str, default='mine_v4_test', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=62, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=16, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1.25, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--beta', type=float, default=0.5, help='weight to balance all losses')
parser.add_argument('--N', type=int, default=4, help='the number of the sample')

args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model)
print('snapshot_path:{}  #is_exists:{}'.format(snapshot_path, os.path.exists(snapshot_path)))

num_classes = 2
train_data_path = args.root_path + 'data/' + args.dataset_name
print('train_data_path:{}  #is_exists:{}'.format(train_data_path, os.path.exists(train_data_path)))

if args.dataset_name == "LA":
    patch_size = (128, 128, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80

elif args.dataset_name == "Pancreas":
    patch_size = (96, 96, 96)
    args.max_samples = 62

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def get_snap_img(imgs: list, _size: object, paint_indx: int = 2, ins_width: int = 2) -> object:
    # imgs:[imgs,x_rs,ds_mps,gts,outputs,uc_mps]
    volume_batch = imgs[0]
    label_batch = imgs[3]
    x_rs = imgs[1]
    ds_mps = imgs[2]  # [b_s,c,h,w,d]
    outputs = imgs[4]  # [i,b_s,2,h,w,d]
    uc_mps = imgs[-1]  # [b_s,c,h,w,d]

    num_outputs = outputs.size()[0]
    B, C, H, W, D = _size
    snapshot_img = torch.ones(
        size=(D, 1, (num_outputs + 5) * H + (num_outputs + 6 + 2) * ins_width, W + 2 * ins_width),
        dtype=torch.float32).cuda()  # [D,1,Hs,Ws]

    # print('volume_batch:',volume_batch.size())
    # print('label_batch:',label_batch.size())
    # print('x_rs:',x_rs.size())
    # print('ds_mps:',ds_mps.size())
    # print('outputs:',outputs.size())
    # print('uc_mps:',uc_mps.size())
    # print('snapshot_img:',snapshot_img.size())

    target = label_batch[paint_indx, ...].permute(2, 0, 1)  # [D,H,W]......taske unlabeled-1th
    train_img = volume_batch[paint_indx, 0, ...].permute(2, 0, 1)  # [D,H,W]......taske unlabeled-1th
    x_rs = x_rs[paint_indx, 0, ...].permute(2, 0, 1)  # [D,H,W]......taske unlabeled-1th

    norm_imgs = (train_img - torch.min(train_img)) / \
                (torch.max(train_img) - torch.min(train_img))  # [D,H,W]
    norm_x_rs = (x_rs - torch.min(x_rs)) / \
                (torch.max(x_rs) - torch.min(x_rs))  # [D,H,W]

    H_now_pos = ins_width
    W_now_pos = ins_width

    def update_H_now_pos():
        return H_now_pos + H + ins_width

    for e in [norm_imgs, norm_x_rs, target, None]:
        if e is not None:
            snapshot_img[:, 0, H_now_pos:H + H_now_pos, W_now_pos:W + W_now_pos] = e
            H_now_pos = update_H_now_pos()
        else:
            for i in range(num_outputs):
                snapshot_img[:, 0, H_now_pos:H_now_pos + H, W_now_pos:W + W_now_pos] = \
                    outputs[i][paint_indx:][0, 1].permute(2, 0, 1)
                H_now_pos = update_H_now_pos()

    snapshot_img[:, :, H_now_pos:H + H_now_pos, W_now_pos:W + W_now_pos] = ds_mps[paint_indx].permute(3, 0, 1, 2)
    H_now_pos = update_H_now_pos()
    snapshot_img[:, :, H_now_pos:H + H_now_pos, W_now_pos:W + W_now_pos] = uc_mps[paint_indx].permute(3, 0, 1, 2)
    snapshot_img = snapshot_img.expand(-1, 3, -1, -1)
    return snapshot_img


def train():
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, input_size=patch_size,N=args.N, mode="train",
                        device='cuda' if torch.cuda.is_available() else 'cpu')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('The model running in {}'.format(args.device))

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter('/root/tf-logs/{}_in_{}'.format(args.model,args.dataset_name))

    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    rebuild_loss = losses.mse_loss
    kl_loss = losses.kl_loss

    iter_num = 0
    best_dice = 0

    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    logging.info("{} itertations per epoch".format(len(trainloader)))
    logging.info("totall epoch:".format(max_epoch))
    iterator = tqdm(range(max_epoch))

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            iterator.set_description('epochs:{}/{}  i_batch:{}/{}  toatal_iteration:{}/{} '.
                                     format(epoch_num, max_epoch, i_batch, len(trainloader), iter_num, max_iterations))

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            volume_batch, label_batch = volume_batch.to(torch.device(args.device)), label_batch.to(
                torch.device(args.device))

            model.train()
            outputs, property = model(volume_batch)

            out1, out2 = outputs
            mean, covar, x_r, Z = property

            y_ori = torch.zeros((2,) + out2.shape).cuda()  #save_probs
            y_pseudo_label = torch.zeros((2,) + out2.shape).cuda()  #save_pseudo_labels
            
            #print('zero_y_ori:',y_ori.size())
            #print('zero_y_pseudo_label:',y_pseudo_label.size())

            out1_N = out1.repeat(args.N,1,1,1,1)  #(b_s*N,c,r,r,d)
            label_batch_N = label_batch[:labeled_bs].repeat(args.N,1,1,1)  #(l_bs*N,r,r,d)
            volume_batch_N = volume_batch.repeat(args.N,1,1,1,1) #(b_s*N,c,r,r,d)

            loss_seg = 0
            loss_seg_dice = 0
            loss_kld = 0
            loss_rebuild = 0
            loss_consist = 0

            loss_kld += kl_loss(mean, covar)
            loss_rebuild += rebuild_loss(volume_batch_N, x_r)
            loss_VAE = 0.001* loss_kld / volume_batch_N.size(0) + 1.0 * loss_rebuild


            y = out1[:labeled_bs, ...]  #(l_bs,c,r,r,d)
            y_prob = F.softmax(y, dim=1)
            loss_seg += F.cross_entropy(y[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_dice += dice_loss(y_prob[:, 1, ...], label_batch[:labeled_bs, ...] == 1)
            
            y_prob_all = F.softmax(out1_N, dim=1)
            #print(y_prob_all.size(),y_ori.size())
            y_ori[0] = y_prob_all
            y_pseudo_label[0] = sharpening(y_prob_all)

            b,c,r,r,d= out1.size()
            y = out2.view(args.N,b,c,r,r,d)[:,0:labeled_bs,...] #(N,l_bs,c,r,r,d)                              
            y = y.reshape(labeled_bs*args.N,c,r,r,d)
            y_prob = F.softmax(y, dim=1)
            loss_seg += F.cross_entropy(y, label_batch_N)
            loss_seg_dice += dice_loss(y_prob[:, 1, ...], label_batch_N == 1)
            
            y_prob_all = F.softmax(out2, dim=1)
            y_ori[1] = y_prob_all
            y_pseudo_label[1] = sharpening(y_prob_all)

            loss_consist += consistency_criterion(y_ori[0], y_pseudo_label[1])
            loss_consist += consistency_criterion(y_ori[1], y_pseudo_label[0])

            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 200)

            loss = args.lamda * ( 2.0 * loss_seg_dice + 1.0 * loss_seg) + args.beta * loss_VAE + consistency_weight * loss_consist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f, loss_VAE: %03f, loss_kld: %03f , loss_rebuild: %03f' % (
                    iter_num, loss, loss_seg_dice, loss_consist, loss_VAE, loss_kld, loss_rebuild))

            writer.add_scalar('{}_{}_Labeled_loss/loss_seg_dice'.format(args.model,args.labelnum), loss_seg_dice, iter_num)
            writer.add_scalar('{}_{}_Labeled_loss/loss_seg_ce'.format(args.model,args.labelnum), loss_seg, iter_num)
            writer.add_scalar('{}_{}_Co_loss/consistency_loss'.format(args.model,args.labelnum), loss_consist, iter_num)
            writer.add_scalar('{}_{}_Co_loss/consist_weight'.format(args.model,args.labelnum), consistency_weight, iter_num)
            writer.add_scalar('{}_{}_Co_loss/kld_loss'.format(args.model,args.labelnum), loss_kld, iter_num)
            writer.add_scalar('{}_{}_Co_loss/rebuild_loss'.format(args.model,args.labelnum), loss_rebuild, iter_num)
            writer.add_scalar('{}_{}_Co_loss/VAE_loss'.format(args.model,args.labelnum), loss_VAE, iter_num)
            
            with torch.no_grad():
                z = torch.mean(Z)
                m = torch.mean(mean)
                _covar = torch.mean(covar,dim=(0,1))
                c_diag = torch.mean(torch.diag(_covar))

            writer.add_scalar('{}_{}_VAE/Z'.format(args.model,args.labelnum), z, iter_num)
            writer.add_scalar('{}_{}_VAE/mean'.format(args.model,args.labelnum), m, iter_num)
            writer.add_scalar('{}_{}_VAE/covar'.format(args.model,args.labelnum), c_diag, iter_num)

            if iter_num >=1000 and iter_num % 1000 == 0:
                # imgs:[imgs,x_rs,ds_mps,gts,outputs,uc_mps]
                with torch.no_grad():
                    eps = 1e-4
                    x_r = torch.mean(x_r.view(((args.N,)+volume_batch.size())),dim=0,keepdim=False)
                    ds_mps = torch.mean(torch.abs(volume_batch - x_r), dim=1, keepdim=True)
                    ds_mps = (ds_mps - torch.min(ds_mps)) / \
                             (torch.max(ds_mps) - torch.min(ds_mps) + eps)  # [b,1,H,W,D]

                    _mean = torch.mean(y_ori, dim=0).expand_as(y_ori)
                    _var = torch.mean(torch.abs(y_ori - _mean), dim=0, keepdim=True)
                    _var = (_var - torch.min(_var)) / \
                           (torch.max(_var) - torch.min(_var) + eps)  # [n,b,2,H,W,D]
                    _var = _var[0, :, 1:, ...]  # [b*N,1,H,W,D]

                    y_ori = torch.mean(y_ori.view(2,args.N,b,2,r,r,d),dim=1,keepdim=False)
                    #print('y_ori:',y_ori.size())
                    _var = torch.mean(_var.view(args.N,b,1,r,r,d),dim=0,keepdim=False)
                    #print('_var:',y_ori.size())

                    imgs = [volume_batch, x_r, ds_mps, label_batch, y_ori, _var]
                    snapshot_img = get_snap_img(imgs, _size=y_ori[0].size(), paint_indx=labeled_bs, ins_width=2)

                writer.add_images(('{}_{}_Epoch_%d_Iter_%d_unlabel'.format(args.model,args.labelnum)) % (epoch_num, iter_num), snapshot_img)

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                sub_model_1 = model.v_net
                sub_model_2 = model.gbdl
                sub_model_2.change_N(1)
                if args.dataset_name == "LA":
                    dice_sample_1 = test_patch.var_all_case(sub_model_1, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                    dice_sample_2 = test_patch.var_all_case(sub_model_2, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas":
                    dice_sample_1 = test_patch.var_all_case(sub_model_1, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas')
                    dice_sample_2 = test_patch.var_all_case(sub_model_2, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=16, stride_z=16, dataset_name='Pancreas')
                dice_sample = dice_sample_1

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_best_path = os.path.join(snapshot_path, '{}_best_model_in_{}.pth'.format(args.model,args.dataset_name))
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_best_path))
                
                writer.add_scalar('{}_{}_Var_dice/Dice_model_1'.format(args.model,args.labelnum), dice_sample_1, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Dice_model_2'.format(args.model,args.labelnum), dice_sample_2, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Best_dice'.format(args.model,args.labelnum), best_dice, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Diff_dice'.format(args.model,args.labelnum), abs(dice_sample_1-dice_sample_2), iter_num)

                sub_model_2.change_N(args.N)
                model.train()

            if(iter_num>=max_iterations): break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

    

    
def train_cps():
    
    snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    'CPSs')
    print('snapshot_path:{}  #is_exists:{}'.format(snapshot_path, os.path.exists(snapshot_path)))
    
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.info(str(args))
      
    Strong_CPS = net_factory(net_type='Strong_CPS', in_chns=1, class_num=num_classes,mode="train",device='cuda' if torch.cuda.is_available() else 'cpu')
    Robust_CPS =  net_factory(net_type='Robust_CPS', in_chns=1, class_num=num_classes,mode="train",device='cuda' if torch.cuda.is_available() else 'cpu')
    Weakly_CPS = net_factory(net_type='Weakly_CPS', in_chns=1, class_num=num_classes,mode="train",device='cuda' if torch.cuda.is_available() else 'cpu')
    Frail_CPS = net_factory(net_type='Frail_CPS', in_chns=1, class_num=num_classes,mode="train",device='cuda' if torch.cuda.is_available() else 'cpu')
    models = [Strong_CPS,Robust_CPS,Weakly_CPS,Frail_CPS]
    
    
    
    def get_log():
        return  OrderedDict([
            ('iter_num', []),
            ('loss_seg_dice', []),
            ('consistency_loss', []),
            ('model_var', []),
            ('now_best_dice', []),
        ])
    logs_list = []
    for i in range(4):logs_list.append(get_log())
    names = [model.name for model in models]
    logs = dict(zip(names,logs_list))
    print(logs)
    
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('The models running in {}'.format(args.device))

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer_Strong_CPS = optim.SGD(models[0].parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_Robust_CPS = optim.SGD(models[1].parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_Weakly_CPS = optim.SGD(models[2].parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_Frail_CPS = optim.SGD(models[-1].sub_net_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizers = [optimizer_Strong_CPS,optimizer_Robust_CPS,optimizer_Weakly_CPS,optimizer_Frail_CPS]
    
    writers = [SummaryWriter('/root/tf-logs/CPSs_in_{}/{}'.format(args.dataset_name,model.name)) for model in models]
    

    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss

    iter_num = 0
    best_dices = [0,0,0,0]

    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    logging.info("{} itertations per epoch".format(len(trainloader)))
    logging.info("totall epoch:".format(max_epoch))
    iterator = tqdm(range(max_epoch))

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            iterator.set_description('epochs:{}/{}  i_batch:{}/{}  toatal_iteration:{}/{} '.
                                     format(epoch_num, max_epoch, i_batch, len(trainloader), iter_num, max_iterations))

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            volume_batch, label_batch = volume_batch.to(torch.device(args.device)), label_batch.to(
                torch.device(args.device))
            
            for which , model in enumerate(models):
                model.train()
                outputs = model(volume_batch) #outputs:out1,out2
                num_outputs = len(outputs)
                y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
                y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)

                loss_seg = 0
                loss_seg_dice = 0 
                model_var = 0
                
                for idx in range(num_outputs):
                    y = outputs[idx][:labeled_bs,...]
                    y_prob = F.softmax(y, dim=1)
                    loss_seg_dice += dice_loss(y_prob[:,1,...], label_batch[:labeled_bs,...] == 1)

                    y_all = outputs[idx]
                    y_prob_all = F.softmax(y_all, dim=1)
                    y_ori[idx] = y_prob_all
                    y_pseudo_label[idx] = sharpening(y_prob_all)

                loss_consist = 0
                for i in range(num_outputs):
                    for j in range(num_outputs):
                        if i != j:
                            loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])

                
                consistency_weight = get_current_consistency_weight(iter_num//150)
                loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist
                
                optimizers[which].zero_grad()
                loss.backward()
                optimizers[which].step()
                
                with torch.no_grad():
                    _mean = torch.mean(y_ori, dim=0).expand_as(y_ori)
                    _var = torch.mean(torch.abs(y_ori - _mean))
                    model_var = _var.item()
                    if which == 3:
                        model.update_params(iter_num)    


                #writer.add_scalar('loss_seg_dice_{}'.format(model.name), loss_seg_dice, iter_num)
                #writer.add_scalar('consistency_loss_{}'.format(model.name), loss_consist, iter_num)
                #writer.add_scalar('model_var_{}'.format(model.name), model_var, iter_num)
                writers[which].add_scalar('consistency_loss',loss_consist,iter_num)
                writers[which].add_scalar('model_vars',model_var,iter_num)
              

                if iter_num >= 100 and iter_num % 50 == 0:
                    model.eval()
                    sub_model_1 = model.sub_net_1
                    sub_model_2 = model.sub_net_2
                    if args.dataset_name == "LA":
                        dice_sample_1 = test_patch.var_part_case(sub_model_1, num_classes=num_classes, patch_size=patch_size,
                                                              stride_xy=18, stride_z=4, dataset_name='LA')
                        dice_sample_2 = test_patch.var_part_case(sub_model_2, num_classes=num_classes, patch_size=patch_size,
                                                              stride_xy=18, stride_z=4, dataset_name='LA')
                    elif args.dataset_name == "Pancreas":
                        dice_sample_1 = test_patch.var_part_case(sub_model_1, num_classes=num_classes, patch_size=patch_size,
                                                              stride_xy=16, stride_z=16, dataset_name='Pancreas')
                        dice_sample_2 = test_patch.var_part_case(sub_model_2, num_classes=num_classes, patch_size=patch_size,
                                                                stride_xy=16, stride_z=16, dataset_name='Pancreas')
                    dice_sample = max(dice_sample_1,dice_sample_2)

                    if dice_sample > best_dices[which]:
                        best_dices[which] = dice_sample
                    writers[which].add_scalar('CPSs/Dice_model_1_{}'.format(model.name), dice_sample_1, iter_num)
                    writers[which].add_scalar('CPSs/Dice_model_2_{}'.format(model.name), dice_sample_2, iter_num)
                    writers[which].add_scalar('CPSs/Best_dice_{}'.format(model.name), dice_sample, iter_num)
                    writers[which].add_scalar('CPSs/diff_dice_{}'.format(model.name), abs(dice_sample_1-dice_sample_2), iter_num)
                    
                    writers[which].add_scalar('best_score',best_dices[which],iter_num)
                    
                    model.train()
                
                if  iter_num % 2 == 0:
                    logs[model.name]['iter_num'].append(iter_num)
                    logs[model.name]['loss_seg_dice'].append(loss_seg_dice.item())
                    logs[model.name]['consistency_loss'].append(loss_consist.item())
                    logs[model.name]['model_var'].append(model_var)
                    logs[model.name]['now_best_dice'].append(best_dices[which])
                    save_log_path = os.path.join(snapshot_path,model.name +'_in_{}'.format(args.dataset_name) + '_log.csv')
                    pd.DataFrame(logs[model.name]).to_csv(save_log_path, index=False)

                if iter_num >=4000 and iter_num %2000 ==0:
                    save_mode_path = os.path.join(snapshot_path,model.name +'iter_' + str(iter_num)+'_in_{}'.format(args.dataset_name) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info("save {} for iter {} to {}".format(model.name,iter_num,save_mode_path))
                    
            iter_num = iter_num + 1
        
        if iter_num >= max_iterations:
            iterator.close()
            break
            
    writer.close()

    
    
def train_all():
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, input_size=patch_size,N=args.N, mode="train",
                        device='cuda' if torch.cuda.is_available() else 'cpu')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('The model running in {}'.format(args.device))

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]))
    elif args.dataset_name == "Pancreas":
        db_train = Pancreas(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter('/root/tf-logs/{}_in_{}'.format(args.model,args.dataset_name))

    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    rebuild_loss = losses.mse_loss
    kl_loss = losses.kl_loss

    iter_num = 0
    best_dice = 0

    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    logging.info("{} itertations per epoch".format(len(trainloader)))
    logging.info("totall epoch:".format(max_epoch))
    iterator = tqdm(range(max_epoch))

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            iterator.set_description('epochs:{}/{}  i_batch:{}/{}  toatal_iteration:{}/{} '.
                                     format(epoch_num, max_epoch, i_batch, len(trainloader), iter_num, max_iterations))

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            volume_batch, label_batch = volume_batch.to(torch.device(args.device)), label_batch.to(
                torch.device(args.device))

            model.train()
            outputs, property = model(volume_batch)

            out1, out2 = outputs
            mean, covar, x_r, Z = property

            y_ori = torch.zeros((2,) + out2.shape).cuda()  #save_probs
            y_pseudo_label = torch.zeros((2,) + out2.shape).cuda()  #save_pseudo_labels
            
            #print('zero_y_ori:',y_ori.size())
            #print('zero_y_pseudo_label:',y_pseudo_label.size())

            out1_N = out1.repeat(args.N,1,1,1,1)  #(b_s*N,c,r,r,d)
            label_batch_N = label_batch.repeat(args.N,1,1,1)  #(l_bs*N,r,r,d)
            volume_batch_N = volume_batch.repeat(args.N,1,1,1,1) #(b_s*N,c,r,r,d)

            loss_seg = 0
            loss_seg_dice = 0
            loss_kld = 0
            loss_rebuild = 0
            loss_consist = 0

            loss_kld += kl_loss(mean, covar)
            loss_rebuild += rebuild_loss(volume_batch_N, x_r)
            loss_VAE = 0.001* loss_kld / volume_batch_N.size(0) + 1.0 * loss_rebuild


            y = out1  #(b_s,c,r,r,d)
            
            
            
            y_prob = F.softmax(y, dim=1)
            loss_seg += F.cross_entropy(y, label_batch)
            loss_seg_dice += dice_loss(y_prob[:, 1, ...], label_batch == 1)
            
            y_prob_all = F.softmax(out1_N, dim=1)
            #print(y_prob_all.size(),y_ori.size())
            y_ori[0] = y_prob_all
            y_pseudo_label[0] = sharpening(y_prob_all)

            b,c,r,r,d= out1.size()
            y = out2.view(args.N,b,c,r,r,d) #(N,b_s,c,r,r,d)                              
            y = y.reshape(b*args.N,c,r,r,d)
            y_prob = F.softmax(y, dim=1)
            loss_seg += F.cross_entropy(y, label_batch_N)
            loss_seg_dice += dice_loss(y_prob[:, 1, ...], label_batch_N == 1)
            
            y_prob_all = F.softmax(out2, dim=1)
            y_ori[1] = y_prob_all
            y_pseudo_label[1] = sharpening(y_prob_all)

            loss_consist += consistency_criterion(y_ori[0], y_pseudo_label[1])
            loss_consist += consistency_criterion(y_ori[1], y_pseudo_label[0])

            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num // 200)

            loss = args.lamda * ( 2.0 * loss_seg_dice + 1.0 * loss_seg) + args.beta * loss_VAE + consistency_weight * loss_consist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(
                'iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f, loss_VAE: %03f, loss_kld: %03f , loss_rebuild: %03f' % (
                    iter_num, loss, loss_seg_dice, loss_consist, loss_VAE, loss_kld, loss_rebuild))

            writer.add_scalar('{}_{}_Labeled_loss/loss_seg_dice'.format(args.model,args.labelnum), loss_seg_dice, iter_num)
            writer.add_scalar('{}_{}_Labeled_loss/loss_seg_ce'.format(args.model,args.labelnum), loss_seg, iter_num)
            writer.add_scalar('{}_{}_Co_loss/consistency_loss'.format(args.model,args.labelnum), loss_consist, iter_num)
            writer.add_scalar('{}_{}_Co_loss/consist_weight'.format(args.model,args.labelnum), consistency_weight, iter_num)
            writer.add_scalar('{}_{}_Co_loss/kld_loss'.format(args.model,args.labelnum), loss_kld, iter_num)
            writer.add_scalar('{}_{}_Co_loss/rebuild_loss'.format(args.model,args.labelnum), loss_rebuild, iter_num)
            writer.add_scalar('{}_{}_Co_loss/VAE_loss'.format(args.model,args.labelnum), loss_VAE, iter_num)
            
            with torch.no_grad():
                z = torch.mean(Z)
                m = torch.mean(mean)
                _covar = torch.mean(covar,dim=(0,1))
                c_diag = torch.mean(torch.diag(_covar))

            writer.add_scalar('{}_{}_VAE/Z'.format(args.model,args.labelnum), z, iter_num)
            writer.add_scalar('{}_{}_VAE/mean'.format(args.model,args.labelnum), m, iter_num)
            writer.add_scalar('{}_{}_VAE/covar'.format(args.model,args.labelnum), c_diag, iter_num)

    
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                sub_model_1 = model.v_net
                sub_model_2 = model.gbdl
                sub_model_2.change_N(1)
                if args.dataset_name == "LA":
                    dice_sample_1 = test_patch.var_all_case(sub_model_1, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                    dice_sample_2 = test_patch.var_all_case(sub_model_2, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                elif args.dataset_name == "Pancreas":
                    dice_sample_1 = test_patch.var_all_case(sub_model_1, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=16, stride_z=16, dataset_name='Pancreas')
                    dice_sample_2 = test_patch.var_all_case(sub_model_2, num_classes=num_classes, patch_size=patch_size,
                                                            stride_xy=16, stride_z=16, dataset_name='Pancreas')
                dice_sample = dice_sample_1

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_best_path = os.path.join(snapshot_path, '{}_best_model_in_{}.pth'.format(args.model,args.dataset_name))
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_best_path))
                
                writer.add_scalar('{}_{}_Var_dice/Dice_model_1'.format(args.model,args.labelnum), dice_sample_1, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Dice_model_2'.format(args.model,args.labelnum), dice_sample_2, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Best_dice'.format(args.model,args.labelnum), best_dice, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Diff_dice'.format(args.model,args.labelnum), abs(dice_sample_1-dice_sample_2), iter_num)

                sub_model_2.change_N(args.N)
                model.train()

            if(iter_num>=max_iterations): break
            
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

    
if __name__ == "__main__":
    train()
    #train_cps()