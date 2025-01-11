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
parser.add_argument('--dataset_name', type=str, default='LA', help='dataset_name')
parser.add_argument('--device', type=str, default='cuda', help='device: cpu or cuda')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='Mine', help='exp_name')
parser.add_argument('--model', type=str, default='mine_v4_train_all_womc', help='model_name')
parser.add_argument('--max_iteration', type=int, default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
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
parser.add_argument('--N', type=int, default=2, help='the number of the sample')
parser.add_argument('--config', type=str, default='ul_0_l80', help='') #ul_0_l80,ul_0_l16,ul_0_l8

args = parser.parse_args()

if args.dataset_name == "LA":
    patch_size = (128, 128, 80)
    #args.root_path = args.root_path + 'data/LA'
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




def train(config='ul_0_l80'):#ul_0_l80,ul_0_l16,ul_0_l8
    
    args.config =config
    
    snapshot_path = args.root_path + "model/{}_{}/{}".format(args.dataset_name,args.config,
                                                                    args.model)
    print('snapshot_path:{}  #is_exists:{}'.format(snapshot_path, os.path.exists(snapshot_path)))

    num_classes = 2
    train_data_path = args.root_path + 'data/' + args.dataset_name
    print('train_data_path:{}  #is_exists:{}'.format(train_data_path, os.path.exists(train_data_path)))
    
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

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    sampler = torch.utils.data.SequentialSampler(range(labelnum))
    trainloader = torch.utils.data.DataLoader(db_train,batch_size=args.batch_size,sampler=sampler,num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter('/root/tf-logs/{}_{}_in_{}'.format(args.model,args.config,args.dataset_name))

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


            #y = out1  #(b_s,c,r,r,d)
            
            y_prob = F.softmax(out1, dim=1)
            loss_seg += F.cross_entropy(out1, label_batch)
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

    
            if iter_num >= 800 and iter_num % 400 == 0:
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
                
                writer.add_scalar('{}_{}_Var_dice/Dice_model_1'.format(args.model,args.config), dice_sample_1, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Dice_model_2'.format(args.model,args.config), dice_sample_2, iter_num)
                writer.add_scalar('{}_{}_Var_dice/Best_dice'.format(args.model,args.config), best_dice, iter_num)
                #writer.add_scalar('{}_{}_Var_dice/Diff_dice'.format(args.model,args.labelnum), abs(dice_sample_1-dice_sample_2), iter_num)

                sub_model_2.change_N(args.N)
                model.train()

            if(iter_num>=max_iterations): break
            
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

    
if __name__ == "__main__":#ul_0_l80,ul_0_l16,ul_0_l8
    
    args.labeled_bs = 0
    labeled_bs=4
    args.labelnum=16
    train(config='ul_0_l16_2')
    
    #args.labeled_bs = 0
    #labeled_bs=4
    #args.labelnum=8
    #train(config='ul_0_l8')