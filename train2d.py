import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)

import sys
sys.path.append('./networks/Archs_2d.py')

torch.autograd.set_detect_anomaly(True)

from networks.net_factory import net_factory_
from utils import losses, metrics, ramps, val_2d


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_gamma(epoch):
    return args.gamma* ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def sharpening(P):
    T = 1 / args.temperature
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen


dev = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--dataset_name', type=str, default='ACDC', help='dataset_name')
parser.add_argument('--exp', type=str, default='Mine', help='experiment_name')
parser.add_argument('--model', type=str, default='MT', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')


# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=7, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')


# costs
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--gamma', type=float, default=1.0, help='gamma')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1.0, help='weight to balance all losses')
parser.add_argument('--alpha', type=float, default=0.05, help='weight to balance all losses')


args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


net_factory = net_factory_

def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes,device=dev)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn) if dev=='cuda' else DataLoader(db_train, batch_sampler=batch_sampler, num_workers=1)
    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss()
    consistency_criterion = losses.mse_loss
    dice_loss = losses.DiceLoss(n_classes=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)


    for _ in iterator:
        for pos, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(dev), label_batch.to(dev)

            model.train()
            (output1,z),output2 = model(volume_batch)

            #output1_pk = F.sigmoid(output1)
            #output1 = output1_pk * ((1-output1_pk).prod(dim=1,keepdims=True).repeat(1,args.num_classes,1,1))/(1-output1_pk)
            #output1 = output1

            y_oris = torch.zeros((2,) + output1.shape)
            y_pseudo_labels = torch.zeros((2,) + output1.shape)

            #loss_seg = 0
            loss_seg_dice = 0

            y1 = output1[:labeled_bs, ...]
            #y1_prob = F.softmax(y1, dim=1)
            y1_prob = F.sigmoid(y1)
            #loss_seg += ce_loss(y1, label_batch[:labeled_bs][:].long())
            loss_seg_dice += dice_loss(y1_prob, label_batch[:labeled_bs].unsqueeze(1))

            y_all_0 =  F.sigmoid(output1)
            y_oris[0]= y_all_0
            y_pseudo_labels[0] = sharpening(y_all_0)

            y2 = output2[:labeled_bs, ...]
            y2_prob = F.softmax(y2, dim=1)
            #loss_seg += ce_loss(y2, label_batch[:labeled_bs][:].long())
            loss_seg_dice += dice_loss(y2_prob, label_batch[:labeled_bs].unsqueeze(1))

            y_all_1 = F.softmax(output2, dim=1)
            y_oris[1] = y_all_1
            y_pseudo_labels[1] = sharpening(y_all_1)


            iter_num = iter_num + 1
            gamma =  get_gamma(iter_num // 150)
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss_consist = gamma * consistency_criterion(y_oris[1], y_pseudo_labels[0]) +(1-gamma)* consistency_criterion(y_oris[0], y_pseudo_labels[1])

            loss_cont =  losses.InfoNCE_loss(z)

            pre_hot = 0 if iter_num<=1500 else 1.0

            loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist + pre_hot * args.alpha * loss_cont

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f,loss_cont: %03f' % (
            iter_num, loss, args.lamda * loss_seg_dice , consistency_weight * loss_consist, loss_cont*args.alpha))


            writer.add_scalar('{}_{}_Labeled_loss/loss_seg_dice'.format(args.model, args.labelnum), loss_seg_dice,
                              iter_num)
            writer.add_scalar('{}_{}_Co_loss/consistency_loss'.format(args.model, args.labelnum), loss_consist,
                              iter_num)
            writer.add_scalar('{}_{}_Co_loss/cont_loss'.format(args.model, args.labelnum), loss_cont, iter_num)

            writer.add_scalar('{}_{}_Co_loss/consist_weight'.format(args.model, args.labelnum), consistency_weight,
                              iter_num)

            if iter_num % 10 == 0:
                with torch.no_grad():
                    image = volume_batch[-1, 0:1, :, :]
                    writer.add_image('train/Image', image, iter_num)
                    pred_1 = (torch.sigmoid(output1)>0.50).float()
                    writer.add_image('train/Prediction_1', pred_1[1, ...] * 50, iter_num)
                    pred_2 = torch.argmax(torch.softmax(output2, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction_2', pred_2[1, ...] * 50, iter_num)
                    labs = label_batch[1, ...].unsqueeze(0) * 50
                    writer.add_image('train/GroundTruth', labs, iter_num)


            if iter_num > 0 and iter_num % 150 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"





def train2(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #model = net_factory(net_type=args.model, in_chns=num_classes, class_num=num_classes,device=dev)

    model = net_factory('MT', in_chns=1, class_num=num_classes, device=dev)
    backbone = net_factory('MT', in_chns=1, class_num=num_classes,device=dev).student

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn) if dev == 'cuda' else DataLoader(db_train,
                                                                                             batch_sampler=batch_sampler,
                                                                                             num_workers=1)
    model.train()
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.student.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_ = optim.SGD(backbone.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    #writer = SummaryWriter(snapshot_path + '/log')
    writer = SummaryWriter('/root/tf-logs/{}_in_{}_labeled_{}'.format(args.model,args.dataset_name,args.labelnum))
    
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)


    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(n_classes=num_classes)

    for _ in iterator:
        for pos, sampled_batch in enumerate(trainloader):
            iter_num = iter_num + 1
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(dev), label_batch.to(dev)
            
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            
            noise_volume_batch = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)+ volume_batch
            

            model.train()
            out_s,out_t  = model(noise_volume_batch.to(dev),unlabeled_volume_batch)

            l_seg = (dice_loss(torch.softmax(out_s[:labeled_bs],dim=1),label_batch[:labeled_bs].unsqueeze(1))
                             + ce_loss(out_s[:labeled_bs],label_batch[:labeled_bs].long()))

            pseudo_labels = torch.argmax(torch.softmax(out_t,dim=1), dim=1, keepdim=False).detach()

            l_con = dice_loss(torch.softmax(out_s[labeled_bs:],dim=1),pseudo_labels.unsqueeze(1)+
                              ce_loss(out_s[labeled_bs:],pseudo_labels.long())
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = 1.0 * l_seg + consistency_weight * l_con

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.update_teacher(0.99,iter_num)
            logging.info('model_iter %d : loss : %03f, loss_seg: %03f, l_con: %03f' % (iter_num, loss, l_seg, l_con))

            backbone.train()

            #out_s_, out_t_ = backbone(noise_volume_batch.to(dev), volume_batch)
            out_s_ =  backbone(noise_volume_batch[:labeled_bs].to(dev))

            #l_seg_ = 0.50 * (dice_loss(torch.softmax(out_s_[:labeled_bs], dim=1), label_batch[:labeled_bs].unsqueeze(1))
                            #+ ce_loss(out_s_[:labeled_bs], label_batch[:labeled_bs].long()))

            #pseudo_labels = torch.argmax(torch.softmax(out_t_[labeled_bs:], dim=1), dim=1, keepdim=False).detach()
            #l_con_ = ce_loss(out_s, pseudo_labels.long())

            l_con_ = 0.0
            l_seg_ = (dice_loss(torch.softmax(out_s_, dim=1), label_batch[:labeled_bs].unsqueeze(1))
                            + ce_loss(out_s_, label_batch[:labeled_bs].long()))
            loss_ = 1.0 * l_seg_ + 0.0 * l_con_

            loss_.backward()
            optimizer_.step()
            optimizer_.zero_grad()
            #backbone.update_teacher(0.99, iter_num // 2)
            logging.info('backbone_iter %d : loss : %03f, loss_seg: %03f, l_con: %03f' % (iter_num, loss_, l_seg_, l_con_))



            writer.add_scalar('{}/model_{}_loss_seg_dice'.format(args.model, args.labelnum), l_seg,
                              iter_num)
            writer.add_scalar('{}/model_{}_loss_con'.format(args.model, args.labelnum), l_con,
                              iter_num)
            
            writer.add_scalar('{}/backbone_{}_loss_seg_dice'.format(args.model, args.labelnum), l_seg_,
                              iter_num)
            

            if iter_num % 100 == 0:
                with torch.no_grad():
                    image = volume_batch[0, :, :, :]
                    writer.add_image('train/Image', image, iter_num)
                    pred_seg = torch.argmax(torch.softmax(out_s, dim=1), dim=1, keepdim=True)
                    writer.add_image('train/Prediction', pred_seg[0, ...] * 80, iter_num)
                    labs = label_batch[0, ...].unsqueeze(0) * 80
                    writer.add_image('train/GroundTruth', labs, iter_num)

                    #print(torch.unique(labs))
                    #print(torch.unique(pred_seg))


                    #re_image = x_r[0,:,:,:]
                    #writer.add_image('train/Re_Image', re_image, iter_num)

            if iter_num > 0 and iter_num % 100 == 0:
                for idx, test_model in enumerate([model.student,backbone]):
                    test_model.eval()
                    metric_list = 0.0
                    for _, sampled_batch in enumerate(valloader):
                        metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], test_model,
                                                             classes=num_classes)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db_val)
                    which = 'model' if idx==0 else 'backbone'
                    for class_i in range(num_classes - 1):
                        writer.add_scalar('{}/{}_{}_info/val_{}_dice'.format(args.model,which, args.labelnum,class_i + 1), metric_list[class_i, 0], iter_num)
                        writer.add_scalar('{}/{}_{}_info/val_{}_hd95'.format(args.model,which, args.labelnum,class_i + 1), metric_list[class_i, 1], iter_num)

                    performance = np.mean(metric_list, axis=0)[0]

                    mean_hd95 = np.mean(metric_list, axis=0)[1]
                    writer.add_scalar('{}/{}_{}_info/val_mean_dice'.format(args.model,which, args.labelnum), performance, iter_num)
                    writer.add_scalar('{}/{}_{}_info/val_mean_hd95'.format(args.model,which, args.labelnum), mean_hd95, iter_num)
                    
                    print(which,': val_mean_dice_{}={};val_mean_hd95={}'.format(iter_num,performance,mean_hd95))

                    if idx == 0 and performance > best_performance:
                        print('the last best_performance:{}, now best_performance_{}:{}'.format(best_performance,iter_num,performance))
                        best_performance = performance
                        save_mode_path = os.path.join(snapshot_path,
                                                      'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                        save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                        torch.save(test_model.state_dict(), save_mode_path)
                        torch.save(test_model.state_dict(), save_best_path)

                    if(idx == 0 ):
                        logging.info('iteration_model %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                    else:
                        logging.info(
                            'iteration_backbone %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                    test_model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    snapshot_path = "./model/ACDC_{}_{}_labeled/{}".format(args.exp, args.labelnum, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    #if os.path.exists(snapshot_path + '/code'):
    #    shutil.rmtree(snapshot_path + '/code')
    #shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train2(args, snapshot_path)