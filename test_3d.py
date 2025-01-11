import os
import argparse
import sys
from tqdm import tqdm
import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
import h5py
sys.path.append(os.path.join(os.path.abspath('.'), 'networks'))
from networks.net_factory import net_factory
from utils.test_patch import test_all_case
import numpy



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='../', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='Mine', help='exp_name')
parser.add_argument('--model', type=str,  default='mine_v4_v4', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=16, help='labeled data')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--load_dir',type=str,default='../model',help='test dir')


FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


snapshot_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
test_save_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}_best_predictions/".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (128, 128, 80)
    FLAGS.root_path = '../data/LA'
    with open('../data/LA/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

elif FLAGS.dataset_name == "Pancreas":
    patch_size = (96, 96, 96)
    FLAGS.root_path = '../data/Pancreas'
    with open('../data/Pancreas/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
#print('test_save_path: ',test_save_path)




def test_calculate_metric(iter=0,_test_save_path='../infer_res/',load_path=None):
    
    if(iter == 0):
        _test_save_path = test_save_path
    else:
        print('iter=',iter)
        _test_save_path = _test_save_path+'{}_{}_{}_{}_predictions/'.format(FLAGS.dataset_name,FLAGS.exp,FLAGS.labelnum,iter)
        
        
    if not os.path.exists(_test_save_path):
        os.makedirs(_test_save_path)
    print('test_save_path: ',_test_save_path)
    
    #net = net_factory(net_type='vnet', in_chns=1, class_num=num_classes, mode="test")
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes,input_size =patch_size,mode="test",
                        device='cuda')
    
    if iter == 0:
        load_dir =os.path.join(FLAGS.load_dir,'{}_Mine_{}_labeled'.format(FLAGS.dataset_name,FLAGS.labelnum)\
                               ,'{}'.format(FLAGS.model),'{}_best_model_in_{}.pth'.format(FLAGS.model,FLAGS.dataset_name))   #_best_model
    else:
        load_dir =os.path.join(FLAGS.load_dir,'{}_Mine_{}_labeled'.format(FLAGS.dataset_name,FLAGS.labelnum)\
                               ,'{}'.format(FLAGS.model),'iter_{}_in_{}.pth'.format(iter,FLAGS.dataset_name))   #_best_model
    
    load_dir = '../model/LA_ul_0_l16_2/mine_v4_train_all_womc/mine_v4_train_all_womc_best_model_in_LA.pth'
    #save_mode_path = os.path.join(FLAGS.load_dir, '{}_best_model.pth'.format(FLAGS.model))
    
    if load_path is not None:
        load_dir = load_path
    
    print("init weight from {}".format(load_dir))
    net.load_state_dict(torch.load(load_dir), strict=False)
    
    net.eval()
    if('CPS' not in FLAGS.model):
        net.test_out()
    avg_metric=None
    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(128, 128, 80), stride_xy=18, stride_z=4,
                        save_result=False, test_save_path=_test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                        save_result=True, test_save_path=_test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


    

            

            
            

    
    
    
    

if __name__ == '__main__':
    #metric = test_calculate_metric()
    #print('mestric: ',metric)
    #test_figures('Strong_CPS','../model/Pancreas_Mine_12_labeled/CPSs/Strong_CPSiter_10000_in_Pancreas.pth')
    
    #models = ['mine_v4']
    #for name in models:
     #   FLAGS.model = name
        
      #  if 'CPS' in name:
       #     load_path = os.path.join(FLAGS.load_dir,'{}_Mine_{}_labeled'.format(FLAGS.dataset_name,FLAGS.labelnum)\
                              #     ,'{}'.format('CPSs'),'{}_CPSiter_10000_in_{}.pth'.format(name.split('_')[0],FLAGS.dataset_name))
        #else:
         #   load_path = os.path.join(FLAGS.load_dir,'{}_Mine_{}_labeled'.format(FLAGS.dataset_name,FLAGS.labelnum)\
                             #      ,'{}'.format('mine_v4_v3'),'mine_v4_v3_best_model_in_Pancreas.pth')
        #test_save_path = '../infer_res/{}_prediction/'.format(name)
        #test_calculate_metric(_test_save_path = test_save_path,load_path =load_path)

    test_calculate_metric(iter=1000)