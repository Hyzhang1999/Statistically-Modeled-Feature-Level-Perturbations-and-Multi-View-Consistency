import torch
#from Arch import My_Archs
#from Arch import Strong_CPS
#from Arch import Robust_CPS
#from Arch import Weakly_CPS
#from Arch import Frail_CPS


from .Archs_2d import  MT

def net_factory_(net_type="unet", in_chns=1, class_num=4, mode = "train",device='cuda'):
    net=None
    if net_type == 'MT':
        net = MT(in_chns,class_num)

    if(net is not None):
        net.to(torch.device(device))
    net  = net.train() if mode == 'train' else net.eval()
    return net




def net_factory(net_type="unet", in_chns=1, class_num=2,input_size =(128,128,80),N=1,mode = "train",device='cuda'):
    net=None
    if 'mine_v3' in net_type and mode == "train":
        print(1)
        net = My_Archs(n_channels=in_chns, n_classes=class_num, input_size=input_size,N=N,normalization='batchnorm', has_dropout=True).cuda()
    elif 'mine_v3' in net_type and mode == "test":
        print(1)
        net = My_Archs(n_channels=in_chns, n_classes=class_num, input_size=input_size,N=1,normalization='batchnorm', has_dropout=False).cuda()
        net.test_out()
    elif 'Strong_CPS' in net_type:
        net = Strong_CPS(n_channels=in_chns, n_classes=class_num).cuda()
    elif 'Robust_CPS' in net_type:
        net = Robust_CPS(n_channels=in_chns, n_classes=class_num).cuda()
    elif 'Weakly_CPS' in net_type:
        net = Weakly_CPS(n_channels=in_chns, n_classes=class_num).cuda()
    elif 'Frail_CPS' in net_type:
        net = Frail_CPS(n_channels=in_chns, n_classes=class_num).cuda()
    elif 'mine_v4' in net_type and mode == "train":
        print(2)
        net = My_Archs(n_channels=in_chns, n_classes=class_num, input_size=input_size,N=N,normalization='batchnorm', has_dropout=True).cuda()
    elif 'mine_v4' in net_type and mode == "test":
        net = My_Archs(n_channels=in_chns, n_classes=class_num, input_size=input_size,N=1,normalization='batchnorm', has_dropout=False).cuda()
        net.test_out()
        
    return net


#change merge statege ; w/0 dropout ;scale = 20.0; loss:2:1