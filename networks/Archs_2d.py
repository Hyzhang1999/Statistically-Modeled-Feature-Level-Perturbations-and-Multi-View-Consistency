# -*- coding: utf-8 -*-
"""
The implementation is borrowed from: https://github.com/HiLab-git/PyMIC
"""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# 计算模型参数量
def cal_params(mod):
    sum = 0
    for _, param in mod.named_parameters():
        sum += torch.prod(torch.tensor(param.data.size()))
    return sum.item()


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p,group=1):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,groups=group),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,groups=group),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Group_UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, n_classes,dropout_p, mode_upsampling=1):
        super(Group_UpBlock, self).__init__()
        self.k = n_classes
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2,groups=n_classes)
        elif mode_upsampling == 1:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1,groups=n_classes)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif mode_upsampling == 2:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1,groups=n_classes)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        elif mode_upsampling == 3:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1,groups=n_classes)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p,group=n_classes)

    def forward(self, x1,x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        b_s,ck,h,w = x1.size()
        x1 = x1.view(b_s*self.k,ck//self.k,h,w)
        x2 = x2.repeat(self.k,1,1,1)
        x = torch.cat([x2, x1], dim=0)
        x = x.view(b_s,2*ck,h,w)
        x = self.conv(x)
        return x



class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]

class Decoder(nn.Module):
    def __init__(self, params,include_classifer=True):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0,
                           mode_upsampling=self.up_type)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0,
                           mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1) if include_classifer else nn.Sequential()

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class UNet(nn.Module):
    def __init__(self, in_chns, class_num,include_classifer=True):
        super(UNet, self).__init__()

        params1 = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}

        self.encoder = Encoder(params1)
        self.decoder1 = Decoder(params1,include_classifer)

    def forward(self, x):
        feature = self.encoder(x)
        output1 = self.decoder1(feature)
        return output1



class Cont_Decoder(nn.Module):
    def __init__(self, params):
        super(Cont_Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.up_type = self.params['up_type']
        self.k =self.n_class
        assert (len(self.ft_chns) == 5)

        self.up1 = Group_UpBlock(self.ft_chns[4]*self.k, self.ft_chns[3]*self.k, self.ft_chns[3]*self.k, dropout_p=0.0,
                           n_classes=self.n_class,mode_upsampling=self.up_type)
        self.up2 = Group_UpBlock(self.ft_chns[3]*self.k, self.ft_chns[2]*self.k, self.ft_chns[2]*self.k, dropout_p=0.0,
                           n_classes=self.n_class,mode_upsampling=self.up_type)
        self.up3 = Group_UpBlock(self.ft_chns[2]*self.k, self.ft_chns[1]*self.k, self.ft_chns[1]*self.k, dropout_p=0.0,
                           n_classes=self.n_class,mode_upsampling=self.up_type)
        self.up4 = Group_UpBlock(self.ft_chns[1]*self.k, self.ft_chns[0]*self.k, self.ft_chns[0]*self.k, dropout_p=0.0,
                           n_classes=self.n_class,mode_upsampling=self.up_type)

        self.out_conv = nn.Conv2d(self.ft_chns[0]*self.k, self.k, kernel_size=3, padding=1,groups=self.k)

        self.intra_chns= 64
        self.project = nn.Sequential(
            nn.Conv2d(self.ft_chns[4],self.intra_chns,kernel_size=1,padding=0),
            nn.BatchNorm2d(self.intra_chns),
            nn.LeakyReLU(),
            nn.Conv2d(self.intra_chns,self.intra_chns,kernel_size=1,padding=0)
        )

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x4 = x4.repeat((1,self.k,1,1))
        b_s,ck,h,w = x4.size()
        z = self.project(x4.view(b_s*self.k,ck//self.k,h,w))
        z = z.view(b_s,self.k,-1) #(b_s,k,D)


        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output,z



class My_Arch(nn.Module):
    def __init__(self,in_chns,class_num):
        super(My_Arch, self).__init__()
        params = {'in_chns': in_chns,
                   'feature_chns': [16, 32, 64, 128, 256],
                   'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                   'class_num': class_num,
                   'up_type': 1,
                   'acti_func': 'relu'}

        self.Shared_Encoder = Encoder(params)
        self.decoder1 = Cont_Decoder(params)
        self.decoder2 = Decoder(params)

    def forward(self, x):
        feature = self.Shared_Encoder(x)
        output1,z = self.decoder1(feature)
        output2 = self.decoder2(feature)
        return (output1,z),output2 if self.training else (output1,output2)



class CVAE(nn.Module):
    def __init__(self,in_chns,class_num,h,w):
        super(CVAE, self).__init__()
        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'up_type': 1,
                  'acti_func': 'relu'}
        self.class_num = class_num
        self.encoder_1 = Encoder(params)

        params_ = {'in_chns': 1,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': 1,
                  'up_type': 1,
                  'acti_func': 'relu'}

        self.encoder_2 = Encoder(params_)

        in_dims= (params['feature_chns'][-1]*(h*w))//256
        self.mid_dims = 384
        out_dims = in_dims

        self.compress = nn.Linear(in_dims,self.mid_dims)
        self.mean_vec = nn.Linear(self.mid_dims, self.mid_dims)
        self.logvar_vec = nn.Linear(self.mid_dims, self.mid_dims)
        self.recover = nn.Linear(self.mid_dims,out_dims)

        self.decoder = Decoder(params)

        self.trans_k = nn.Conv2d(params['feature_chns'][-1],params['feature_chns'][-1],1,1)
        self.trans_ =  nn.Sequential(
            nn.Conv2d(2*params['feature_chns'][-1],params['feature_chns'][-1],1,1),
            nn.BatchNorm2d(params['feature_chns'][-1]),
            nn.LeakyReLU()
        )


    def reparameterize(self, mu=None, logvar=None):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return torch.randn_like(mu)


    def llm(self,f):
        z = torch.flatten(f,start_dim=1)
        z = self.compress(z)
        mean = self.mean_vec(z)
        logvar = self.logvar_vec(z)

        return self.reparameterize(mean,logvar),mean,logvar


    def forward(self, x, label=None):

        mu,logvar =None,None
        xr =None
        x_feature = self.encoder_2(x)
        x_f_final = x_feature[-1]

        if self.training:
            y = F.one_hot(label, num_classes=self.class_num).permute(0, 3, 1,2).float()
            xy = x*y

            xy_feature = self.encoder_1(xy)
            xy_f_final = xy_feature[-1]

            z,mu,logvar = self.llm(xy_f_final) #D=256

        else:
            z = torch.randn(x_f_final.size(0),self.mid_dims) #D=256

        z = self.recover(z) #ori_depth
        z = z.view(x_f_final.size())
        zy = self.trans_(torch.cat([z,self.trans_k(x_f_final)],dim=1))

        #q_y = self.trans_k(x_f_final).flatten(1)

        #cosine_sim = F.cosine_similarity(z, q_y, dim=1)
        #attention_weights = F.softmax(cosine_sim, dim=0)
        #zy = attention_weights.unsqueeze(1) * z

        #zy = zy.view(x_f_final.size())
        # zy = self.trans_(zy)

        pred_y = self.decoder([x_feature[0],x_feature[1],x_feature[2],x_feature[3],zy])

        return pred_y,mu,logvar if self.training else pred_y




class MT(nn.Module):
    def __init__(self, in_chns, class_num,include_classifer=True):
        super(MT, self).__init__()
        self.student = UNet(in_chns,class_num,include_classifer)
        self.teacher = UNet(in_chns,class_num,include_classifer)
        self.name = 'MT'
        self.is_include_classifer = include_classifer
        for param in self.teacher.parameters():
            param.detach_()

    def forward(self, x_s,x_t):
        out_s = self.student(x_s)
        with torch.no_grad():
            out_t = self.teacher(x_t)

        return out_s,out_t

    def update_teacher(self,alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)




if __name__ == '__main__':
    x = torch.randn((8,1,256,256))
    noise = torch.clamp(torch.randn_like(x) * 0.1, -0.2, 0.2)

    x_ =x +noise

    model = MT(1,4,False).train()

    print(cal_params(model))
    out_s,out_t = model(x,x_)
    print(out_s.size())

