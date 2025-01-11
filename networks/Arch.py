import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', hold_depth=False, padding=0):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if not hold_depth:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=padding, stride=stride))
        else:
            h_stride = (stride, stride, 1)
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, h_stride, padding=padding, stride=h_stride))

        if normalization != 'none':
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=(2, 2, 1), normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, hold_depth=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization,
                                                  hold_depth=hold_depth)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization,
                                                  hold_depth=hold_depth)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization,
                                                    hold_depth=hold_depth)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization,
                                                   hold_depth=hold_depth)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0, hold_depth=False,N=1):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        stride = (2, 2, 1) if hold_depth else (2, 2, 2)
        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, stride, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, stride, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, stride, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, stride, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.10, inplace=False)

        self.hold_depth = hold_depth
        self.N =N

    def forward(self, features):
        N =self.N
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x4 = x4.repeat(N,1,1,1,1) if N>1 else x4
        x5_up = x5_up + x4 if x4 is not None else x5_up

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x3 = x3.repeat(N, 1, 1, 1, 1) if N > 1 else x3
        x6_up = x6_up + x3 if x3 is not None else x6_up

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x2 = x2.repeat(N, 1, 1, 1, 1) if N > 1 else x2
        x7_up = x7_up + x2 if x2 is not None else x7_up

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x1 = x1.repeat(N, 1, 1, 1, 1) if N > 1 else x1
        x8_up = x8_up + x1 if x1 is not None else x8_up
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out_seg = self.out_conv(x9)

        return out_seg


# 计算模型参数量
def cal_params(mod):
    sum = 0
    for _, param in mod.named_parameters():
        sum += torch.prod(torch.tensor(param.data.size()))
    return sum.item()



class Embedding_VAE_v2(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.name = 'Embedding_VAE_v2'
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.mean_vec = nn.Linear(in_dims, out_dims)
        self.covar_half = nn.Linear(in_dims, out_dims)
        # self.dropout = nn.Dropout(p=0.10, inplace=False)

    def reparameterize(self, mu, covar_half,N = 1):
        B, depth, D = mu.size()
        if N == 1:
            eps = torch.randn(B * depth, 1, D).cuda()
            eps_ = torch.bmm(eps, covar_half)
            return eps_.view(B, depth, D) + mu  # (b,depth,D)
        else:
            latent = None
            # latent will be (M, B, depth, D)
            for i in range(N):
                eps = torch.randn(B * depth, 1, D).cuda()
                eps_ = torch.bmm(eps, covar_half)
                lat = eps_.view(B, depth, D) + mu
                if i == 0:
                    latent = lat.unsqueeze(0)
                else:
                    latent = torch.cat((latent, lat.unsqueeze(0)), 0)
            return latent.view(B * N, depth, D)


    def forward(self, x_d, scale=20.0,N = 1):
        # mean:
        mean = self.mean_vec(x_d)  # mean:（b_s,depth,D）
        B, depth, D = mean.size()
        # covar:
        covar_half_vec = self.covar_half(x_d)
        covar_half_vec = covar_half_vec.view(-1, D).unsqueeze(2)  # (b_s,D,1)
        I = (torch.eye(D).unsqueeze(0).expand(B * depth, -1, -1) * scale).cuda()  # （b_s,D,D）I
        covar_half = torch.bmm(covar_half_vec, covar_half_vec.transpose(1, 2)) + I  # v=v+I
        covar = torch.bmm(covar_half, covar_half)  # （b_s,D,D）  #covar = (v+I)^2

        Z = self.reparameterize(mean, covar_half,N)
        return Z, mean, covar  #z :(b_s*N,d,D)



class GBDL_v4(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, input_size=(128, 128, 80),N=1, normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(GBDL_v4, self).__init__()
        self.encoder_v = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                                 hold_depth=False)
        self.decoder_v = Decoder(n_channels, n_channels, n_filters, normalization, has_dropout, has_residual, up_type=0,
                                 hold_depth=False,N=N)  # use trans_conv

        self.encoder_d = Encoder(n_channels, n_classes, n_filters, normalization, True, has_residual,
                                 hold_depth=False)
        self.decoder_d = Decoder(n_channels, n_classes, n_filters, normalization, True, has_residual, up_type=0,
                                 hold_depth=False,N=N)  # use trans_conv

        D_count_in = 16 * n_filters * (input_size[0] // 32) * (input_size[1] // 32)
        D_count_out = 32 * n_filters

        self.compress = nn.Linear(D_count_in, D_count_out)

        self.down = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0)
        self.up = nn.Upsample(scale_factor=(2, 2, 1), mode="trilinear", align_corners=True)

        self.VAE = Embedding_VAE_v2(D_count_out, D_count_out)
        self.recover = nn.Linear(D_count_out, D_count_in)

        self.merge = nn.Sequential(
            nn.Conv3d(2 * 16 * n_filters, 1 * 16 * n_filters, 1, padding=0),
            nn.BatchNorm3d(1 * 16 * n_filters),
            nn.ReLU(inplace=True)
        )
        self.N = N
        self.th = nn.Tanh()

    def change_N(self,N):
        self.N = N
        self.decoder_v.N=N
        self.decoder_d.N=N

    def forward(self, input):
        N=self.N
        features_v = self.encoder_v(input)
        fz = features_v[-1]  # r=1/16 d=1/16 c=256=16*n_filters [b_s,c,r,r,d]
        fz = self.down(fz)
        fz_size = fz.size()
        fz = torch.permute(fz, [0, 4, 1, 2, 3])
        fz_d = torch.flatten(fz, start_dim=2)  # (b_s,D,D_count_in)
        fz_d = self.compress(fz_d)  # (b_s,D,D_count_out)
        Z, mean, covar = self.VAE(fz_d,N=N)  # Z:(N*b_s,D,D_count_out)
        _, depth, D = mean.size()

        Z_f = self.recover(Z).view(N * fz_size[0], fz_size[4], fz_size[1], fz_size[2],
                                   fz_size[3])  # r=1/32 c=256=16*n_filters [N*b_s,d,c,r,r]
        Z_f = torch.permute(Z_f, [0, 2, 3, 4, 1])   #[N*b_s,c,r,r,d]

        features_d = self.encoder_d(input)
        f_d = features_d[-1].repeat(N,1,1,1,1) #[N*b_s,c,r,r,d]

        # merge Z_d and Z_f
        Z_f = self.up(Z_f)
        f_d = self.merge(torch.cat([f_d, Z_f], dim=1))
        #f_d = f_d +Z_f*f_d

        x_r = self.decoder_v([features_v[0], features_v[1], features_v[2], features_v[3], Z_f])  # VAE_stream: out
        out = self.decoder_d([features_d[0], features_d[1], features_d[2], features_d[3], f_d])  # Vent_stream: out

        return out, mean, covar.view(-1, depth, D, D), self.th(x_r), Z

    
    
    
    
    
    
    
    
    
class GBDL_v5(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, input_size=(128, 128, 80),N=1, normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super(GBDL_v5, self).__init__()
        self.encoder_v = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                                 hold_depth=False)
        self.decoder_v = Decoder(n_channels, n_channels, n_filters, normalization, has_dropout, has_residual, up_type=0,
                                 hold_depth=False,N=N)  # use trans_conv

        self.encoder_d = Encoder(n_channels, n_classes, n_filters, normalization, True, has_residual,
                                 hold_depth=False)
        self.decoder_d = Decoder(n_channels, n_classes, n_filters, normalization, True, has_residual, up_type=0,
                                 hold_depth=False,N=N)  # use trans_conv

        D_count_in = 16 * n_filters * (input_size[0] // 32) * (input_size[1] // 32)
        D_count_out = 32 * n_filters

        self.compress = nn.Linear(D_count_in, D_count_out)

        self.down = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0)
        self.up = nn.Upsample(scale_factor=(2, 2, 1), mode="trilinear", align_corners=True)

        self.VAE = Embedding_VAE_v2(D_count_out, D_count_out)
        self.recover = nn.Linear(D_count_out, D_count_in)

        self.merge = nn.Sequential(
            nn.Conv3d(2 * 16 * n_filters, 1 * 16 * n_filters, 1, padding=0),
            nn.BatchNorm3d(1 * 16 * n_filters),
            nn.ReLU(inplace=True)
        )
        self.N = N
        self.th = nn.Tanh()

    def change_N(self,N):
        self.N = N
        self.decoder_v.N=N
        self.decoder_d.N=N

    def forward(self, input):
        N=self.N
        features_v = self.encoder_v(input)
        fz = features_v[-1]  # r=1/16 d=1/16 c=256=16*n_filters [b_s,c,r,r,d]
        fz = self.down(fz)
        fz_size = fz.size()
        fz = torch.permute(fz, [0, 4, 1, 2, 3])
        fz_d = torch.flatten(fz, start_dim=2)  # (b_s,D,D_count_in)
        fz_d = self.compress(fz_d)  # (b_s,D,D_count_out)
        Z, mean, covar = self.VAE(fz_d,N=N)  # Z:(N*b_s,D,D_count_out)
        _, depth, D = mean.size()

        Z_f = self.recover(Z).view(N * fz_size[0], fz_size[4], fz_size[1], fz_size[2],
                                   fz_size[3])  # r=1/32 c=256=16*n_filters [N*b_s,d,c,r,r]
        Z_f = torch.permute(Z_f, [0, 2, 3, 4, 1])   #[N*b_s,c,r,r,d]

        features_d = self.encoder_d(input)
        f_d = features_d[-1].repeat(N,1,1,1,1) #[N*b_s,c,r,r,d]

        # merge Z_d and Z_f
        Z_f = self.up(Z_f)
        f_d = self.merge(torch.cat([f_d, Z_f], dim=1))
        #f_d = f_d +Z_f*f_d

        x_r = self.decoder_v([features_v[0], features_v[1], features_v[2], features_v[3], Z_f])  # VAE_stream: out
        out = self.decoder_d([features_d[0], features_d[1], features_d[2], features_d[3], f_d])  # Vent_stream: out

        return out, mean, covar.view(-1, depth, D, D), self.th(x_r), Z
    
    
    

class My_Archs(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, input_size=(128, 128, 80), normalization='none',
                 has_dropout=False,
                 has_residual=False,N=1):
        super(My_Archs, self).__init__()
        self.gbdl = GBDL_v4(n_channels, n_classes, n_filters, input_size, N,normalization, False, has_residual)

        self.v_net = nn.Sequential(
            Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                    hold_depth=False),
            Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0,
                    hold_depth=False)  # use trans_conv
        )

        self.test_out_ = False

    def test_out(self):
        self.test_out_ = True

    def forward(self, input):
        out_seg1, mean, covar, x_r, Z = self.gbdl(input)
        property = (mean, covar, x_r, Z)
        out_seg2 = self.v_net(input)
        out_put = (out_seg2, out_seg1)

        if not self.test_out_:
            return out_put, property
        else:
            return out_put

 
    
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.relu = act
        self.conv1 = nn.Conv3d(in_channels, middle_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in1 = nn.BatchNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))
        self.in2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, input_channels=3,num_classes=2, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 96, 192, 384]

        self.pool = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.up = nn.Upsample(scale_factor=(2.0, 2.0, 2.0), mode='trilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])

        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[0], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[2], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[3], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, 1, padding=0)

    def forward(self, input, train=True, T=1):

        x0_0 = self.conv0_0(input)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        output_final = None

        for i in range(0, T):

            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

            output = self.final(x0_4)

            if T <= 1:
                output_final = output
            else:
                if i == 0:
                    output_final = output.unsqueeze(0)
                else:
                    output_final = torch.cat((output_final, output.unsqueeze(0)), dim=0)

        return output_final
    
class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=False, has_residual=False,up_type=0):
        super(VNet, self).__init__()

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type)
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        return out_seg1
    
class Strong_CPS(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(Strong_CPS, self).__init__()
        self.sub_net_1 =VNet(n_channels,n_classes,up_type=0,has_dropout=True)
        self.sub_net_2 =UNet(n_channels,n_classes)
        self.name = 'Strong_CPS'

    def forward(self, input):
        out_seg1 = self.sub_net_1(input)
        out_seg2 = self.sub_net_2(input)
        return out_seg1, out_seg2


class Robust_CPS(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(Robust_CPS, self).__init__()
        self.sub_net_1 = VNet(n_channels,n_classes,up_type=0,has_dropout=True)
        self.sub_net_2 = VNet(n_channels,n_classes,up_type=1,has_dropout=True)
        self.name = 'Robust_CPS'

    def forward(self, input):
        out_seg1 = self.sub_net_1(input)
        out_seg2 = self.sub_net_2(input)
        return out_seg1, out_seg2
    
    
class Weakly_CPS(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(Weakly_CPS, self).__init__()
        self.encoder = Encoder(n_channels, n_classes,normalization='batchnorm')
        self.decoder1 = Decoder(n_channels, n_classes,up_type=0,normalization='batchnorm',has_dropout=True)
        self.decoder2 = Decoder(n_channels, n_classes,up_type=1,normalization='batchnorm',has_dropout=True)
        
        self.sub_net_1= nn.Sequential(
            self.encoder,
            self.decoder1
        )

        self.sub_net_2= nn.Sequential(
            self.encoder,
            self.decoder2
        )

        self.name = 'Weakly_CPS'
    
    def forward(self, input):
        features = self.encoder(input)
        out_seg1 = self.decoder1(features)
        out_seg2 = self.decoder2(features)
        return out_seg1, out_seg2
    


class Frail_CPS(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(Frail_CPS, self).__init__()
        self.sub_net_1 = VNet(n_channels,n_classes,up_type=0)
        self.sub_net_2 = VNet(n_channels,n_classes,up_type=0)
        self.name = 'Frail_CPS'
        for param in self.sub_net_2.parameters():
            param.detach_()

    def forward(self, input):
        out_seg1 = self.sub_net_1(input)
        with torch.no_grad():
            out_seg2 = self.sub_net_2(input)
        return out_seg1, out_seg2

    def update_params(self,global_step,ema_weight=0.99):
        with torch.no_grad():
            alpha = min(1 - 1 / (global_step + 1), ema_weight)
            alpha = ema_weight
            for ema_param, param in zip(self.sub_net_2.parameters(), self.sub_net_1.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data)        
        

        
class GBDL_v4_2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, input_size=(128, 128, 80),N=1, normalization='none',
                 has_dropout=False,
                 has_residual=False):
        super().__init__()
        self.encoder_v = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                                 hold_depth=False)
        self.decoder_v = Decoder(n_channels, n_channels, n_filters, normalization, has_dropout, has_residual, up_type=0,
                                 hold_depth=False,N=N)  # use trans_conv

        self.encoder_d = Encoder(n_channels, n_classes, n_filters, normalization, True, has_residual,
                                 hold_depth=False)
        self.decoder_d = Decoder(n_channels, n_classes, n_filters, normalization, True, has_residual, up_type=0,
                                 hold_depth=False,N=N)  # use trans_conv

        D_count_in = 16 * n_filters * (input_size[0] // 32) * (input_size[1] // 32)
        D_count_out = 32 * n_filters

        self.compress = nn.Linear(D_count_in, D_count_out)

        self.down = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0)
        self.up = nn.Upsample(scale_factor=(2, 2, 1), mode="trilinear", align_corners=True)

        self.VAE = Embedding_VAE_v2(D_count_out, D_count_out)
        self.recover = nn.Linear(D_count_out, D_count_in)

        self.merge = nn.Sequential(
            nn.Conv3d(2 * 16 * n_filters, 1 * 16 * n_filters, 1, padding=0),
            nn.BatchNorm3d(1 * 16 * n_filters),
            nn.ReLU(inplace=True)
        )
        self.N = N
        self.th = nn.Tanh()

    def change_N(self,N):
        self.N = N
        self.decoder_v.N=N
        self.decoder_d.N=N

    def forward(self, input):
        N=self.N
        features_v = self.encoder_v(input)
        fz = features_v[-1]  # r=1/16 d=1/16 c=256=16*n_filters [b_s,c,r,r,d]
        fz = self.down(fz)
        fz_size = fz.size()
        fz = torch.permute(fz, [0, 4, 1, 2, 3])
        fz_d = torch.flatten(fz, start_dim=2)  # (b_s,D,D_count_in)
        fz_d = self.compress(fz_d)  # (b_s,D,D_count_out)
        Z, mean, covar = self.VAE(fz_d,N=N)  # Z:(N*b_s,D,D_count_out)
        _, depth, D = mean.size()

        Z_f = self.recover(Z).view(N * fz_size[0], fz_size[4], fz_size[1], fz_size[2],
                                   fz_size[3])  # r=1/32 c=256=16*n_filters [N*b_s,d,c,r,r]
        Z_f = torch.permute(Z_f, [0, 2, 3, 4, 1])   #[N*b_s,c,r,r,d]

        features_d = self.encoder_d(input)
        f_d = features_d[-1].repeat(N,1,1,1,1) #[N*b_s,c,r,r,d]

        # merge Z_d and Z_f
        Z_f = self.up(Z_f)
        f_d = self.merge(torch.cat([f_d, Z_f], dim=1))
        #f_d = f_d +Z_f*f_d

        x_r = self.decoder_v([features_v[0], features_v[1], features_v[2], features_v[3], Z_f])  # VAE_stream: out
        out = self.decoder_d([features_d[0], features_d[1], features_d[2], features_d[3], f_d])  # Vent_stream: out

        return out, mean, covar.view(-1, depth, D, D), self.th(x_r), Z,features_d   
    
    
        
class My_Archs_v2(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, input_size=(128, 128, 80), normalization='none',
                 has_dropout=False,
                 has_residual=False,N=1):
        super(My_Archs_v2, self).__init__()
        self.gbdl = GBDL_v4(n_channels, n_classes, n_filters, input_size, N,normalization, False, has_residual)

        self.v_net = nn.Sequential(
            Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual,
                    hold_depth=False),
            Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0,
                    hold_depth=False)  # use trans_conv
        )

        self.test_out_ = False

    def test_out(self):
        self.test_out_ = True

    def forward(self, input):
        out_seg1, mean, covar, x_r, Z = self.gbdl(input)
        property = (mean, covar, x_r, Z)
        out_seg2 = self.v_net(input)
        out_put = (out_seg2, out_seg1)

        if not self.test_out_:
            return out_put, property
        else:
            return out_put
        
        
if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    mod1 = My_Archs(input_size=(80, 128, 128), normalization='batchnorm', has_dropout=False, has_residual=False,N=1)
    print(cal_params(mod1.v_net))
    
