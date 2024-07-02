import torch
import torch.nn.functional as F
from torch.optim import Adam
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from torchvision import models
from COA_SOD.Shunted_Transformer_master.SSA import *
from COA_SOD.Backbone.p2t import *
from thop import profile
from COA_SOD.al.models.mix_transformer import *
from COA_SOD.al.models.Thrid_models.Dynamic_Cokener_Segformer import *
# from COA_SOD.al.models.Second_model.module_L import Decoder, block, block4


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)

        return out


class CoNet(nn.Module):

    def __init__(self):
        super(CoNet, self).__init__()
        self.conv_rgb = mit_b4().eval()
        self.conv_depth = mit_b4()
        self.channels = [64, 128, 320, 512]

        self.mpg1 = MPG(self.channels[0])
        self.mpg2 = MPG(self.channels[1])
        self.mpg3 = MPG(self.channels[2])
        self.mpg4 = MPG(self.channels[3])
        self.co_sal4 = Cosal_Module(self.channels[3])

        self.SISP = Decoder1(self.channels)

        self.conv4 = DepthWiseConv(self.channels[3] * 2, self.channels[3])

        self.conv4_3 = DepthWiseConv(self.channels[3], self.channels[2])
        self.conv3_2 = DepthWiseConv(self.channels[2], self.channels[1])
        self.conv2_1 = DepthWiseConv(self.channels[1], self.channels[0])
        self.conv64 = DepthWiseConv(self.channels[0], 1)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, rgb, depth):

        B = depth.shape[0]
        # Segformer
        with torch.no_grad():
            e1, e2, e3, e4 = self.conv_rgb(rgb)

        S = self.SISP([e1, e2, e3, e4])

        # Encoder 1
        # with torch.no_grad():
        #     e1_rgb, H, W = self.conv_rgb.patch_embed1(rgb)
        #     for i, blk in enumerate(self.conv_rgb.block1):
        #         e1_rgb = blk(e1_rgb, H, W)
        #     e1_rgb = self.conv_rgb.norm1(e1_rgb)
        #     e1_rgb = e1_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        e1_depth, H, W = self.conv_depth.patch_embed1(depth)
        e1_depth = self.mpg1(e1.flatten(2).transpose(1, 2), e1_depth)

        for i, blk in enumerate(self.conv_depth.block1):
            e1_depth = blk(e1_depth, H, W)
        e1_depth = self.conv_depth.norm1(e1_depth)
        e1_depth = e1_depth.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e1_rd = e1 + e1_depth

        # Encoder 2
        # with torch.no_grad():
        #     e2_rgb, H, W = self.conv_rgb.patch_embed2(e1_rgb)
        #     for i, blk in enumerate(self.conv_rgb.block2):
        #         e2_rgb = blk(e2_rgb, H, W)
        #     e2_rgb = self.conv_rgb.norm2(e2_rgb)
        #     e2_rgb = e2_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        e2_depth, H, W = self.conv_depth.patch_embed2(e1_rd)
        e2_depth = self.mpg2(e2.flatten(2).transpose(1, 2), e2_depth)

        for i, blk in enumerate(self.conv_depth.block2):
            e2_depth = blk(e2_depth, H, W)
        e2_depth = self.conv_depth.norm2(e2_depth)
        e2_depth = e2_depth.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e2_rd = e2 + e2_depth

        # Encoder 3
        # with torch.no_grad():
        #     e3_rgb, H, W = self.conv_rgb.patch_embed3(e2_rgb)
        #     for i, blk in enumerate(self.conv_rgb.block3):
        #         e3_rgb = blk(e3_rgb, H, W)
        #     e3_rgb = self.conv_rgb.norm3(e3_rgb)
        #     e3_rgb = e3_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        e3_depth, H, W = self.conv_depth.patch_embed3(e2_rd)
        e3_depth = self.mpg3(e3.flatten(2).transpose(1, 2), e3_depth)

        for i, blk in enumerate(self.conv_depth.block3):
            e3_depth = blk(e3_depth, H, W)
        e3_depth = self.conv_depth.norm3(e3_depth)
        e3_depth = e3_depth.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e3_rd = e3 + e3_depth
        # Encoder 4
        # with torch.no_grad():
        #     e4_rgb, H, W = self.conv_rgb.patch_embed4(e3_rgb)
        #     for i, blk in enumerate(self.conv_rgb.block4):
        #         e4_rgb = blk(e4_rgb, H, W)
        #     e4_rgb = self.conv_rgb.norm4(e4_rgb)
        #     e4_rgb = e4_rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


        e4_depth, H, W = self.conv_depth.patch_embed4(e3_rd)
        e4_depth = self.mpg4(e4.flatten(2).transpose(1, 2), e4_depth)

        for i, blk in enumerate(self.conv_depth.block4):
            e4_depth = blk(e4_depth, H, W)
        e4_depth = self.conv_depth.norm4(e4_depth)
        e4_depth = e4_depth.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        e4_rd = e4 + e4_depth
        # Intra-group consistency Inter-group difference

        feat4, feat_f4, feat_b4 = self.co_sal4(e4_rd, S)
        # feat3, feat_f3, feat_b3 = self.co_sal3(frd3, S)
        # feat2, feat_f2, feat_b2 = self.co_sal2(frd2, S)
        # feat1, feat_f1, feat_b1 = self.co_sal1(frd1, S)

        e4 = self.conv4(torch.cat([feat4, e4_rd], dim=1))
        e3 = e3_rd + self.up2(self.conv4_3(e4))
        e2 = e2_rd + self.up2(self.conv3_2(e3))
        e1 = e1_rd + self.up2(self.conv2_1(e2))
        S1 = self.up4(self.conv64(e1))


        return S1, S, feat4, feat_f4, feat_b4


    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.conv_rgb.state_dict()
        state_dict_r = {k:v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.conv_rgb.load_state_dict(model_dict_r)
        print(f"RGB Loading pre_model ${pre_model}")

        save_model = torch.load(pre_model)
        model_dict_d = self.conv_depth.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_d.keys()}
        model_dict_d.update(state_dict_d)
        self.conv_depth.load_state_dict(model_dict_d)
        print(f"Depth Loading pre_model ${pre_model}")

if __name__ == "__main__":
    model = CoNet()

    # def print_network(model, name):
    #     num_params = 0
    #     for p in model.parameters():
    #         num_params += p.numel()
    #     print(name)
    #     print("The number of parameters:{}M".format(num_params / 1e6))

    model.train()
    depth = torch.randn(5, 3, 256, 256)
    input = torch.randn(5, 3, 256, 256)
    # model.load_pre('/home/map/Alchemist/COA/COA_RGBD_SOD/al/Pretrain/segformer.b4.512x512.ade.160k.pth')
    flops, params = profile(model, inputs=(input, depth))
    print("the number of Flops {} G ".format(flops / 1e9))
    print("the number of Parameter {}M ".format(params / 1e6)) #1048576

    # print_network(model, 'ccc')

    out = model(input, depth)
    for i in range(len(out)):
        print(out[i].shape)