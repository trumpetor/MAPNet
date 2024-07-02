import torch
import torch.nn.functional as F
from torch import nn
from COA_SOD.al.models.Thrid_models import pmath as pmath
# from COA_RGBD_SOD.al.models.Third_model.kmeans_pytorch import kmeans


class Decoder1(nn.Module):
    def __init__(self, in_channels):
        super(Decoder1, self).__init__()

        # lat_layers = []
        # for idx in range(4):
        #     lat_layers.append(DepthWiseConv(in_channels[idx] * 2, in_channels[idx]))
        # self.lat_layers = nn.ModuleList(lat_layers)

        down_layers = []
        for idx in range(3):
            down_layers.append(DepthWiseConv(in_channels[idx + 1], in_channels[idx]))
        self.down_layers = nn.ModuleList(down_layers)


        # out_layer = []
        # for idx in range(4):
        #     out_layer.append(nn.Sequential(
        #         DepthWiseConv(in_channels[idx], 1),
        #         nn.UpsamplingBilinear2d(scale_factor=pow(2, idx + 2))
        #
        #     ))
        self.out_layer = nn.Sequential(
                DepthWiseConv(in_channels[0], 1),
                nn.UpsamplingBilinear2d(scale_factor=4)
        )



    def forward(self, feat_list):

        feat_top = feat_list[-1]
        p = feat_top


        for idx in [2, 1, 0]:
            p = self._upsample_add(self.down_layers[idx](p), feat_list[idx])


        pre = self.out_layer(p)



        return pre

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear') + y

class MPG(nn.Module):
    def __init__(self, dim):
        super(MPG, self).__init__()
        self.rgb = nn.Linear(dim, dim // 2)
        self.depth = nn.Linear(dim, dim // 2)
        self.up = nn.Linear(dim, dim * 2)
        self.poin = ToPoincare()
        self.distance = HyperbolicDistanceLayer()
        self.dim = dim


    def forward(self, r, d):
        r_hyperbolic = self.poin(self.rgb(r))
        d_hyperbolic = self.poin(self.depth(d))
        # rd = self.distance(r_hyperbolic, d_hyperbolic)
        rd = torch.cat([r_hyperbolic, d_hyperbolic], dim=2)
        rd = self.up(rd)
        rd = F.softmax(rd)
        rd_r, rd_d = rd.split(self.dim, dim=2)
        rd = r * rd_r + d * rd_d + r + d
        return rd


"""
   Based on the implementation in https://github.com/leymir/hyperbolic-image-embeddings
    """
class HyperbolicDistanceLayer(nn.Module):
    def __init__(self, c=1):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.dist(x1, x2, c=c, keepdim=False)

    def extra_repr(self):
        return "c={}".format(self.c)

class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """

    def __init__(self, c=1, train_c=False, train_x=False, ball_dim=None, riemannian=True):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c

        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))



# class Decoder(nn.Module):
#     def __init__(self, in_channels):
#         super(Decoder, self).__init__()
#
#         lat_layers = []
#         for idx in range(4):
#             lat_layers.append(DepthWiseConv(in_channels[idx] * 2, in_channels[idx]))
#         self.lat_layers = nn.ModuleList(lat_layers)
#
#         down_layers = []
#         for idx in range(3):
#             down_layers.append(DepthWiseConv(in_channels[idx + 1], in_channels[idx]))
#         self.down_layers = nn.ModuleList(down_layers)
#
#
#         out_layer = []
#         for idx in range(4):
#             out_layer.append(nn.Sequential(
#                 DepthWiseConv(in_channels[idx], 1),
#                 nn.UpsamplingBilinear2d(scale_factor=pow(2, idx + 2))
#
#             ))
#         self.out_layer = nn.ModuleList(out_layer)
#
#
#
#     def forward(self, feat_list, cosal_list):
#         all_feat = []
#         all_pre = []
#         feat_top = self.lat_layers[-1](torch.cat([feat_list[-1], cosal_list[-1]], dim=1))
#         p = feat_top
#         pre = self.out_layer[-1](p)
#         all_feat.append(p)
#         all_pre.append(pre)
#
#         for idx in [2, 1, 0]:
#             p = self._upsample_add(self.down_layers[idx](p), self.lat_layers[idx](torch.cat([feat_list[idx], cosal_list[idx]], dim=1)))
#             all_feat.append(p)
#             pre = self.out_layer[idx](p)
#             all_pre.append(pre)
#
#
#         return all_feat, all_pre
#
#     def _upsample_add(self, x, y):
#         [_, _, H, W] = y.size()
#         return F.interpolate(
#             x, size=(H, W), mode='bilinear') + y

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = max_out
#         return self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = max_out
#         x = self.conv1(x)
#         return self.sigmoid(x)


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)

        return out

# class DDCM(nn.Module):
#     def __init__(self, in_channels, in_channels1=512):
#         super(DDCM, self).__init__()
#         self.in_channels = in_channels
#         self.AdAvgpool1 = nn.AdaptiveAvgPool2d((3, 3))
#         self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
#         self.conv_r = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.BatchNorm2d(in_channels), nn.ReLU())
#         self.conv_d = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1, 0), nn.BatchNorm2d(in_channels), nn.ReLU())
#         self.conv = nn.Sequential(DepthWiseConv(in_channels * 3, in_channels), nn.BatchNorm2d(in_channels), nn.ReLU())
#         self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.conv1 = DepthWiseConv(in_channels + in_channels1, in_channels)
#         self.channel_attention = ChannelAttention(in_channels)
#         self.spatial_attention = SpatialAttention()
#
#     # def upsample(self, x, y):
#     #     [_, _, H, W] = y.size()
#     #     return F.interpolate(
#     #         x, size=(H, W), mode='bilinear')
#
#     def forward(self, fr, fd, frd=None):
#         B, C, H, W = fr.size()
#
#         kerner1 = self.AdAvgpool1(fr)
#         B1, C1, H1, W1 = kerner1.size()
#         fd_new = fd.clone()
#         for i in range(1, B):
#             kernel1 = kerner1[i, :, :, :]
#             kernel1 = kernel1.view(C1, 1, H1, W1)
#             x4_r1 = F.conv2d(fd[i, :, :, :].view(1, C, H, W), kernel1, stride=1, padding=1, dilation=1, groups=C)
#             x4_r2 = F.conv2d(fd[i, :, :, :].view(1, C, H, W), kernel1, stride=1, padding=2, dilation=2, groups=C)
#             x4_r4 = F.conv2d(fd[i, :, :, :].view(1, C, H, W), kernel1, stride=1, padding=4, dilation=4, groups=C)
#
#             fd_new[i, :, :, :] = x4_r1 + x4_r2 + x4_r4
#         fd1 = self.conv_d(fd_new) * fd
#
#
#         kerner2 = self.AdAvgpool1(fd)
#         B1, C1, H1, W1 = kerner2.size()
#         fr_new = fr.clone()
#         for i in range(1, B):
#             kernel2 = kerner2[i, :, :, :]
#             kernel2 = kernel2.view(C1, 1, H1, W1)
#             # DDconv
#             x4_r1 = F.conv2d(fr[i, :, :, :].view(1, C, H, W), kernel2, stride=1, padding=1, dilation=1, groups=C)
#             x4_r2 = F.conv2d(fr[i, :, :, :].view(1, C, H, W), kernel2, stride=1, padding=2, dilation=2, groups=C)
#             x4_r4 = F.conv2d(fr[i, :, :, :].view(1, C, H, W), kernel2, stride=1, padding=4, dilation=4, groups=C)
#
#             fr_new[i, :, :, :] = x4_r1 + x4_r2 + x4_r4
#
#         fr1 = self.conv_r(fr_new) * fr
#
#         if frd == None:
#
#             frd_mut = fr1 * fd1
#             sa = self.spatial_attention(frd_mut)
#             r_f = frd_mut * sa
#             r_f = frd_mut + r_f
#             r_ca = self.channel_attention(frd_mut)
#             r_out = frd_mut * r_ca
#             r_out = r_out + frd_mut
#             frd_add = r_f + r_out
#             fusion = self.conv(torch.cat([fr, fd, frd_add], dim=1))
#
#
#             return fusion
#         else:
#             frd_mut = fr1 * fd1
#             sa = self.spatial_attention(frd_mut)
#             r_f = frd_mut * sa
#             r_f = frd_mut + r_f
#             r_ca = self.channel_attention(frd_mut)
#             r_out = frd_mut * r_ca
#             r_out = r_out + frd_mut
#             frd_add = r_f + r_out
#             fusion = self.conv(torch.cat([fr, fd, frd_add], dim=1))
#             fusion = self.conv1(torch.cat([fusion, self.up2(frd)], dim=1))
#             return fusion




def resize(input, target_size=(256, 256)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)



class Cosal_Module(nn.Module):
    def __init__(self, in_channels):
        super(Cosal_Module, self).__init__()
        self.cosal_feat = Cosal_Sub_Module(in_channels)
        self.conv = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1))

    def forward(self, feats, SISMs):

        SISMs = torch.sigmoid(SISMs)
        fore_cosal_feats = self.cosal_feat(feats, SISMs)
        back_cosal_feats = self.cosal_feat(feats, 1.0 - SISMs)
        cosal_enhanced_feats = self.conv(torch.cat([fore_cosal_feats, back_cosal_feats], dim=1))
        return cosal_enhanced_feats, fore_cosal_feats, back_cosal_feats


class Cosal_Sub_Module(nn.Module):
    def __init__(self, in_channels):
        super(Cosal_Sub_Module, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(32, in_channels, 1))

    def forward(self, feats, SISMs):
        N, C, H, W = feats.shape
        HW = H * W
        if SISMs == None:
            SISMs = torch.ones_like(feats)
        # Resize SISMs to the same size as the input feats.
        SISMs = resize(SISMs, [H, W])  # shape=[N, 1, H, W], SISMs are the saliency maps generated by saliency head.

        # NFs: L2-normalized features.
        NFs = F.normalize(feats, dim=1)  # shape=[N, C, H, W]

        # Co_attention_maps are utilized to filter more background noise.
        def get_co_maps(co_proxy, NFs):
            correlation_maps = F.conv2d(NFs, weight=co_proxy)  # shape=[N, N, H, W]

            # Normalize correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, HW), dim=2)  # shape=[N, N, HW]
            co_attention_maps = torch.sum(correlation_maps, dim=1)  # shape=[N, HW]

            # Max-min normalize co-attention maps.
            min_value = torch.min(co_attention_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(co_attention_maps, dim=1, keepdim=True)[0]
            co_attention_maps = (co_attention_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[N, HW]
            co_attention_maps = co_attention_maps.view(N, 1, H, W)  # shape=[N, 1, H, W]
            return co_attention_maps

        # Use co-representation to obtain co-saliency features.
        def get_CoFs(NFs, co_rep):
            SCFs = F.conv2d(NFs, weight=co_rep)
            return SCFs

        # Find the co-representation proxy.
        co_proxy = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2), dim=1).view(N, C, 1, 1)  # shape=[N, C, 1, 1]

        # Reshape the co-representation proxy to compute correlations between all pixel embeddings and the proxy.
        r_co_proxy = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2).mean(dim=0), dim=0)
        r_co_proxy = r_co_proxy.view(1, C)
        all_pixels = NFs.reshape(N, C, HW).permute(0, 2, 1).reshape(N*HW, C)
        correlation_index = torch.matmul(all_pixels, r_co_proxy.permute(1, 0))

        # Employ top-K pixel embeddings with high correlation as co-representation.
        ranged_index = torch.argsort(correlation_index, dim=0, descending=True).repeat(1, C)
        co_representation = torch.gather(all_pixels, dim=0, index=ranged_index)[:32, :].view(32, C, 1, 1)

        co_attention_maps = get_co_maps(co_proxy, NFs)  # shape=[N, 1, H, W]
        CoFs = get_CoFs(NFs, co_representation)  # shape=[N, HW, H, W]
        co_saliency_feat = self.conv(CoFs * co_attention_maps)  # shape=[N, 128, H, W]

        return co_saliency_feat



#     return max_centers


from thop import profile
if __name__ == "__main__":
    # inputs = torch.rand(5, 512, 8, 8).cuda()
    # depths = torch.rand(5, 512, 8, 8).cuda()
    # models4 = DDCM(512).cuda()
    # outs = models4(inputs, depths)
    # print("outs shape", outs.shape)
    # flops, params = profile(models4, inputs=(inputs, depths))
    # print("the number of Flops {} G ".format(flops / 1e9))
    # print("the number of Parameter {}M ".format(params / 1e6))  # 1048576
    inputs1 = torch.rand(5, 64, 64, 64).cuda()
    depths1 = torch.rand(5, 64, 64, 64).cuda()
    # models1 = DDCM(64).cuda()
    # outs1 = models1(inputs1, depths1)
    # print("outs1 shape", outs1.shape)
    inputs4 = torch.rand(5, 256, 16, 16).cuda()
    sisms = torch.rand(5, 1, 8, 8).cuda()
    # model = Cosal_Module(512).cuda()
    # out = model(inputs4, sisms)
    # for i in range(len(out)):
    #     print(out[i].shape)


    # flops, params = profile(models1, inputs=(inputs1, depths1))
    # print("the number of Parameter1 {}M ".format(params / 1e6))