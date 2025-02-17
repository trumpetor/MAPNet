import os
import argparse
from collections import OrderedDict
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch import nn
from thop import profile
import numpy as np
import cv2

from COA_RGBD_SOD.al.dataset import get_loader
# from Comparision_Methods.UniTR.model_image_depth import build_model as CoNet
from COA_RGBD_SOD.al.models.Fourth_model.CoNet import CoNet


from COA_RGBD_SOD.al.util import save_tensor_img
from COA_RGBD_SOD.al.config import Config


def main(args):
    # Init model
    config = Config()

    device = torch.device("cuda")
    # model = GCoNet()
    # model = CoNet()
    model = CoNet()
    model = model.to(device)
    print('Testing with model {}'.format(args.ckpt))

    base_weights = torch.load(args.ckpt)
    # new_state_dict = OrderedDict()
    # for k, v in base_weights.items():
    #     name = k  # remove 'module.'
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict["state_dict"])
    model.load_state_dict(base_weights)

    model.eval()

    for testset in args.testsets.split('+'):
        print('Testing {}...'.format(testset))
        root_dir = '/home/map/Alchemist/COA/data/'
        if testset == 'CoCA':
            test_img_path = os.path.join(root_dir, 'images/CoCA')
            test_gt_path = os.path.join(root_dir, 'gts/CoCA')
            test_depth_path = os.path.join(root_dir, 'depths/CoCA')
            saved_root = os.path.join(args.pred_dir, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = os.path.join(root_dir, 'images/CoSOD3k')
            test_gt_path = os.path.join(root_dir, 'gts/CoSOD3k')
            saved_root = os.path.join(args.pred_dir, 'CoSOD3k')
        elif testset == 'Cosal2015':
            test_img_path = os.path.join(root_dir, 'images/Cosal2015')
            test_gt_path = os.path.join(root_dir, 'gts/Cosal2015')
            saved_root = os.path.join(args.pred_dir, 'Cosal2015')
        elif testset == 'RGBD_CoSal150':
            test_img_path = os.path.join(root_dir, 'images/RGBD_CoSal150')
            test_depth_path = os.path.join(root_dir, 'depths/RGBD_CoSal150')
            test_gt_path = os.path.join(root_dir, 'gts/RGBD_CoSal150')
            saved_root = os.path.join(args.pred_dir, 'RGBD_CoSal150')
        elif testset == 'RGBD_CoSal1k':
            test_img_path = os.path.join(root_dir, 'images/RGBD_CoSal1k')
            test_depth_path = os.path.join(root_dir, 'depths/RGBD_CoSal1k')
            test_gt_path = os.path.join(root_dir, 'gts/RGBD_CoSal1k')
            saved_root = os.path.join(args.pred_dir, 'RGBD_CoSal1k')
        elif testset == 'RGBD_CoSeg183':
            test_img_path = os.path.join(root_dir, 'images/RGBD_CoSeg183')
            test_depth_path = os.path.join(root_dir, 'depths/RGBD_CoSeg183')
            test_gt_path = os.path.join(root_dir, 'gts/RGBD_CoSeg183')
            saved_root = os.path.join(args.pred_dir, 'RGBD_CoSeg183')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)

        test_loader = get_loader(
            test_img_path, test_gt_path, test_depth_path, test_gt_path, args.size, 1, istrain=False, shuffle=False, num_workers=8,
            pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            depths = batch[2].to(device).squeeze(0)
            edges = batch[3].to(device).squeeze(0)
            gt_ground = 1 - gts

            subpaths = batch[4]
            ori_sizes = batch[-1]
            with torch.no_grad():
                # scaled_preds = model(inputs, depths)[1]
                scaled_preds = model(inputs, depths)[0]

            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)

            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                    print("11")
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                    # gt_ground = nn.functional.interpolate(gt_ground[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                    #                                       align_corners=True)
                    # res = gt_ground
                else:
                    ### 原本的输出到原图尺寸
                    # print("22") # this
                    # res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                    #                                 align_corners=True)
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True).sigmoid()
                    # # print(out.shape)
                    # out = F.interpolate(scaled_preds[inum].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
                    # out_img = out.cpu().detach().numpy()
                    # out_img = np.max(out_img, axis=1).reshape(256, 256)
                    # out_img = (((out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
                    # out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
                    # res = out_img




                # cv2.imwrite(os.path.join(saved_root, subpath), out_img)
                save_tensor_img(res, os.path.join(saved_root, subpath))



if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet_plus',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--testsets',
                        default='RGBD_CoSeg183',  # +CoSOD3k+Cosal2015 RGBD_CoSal1k' RGBD_CoSal150 RGBD_CoSeg183
                        type=str,
                        help="Options: 'CoCA','Cosal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size',
                        default=224,
                        type=int,
                        help='input size')
    parser.add_argument('--ckpt', default='/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/Alchemist/COA/COA_RGBD_SOD/ckpt/Fourth_models/CoNet_shunted_s_co/Pth/CoNet_best_ep149_Smeasure0.8607.pth', type=str,
                        help='model folder')
    parser.add_argument('--pred_dir',
                        default='/home/map/Alchemist/COA/COA_RGBD_SOD/ckpt/Fourth_models/CoNet_shunted_s_co_8607/RGBD_CoSal1k',
                        # default='/home/map/Alchemist/COA/COA_RGBD_SOD/ckpt/Thrid_models/CoNet_baseline_8461/RGBD_CoSal1k'
                        type=str, help='Output folder')

    args = parser.parse_args()

    main(args)
# [cost:20.5195s] RGBD_CoSeg183 (M-epRGBD_CoSeg183): 0.9118 max-Emeasure || 0.8532 S-measure  || 0.7761 max-fm || 0.0329 mae || 0.8780 mean-Emeasure || 0.7316 mean-fm.