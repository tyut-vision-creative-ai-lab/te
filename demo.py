from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import warnings
from utils.Functions import *
import utils.lib.pytorch_ssim as pytorch_ssim
import imageio
from skimage import metrics 
import numpy as np
import scipy.io as scio 
import scipy.misc as scim
import os,time
import logging,argparse
from datetime import datetime
from collections import OrderedDict
from einops import rearrange
from utils.LFDataset import LoadTestData
from utils.DeviceParameters import to_device

from model.HLFRN import HLFRN
from model.MSP import MSP
from model.DRLF import DRLF
from model.PFE import PFE
from model.LFFG import LF_WXformerCHUAN


# Testing settings
parser = argparse.ArgumentParser(description="Light Field Restoration")
parser.add_argument("--model_name", type=str, default='LFWXformerCJ', help="Path for saving training log ")
parser.add_argument("--sigma", type=int, default=20, help="The number of stages")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")

parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("--cropPatchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")

parser.add_argument("--modelPath", type=str, default='./pretrained_models/LFWXformer/model_sigma_20.pth', help="Path for loading trained model ")
parser.add_argument("--dataPath", type=str, default='./data/', help="Path for loading testing data ")
parser.add_argument("--savePath", type=str, default='./results/demo_real_img2', help="Path for saving results ")
parser.add_argument("--cropImage", type=bool, default=True, help="Crop image to save memory during inference")

parser.add_argument("--patch_size", type=int, default=32, help="patch size for train")
parser.add_argument("--channels", type=int, default=64, help="channels , embed_dim for transformer —— C")
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--attn_drop_rate", type=float, default=0.05, help="drop rate for attention calculation")
parser.add_argument("--drop_rate", type=float, default=0.05, help="common drop rate")
parser.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth decay rule")
parser.add_argument("--ang_num_heads", type=int, default=4, help="number of multi heads for angular transformer —— P")
parser.add_argument("--spa_num_heads", type=int, default=4, help="number of multi heads for spatial transformer —— P")
parser.add_argument("--ang_mlp_ratio", type=int, default=4, help="scale ratio in MLP for angular transformer")
parser.add_argument("--spa_mlp_ratio", type=int, default=4, help="scale ratio in MLP for spatial transformer")
parser.add_argument("--depth", type=int, default=4, help="number of spatial-angular transformer encoder —— N")
parser.add_argument("--ang_sr_ratio", type=int, default=1, help="reduce patches scale for angular transformer")
parser.add_argument("--spa_sr_ratio", type=int, default=2, help="reduce patches scale for spatial transformer —— S")
parser.add_argument("--spa_trans_num", type=int, default=2, help="number of spatial transformer in transformer encoder —— K")
parser.add_argument("--attn_ratio", type=float, default=0.5, help="drop rate for attention calculation")

#  HLFRN parameters
parser.add_argument("--n_groups", type=int, default=5, help="The number of HGAG groups")
parser.add_argument("--n_blocks", type=int, default=5, help="The number of HFEB blocks")
parser.add_argument("--n_channels", type=int, default=32, help="The number of convolution filters")

#  DRLF parameters
parser.add_argument("--stageNum", type=int, default=3, help="The number of stages")
parser.add_argument("--channelNum", type=int, default=3, help="The number of input channels")

# PFE parameters
parser.add_argument("--temperature_1", type=float, default=1, help="The number of temperature_1")
parser.add_argument("--temperature_2", type=float, default=1, help="The number of temperature_2")
parser.add_argument("--component_num", type=int, default=4, help="The number of pfe component")
parser.add_argument("--sasLayerNum", type=int, default=6, help="The number of stages")
parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")

opt = parser.parse_args()

save_dir = opt.savePath + '/' + opt.model_name + '_' + str(opt.sigma)
if not os.path.exists(save_dir): 
		os.makedirs(save_dir) 

# warnings.filterwarnings("ignore")
# plt.ion()
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# log = logging.getLogger()
# fh = logging.FileHandler( save_dir +  '/Testing_' + opt.model_name + '_' + str(opt.sigma) + '.log')
# log.addHandler(fh)

# logging.info(opt)
if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------ 构建模型 ------
    if opt.model_name == 'DRLF':
        model = DRLF(opt)
    elif opt.model_name == 'MSP':
        model = MSP(opt)
    elif opt.model_name == 'PFE':
        model = PFE(opt)
    elif opt.model_name == 'HLFRN':
        model = HLFRN(opt)
    elif opt.model_name == 'LFWXformerCJ':
        model = LF_WXformerCHUAN(opt)
    else:
        raise ValueError(f"Unknown model_name: {opt.model_name}")

    model.load_state_dict(torch.load(opt.modelPath, map_location='cuda:0'))
    model.eval()
    to_device(model, device)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # log.info("Training parameters: %d" %total_trainable_params)

    scene_list = os.listdir(opt.dataPath)

    for scenes in scene_list:
        print('Working on scene: ' + scenes + '...')

        # 读取一个中间视角确定尺寸
        temp = imageio.imread(os.path.join(opt.dataPath, scenes, '05_05.png'))
        angRes = 9
        noiLF = np.zeros(
            shape=(angRes, angRes, temp.shape[0], temp.shape[1], 3),
            dtype=np.float32
        )

        # 读取 9x9 视角图像：01_01.png ... 09_09.png
        for i in range(angRes):
            for j in range(angRes):
                name = "%.2d_%.2d" % (i + 1, j + 1)
                img_path = os.path.join(opt.dataPath, scenes, name + '.png')
                noiLF[i, j, :, :, :] = imageio.imread(img_path)

        # 转为 tensor，归一化到 [0,1]
        noiLF = torch.from_numpy(noiLF) / 255.0
        noiLF = noiLF.unsqueeze(0)  # [1, u, v, h, w, c]
        b, u, v, h, w, c = noiLF.shape

        # 按 opt.angResolution 居中裁剪角度（比如从 9 裁剪到 5）
        noiLF = noiLF[
            :,
            (u - opt.angResolution) // 2:(u + opt.angResolution) // 2,
            (v - opt.angResolution) // 2:(v + opt.angResolution) // 2,
            :,
            :,
            0:3
        ]  # [1, U', V', H, W, 3]
        # 这里的 noiLF 是裁过角度的 noisy 光场

        with torch.no_grad():
            if opt.cropImage:
                cropStride = opt.cropPatchSize - opt.overlap
                # [b,n,u,v,x,y,c]
                noiLFStack, coordinate = CropLF(noiLF, opt.cropPatchSize, cropStride)
                b, n, u, v, x, y, c = noiLFStack.shape
                denoilfStack = torch.zeros(b, n, u, v, x, y, c)  # [b,n,u,v,x,y,c]

                # reconstruction
                avg_time_patch = 0
                for i in range(noiLFStack.shape[1]):
                    inp = noiLFStack[:, i, :, :, :, :, :].permute(0, 1, 2, 5, 3, 4).cuda()  # [b,u,v,c,x,y]
                    if opt.model_name == 'MSP':
                        _, _, denoiLFPatch = model(inp)  # [b,u,v,c,x,y]
                    else:
                        if opt.model_name == 'PFE':
                            epoch = 10000
                            denoiLFPatch = model(inp, epoch)  # [b,u,v,c,x,y]
                        else:
                            denoiLFPatch = model(inp)  # [b,u,v,c,x,y]

                    denoilfStack[:, i, :, :, :, :, :] = denoiLFPatch.permute(
                        0, 1, 2, 4, 5, 3
                    )  # [b,n,u,v,x,y,c]

                # 去噪结果：patch + overlap 合成
                denoiLF = MergeLF(denoilfStack, coordinate, opt.overlap)  # [b,u,v,x,y,c]

                # 🔴 noisy 也用同样的 patch + overlap 合成（裁掉边缘）
                noisyLF_merged = MergeLF(noiLFStack, coordinate, opt.overlap)  # [b,u,v,x,y,c]

                b, u, v, x, y, c = denoiLF.shape

            else:
                # 直接整幅推理，不需要裁边
                # noiLF: [1,u,v,h,w,c] -> [1,u,v,c,h,w]
                denoiLF = model(noiLF.permute(0, 1, 2, 5, 3, 4).cuda())  # [b,u,v,c,x,y]
                denoiLF = denoiLF.permute(0, 1, 2, 4, 5, 3)  # [b,u,v,x,y,c]
                b, u, v, x, y, c = denoiLF.shape

                # 对应的 noisy 就是当前 noiLF，不再额外裁边
                noisyLF_merged = noiLF  # [b,u,v,h,w,c]，这里 h=x, w=y

        # ============ 保存部分：分 noisy / denoised 子文件夹 ============
        save_png_dir = os.path.join(save_dir, scenes)
        os.makedirs(save_png_dir, exist_ok=True)

        noisy_dir = os.path.join(save_png_dir, "noisy")
        denoised_dir = os.path.join(save_png_dir, "denoised")
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(denoised_dir, exist_ok=True)

        # 把 tensor 转回 numpy、uint8
        # 去噪结果：denoiLF [1,u,v,h,w,c]
        denoiLF_np = denoiLF.squeeze(0).cpu().numpy()  # [u,v,h,w,c]
        denoiLF_np = np.clip(denoiLF_np * 255.0, 0, 255).astype(np.uint8)

        # 噪声输入：noisyLF_merged [1,u,v,h,w,c]（cropImage=True 时是 Merge 后的；否则是原图）
        noisyLF_np = noisyLF_merged.squeeze(0).cpu().numpy()  # [u,v,h,w,c]
        noisyLF_np = np.clip(noisyLF_np * 255.0, 0, 255).astype(np.uint8)

        # save all views
        for i in range(opt.angResolution):
            for j in range(opt.angResolution):
                # 去噪图像
                img_den = denoiLF_np[i, j, :, :, :]
                den_path = os.path.join(denoised_dir, f"View_{i}_{j}.png")
                imageio.imwrite(den_path, img_den)

                # 噪声图像（已经经过 overlap Merge，边缘一致）
                img_noi = noisyLF_np[i, j, :, :, :]
                noi_path = os.path.join(noisy_dir, f"View_{i}_{j}.png")
                imageio.imwrite(noi_path, img_noi)

        print(f"Scene {scenes} done. Saved to {save_png_dir}")

    print('Finish.')
