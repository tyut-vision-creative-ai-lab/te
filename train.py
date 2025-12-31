from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging

from model.LFDET import LFDET
from model.LFWXformerV2 import LFWXformerV2
from model.LFXformer import LFXformer
from model.LFWXformerCJ_noDWT_no1x1 import LF_WXformerCHUAN
from utils.LFDataset import LoadTrainData
from utils.Functions import *
from utils.DeviceParameters import to_device

from model.MSP import MSP
from model.DRLF import DRLF
from model.PFE import PFE
from model.HLFRN import HLFRN
from model.LF_WXformer import LF_WXformer
import copy
import mat73
from skimage import metrics
import itertools, argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
# 日志目录：放在 train.py 所在目录的 log 子目录
LOG_DIR = Path(__file__).resolve().parent / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 文件名带到秒，并附上 PID，避免同一分钟二次运行冲突
log_path = LOG_DIR / f"Training_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log"

fh = logging.FileHandler(log_path, encoding="utf-8")
log.addHandler(fh)

# Training settings
parser = argparse.ArgumentParser(description="Hybrid Light Field Restoration")
parser.add_argument("--model_name", type=str, default='LFWXformerCJ_noDWT_no1x1', help="Path for saving training log ")
parser.add_argument("--learningRate", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--sigma", type=int, default=20, help="Noise level")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")

parser.add_argument("--batchSize", type=int, default=2, help="Batch size")
parser.add_argument("--sampleNum", type=int, default=70, help="The number of LF in training set")
parser.add_argument("--patchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--patch_size", type=int, default=32, help="patch size for train")
parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")
parser.add_argument("--num_steps", type=int, default=2500, help="The number of step size reduce learning rate")

parser.add_argument("--summaryPath", type=str, default='./log/', help="Path for saving training log ")
parser.add_argument("--saveCheckpointsDir", type=str, default='./checkpoints/', help="Path for saving training log ")
parser.add_argument("--dataPath", type=str, default='./datasets/train_noiseLevel_10-20-50_4-11_color_5x5.mat',
                    help="Path for loading training data ")

parser.add_argument('--resume', type=str, default=True, help='Resume training from saved checkpoint(s).')
parser.add_argument('--modelPath', type=str, default='./checkpoints/LFWXformerCJ_noDWT_no1x1_20/model_sigma_20_group128_best.pth',
                    help='Resume training from saved checkpoint(s).')
parser.add_argument('--optimizerPath', type=str, default='./checkpoints/LFWXformerCJ_noDWT_no1x1_20/optimizer_group128_best.pth',
                    help='Resume optimizer from saved optimizer(s).')
parser.add_argument("--scale_factor", type=int, default=1, help="4, 2")
parser.add_argument("--channels", type=int, default=64, help="channels , embed_dim for transformer —— C")
parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
parser.add_argument("--attn_drop_rate", type=float, default=0.1, help="drop rate for attention calculation")
parser.add_argument("--drop_rate", type=float, default=0.1, help="common drop rate")
parser.add_argument("--drop_path_rate", type=float, default=0.2, help="stochastic depth decay rule")
parser.add_argument("--ang_num_heads", type=int, default=4, help="number of multi heads for angular transformer —— P")
parser.add_argument("--spa_num_heads", type=int, default=4, help="number of multi heads for spatial transformer —— P")
parser.add_argument("--ang_mlp_ratio", type=int, default=4, help="scale ratio in MLP for angular transformer")
parser.add_argument("--spa_mlp_ratio", type=int, default=4, help="scale ratio in MLP for spatial transformer")
# parser.add_argument("--depth", type=int, default=4, help="number of spatial-angular transformer encoder —— N")
# parser.add_argument("--ang_sr_ratio", type=int, default=1, help="reduce patches scale for angular transformer")
# parser.add_argument("--spa_sr_ratio", type=int, default=2, help="reduce patches scale for spatial transformer —— S")
parser.add_argument("--attn_ratio", type=float, default=0.5, help="drop rate for attention calculation")
parser.add_argument("--spa_trans_num", type=int, default=2,
                    help="number of spatial transformer in transformer encoder —— K")

#  HLRN parameters
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

opt = parser.parse_args()
logging.info(opt)

# make save ckpt
saveCheckpointsDir = opt.saveCheckpointsDir + '/' + opt.model_name + '_' + str(opt.sigma)
if not os.path.exists(saveCheckpointsDir):
    os.makedirs(saveCheckpointsDir)

if __name__ == '__main__':

    lfDataset = LoadTrainData(opt)
    dataloader = DataLoader(lfDataset, batch_size=opt.batchSize, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if opt.model_name == 'LFWXformerCJ_noDWT_no1x1':
        model = LF_WXformerCHUAN(opt)

    if opt.model_name == 'LFXformer':
        model = LFXformer(opt)

    if opt.model_name == 'LFWXformerV3':
        model = LFWXformerV2(opt)

    if opt.model_name == 'HLFRN':
        model = HLFRN(opt)

    if opt.model_name == 'LF_WXformer':
        model = LF_WXformer(opt)

    to_device(model, device)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Training parameters: %d" % total_trainable_params)

    criterion = torch.nn.L1Loss()  # Loss

    optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=opt.learningRate)  # optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.num_steps, gamma=0.5)
    writer = SummaryWriter(opt.summaryPath)

    if opt.resume:
        print('Resume training from' + opt.modelPath)
        model.load_state_dict(torch.load(opt.modelPath), strict=False)

        optimizerPath = torch.load(opt.optimizerPath)
        optimizer.load_state_dict(optimizerPath['optimizer_state_dict'])
        epoch_start = optimizerPath['epoch'] + 1
        model.train()
    else:
        print('Start training from scratch.')
        epoch_start = 1

    lossLogger = defaultdict(list)
    # 全局最优
    best_loss = float('inf')
    best_model_path = saveCheckpointsDir + '/model_sigma_%d_best.pth' % opt.sigma

    # ------- 每 50 轮一份局部最优 -------
    # 当前所在的“50 轮分组”索引（从 0 开始）：0->第 1~50 轮，1->第 51~100 轮，以此类推
    local_group_idx = (epoch_start - 1) // 50
    local_best_loss = float('inf')
    # --------------------------------

    for epoch in range(epoch_start, opt.epochNum):
        batch = 0
        lossSum = 0
        for _, sample in enumerate(dataloader):
            batch = batch + 1
            LF = sample['LFPatch']
            noiLF = sample['noiLFPatch']
            LF = to_device(LF, device)  # label:[u v c x y]
            noiLF = to_device(noiLF, device)  # input:[u v c x/s y/s]

            b, u, v, k, h, w, c = noiLF.shape
            noiLF = noiLF.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
            noiLF = noiLF.view(b, u, v, h, w, c * k)
            noiLF = noiLF.permute(0, 1, 2, 5, 3, 4).contiguous()

            LF = LF.permute(0, 1, 2, 4, 5, 6, 3).contiguous()
            LF = LF.view(b, u, v, h, w, c * k).contiguous()
            LF = LF.permute(0, 1, 2, 5, 3, 4).contiguous()

            if opt.model_name == 'MSP':
                estimatedLF_1, estimatedLF_2, estimatedLF = model(noiLF)

                l1loss_1 = criterion(estimatedLF_1, LF)
                l1loss_2 = criterion(estimatedLF_2, LF)
                l1loss = criterion(estimatedLF, LF)

                ssim_loss = 0
                for ind_uv in range(u * v):
                    ssim_loss += metrics.structural_similarity(
                        np.squeeze((estimatedLF.reshape(b, u * v, h, w, c)[0, ind_uv]
                                    .detach().cpu().numpy() * 255.0).astype(np.uint8)),
                        np.squeeze((LF.reshape(b, u * v, h, w, c)[0, ind_uv]
                                    .detach().cpu().numpy() * 255.0).astype(np.uint8)),
                        gaussian_weights=True, sigma=1.5,
                        use_sample_covariance=False, channel_axis=-1
                    ) / (u * v)

                diff_ssim_loss = (1 - ssim_loss)

                loss = l1loss + diff_ssim_loss + 0.05 * l1loss_1 + 0.1 * l1loss_2
            else:
                if opt.model_name == 'PFE':
                    estimatedLF = model(noiLF, epoch)
                else:
                    estimatedLF = model(noiLF)
                loss = criterion(estimatedLF, LF)

            lossSum += loss.item()

            writer.add_scalar('loss', loss, opt.sampleNum // opt.batchSize * epoch + batch)
            print("Epoch: %d Batch: %d Loss: %.6f" % (epoch, batch, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算当前 epoch 平均 loss
        epoch_loss = lossSum / len(dataloader)
        log.info("Epoch: %d Loss: %.6f" % (epoch, epoch_loss))

        # -------- 每轮检查并保存 全局 best 模型 --------
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            torch.save(
                {'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'best_loss': best_loss},
                saveCheckpointsDir + '/optimizer_best.pth'
            )
            log.info(">>> New best model saved at epoch %d, best_loss = %.6f" % (epoch, best_loss))
        # ------------------------------------------------

        # -------- 每 50 轮一份“局部最优”（当前 50 轮组里的最小 loss） --------
        # 当前属于第几个 50-epoch 分组（从 0 开始）：0 -> 1~50, 1 -> 51~100, ...
        cur_group_idx = (epoch - 1) // 50

        # 如果进入了新的 50 轮分组，重置局部最优
        if cur_group_idx != local_group_idx:
            local_group_idx = cur_group_idx
            local_best_loss = float('inf')

        # 更新当前 50 轮分组内的最优模型
        if epoch_loss < local_best_loss:
            local_best_loss = epoch_loss
            group_no = local_group_idx + 1  # 分组编号从 1 开始
            local_ckpt_path = saveCheckpointsDir + '/model_sigma_%d_group%02d_best.pth' % (opt.sigma, group_no)
            torch.save(model.state_dict(), local_ckpt_path)
            torch.save(
                {'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(), 'local_best_loss': local_best_loss},
                saveCheckpointsDir + '/optimizer_group%02d_best.pth' % group_no
            )
            log.info(
                ">>> [Group %02d] new local best at epoch %d, local_best_loss = %.6f"
                % (group_no, epoch, local_best_loss)
            )
        # -----------------------------------------------------------

        # -------- 每 50 轮保存一次普通 checkpoint（非 best）--------
        if (epoch + 1) % 50 == 0:
            save_ckpt_path = saveCheckpointsDir + '/model_sigma_%d_epoch_%02d.pth' % (
                opt.sigma, epoch + 1)
            torch.save(model.state_dict(), save_ckpt_path)

            save_optimizer_path = saveCheckpointsDir + '/optimizer.pth'
            torch.save({'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict()},
                       save_optimizer_path)
        # ---------------------------------------------------------

        scheduler.step()

        # Record the training loss
        lossLogger['Epoch'].append(epoch)
        lossLogger['Loss'].append(epoch_loss)
        plt.figure()
        plt.title('Loss')
        plt.plot(lossLogger['Epoch'], lossLogger['Loss'])
        plt.savefig('./log/Training_{}.jpg'.format(opt.sigma))
        plt.close()
