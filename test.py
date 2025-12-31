from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import utils
import warnings

from model.LFXformer import LFXformer

from model.LF_WXformer import LF_WXformer
from model.LFFG import LF_WXformerCHUAN
from utils.Functions import *
import utils.lib.pytorch_ssim as pytorch_ssim

from skimage import metrics 
import numpy as np
import scipy.io as scio 
import scipy.misc as scim
import os,time
import logging,argparse
from datetime import datetime
from collections import OrderedDict
from utils.LFDataset import LoadTestData
from utils.DeviceParameters import to_device
import imageio

from model.MSP import MSP
from model.DRLF import DRLF
from model.PFE import PFE
from model.HLFRN import HLFRN


# Testing settings
parser = argparse.ArgumentParser(description="Light Field Restoration")
parser.add_argument("--model_name", type=str, default='LFFG', help="Path for saving training log ")
parser.add_argument("--sigma", type=int, default=20, help="The number of stages")
parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")

parser.add_argument("--batchSize", type=int, default=1, help="Batch size")
parser.add_argument("--cropPatchSize", type=int, default=32, help="The size of croped LF patch")
parser.add_argument("--patch_size", type=int, default=32, help="patch size for train")
parser.add_argument("--overlap", type=int, default=4, help="The size of croped LF patch")

parser.add_argument("--modelPath", type=str, default='./pretrained_models/LFWXformer/model_sigma_20.pth', help="Path for loading trained model ")
parser.add_argument("--dataPath", type=str, default='./datasets/test_noiseLeve_10-20-50_4-11_5x5.mat', help="Path for loading testing data ")
parser.add_argument("--savePath", type=str, default='./results/sythesis_img_test', help="Path for saving results ")
parser.add_argument("--save_png", type=str, default=False, help="save png results")
parser.add_argument("--save_mat_files", type=str, default=False, help="save mat results")

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
parser.add_argument("--n_groups", type=int, default=5, help="The number of HGAG groups") # Large: 5; Small: 3
parser.add_argument("--n_blocks", type=int, default=5, help="The number of HFEB blocks") # Large: 5; Small: 3
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

warnings.filterwarnings("ignore")
plt.ion()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
fh = logging.FileHandler( save_dir +  '/Testing_' + opt.model_name + '_' + str(opt.sigma) + '.log')
log.addHandler(fh)

logging.info(opt)


if __name__ == '__main__':

	lf_dataset = LoadTestData(opt)
	dataloader = DataLoader(lf_dataset, batch_size=opt.batchSize,shuffle=False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if opt.model_name == 'LFXformer':
		model= LFXformer(opt)

	if opt.model_name == 'DRLF':
		model= DRLF(opt)
	
	if opt.model_name == 'PFE':
		model= PFE(opt)

	if opt.model_name == 'HLFRN':
		model=HLFRN(opt)

	if opt.model_name == 'LF_WXformer':
		model=LF_WXformer(opt)
	if opt.model_name == 'LFFG':
		model=LF_WXformerCHUAN(opt)
	model.load_state_dict(torch.load(opt.modelPath, map_location='cuda:0'))
	model.eval()
	to_device(model,device)

	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	log.info("Training parameters: %d" %total_trainable_params)

	with torch.no_grad():
		num = 0
		avg_psnr_y = 0
		avg_ssim_y = 0
		avg_lpips = 0
		for _,sample in enumerate(dataloader):
			num=num+1
			LF=sample['LF'] #test lf
			noiLF=sample['noiLF'] #test lf
			lfName=sample['lfName']

			b,u,v,k,h,w,c = noiLF.shape
			noiLF = noiLF.permute(0, 1, 2, 4, 5, 6,3).contiguous()
			noiLF = noiLF.view(b, u, v,h,w,c*k)
			noiLF = noiLF.permute(0, 1, 2, 5, 3, 4).contiguous()

			LF = LF.permute(0, 1, 2, 4, 5, 6,3).contiguous()
			LF = LF.view(b, u, v,h,w,c*k).contiguous() 
			LF = LF.permute(0, 1, 2, 5, 3, 4).contiguous()
						 
			cropStride=opt.cropPatchSize-opt.overlap
			noiLFStack,coordinate=CropLF(noiLF,opt.cropPatchSize, cropStride) #[b,n,u,v,x,y,c]
			b,n,u,v,x,y,c=noiLFStack.shape
		
			denoilfStack=torch.zeros(b,n,u,v,x,y,c)#[b,n,u,v,x,y,c]
								
			# reconstruction
			for i in range(noiLFStack.shape[1]):
				if opt.model_name == 'MSP':
					_,_,denoiLFPatch=model(noiLFStack[:,i,:,:,:,:].permute(0,1,2,5,3,4).cuda())  #[b,u,v,c,x,y]
				else:
					if opt.model_name == 'PFE':
						epoch = 10000
						denoiLFPatch=model(noiLFStack[:,i,:,:,:,:].permute(0,1,2,5,3,4).cuda(),epoch)  #[b,u,v,c,x,y]
					else:
						denoiLFPatch=model(noiLFStack[:,i,:,:,:,:].permute(0,1,2,5,3,4).cuda())  #[b,u,v,c,x,y]
				denoilfStack[:,i,:,:,:,:,:]= denoiLFPatch.permute(0,1,2,4,5,3) #[b,n,u,v,x,y,c]
				
			denoiLF=MergeLF(denoilfStack,coordinate,opt.overlap) #[b,u,v,x,y,c]
			b,u,v,x,y,c=denoiLF.shape            
			LF=LF[:,:,:, opt.overlap//2:opt.overlap//2+x,opt.overlap//2:opt.overlap//2+y,:]
			noiLF=noiLF[:,:,:, opt.overlap//2:opt.overlap//2+x,opt.overlap//2:opt.overlap//2+y,:]
			   
			lf_psnr_y = 0
			lf_ssim_y = 0

			denoiLF = torch.clamp(denoiLF, 0, 1)
			for ind_uv in range(u*v):
				lf_psnr_y += ComptPSNR(np.squeeze(denoiLF.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()),
									   np.squeeze(LF.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()))  / (u*v)
									   
				lf_ssim_y += metrics.structural_similarity(np.squeeze((denoiLF.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8)),
										  np.squeeze((LF.reshape(b,u*v,x,y,c)[0,ind_uv].cpu().numpy()*255.0).astype(np.uint8)),gaussian_weights=True, sigma=1.5, use_sample_covariance=False, multichannel=True, data_range=255.0, channel_axis = 2) / (u*v)
			avg_psnr_y += lf_psnr_y / len(dataloader)           
			avg_ssim_y += lf_ssim_y / len(dataloader) 
			
			if opt.save_mat_files:
				#save reconstructed LF
				scio.savemat(os.path.join(save_dir,str(lfName[0])+'.mat'),
							{'lf_recons':torch.squeeze(denoiLF).numpy()}) #[u,v,x,y,c]
				
			if opt.save_png:
				save_png_dir = os.path.join(save_dir,str(lfName[0]))
				if not os.path.exists(save_png_dir): 
					os.makedirs(save_png_dir) 
				
				# # ''' Save RGB '''
				if save_png_dir is not None:
					denoiLF = denoiLF.squeeze(0)
					denoiLF = 255 * denoiLF.cpu().numpy()
					denoiLF = np.clip(denoiLF, 0, 255)
					# print(denoiLF.shape)
					
					# save all views
					for i in range(opt.angResolution):
						for j in range(opt.angResolution):
							img = np.uint8(denoiLF[i, j, :, :, :])
							path = str(save_png_dir) + '/' + 'View' + '_' + str(i) + '_' + str(j) + '.png'
							imageio.imwrite(path, img)
							pass
						pass
					pass
			
			log.info('Index: %d  Scene: %s  PSNR: %.2f  SSIM: %.4f '%(num,lfName[0],lf_psnr_y,lf_ssim_y))
		log.info('Average PSNR: %.2f  SSIM: %.4f '%(avg_psnr_y,avg_ssim_y))            