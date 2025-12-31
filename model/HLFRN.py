import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np 
import warnings
import scipy.io as scio
from utils.Functions import weights_init
from einops import rearrange
from utils.dct import *


class HLFRN(nn.Module):
	def __init__(self, opt):
		super(HLFRN, self).__init__()
		self.angRes = opt.angResolution
		self.n_blocks = opt.n_blocks
		self.n_groups = opt.n_groups
		self.n_blocks = opt.n_blocks
		self.n_channels = opt.n_channels
		
		self.FreqNet = FreqNet(self.n_blocks, self.n_channels)
		self.PixNet = PixNet(self.n_groups, self.n_blocks, self.n_channels)

	def forward(self, x, info=None):
		b,u,v,c,h,w = x.shape
		out = self.FreqNet(x)
		out = out.reshape(b, c, u, v, h, w)
		out = self.PixNet(out)
		out = out.reshape(b,  u, v, c, h, w)
		return out

class FreqNet(nn.Module):
	def __init__(self, n_blocks, n_channels):
		super(FreqNet, self).__init__()
		
		self.n_channels = n_channels
		self.n_blocks = n_blocks
		# define DC feature extraction module
		DC_init = [
			nn.Conv3d(3, self.n_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)]
		DC_body = [
			ResBlock3D(self.n_channels) for _ in range(self.n_blocks)
		]

		# define low-freq feature extraction module
		low_init = [
			nn.Conv3d(3, self.n_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)]
		low_body = [
			ResBlock3D(self.n_channels) for _ in range(self.n_blocks)
		]

		# define mid-freq feature extraction module
		mid_init = [
			nn.Conv3d(3, self.n_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)]
		mid_body = [
			ResBlock3D(self.n_channels) for _ in range(self.n_blocks)
		]

		# define high-freq feature extraction module
		high_init = [
			nn.Conv3d(3, self.n_channels, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)]
		high_body = [
			ResBlock3D(self.n_channels) for _ in range(self.n_blocks)
		]

		self.DC_init = nn.Sequential(*DC_init)
		self.DC_body = nn.Sequential(*DC_body)
		self.low_init = nn.Sequential(*low_init)
		self.low_body = nn.Sequential(*low_body)
		self.mid_init = nn.Sequential(*mid_init)
		self.mid_body = nn.Sequential(*mid_body)
		self.high_init = nn.Sequential(*high_init)
		self.high_body = nn.Sequential(*high_body)
		self.last_cov = nn.Conv2d(self.n_channels, 3, kernel_size=1,stride =1,dilation=1,padding=0,bias=False)
	
	def forward(self, x):
		b, u, v, c, h, w = x.shape
		x_freq = SpaAngDCT(x).to('cuda')
		x_freq = rearrange(x_freq, 'b u v c h w -> (b c) u v h w')
		angRes = u
		x_rec = torch.zeros((b, self.n_channels, angRes, angRes, int(h), int(w)), dtype=torch.float32)
		x_rec = x_rec.to(x.device)
		pos_freq = get_freq_position(angRes)

		freq_idx = 0
		for item in pos_freq:
			if freq_idx == 0:
				dc_cut = x_freq[:, item[0], item[1], :, :]
				dc_cut = dc_cut.view(b, c, len(item[0]), h, w)
				dc_cut = self.DC_init(dc_cut)
				out = dc_cut + self.DC_body(dc_cut)

			if 1 <= freq_idx <= 2:
				low_cut = x_freq[:, item[0], item[1], :, :]
				low_cut = low_cut.view(b, c, len(item[0]), h, w)
				low_cut = self.low_init(low_cut)
				out = low_cut + self.low_body(low_cut)

			if 3 <=  freq_idx <= (angRes-1):
				mid_cut = x_freq[:, item[0], item[1], :, :]
				mid_cut = mid_cut.view(b, c, len(item[0]), h, w)
				mid_cut = self.mid_init(mid_cut)
				out = mid_cut + self.mid_body(mid_cut)

			if  angRes <= freq_idx:
				high_cut = x_freq[:, item[0], item[1], :, :]
				high_cut = high_cut.view(b, c, len(item[0]), h, w)
				high_cut = self.high_init(high_cut)
				out = high_cut + self.high_body(high_cut)

			for i in range(len(item[0])):
				x_rec[:, :, item[0][i], item[1][i], :, :] = out[:, :, i, :, :]
			freq_idx +=1

		x_rec = rearrange(x_rec, 'b c u v h w -> b c (u h) (v w)', c = self.n_channels)
		x_rec = self.last_cov(x_rec)
		x_rec = x_rec.view(b, c, u, h, v, w).permute(0, 2, 4, 1, 3, 5).contiguous() 
		x_rec = InverseSpaAngDCT(x_rec).to('cuda:0')
		x_rec +=  x
		return x_rec 
	

class PixNet(nn.Module):
	def __init__(self, n_groups, n_blocks,  n_channels):
		super(PixNet, self).__init__()

		self.n_groups = n_groups
		self.n_blocks = n_blocks
		self.n_channels  = n_channels
		self.conv_first = nn.Conv2d(3, n_channels,  kernel_size=3, stride=1, dilation=1, padding=1, bias=True)

		Groups = [
			HGAG(self.n_blocks, n_channels) \
			for _ in range(self.n_groups)]
		self.Group = nn.Sequential(*Groups)

		self.conv_last = nn.Conv2d(n_channels, 3,  kernel_size=3, stride=1, dilation=1, padding=1, bias=True)

	def forward(self, x):

		b,c,u,v,h,w = x.shape
		out = rearrange(x, 'b c u v h w -> b u v c h w')
		out = rearrange(out, 'b u v c h w -> (b u v) c h w')

		out = self.conv_first(out)
		out = rearrange(out, '(b u v) c h w  -> b u v c h w ',b=b,u = u,v=v, c=self.n_channels,h=h,w =w )
		out = rearrange(out, 'b u v c h w   -> b c u v  h w ',b=b,u = u,v=v, c=self.n_channels,h=h,w =w )
		
		out = self.Group(out)
		out = rearrange(out, 'b c u v h w -> b u v c h w')
		out = rearrange(out, 'b u v c h w -> (b u v) c h w')

		out = self.conv_last(out)
		out = rearrange(out, '(b u v) c h w  -> b u v c h w ',b=b,u = u,v=v, c=3,h=h,w =w )
		out = rearrange(out, 'b u v c h w   -> b c u v  h w ',b=b,u = u,v=v, c=3,h=h,w =w )

		out += x
		return out
	
## Hybird Geometric-aware Attention Group
class HGAG(nn.Module):
	def __init__(self ,n_blocks, n_channels):
		super(HGAG, self).__init__()

		self.fea_block = make_layer(HFEM, n_channels, n_blocks)
		self.GAM = GAM(n_channels)
		
	def forward(self, x):
		out = self.fea_block(x)
		out = self.GAM(out)
		out += x
		return out

## Hybird Feature Exatraction Module
class HFEM(nn.Module):
	def __init__(self, n_channels):
		super(HFEM, self).__init__()
		
		self.RSAB = RSAB(n_channels)
		self.REB = REB(n_channels)
	
	def forward(self, x):
		out_SAB = self.RSAB(x)
		out_EAB = self.REB(x)
		out = out_SAB + out_EAB + x
		return out
	
# Residual Spa-Ang Feature Extraction Block
class RSAB(nn.Module):
	def __init__(self, n_channels):
		super(RSAB, self).__init__()
		
		self.conv_spa = torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
		self.conv_ang = torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
		self.act = nn.LeakyReLU(0.1,inplace=True)
		self.n_channels = n_channels
	
	def forward(self, x):
		b,c,u,v,h,w = x.shape
		out= self.act(self.conv_spa(x.permute(0,2,3,1,4,5).reshape(b*u*v,c,h,w))) #[64]
		out= self.act(self.conv_ang(out.reshape(b,u,v,self.n_channels,h,w).permute(0,4,5,3,1,2).reshape(b*h*w,self.n_channels,u,v))) #[64]
		out = out.reshape(b,h,w,self.n_channels,u,v).permute(0,3,4,5,1,2)
		return out

# Residual EPI Feature Extraction Block
class REB(nn.Module):
	def __init__(self, n_channels):
		super(REB, self).__init__()
		
		self.conv_epi_h =torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
		self.conv_epi_v =torch.nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=3,stride=1,padding=1,bias=True)
		self.act = nn.LeakyReLU(0.1,inplace=True)

		self.n_feat = n_channels
	
	def forward(self, x):
		b,c,u,v,h,w = x.shape

		out = x.permute(0, 3, 5, 1, 2, 4).contiguous()
		out = out.reshape(b * v * w, c, u, h)
		out = self.act(self.conv_epi_v(out))

		out = out.reshape(b, v, w, c, u, h).permute(0,4,5,3,1,2)
		out = out.reshape(b * u * h, c, v, w)  

		out = self.act(self.conv_epi_h(out))
		out = out.reshape(b, u, h, c, v, w).permute(0, 3, 1, 4, 2, 5).contiguous() 
		return out

# Geometric-aware attention module
class GAM(nn.Module):
	def __init__(self,in_channels, eps=1e-5):
		super(GAM, self).__init__()
	
		self.spa_att = JCSA_Module()
		self.ang_att = JCSA_Module()
		self.epi_h_att = JCSA_Module()
		self.epi_v_att = JCSA_Module()

		self.conv = nn.Conv2d(in_channels*4, in_channels,  kernel_size=3, stride=1, dilation=1, padding=1, bias=True)

	def forward(self,x):
		b,c,u,v,h,w = x.shape

		spa_input = x.permute(0,2,3,1,4,5).reshape(b*u*v,c,h,w)
		spa_out = self.spa_att(spa_input)  #[b*u*v,c,h,w]
		
		ang_input = x.permute(0,4,5,1,2,3).reshape(b*h*w,c,u,v)
		ang_out = self.ang_att(ang_input)
		ang_out = ang_out.reshape(b, h, w, c, u, v).permute(0, 4, 5, 3, 1, 2).reshape(b*u*v,c,h,w) #[b*u*v,c,h,w]

		epi_h_input = x.permute(0,3,5,1,2,4).reshape(b*v*w,c,u,h)
		epi_h_ouput = self.ang_att(epi_h_input)
		epi_h_ouput = epi_h_ouput.reshape(b, v, w, c, u, h).permute(0, 4, 1, 3, 5, 2).reshape(b*u*v,c,h,w) #[b*u*v,c,h,w]

		epi_v_input = x.permute(0,3,5,1,2,4).reshape(b*u*h,c,v,w)
		epi_v_ouput = self.ang_att(epi_v_input)
		epi_v_ouput = epi_v_ouput.reshape(b, u, h, c, v, w).permute(0, 1, 4, 3, 2, 5).reshape(b*u*v,c,h,w) #[b*u*v,c,h,w]

		merged = torch.cat((spa_out, ang_out, epi_h_ouput, epi_v_ouput), 1)
		out = self.conv(merged)
		out = out.reshape(b, u, v, c, h, w).permute(0, 3, 1, 2, 4, 5).contiguous() #[b,c,u,v,h,w]

		return out

class JCSA_Module(nn.Module):
	""" Joint Channel-Spatial Attention Module"""
	def __init__(self,eps=1e-5):
		super(JCSA_Module, self).__init__()
		
		self.conv3D = nn.Conv3d(1, 1, 3, 1, 1)
		self.epsilon = eps

		self.gamma = nn.Parameter(torch.zeros(1))
		self.beta = nn.Parameter(torch.zeros(1))
		self.tanh = nn.Tanh()

	def forward(self,x):
		b, c, height, width = x.size()
		out = x.unsqueeze(1)
		out = self.conv3D(out)
		norm = out/((out.pow(2).mean((3,4),keepdim=True) + self.epsilon).pow(0.5))
		attention = self.tanh(self.gamma*norm + self.beta)
		attention = attention.view(b, -1, height, width)
		x = x * attention + x
		return x


class ResBlock3D(nn.Module):
	def __init__(self, n_feats):
		super(ResBlock3D, self).__init__()
		m = []
		m.append(nn.PReLU())
		m.append(nn.Conv3d(n_feats, n_feats, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=True))
		self.body = nn.Sequential(*m)
	
	def forward(self, x):
		res = self.body(x)
		res += x
		return res
	

class Upsampler(nn.Sequential):
	def __init__(self, scale, n_feat,kernel_size, stride, dilation, padding,  bn=False, act=False, bias=True):

		m = []
		if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
			for _ in range(int(math.log(scale, 2))):
				m.append(nn.Conv2d(n_feat, 4 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
				m.append(nn.PixelShuffle(2))
				if bn: m.append(nn.BatchNorm2d(n_feat))
				if act: m.append(act())
		elif scale == 3:
			m.append(nn.Conv2d(n_feat, 9 * n_feat, kernel_size=kernel_size,stride=stride,dilation=dilation, padding=padding, bias=True))
			m.append(nn.PixelShuffle(3))
			if bn: m.append(nn.BatchNorm2d(n_feat))
			if act: m.append(act())
		else:
			raise NotImplementedError

		super(Upsampler, self).__init__(*m)


def make_layer(block, nf, n_layers ):
	layers = []
	for _ in range(n_layers):
		layers.append(block(nf))
	return nn.Sequential(*layers)	

def get_freq_position(view_n):
	start_position_list = []
	for i in range(view_n):
		start_position_list.append(([i], [0]))
	for j in range(1, view_n):
		start_position_list.append(([view_n - 1], [j]))
	for item in start_position_list:
		while item[0][0] > 0 and item[1][0] < view_n - 1:
			item[0].insert(0, item[0][0] - 1)
			item[1].insert(0, item[1][0] + 1)
	return start_position_list


def SpaAngDCT(x):
	b,u,v,c, h,w = x.shape
	out = rearrange(x, 'b u v c h w -> (b u v) c h w')
	out = dct_2d(out)
	out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v, h=h, w=w)
	out = rearrange(out, 'b u v c h w -> (b h w) c u v')
	out = dct_2d(out)
	out = rearrange(out, '(b h w) c u v -> b u v c h w', b=b, h=h, w=w)
	return out		

def InverseSpaAngDCT(x):
	b,u,v,c, h,w = x.shape
	out = rearrange(x, 'b u v c h w -> (b h w) c u v')
	out = idct_2d(out)
	out = rearrange(out, '(b h w) c u v -> b u v c h w', b=b, h=h, w=w)
	out = rearrange(out, 'b u v c h w -> (b u v) c h w')
	out = idct_2d(out)
	out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v, h=h, w=w)
	return out		


def MacPI2SAI(x, angRes):
	out = []
	for i in range(angRes):
		out_h = []
		for j in range(angRes):
			out_h.append(x[:, :, i::angRes, j::angRes])
		out.append(torch.cat(out_h, 3))
	out = torch.cat(out, 2)
	return out


def SAI2MacPI(x, angRes):
	b, c, hu, wv = x.shape
	h, w = hu // angRes, wv // angRes
	tempU = []
	for i in range(h):
		tempV = []
		for j in range(w):
			tempV.append(x[:, :, i::h, j::w])
		tempU.append(torch.cat(tempV, dim=3))
	out = torch.cat(tempU, dim=2)
	return out



def weights_init(m):
	pass

class get_loss(nn.Module):
	def __init__(self, args):
		super(get_loss, self).__init__()
		self.criterion_Loss = torch.nn.L1Loss()

	def forward(self, SR, HR, criterion_data=[]):
		loss = self.criterion_Loss(SR, HR)

		return loss

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from fvcore.nn import FlopCountAnalysis

    # Training settings
    parser = argparse.ArgumentParser(description="Hybrid Light Field Restoration")
    parser.add_argument("--model_name", type=str, default='LFWXformerV3', help="Path for saving training log ")
    parser.add_argument("--learningRate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--sigma", type=int, default=50, help="Noise level")
    parser.add_argument("--angResolution", type=int, default=5, help="The angular resolution of original LF")

    parser.add_argument("--batchSize", type=int, default=4, help="Batch size")
    parser.add_argument("--sampleNum", type=int, default=70, help="The number of LF in training set")
    parser.add_argument("--patchSize", type=int, default=32, help="The size of croped LF patch")
    parser.add_argument("--patch_size", type=int, default=32, help="patch size for train")
    parser.add_argument("--epochNum", type=int, default=10000, help="The number of epoches")
    parser.add_argument("--num_steps", type=int, default=2500, help="The number of step size reduce learning rate")

    parser.add_argument("--summaryPath", type=str, default='./log/', help="Path for saving training log ")
    parser.add_argument("--saveCheckpointsDir", type=str, default='./checkpoints/',
                        help="Path for saving training log ")
    parser.add_argument("--dataPath", type=str, default='./datasets/train_noiseLevel_10-20-50_4-11_color_5x5.mat',
                        help="Path for loading training data ")

    parser.add_argument('--resume', type=str, default=False, help='Resume training from saved checkpoint(s).')
    parser.add_argument('--modelPath', type=str, default='./checkpoints/LF_WXformer_50/model_sigma_50_best.pth',
                        help='Resume training from saved checkpoint(s).')
    parser.add_argument('--optimizerPath', type=str, default='./checkpoints/LF_WXformer_50/optimizer_best.pth',
                        help='Resume optimizer from saved optimizer(s).')
    parser.add_argument("--scale_factor", type=int, default=1, help="4, 2")
    parser.add_argument("--channels", type=int, default=48, help="channels , embed_dim for transformer —— C")
    parser.add_argument("--angRes", type=int, default=5, help="angular resolution")
    parser.add_argument("--attn_drop_rate", type=float, default=0.1, help="drop rate for attention calculation")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="common drop rate")
    parser.add_argument("--drop_path_rate", type=float, default=0.2, help="stochastic depth decay rule")
    parser.add_argument("--ang_num_heads", type=int, default=4,
                        help="number of multi heads for angular transformer —— P")
    parser.add_argument("--spa_num_heads", type=int, default=4,
                        help="number of multi heads for spatial transformer —— P")
    parser.add_argument("--ang_mlp_ratio", type=int, default=4, help="scale ratio in MLP for angular transformer")
    parser.add_argument("--spa_mlp_ratio", type=int, default=4, help="scale ratio in MLP for spatial transformer")
    parser.add_argument("--depth", type=int, default=4, help="number of spatial-angular transformer encoder —— N")
    parser.add_argument("--ang_sr_ratio", type=int, default=1, help="reduce patches scale for angular transformer")
    parser.add_argument("--spa_sr_ratio", type=int, default=2, help="reduce patches scale for spatial transformer —— S")
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
    net = HLFRN(opt).cuda()
    input = torch.randn(1,5,5, 3, 32, 32).cuda()
    flops = FlopCountAnalysis(net, input)
    print("Flops: ", flops.total())
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.4fM" % (total / 1e6))
