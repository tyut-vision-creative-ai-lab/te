from ctypes import sizeof
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import scipy.io as scio

# from RefNet_pfe_ver0 import RefNet
# from Functions import weights_init

import warnings
from scipy import sparse
import random
import numpy as np
warnings.filterwarnings("ignore")
plt.ion()

class StageBlock(torch.nn.Module):
	def __init__(self, opt, bs):
		super(StageBlock,self).__init__()
		# Regularization sub-network
		self.refnet=RefNet(opt,bs)
		self.refnet.apply(weights_init)

	def forward(self, mResidual,sampleLF,epoch):
		lfRedisual = self.refnet(mResidual,sampleLF,epoch) #[b,uv,c,x,y]
		return lfRedisual

def CascadeStages(block, opt, bs):
	blocks = torch.nn.ModuleList([])
	for _ in range(opt.stageNum):
		blocks.append(block(opt, bs))
	return blocks       

class PFE(torch.nn.Module):
	def __init__(self,opt):
		super(PFE,self).__init__()
		self.kernelSize=[opt.angResolution,opt.angResolution]
		self.angularnum = opt.angResolution

		# # if opt.sasLayerNum is None:
		# opt.sasLayerNum = 6
		
		# # if opt.temperature_1 is None:
		# opt.temperature_1 = 1

		# # if opt.temperature_2 is None:
		# opt.temperature_2 = 1

		# # if opt.component_num is None:
		# opt.component_num = 6
		
		# # if opt.stageNum is None:
		# opt.stageNum = 3
		
		# # if opt.channelNum is None:
		# opt.channelNum = 3
		

		# global average
		self.avglf = torch.nn.AvgPool2d(kernel_size=self.kernelSize,stride = None, padding = 0)   
		self.proj_init = torch.nn.Conv2d(in_channels=3,out_channels=7,kernel_size=self.kernelSize,bias=False)
		torch.nn.init.xavier_uniform_(self.proj_init.weight.data)

		self.initialRefnet=RefNet(opt, True)
		self.initialRefnet.apply(weights_init) 
		# Iterative stages
		self.iterativeRecon = CascadeStages(StageBlock, opt, False)
 
	def forward(self, noiself, epoch):
		b,u,v,c,x,y=noiself.shape
		avgLF = self.avglf(noiself.permute(0,3,4,5,1,2).reshape(b,x*y*c,u,v))
		avgLF = avgLF.reshape(b,x,y,c).permute(0,3,1,2)

		projLF = self.proj_init(noiself.permute(0,4,5,3,1,2).reshape(b*x*y,c,u,v))
		projLF = projLF.reshape(b,x,y,7).permute(0,3,1,2)
		sampleLF = torch.cat([avgLF,projLF],1)
		# Initialize LF 
		out = self.initialRefnet(noiself,sampleLF,epoch) 

		# Reconstructing iteratively
		for stage in self.iterativeRecon:
			avgLF = self.avglf(noiself.permute(0,3,4,5,1,2).reshape(b,x*y*c,u,v))
			avgLF = avgLF.reshape(b,x,y,c).permute(0,3,1,2)

			projLF = self.proj_init(noiself.permute(0,4,5,3,1,2).reshape(b*x*y,c,u,v))
			projLF = projLF.reshape(b,x,y,7).permute(0,3,1,2)
			sampleLF = torch.cat([avgLF,projLF],1)

			# avgLF = self.avglf(out.permute(0,3,4,1,2).reshape(b,x*y,u,v))
			# avgLF = avgLF.reshape(b,x,y,1).permute(0,3,1,2) 
			# projLF = self.proj_init(out.permute(0,3,4,1,2).reshape(b*x*y,1,u,v))
			# projLF = projLF.reshape(b,x,y,7).permute(0,3,1,2)
			# sampleLF = torch.cat([avgLF,projLF],1)
			out = out + stage(out,sampleLF,epoch)
		return out 





class ChannelSELayer3D(nn.Module):
	def __init__(self, num_channels, reduction_ratio=2):
		"""
		:param num_channels: No of input channels
		:param reduction_ratio: By how much should the num_channels should be reduced
		"""
		super(ChannelSELayer3D, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool3d(1)
		num_channels_reduced = num_channels // reduction_ratio
		self.reduction_ratio = reduction_ratio
		self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
		self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, input_tensor):
		"""
		:param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
		:return: output tensor
		"""
		batch_size, num_channels, D, H, W = input_tensor.size()
		# Average along each channel
		squeeze_tensor = self.avg_pool(input_tensor)

		# channel excitation
		fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
		fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

		output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

		return output_tensor

class ChannelSELayer(nn.Module):
	def __init__(self, num_channels, reduction_ratio=2):
		"""
		:param num_channels: No of input channels
		:param reduction_ratio: By how much should the num_channels should be reduced
		"""
		super(ChannelSELayer, self).__init__()
		num_channels_reduced = num_channels // reduction_ratio
		self.reduction_ratio = reduction_ratio
		self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
		self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, input_tensor):
		"""
		:param input_tensor: X, shape = (batch_size, num_channels, H, W)
		:return: output tensor
		"""
		batch_size, num_channels, H, W = input_tensor.size()
		# Average along each channel
		squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

		# channel excitation
		fc_out_1 = self.relu(self.fc1(squeeze_tensor))
		fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

		a, b = squeeze_tensor.size()
		output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
		return output_tensor


class SpatialSELayer(nn.Module):
	def __init__(self, num_channels):
		"""
		:param num_channels: No of input channels
		"""
		super(SpatialSELayer, self).__init__()
		self.conv = nn.Conv2d(num_channels, 1, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, input_tensor, weights=None):
		"""
		:param weights: weights for few shot learning
		:param input_tensor: X, shape = (batch_size, num_channels, H, W)
		:return: output_tensor
		"""
		# spatial squeeze
		batch_size, channel, a, b = input_tensor.size()

		if weights is not None:
			weights = torch.mean(weights, dim=0)
			weights = weights.view(1, channel, 1, 1)
			out = F.conv2d(input_tensor, weights)
		else:
			out = self.conv(input_tensor)
		squeeze_tensor = self.sigmoid(out)

		# spatial excitation
		# print(input_tensor.size(), squeeze_tensor.size())
		squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
		output_tensor = torch.mul(input_tensor, squeeze_tensor)
		#output_tensor = torch.mul(input_tensor, squeeze_tensor)
		return output_tensor

class Conv_spa(nn.Module):
	def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
		super(Conv_spa, self).__init__()
		self.op = nn.Sequential(
			nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
			nn.ReLU(inplace = True)
		)
	def forward(self,x):   
		N,c,uv,h,w = x.shape
		x = x.permute(0,2,1,3,4).reshape(N*uv,c,h,w)  
		out = self.op(x)
		#print(out.shape)
		out = out.reshape(N,uv,32,h,w).permute(0,2,1,3,4)
		return out

class Conv_ang(nn.Module):
	def __init__(self, C_in, C_out, kernel_size, stride, padding, angular, bias):
		super(Conv_ang, self).__init__()
		self.angular = angular
		self.op = nn.Sequential(
			nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
			nn.ReLU(inplace = True)
		)
	def forward(self,x):    
		N,c,uv,h,w = x.shape
		x = x.permute(0,3,4,1,2).reshape(N*h*w,c,self.angular,self.angular)   
		out = self.op(x)
		out = out.reshape(N,h,w,32,uv).permute(0,3,4,1,2)
		return out

class Conv_epi_h(nn.Module):
	def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
		super(Conv_epi_h, self).__init__()
		self.op = nn.Sequential(
			nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
			nn.ReLU(inplace = True)
		)
	def forward(self,x): 
		N,c,uv,h,w = x.shape
		x = x.permute(0,3,1,2,4).reshape(N*h,c,uv,w)
		out = self.op(x)
		out = out.reshape(N,h,32,uv,w).permute(0,2,3,1,4)
		return out

class Conv_epi_v(nn.Module):
	def __init__(self, C_in, C_out, kernel_size, stride, padding, bias):
		super(Conv_epi_v, self).__init__()
		self.op = nn.Sequential(
			nn.Conv2d(C_in, C_out, kernel_size, stride = stride, padding = padding, bias = bias),
			nn.ReLU(inplace = True)
		)
	def forward(self,x):
		N,c,uv,h,w = x.shape
		x = x.permute(0,4,1,2,3).reshape(N*w,c,uv,h)
		out = self.op(x)
		out = out.reshape(N,w,32,uv,h).permute(0,2,3,4,1)
		return out
	

class Autocovnlayer(nn.Module):
	def __init__(self,dence_num,component_num,angular,bs):
		super(Autocovnlayer, self).__init__()
		self.dence_num = dence_num
		self.component_num = component_num
		self.dence_weight = nn.Parameter(torch.rand(dence_num),requires_grad=True)
		self.component_weight = nn.Parameter(torch.rand(component_num),requires_grad=True)
		self.angular = angular
		self.kernel_size = 3

		self.naslayers = nn.ModuleList([
		   Conv_spa(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
		   Conv_ang(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, angular = self.angular, bias = bs),
		   Conv_epi_h(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs),
		   Conv_epi_v(C_in = 64, C_out = 32, kernel_size = self.kernel_size, stride = 1, padding = 1, bias = bs)
		])
		self.Conv_all = nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=1, padding=1, bias = bs)
		self.softmax1 = nn.Softmax(1)
		self.softmax0 = nn.Softmax(0) 
		self.Conv_mixdence = nn.Conv2d(in_channels = 64*self.dence_num, out_channels=64, kernel_size=1, stride=1, padding=0, bias = False)
		self.Conv_mixnas = nn.Conv2d(in_channels = 32*4, out_channels=64, kernel_size=1, stride=1, padding=0, bias = False)     ## 1*1 paddding!!
		self.relu = nn.ReLU(inplace=True)

	def forward(self,x,temperature_1,temperature_2):
		x = torch.stack(x,dim = 0) 
		[fn, N, C, uv, h, w] = x.shape 
		## generate 2 noise    dim of noise !!!       
		dence_weight = self.dence_weight.clamp(0.02,0.98)
		dence_weight = dence_weight[:,None,None,None,None,None] 
		component_weight = self.component_weight.clamp(0.02,0.98)
		component_weight = component_weight[:,None,None,None,None,None]
		
		noise_dence_r1 = torch.rand((self.dence_num,N))[:,:,None,None,None,None].cuda()   
		noise_dence_r2 = torch.rand((self.dence_num,N))[:,:,None,None,None,None].cuda()
		noise_dence_logits = torch.log(torch.log(noise_dence_r1) / torch.log(noise_dence_r2))
		dence_weight_soft = torch.sigmoid((torch.log(dence_weight / (1 - dence_weight)) + noise_dence_logits) / temperature_1)
		
		noise_component_r1 = torch.rand((self.component_num,N))[:,:,None,None,None,None].cuda() 
		noise_component_r2 = torch.rand((self.component_num,N))[:,:,None,None,None,None].cuda()
		noise_component_logits = torch.log(torch.log(noise_component_r1) / torch.log(noise_component_r2))
		component_weight_gumbel = torch.sigmoid((torch.log(component_weight / (1 - component_weight)) + noise_component_logits) / temperature_2)
		
		x = x * dence_weight_soft
		x = x.permute([1,3,0,2,4,5]).reshape([N*uv,fn*C,h,w])    
		x = self.relu(self.Conv_mixdence(x))                               
		x_mix = x.reshape([N,uv,C,h,w]).permute([0,2,1,3,4]) 
		layer_label = 0
		nas = []
		for layer in self.naslayers:
			nas_ = layer(x_mix)
			nas.append(nas_)
		
		nas = torch.stack(nas,dim = 0)
		nas = nas * component_weight_gumbel 
		nas = nas.permute([1,3,0,2,4,5]).reshape([N*uv,self.component_num*32,h,w])
		nas = self.relu(self.Conv_mixnas(nas))           
		####### add a spa conv  #######
		nas = self.Conv_all(nas)
		nas = nas.reshape(N,uv,C,h,w).permute(0,2,1,3,4)
		nas = self.relu(nas + x_mix)
		return nas

def make_autolayers(opt,bs):
	layers = []
	for i in range( opt.sasLayerNum ):
		layers.append(Autocovnlayer(i+1, opt.component_num, opt.angResolution, bs))
	return nn.Sequential(*layers)
	
class RefNet(nn.Module):
	def __init__(self, opt, bs):        
		super(RefNet, self).__init__()
		self.angResolution = opt.angResolution
		self.lfNum = opt.angResolution * opt.angResolution
		self.epochNum = opt.epochNum
		self.temperature_1 = opt.temperature_1
		self.temperature_2 = opt.temperature_2
		
		self.relu = nn.ReLU(inplace=True)
		self.conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
		self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, stride=1, padding=1, bias = bs)
		self.dence_autolayers = make_autolayers(opt,bs)
		self.sptialSE = SpatialSELayer(32)
		self.channelSE = ChannelSELayer3D(32*2,2)
		self.syn_conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias = bs)

	def forward(self,input,sampleLF,epoch):
		N,u,v,c,h,w = input.shape 
		# _,c,_,_ = sampleLF.shape
		if epoch <= 3800: 
			temperature_1 = self.temperature_1 * (1 - epoch / 4000)
			temperature_2 = self.temperature_2 * (1 - epoch / 4000)
		else:
			temperature_1 = 0.05
			temperature_2 = 0.05
		# feature extraction sample
		feat1 = self.relu(self.conv1(sampleLF))
		feat1 = self.sptialSE(feat1).unsqueeze(2)
		feat1 = feat1.expand(-1,-1,u*v,-1,-1)
		# feature extraction LF
		feat2 = input.reshape(N*u*v,3,h,w) 
		feat2 = self.relu(self.conv0(feat2)) 
		feat2 = feat2.reshape(N,u*v,32,h,w).permute(0,2,1,3,4)   
		feat = torch.cat([feat2,feat1],1)
		feat = self.channelSE(feat)
		feat = [feat]
		for index, layer in enumerate(self.dence_autolayers):
			feat_ = layer(feat,temperature_1,temperature_2)
			feat.append(feat_)
		feat = feat[-1].permute(0,2,1,3,4)
		feat = self.syn_conv2(feat.reshape(N*self.lfNum,64,h,w))
		out = feat.reshape(N,u,v,c,h,w)
		return out
	

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		#torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('ConvTranspose2d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		#torch.nn.init.constant_(m.bias.data, 0.0)
	if classname.find('Conv3d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		#torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('ConvTranspose3d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		#torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		#torch.nn.init.constant_(m.bias.data, 0.0)

def SetupSeed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True

def ExtractPatch(lf,noiself, H, W, patchSize):
	indx=random.randrange(0,H-patchSize,8)
	indy=random.randrange(0,W-patchSize,8)
#     indc=random.randint(0,2)

	lfPatch=lf[:,:,indx:indx+patchSize, indy:indy+patchSize]
	noiselfPatch=noiself[:,:,indx:indx+patchSize,indy:indy+patchSize]
	return lfPatch,noiselfPatch #[u v x y] 
	
def ResizeLF(lf,scale_factor):
	u,v,x,y,c=lf.shape
	resizedLF=np.zeros((u,v,int(scale_factor*x),int(scale_factor*y),c),dtype=np.int)
	for ind_u in range(u):
		for ind_v in range(v):
			view=lf[ind_u,ind_v,:,:,:]
			resizedView=cv2.resize(view, (int(scale_factor*x),int(scale_factor*y)), interpolation=cv2.INTER_CUBIC)
			resizedLF[ind_u,ind_v,:,:,:]=resizedView.reshape(int(scale_factor*x),int(scale_factor*y),-1)
	return resizedLF


def CropLF(lf,patchSize, overlap): #lf [b,u,v,c,x,y]
	b,u,v,x,y=lf.shape
	numX=0
	numY=0
	while (patchSize-overlap)*numX < x:
		numX = numX + 1 
	while (patchSize-overlap)*numY < y:
		numY = numY + 1 
	lfStack=torch.zeros(b,numX*numY,u,v,patchSize,patchSize)
	indCurrent=0
	for i in range(numX):
		for j in range(numY):
			if (i != numX-1)and(j != numY-1): 
				lfPatch=lf[:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
			elif (i != numX-1)and(j == numY-1): 
				lfPatch=lf[:,:,:,i*(patchSize-overlap):(i+1)*patchSize-i*overlap,-patchSize:]
			elif (i == numX-1)and(j != numY-1): 
				lfPatch=lf[:,:,:,-patchSize:,j*(patchSize-overlap):(j+1)*patchSize-j*overlap]
			else : 
				lfPatch=lf[:,:,:,-patchSize:,-patchSize:]
			# print(numX,numY,i,j,lfPatch.shape)
			lfStack[:,indCurrent,:,:,:,:]=lfPatch
			indCurrent=indCurrent+1
	return lfStack, [numX,numY] #lfStack [b,n,u,v,c,x,y] 


def MergeLF(lfStack, coordinate, overlap, x, y):
	b,n,u,v,patchSize,_=lfStack.shape
	lfMerged=torch.zeros(b,u,v,x-overlap,y-overlap)
	for i in range(coordinate[0]):
		for j in range(coordinate[1]):
			if (i != coordinate[0]-1)and(j != coordinate[1]-1): 
				lfMerged[:,:,:,
				i*(patchSize-overlap):(i+1)*(patchSize-overlap),
				j*(patchSize-overlap):(j+1)*(patchSize-overlap)]=lfStack[:,i*coordinate[1]+j,:,:,
																			overlap//2:-overlap//2,
																			overlap//2:-overlap//2] 
			elif (i == coordinate[0]-1)and(j != coordinate[1]-1): 
				lfMerged[:,:,:,i*(patchSize-overlap):,
								 j*(patchSize-overlap):(j+1)*(patchSize-overlap)]=lfStack[:,i*coordinate[1]+j,:,:,
																							-((x-overlap)-i*(patchSize-overlap))-overlap//2:-overlap//2,
																							overlap//2:-overlap//2]            
			elif (i != coordinate[0]-1)and(j == coordinate[1]-1): 
				lfMerged[:,:,:,i*(patchSize-overlap):(i+1)*(patchSize-overlap),
								 j*(patchSize-overlap):]=lfStack[:,i*coordinate[1]+j,:,:,
																	overlap//2:-overlap//2,
																	-((y-overlap)-j*(patchSize-overlap))-overlap//2:-overlap//2]
			else: 
				lfMerged[:,:,:,i*(patchSize-overlap):,
								 j*(patchSize-overlap):]=lfStack[:,i*coordinate[1]+j,:,:,
																	-((x-overlap)-i*(patchSize-overlap))-overlap//2:-overlap//2,
																	-((y-overlap)-j*(patchSize-overlap))-overlap//2:-overlap//2]    
	return lfMerged # [b,u,v,c,x,y]

def ComptPSNR(img1, img2):
	mse = np.mean( (img1 - img2) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 1.0
	
	if mse > 1000:
		return -100
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def rgb2ycbcr(rgb):
	m = np.array([[ 65.481, 128.553, 24.966],
				  [-37.797, -74.203, 112],
				  [ 112, -93.786, -18.214]])
	shape = rgb.shape
	if len(shape) == 3:
		rgb = rgb.reshape((shape[0] * shape[1], 3))
	ycbcr = np.dot(rgb, m.transpose() / 255.)
	ycbcr[:,0] += 16.
	ycbcr[:,1:] += 128.
	return ycbcr.reshape(shape)
