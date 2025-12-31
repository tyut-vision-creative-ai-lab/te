from __future__ import print_function, division
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
# Ignore warnings
import warnings
from scipy import sparse
import random
import numpy as np
warnings.filterwarnings("ignore")
plt.ion()
from math import exp
from utils.DeviceParameters import to_device


#Initiate parameters in model 
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('ConvTranspose2d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		torch.nn.init.constant_(m.bias.data, 0.0)
	if classname.find('Conv3d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('ConvTranspose3d') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('Linear') != -1:
		torch.nn.init.xavier_uniform_(m.weight.data)
		torch.nn.init.constant_(m.bias.data, 0.0)

	
def SetupSeed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


 
def ExtractPatch(LF, noiLF, patchSize):
	u,v,H,W,c=LF.shape
	indx=random.randrange(0,H-patchSize,8)
	indy=random.randrange(0,W-patchSize,8)
	# indx=random.randint(0,H-patchSize[2])
	# indy=random.randint(0,W-patchSize[3])
	LF=LF[:,:,np.newaxis,
				   indx:indx+patchSize,
				   indy:indy+patchSize,:]
				   
	noiLF=noiLF[:,:,np.newaxis,
				   indx:indx+patchSize,
				   indy:indy+patchSize,:]
				   
	return LF,noiLF #[u v c x y,c]

 
def ResizeLF(lf,scale_factor):
	u,v,x,y,c=lf.shape
	resizedLF=np.zeros((u,v,int(scale_factor*x),int(scale_factor*y),c),dtype=np.int)
	for ind_u in range(u):
		for ind_v in range(v):
			view=lf[ind_u,ind_v,:,:,:]
			resizedView=cv2.resize(view, (int(scale_factor*x),int(scale_factor*y)), interpolation=cv2.INTER_CUBIC)
			resizedLF[ind_u,ind_v,:,:,:]=resizedView.reshape(int(scale_factor*x),int(scale_factor*y),-1)
	return resizedLF

def CropLF(lf,patchSize, stride): #lf [b,u,v,x,y,c]
    b,u,v,x,y,c=lf.shape
    numX=len(range(0,x-patchSize,stride))
    numY=len(range(0,y-patchSize,stride))

    lfStack=torch.zeros(b,numX*numY,u,v,patchSize,patchSize,c)

    indCurrent=0
    for i in range(0,x-patchSize,stride):
        for j in range(0,y-patchSize,stride):
            lfPatch=lf[:,:,:,i:i+patchSize,j:j+patchSize,:]
            lfStack[:,indCurrent,:,:,:,:,:]=lfPatch
            indCurrent=indCurrent+1

    return lfStack, [numX,numY] #lfStack [b,n,u,v,x,y,c] 


def MergeLF(lfStack, coordinate, overlap):
    b,n,u,v,x,y,c=lfStack.shape
    
    xMerged=coordinate[0]*x-coordinate[0]*overlap
    yMerged=coordinate[1]*y-coordinate[1]*overlap

    lfMerged=torch.zeros(b,u,v,xMerged,yMerged,c)
    for i in range(coordinate[0]):
        for j in range(coordinate[1]):
            lfMerged[:,
                     :,
                     :,
                     i*(x-overlap):(i+1)*(x-overlap),
                     j*(y-overlap):(j+1)*(y-overlap),
                     :]=lfStack[:,
                                i*coordinate[1]+j,
                                :,
                                :,
                                overlap//2:-overlap//2,
                                overlap//2:-overlap//2,
                                :] 
            
    return lfMerged # [b,u,v,x,y,c]




def ComptPSNR(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    
    # PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    # img1 = img1.clip(0,1)
    # img2 = img2.clip(0,1)
    # return metrics.peak_signal_noise_ratio(img1, img2)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))