import torch
from torch import nn 
from torch.nn import functional as F
import lightning as L
import dataclasses
from typing import Tuple
from einops import reduce,rearrange
from torchsummary import summary
from pytorch_lightning.utilities.model_summary import ModelSummary
from einops import rearrange,reduce

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb"""

class EEGNet(L.LightningModule):
	def __init__(self,c,do):
		super(EEGNet, self).__init__()
		self.T = 120
		
		# Layer 1
		self.conv1 = nn.Conv2d(1, 16, (c, 64), padding = 0)
		self.batchnorm1 = nn.BatchNorm2d(16, False)
		
		# Layer 2
		self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
		self.conv2 = nn.Conv2d(1, 4, (2, 32))
		self.batchnorm2 = nn.BatchNorm2d(4, False)
		self.pooling2 = nn.MaxPool2d(2, 4)
		
		# Layer 3
		self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
		self.conv3 = nn.Conv2d(4, 4, (8, 4))
		self.batchnorm3 = nn.BatchNorm2d(4, False)
		self.pooling3 = nn.MaxPool2d((2, 4))
		self.out_proj = nn.Linear(do,2)

	def forward(self, x):
		# Layer 1
		x = F.elu(self.conv1(x))
		x = self.batchnorm1(x)
		x = F.dropout(x, 0.25)
		x = rearrange(x,"b c h w ->b h c w")

		# Layer 2
		x = self.padding1(x)
		x = F.elu(self.conv2(x))
		x = self.batchnorm2(x)
		x = F.dropout(x, 0.25)
		x = self.pooling2(x)
		
		# Layer 3
		x = self.padding2(x)
		x = F.elu(self.conv3(x))
		x = self.batchnorm3(x)
		x = F.dropout(x, 0.25)
		x = self.pooling3(x)
		x = rearrange(x,"b d1 d2 t -> b (d1 d2 t)")
		x = self.out_proj(x)
		return x
	
	def classify(self,x):
		x = rearrange(x,"b d t -> b 1 d t")
		return self.forward(x)
	
if __name__ == "__main__":

	model = EEGNet(2,224)

	print(ModelSummary(model))
	 
	model.to("cuda")

	summary(model,(1,2,512))

	x = torch.rand(2,1,2,512).to("cuda")
	y = model(x)
	print(y.shape)