import numpy as np
import pandas as pd
import os
import scipy.io
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
# import pywt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from lightning import Fabric
from pytorch_lightning.utilities.model_summary import ModelSummary
import lightning as L
from einops import repeat

import sys

sys.path.append("../../../motor-imagery-classification-2024/")

from classification.loaders import EEGDataset,load_data
from models.unet.eeg_unets import Unet,UnetConfig, BottleNeckClassifier, Unet1D
from classification.classifiers import DeepClassifier , SimpleCSP
from classification.loaders import subject_dataset
from ntd.networks import SinusoidalPosEmb
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess

@torch.jit.script
def double_inputs(x:torch.Tensor,
				  t:torch.Tensor,
				  cond:torch.Tensor):
	
	x = torch.cat([x,x],0)
	t = torch.cat([t,t],0)
	cond = torch.cat([cond,0*cond],0)
	return x,t,cond

@torch.jit.script
def dedouble_outputs(x:torch.Tensor,
					 w:float):
	
	conditional = x[0:len(x)//2]
	unconditinoal = x[len(x)//2:]
	return (1+w)*conditional-w*unconditinoal

from lightning import LightningDataModule


from models.unet.eeg_unets import (UnetConfig,
								   Encode,
								   Decode,
								   ConvConfig,
								   EncodeConfig,
								   DecodeConfig,
								   Convdown)


class DiffusionUnet(L.LightningModule):

	"""
	base Unet model with adaptable topology and dimension in nnUnet style.
	
	Attributes:
		config: configuration for Unet
	
	"""

	def __init__(self,
				 config: UnetConfig,
				 classifier: L.LightningDataModule,
				 time_dim=12,
				 ):
		
		super().__init__()

		self.time_dim = 12
		self.time_embbeder = SinusoidalPosEmb(time_dim)
		self.class_dim = 1
		self.signal_length = config.input_shape
		self.input_features = [config.starting_channels]
		self.signal_channel = config.input_channels
		size = torch.tensor(config.input_shape)

		self.class_embed_product = nn.Linear(1,config.input_channels,
									   bias=True)
		self.class_embed_addition = nn.Linear(1,config.input_channels,
									   bias=False)

		"""
		possible input shapes:
		1. N x D x L
			starting with N examples D=2 (2 electrodes) with length L (time)
			going to N x 32 x L after 1st layer
			
		2. N X D x C x L
			starting with N examples D = 1 (one feature) with 2 electrodes with lenght L
			going to N x 32 x 2 x L

		3. N x D x C x F x L
			N examples D =1 feature 2 electrodes x frequency x L

		4. N x D x F x L
			we don't do the distinction between channels and features but add a frequency dimension
		"""

		# can't divice 0-d tensor
		get_min = lambda x: min(x) if len(size.shape)>0 else x

		while (get_min(size) > 8) & (len(self.input_features)<=6):
			if 2*self.input_features[-1] < config.max_channels:
				self.input_features.append(2*self.input_features[-1])
			else:
				self.input_features.append(config.max_channels)
			size = size/config.pool_fact

		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()
		self.out_shape = [get_min(size),self.input_features[-1]]

		# input_channels = [32, 64, 128, 256,256....]

		self.auxiliary_clf = classifier

		self.base_conv_config = ConvConfig(
			input_channels=1,
			output_channels=1,
			conv_op=config.conv_op,
			norm_op=config.norm_op,
			non_lin=config.non_lin,
			groups=config.conv_group,
			padding=config.conv_padding,
			kernel_size=config.conv_kernel,
			pool_fact=config.pool_fact,
			pool_op=config.pool_op,
			residual=config.residual,
			p_drop=config.conv_pdrop
		)

		self.base_decode_config = DecodeConfig(
			x_channels=1,
			g_channels=1,
			output_channels=1,
			up_conv=config.up_op,
			groups=config.deconv_group,
			padding=config.deconv_padding,
			kernel_size=config.deconv_kernel,
			stride=config.deconv_stride,
			conv_config=self.base_conv_config
		)

		input_channels = config.input_channels + self.time_dim

		for idx,i in enumerate(self.input_features[:-1]):
			encode_config = self.base_conv_config.new_shapes(input_channels,i)
			self.encoder.append(Encode(encode_config))
			input_channels = i

		bottleneck_conv_config = self.base_conv_config.new_shapes(
			input_channels=self.input_features[-2],
			ouput_channels=self.input_features[-1]
		)

		self.middle_conv = Convdown(bottleneck_conv_config)

		output_features = self.input_features[::-1]

		for i in range(len(self.input_features)-1):

			decode_config = self.base_decode_config.new_shapes(
				x_channels=output_features[i+1],
				g_channels=output_features[i],
				output_channels=output_features[i+1]
			)

			self.decoder.append(Decode(decode_config))

		self.output_conv = config.conv_op(config.starting_channels,config.input_channels,1)

	def forward(self,
			 x,
			 t,
			 cond):

		"""
		Full U-net forward pass to get the reconstructed datas
		"""
		
		factor = repeat(self.class_embed_product(cond[...,0]),"b d -> b d l",l=x.shape[-1])
		bias = repeat(self.class_embed_addition(cond[...,0]),"b d -> b d l",l=x.shape[-1])
		x = x*factor+bias
		if self.time_embbeder is not None:
			time_emb = self.time_embbeder(t)  # (-1, time_dim)
			time_emb_repeat = repeat(time_emb, "b t -> b t l", l=x.shape[2])
			x = torch.cat([x, time_emb_repeat], dim=1)

		skip_connections = []
		for encode in self.encoder:
			x,skip = encode(x)
			skip_connections.append(skip)

		x = self.middle_conv(x)

		for decode,skip in zip(self.decoder,reversed(skip_connections)):
			x = decode(skip,x)

		x = self.output_conv(x)
		return x
	
	def conditional_forward(self,
			 x,
			 t,
			 cond,
			 w):

		"""
		Full U-net forward pass to get the reconstructed datas
		"""

		n = len(x)
		x,t,cond = double_inputs(x,t,cond)
		x = self.forward(x,t,cond)
		x = dedouble_outputs(x,w)
		return x
	
	def classify(self,x):

		cond = torch.zeros((x.shape[0],1,x.shape[-1]),device=x.device)
		factor = repeat(self.class_embed_product(cond[...,0]),"b d -> b d l",l=x.shape[-1])
		bias = repeat(self.class_embed_addition(cond[...,0]),"b d -> b d l",l=x.shape[-1])
		x = x*factor+bias
		t = torch.zeros((len(x)),device=x.device)
		if self.time_embbeder is not None:
			time_emb = self.time_embbeder(t)
			time_emb_repeat = repeat(time_emb, "b t -> b t l", l=x.shape[2])
			x = torch.cat([x, time_emb_repeat], dim=1)

		skip_connections = []
		for encode in self.encoder:
			x,skip = encode(x)
			skip_connections.append(skip)

		x = self.middle_conv(x)
		y = self.auxiliary_clf(x)

		return y