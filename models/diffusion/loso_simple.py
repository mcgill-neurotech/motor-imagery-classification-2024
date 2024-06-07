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
from lightning_fabric.utilities.seed import seed_everything
import lightning as L
from einops import repeat
import argparse
import pickle
import wandb

import sys

sys.path.append("../../../motor-imagery-classification-2024/")

from classification.loaders import EEGDataset,load_data
from models.unet.eeg_unets import Unet,UnetConfig, BottleNeckClassifier, Unet1D
from classification.classifiers import DeepClassifier , SimpleCSP, k_fold_splits
from classification.loaders import subject_dataset
from ntd.networks import SinusoidalPosEmb
from ntd.diffusion_model import Diffusion
from ntd.utils.kernels_and_diffusion_utils import WhiteNoiseProcess
from simple_diff import DiffusionUnet

torch.set_float32_matmul_precision('medium')
seed_everything(0)

FS = 250
DEVICE = "cuda"
sns.set_style("darkgrid")


DEBUG = False

if DEBUG:
	print("---\n---\nCurrently in debug mode\n---\n---")

NUM_TIMESTEPS = 100
DIFFUSION_LR = 6E-4
SCHEDULE = "linear"
START_BETA = 1E-4
END_BETA = 8E-2
DIFFUSION_NUM_EPOCHS = 180 if not DEBUG else 1
DIFFUSION_BATCH_SIZE = 64
CLASSIFICATION_MAX_EPOCHS = 150 if not DEBUG else 1
CHANNELS = [0,1,2]

dataset = {}
for i in range(1,10):
    mat_train,mat_test = load_data("../../data/2b_iv",i)
    dataset[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

REAL_DATA = "../../data/2b_iv/raw"

SAVE_PATH = "../../saved_models"

def generate_samples(fabric,
                     diffusion_model, 
					 condition,
                     batch_size=200,
                     n_iter=20,
                     w=0):
	# it's a bit hard to predict memory consumption so splitting in mini-batches to be safe
	num_samples = batch_size
	cond = 0
	if (condition == 0):
		cond = (torch.zeros(num_samples, 1, 512)+1).to(dtype=torch.float16,
														device=DEVICE)
	elif (condition == 1):
		cond = (torch.ones(num_samples, 1, 512)+1).to(dtype=torch.float16,
														device=DEVICE)

	diffusion_model.eval()

	print(f"Generating samples: cue {condition + 1}")
	k = 1 if DEBUG else n_iter
	complete_samples = []
	with fabric.autocast():
		with torch.no_grad():
			for i in range(k):
				samples, _ = diffusion_model.sample(num_samples, cond=cond,w=w)
				samples = samples.cpu().numpy()
				print(samples.shape)
				complete_samples.append(samples)
	complete_samples = np.float32(np.concatenate(complete_samples))
	if DEBUG:
		complete_samples = repeat(complete_samples,"n ... -> (n k) ...",k=n_iter)
	print(complete_samples.shape)
	return complete_samples
      
def train_diffusion(fabric,
					unet,
					lr,
					diffusion_model,
					train_set,
					num_epochs,
					batch_size,
					subject_id,):
	
	train_loader = DataLoader(
		train_set,
		batch_size=batch_size
	)

	optimizer = optim.AdamW(
		unet.parameters(),
		lr=lr,
	)

	diffusion_model,optimizer = fabric.setup(diffusion_model,optimizer)
	train_loader = fabric.setup_dataloaders(train_loader)

	loss_per_epoch = []

	stop_counter = 0
	min_delta = 0.075
	tolerance = 30
			
		# Train model
	for i in range(num_epochs):
		
		epoch_loss = []
		for batch in train_loader:
			
			with fabric.autocast():
			# Repeat the cue signal to match the signal length
				# print(batch["signal"].shape)
				signal,cue = batch
				cue = (cue + 1).to(signal.dtype)
				cond = cue.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 512).to(DEVICE)
				
				loss = diffusion_model.train_batch(signal.to(DEVICE), cond=cond,
									   p_uncond=0.15)
			loss = torch.mean(loss)
			
			epoch_loss.append(loss.item())
			
			fabric.backward(loss)
			# loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			
		epoch_loss = np.mean(epoch_loss)
		loss_per_epoch.append(epoch_loss)

		wandb.log({f"loss_{subject_id}": epoch_loss,
                f"epoch":i})
		
		print(f"Epoch {i} loss: {epoch_loss}")

		print(f"diff: {epoch_loss - min(loss_per_epoch)}")

		if epoch_loss - min(loss_per_epoch) >= min_delta*min(loss_per_epoch):
			stop_counter += 1
		if stop_counter > tolerance:
			break

def check(train_split,
		  test_split,
		  fake_paths,
		  channels=CHANNELS):

	generated_signals_one = np.load(fake_paths[0])
	generated_signals_zero = np.load(fake_paths[1])

	accuracies = []
		
	test_classifier = SimpleCSP(train_split=train_split,
								test_split=test_split,
								dataset=None,
								save_paths=[REAL_DATA],
								channels=channels,
								length=2.05)

	full_x,full_y = test_classifier.get_train()

	print(f"full x shape: {full_x.shape}")

	real_acc = test_classifier.fit()

	print(f"reaching an accuracy of {real_acc} without fake data")

	for real_fake_split in range(15, 46, 15):
		
		# Train new classifier with a mix of generated and real data
		
		# Change real_fake_split percent of the test_classifier data to generated signals
		n = int(len(full_x) * real_fake_split / 100)

		shuffling = np.random.permutation(full_x.shape[0])

		split_x = full_x[shuffling]
		split_y = full_y[shuffling]
		split_x[0:n//2] = generated_signals_one[0:n//2]
		split_y[0:n//2] = 1

		split_x[n//2:2*(n//2)] = generated_signals_zero[0:n//2]
		split_y[n//2:2*(n//2)] = 0

		print(f"split x shape: {split_x.shape}")

		acc = test_classifier.fit((split_x,split_y))

		accuracies.append(acc)
					
		print(f"Reaching an accuracy of {acc} using {real_fake_split}% fake data")

	return real_acc,accuracies

def train_classification(fabric,
						 unet,
						 fake_percentage,
						 fake_paths,
						 train_split,
						 test_split,
						 train_real,
						 fine_tune,):
	
	deep_clf = DeepClassifier(
		model=unet,
		save_paths=["../../data/2b_iv/raw/"],
		fake_data=fake_paths,
		train_split=train_split,
		test_split=test_split,
		fake_percentage=fake_percentage,
		dataset=None,
		dataset_type=subject_dataset,
		length=2.05,
		index_cutoff=512
	)

	with_fake = deep_clf.fit(fabric=fabric,
			 num_epochs=CLASSIFICATION_MAX_EPOCHS,
			 lr=1E-3,
			 weight_decay=1E-4,
			 verbose=False,
			 optimizer=None,
			 stop_threshold=25)
	
	if fine_tune:
		print("\n---\nFine-tuning model\n---\n")
		to_fine_tune = [unet.encoder,
			unet.decoder,
			unet.middle_conv,
			unet.class_embed,]

		to_optimize = [{"params":i.parameters(),
			"lr":2E-5,
			"weight_decay":1E-4} for i in to_fine_tune]

		to_optimize.append({"params":unet.auxiliary_clf.parameters(),
			"lr":1E-3,
			"weight_decay":1E-4})

		optimizer = optim.AdamW(to_optimize)
	else:
		optimizer = None
	
	deep_clf.setup_dataloaders(use_fake=False)
	if train_real:
		without_fake = deep_clf.fit(fabric=fabric,
				num_epochs=CLASSIFICATION_MAX_EPOCHS,
				lr=1E-3,
				weight_decay=1E-4,
				verbose=False,
				optimizer=optimizer,
				stop_threshold=25,)
		
		return with_fake,without_fake
	else:
		return with_fake

def loso_trial(fabric,
			   experiment_name,
			   train_split,
			   test_split,
			   subject_id,
			   w,
			   train=True,
			   fine_tune=False,
			   train_real=None):

	UnetDiff1D = UnetConfig(
		input_shape=(512),
		input_channels=3,
		conv_op=nn.Conv1d,
		norm_op=nn.InstanceNorm1d,
		non_lin=nn.ReLU,
		pool_op=nn.AvgPool1d,
		up_op=nn.ConvTranspose1d,
		starting_channels=32,
		max_channels=256,
		conv_group=1,
		conv_padding=(1),
		conv_kernel=(3),
		pool_fact=2,
		deconv_group=1,
		deconv_padding=(0),
		deconv_kernel=(2),
		deconv_stride=(2),
		residual=True
	)

	train_set = EEGDataset(subject_splits=train_split,
                    dataset=None,
                    save_paths=[REAL_DATA],
                    dataset_type=subject_dataset,
                    channels=CHANNELS,
                    sanity_check=False,
                    length=2.05)

	test_set = EEGDataset(subject_splits=test_split,
						dataset=None,
						save_paths=[REAL_DATA],
						channels=CHANNELS,
						sanity_check=False,
						length=2.05)

	classifier = BottleNeckClassifier((2048,1024),)
	unet = DiffusionUnet(UnetDiff1D,classifier)

	noise_sampler = WhiteNoiseProcess(1.0, 512)

	diffusion_model = Diffusion(
		network=unet,
		diffusion_time_steps=NUM_TIMESTEPS,
		noise_sampler=noise_sampler,
		mal_dist_computer=noise_sampler,
		schedule=SCHEDULE,
		start_beta=START_BETA,
		end_beta=END_BETA,
	)

	save_path = os.path.join(experiment_name,f"subject_{subject_id}")

	if not os.path.isdir(save_path):
		os.makedirs(save_path)

	if train:

		train_diffusion(fabric,
					unet=unet,
					diffusion_model=diffusion_model,
					train_set=train_set,
					num_epochs=DIFFUSION_NUM_EPOCHS,
					batch_size=DIFFUSION_BATCH_SIZE,
					lr=DIFFUSION_LR,
					subject_id=subject_id)

		torch.save(diffusion_model.state_dict(),os.path.join(save_path,f"unet_diff_{subject_id}.pt"))
		torch.save(unet.state_dict(),os.path.join(save_path,f"unet_state_dict_{subject_id}.pt"))
	else:
		print(f"\n---\n\nloading diffusion model for {subject_id}, w = {w}\n\n---\n")
		diffusion_model = fabric.setup(diffusion_model)

	diffusion_model.load_state_dict(torch.load(os.path.join(save_path,f"unet_diff_{subject_id}.pt")))
	unet.load_state_dict(torch.load(os.path.join(save_path,f"unet_state_dict_{subject_id}.pt")))

	generated_signals_zero = generate_samples(fabric,diffusion_model, condition=0,n_iter=10,
										  batch_size=250,w=w)
	generated_signals_one = generate_samples(fabric,diffusion_model, condition=1,n_iter=10,
											batch_size=250,w=w)
	
	zeros_path = os.path.join(save_path,f"generated_zeros_{w}.npy")
	ones_path = os.path.join(save_path,f"generated_ones_{w}.npy")
	
	
	np.save(zeros_path,generated_signals_zero)
	np.save(ones_path,generated_signals_one)

	fake_paths = [ones_path,zeros_path]

	csp_real,accuracies = check(train_split=train_split,
						test_split=test_split,
						fake_paths=fake_paths,
						channels=CHANNELS)
	
	max_acc = np.argmax(accuracies)
	print(f"Reaching a maximal accuracy of {accuracies[max_acc]} for CSP using {(max_acc+1)*15}% fake vs {csp_real}")	

	# resetting to randomly initialized model
	if not fine_tune:
		classifier = BottleNeckClassifier((2048,1024),)
		unet = DiffusionUnet(UnetDiff1D,classifier)

	results = {"accuracies_csp":accuracies}

	cnn_results = {}

	if train_real is not None:
		print(f"Training without fake data: {train_real}")

	for idx,p in enumerate([0.5,1]):
		train_real = train_real if train_real is not None else (idx==0)
		if train_real:
			fake,real = train_classification(fabric=fabric,
									unet=unet,
									fake_percentage=p,
									fake_paths=fake_paths,
									train_split=train_split,
									test_split=test_split,
									train_real=train_real,
									fine_tune=fine_tune)
			cnn_results["real"] = real
			print(f"Reaching an accuracy of {real} without fake data")
			wandb.log({f"accuracy_{0}":real})
		else:
			fake = train_classification(fabric=fabric,
									unet=unet,
									fake_percentage=p,
									fake_paths=fake_paths,
									train_split=train_split,
									test_split=test_split,
									train_real=train_real,
									fine_tune=fine_tune)
		print(f"Reaching an accuracy of {fake} with {p} fake")
		wandb.log({f"accuracy_{p}":fake})
		cnn_results[p] = fake
	
	results["cnn"] = cnn_results
	return results

def k_fold(experiment_name,
		   k=9,
		   n=9,
		   w=3,
		   train=True,
		   fine_tune=False,
		   train_real=None):
	
	fabric = Fabric(accelerator="cuda",precision="bf16-mixed")
	
	splits = k_fold_splits(k,n,leave_out=True)

	results = {}

	full_folder = os.path.join(SAVE_PATH,experiment_name)

	for idx,split in enumerate(splits):

		results_k = loso_trial(fabric=fabric,
			 train_split=split[0],
			 test_split=split[0],
			 subject_id=idx,
			 experiment_name=full_folder,
			 w=w,
			 train=train,
			 fine_tune=fine_tune,
			 train_real=train_real)
		results[f"split_{idx}"] = results_k

	with open(os.path.join(full_folder,f"results_{w}.p"),"wb") as f:
		pickle.dump(results,f)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-n","--name",help="experiment name",type=str)
	parser.add_argument("-k","--k_fold",help="number of folds",type=int)

	args = parser.parse_args() 

	wandb.init(project="cnn-diffusion-mi", mode="online",
			name=args.name)
	
	k_fold(args.name,args.k_fold,9,w=0,train=True)
	k_fold(args.name,args.k_fold,9,w=3,train=False,train_real=False)