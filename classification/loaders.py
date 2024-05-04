from typing import Optional, Iterable
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,dataloader
from scipy.signal import filtfilt, iirnotch, butter
from einops import rearrange
import os
import scipy

def load_data(folder,idx):

	"""
	load the train and validatation mat files for a subject
	"""

	path_train = os.path.join(folder,f"B0{idx}T.mat")
	path_test = os.path.join(folder,f"B0{idx}E.mat")
	mat_train = scipy.io.loadmat(path_train)["data"]
	mat_test = scipy.io.loadmat(path_test)["data"]
	return mat_train,mat_test

class subject_dataset:

	def __init__(self,
			  mat,
			  fs = 250,
			  t_baseline = 0.3,
			  t_epoch = 4):

		self.fs = fs
		self.t_baseline = t_baseline
		self.t_epoch = t_epoch

		self.epochs = []
		self.cues = []
		self.dfs = []
		self.timestamps = []

		for i in range(mat.shape[-1]):
			epochs,cues,df = self.load_data(mat,i)
			self.epochs.append(epochs)
			self.cues.append(cues)
			self.dfs.append(df)

		self.epochs = np.concatenate(self.epochs,axis=0)
		# removing 1 to get 0,1 instead of 1,2
		self.cues = np.concatenate(self.cues,0)	- 1	

	def  load_data(self,
				   mat,
				   index=0):

		"""
		Loading the epochs,cues, and dataframe for one trial
		For now, we are not removing samples with artifacts
		since it can't be done on the fly.

		Args:
			mat: array
			index: trial index

		Returns:
			epochs: array of epoch electrode data
			cues: array of cues (labels) for each epoch
			df: dataframe of trial data
		"""
		
		electrodes = mat[0][index][0][0][0].squeeze()
		timestamps = mat[0][index][0][0][1].squeeze()
		cues = mat[0][index][0][0][2].squeeze()
		artifacts = mat[0][index][0][0][5]

		epochs,cues = self.create_epochs(electrodes,timestamps,artifacts,cues,
							  self.t_baseline,self.t_epoch,self.fs)
		
		epochs,cues = self.epoch_preprocess(epochs,cues)
		
		df = self.load_df(timestamps,electrodes,cues)
		return epochs,cues,df

	def create_epochs(self,
				   electrodes,
				   timestamps,
				   artifacts,
				   cues,
				   t_baseline,
				   t_epoch,
				   fs):
		
		"""
		creating an array with all epochs from a trial

		Args:
			electrode: electrode data
			timestamps: cue indices
			artifacts: artifact presence array
			cues: labels for cues
			t_baseline: additional time for baseline
			t_epoch: time of epoch
		"""
		
		n_samples = int(fs*t_epoch)
		n_samples_baseline = int(fs*t_baseline)
		epochs = []
		cues_left = []
		for i,j,c in zip(timestamps,artifacts,cues):
			if j==0:
				epochs.append(electrodes[i-n_samples_baseline:i+n_samples,:])
				cues_left.append(c)
		epochs = np.stack(epochs)
		cues_left = np.asarray(cues_left)
		return epochs,cues_left

	def load_df(self,
			 timestamps,
			 electrodes,
			 cues):

		"""
		Loading the channel values and adding a timestamp column.
		Useful for visualizing an entire trial

		Args:
			None
		Returns:
			None
		"""

		timeline = np.zeros_like(electrodes[:,0])

		for t,c in zip(timestamps,cues):
			timeline[t] = c

		df = pd.DataFrame(electrodes)

		df["timestamps"] = timeline

		return df
	
	def trial_preprocess(x,*args,**kwargs):
		"""
		Pre-processing step to be applied to entire trial.
		"""
		return x
	
	def epoch_preprocess(self,x,y):
		"""
		Apply pre-processing before concatenating everything in a single array.
		Easier to manage multiple splits
		"""
		return x,y
	
class CSP_subject_dataset(subject_dataset):

	def __init__(self, mat, fs=250, t_baseline=0.3, t_epoch=4):
		super().__init__(mat, fs, t_baseline, t_epoch)

	def epoch_preprocess(self, x, y):

		x,y = super().epoch_preprocess(x, y)
	
		ax = []

		for i in range(1,10):
			ax.append(self.bandpass(x,self.fs,4*i,4*i+4))
		x = np.concatenate(ax,-1)
		print(x.shape)
		mu = np.mean(x,axis=-1)
		sigma = np.std(x,axis=-1)
		x = (x-rearrange(mu,"n d -> n d 1"))/rearrange(sigma,"n d -> n d 1")
		return x,y
	
	def bandpass(self,
			  x,
			  fs,
			  low,
			  high,):
		
		nyquist = fs/2
		b,a = butter(4,[low/nyquist,high/nyquist],"bandpass",analog=False)
		n,d,t = x.shape
		x = rearrange(x,"n t d -> (n d) t")
		x = filtfilt(b,a,x)
		x = rearrange(x,"(n d) t -> n t d",n=n)
		return x


class EEGDataset(Dataset):
	def __init__(self,
			     subject_splits:list[list[str]],
				 dataset:Optional[dict] = None,
				 save_path:Optional[str] = None,
				 dataset_type: Optional[subject_dataset] = None,
				 fs:float = 250, 
				 t_baseline:float = 0, 
				 t_epoch:float = 9,
				 start:float = 3.5,
				 length:float = 2,
				 channels:Iterable = np.array([0,1,2])):
		
		"""
		Args:
			subject_splits: splits to use for train and test
			save_path: path to save/load pre-processed data
			dataset: dictionnary of train and test splits for all subjects
			dataset_type: type of subject dataset for pre-processing
			pickled: load pickled dataset instead of saving
			fs: sampling frequency
			t_baseline: start of motor imagery trial
			t_epoch: length of motor imagery trial
			start: start of data
			length: duration of data
			channels: chanel indices to include
		"""
		
		self.fs = fs
		self.t_baseline = t_baseline
		self.t_epoch = t_epoch
		if dataset is None:
			self.data = self.load_data(save_path,subject_splits,channels)
		else:
			self.save_dataset(dataset,save_path,dataset_type)
			self.data = self.load_data(save_path,subject_splits,channels)

		self.set_epoch(start,length)

	def __len__(self):
		return self.data[0].shape[0]

	def __getitem__(self, idx):
		return self.data[0][idx,:,int(self.input_start*250):int(self.input_end*250)], self.data[1][idx]

	def save_dataset(self,
				  dataset:dict,
				  path:str,
				  dataset_type:subject_dataset):
		
		"""
		Function for pre-processing a dataset and saving it
		It will save all trial, not just the ones used in this dataset

		Args:
			dataset: dictionnary of train and test splits for all subjects
			path: path to save the pre-processed data
			dataset_type: type of subject dataset for pre-processing

		Return:
			None
		"""
		
		if not os.path.isdir(path):
			os.makedirs(path)
		
		for idx,(k,subject) in enumerate(dataset.items()):
			for split in ["train","test"]:
				set = dataset_type(subject[split],self.fs,self.t_baseline,self.t_epoch)
				epochs = np.float32(set.epochs)
				cues = set.cues
				np.save(os.path.join(path,f"subject_{idx}_{split}_epochs.npy"),epochs)
				np.save(os.path.join(path,f"subject_{idx}_{split}_cues.npy"),cues)

	def load_data(self,
			   path,
			   subject_splits,
			   channels):
		
		epochs = []
		cues = []

		for idx,splits in enumerate(subject_splits):
			for split in splits:
				epochs.append(np.load(os.path.join(path,f"subject_{idx}_{split}_epochs.npy")))
				cues.append(np.load(os.path.join(path,f"subject_{idx}_{split}_cues.npy")))

		epochs = rearrange(np.concatenate(epochs,0),"n t d -> n d t")[:,channels,:]
		cues = np.concatenate(cues,0)

		print(epochs.shape)
		print(cues.shape)

		epochs, cues = self.preprocess(epochs,cues)
		return ((np.float32(epochs).copy()),cues.copy())
	
	def set_epoch(self,start,length):
		self.input_start = start + self.t_baseline
		self.input_end = self.input_start + length
	
	def filter(self,
			x,
			notch_freq=50):
		
		nyquist = self.fs/2
		b,a = iirnotch(notch_freq,30,self.fs)
		x = filtfilt(b,a,x)
		return x

	def preprocess(self,x,y):
		"""
		Apply filters and additional preprocessing to measurements
		"""
		n,t,d = x.shape
		x = rearrange(x,"n d t -> (n d) t")
		x = self.filter(x)
		x = rearrange(x,"(n d) t -> n d t",n=n)
		return x,y
	
if __name__ == "__main__":

	dataset = {}
	for i in range(1,10):
		mat_train,mat_test = load_data("../data/2b_iv",i)
		dataset[f"subject_{i}"] = {"train":mat_train,"test":mat_test}

	save_path = "../data/2b_iv/csp"

	train_split = 3*[["train","test"]] + 6*[["train"]]
	test_split = 3*[[]] + 6* [["test"]]

	channels = np.split(np.arange(0,6*9),6)
	channels = np.concatenate([channels[0],channels[2]])

	train_dataset = EEGDataset(subject_splits=train_split,
					  dataset=None,
					  save_path=save_path,
					  dataset_type=CSP_subject_dataset,
					  channels=channels)
	
	test_dataset = EEGDataset(subject_splits=test_split,
					  dataset=None,
					  save_path=save_path,
					  channels=channels)