import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import nibabel as nib
import pandas as pd
from torch.utils import data
import random
from scipy.ndimage import rotate, zoom
from sklearn.preprocessing import MinMaxScaler
from .resnet3d import *
from sklearn.model_selection import KFold
import random
from ast import literal_eval
torch.cuda.empty_cache() 


def init_he(m):
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.kaiming_uniform_(m.bias)

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


def get_resnet(which_model, channel, num_classes, head):
	if which_model == "resnet6":
		num_block = [1,1]
		block_channel = [channel,channel,channel,channel]
		model = MyResNet(BasicBlock, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)
	elif which_model == "resnet10":
		num_block = [1,1,1,1]
		block_channel = [channel,channel,channel,channel]
		model = MyResNet(BasicBlock, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)
	elif which_model == "resnet18":
		num_block = [2,2,2,2]
		block_channel = [channel,channel,channel,channel]
		model = MyResNet(BasicBlock, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)
	elif which_model == "resnet26":
		num_block = [3, 4, 4, 2]
		block_channel = [channel,channel,channel,channel]
		model = MyResNet(BasicBlock, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)
	elif which_model == "resnet34":
		num_block = [3, 4, 6, 3]
		block_channel = [channel,channel,channel,channel]
		model = MyResNet(BasicBlock, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)
	elif which_model == "resnet50":
		num_block = [3, 4, 6, 3]
		block_channel = [channel,channel,channel,channel]
		model = MyResNet(Bottleneck, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)

	elif which_model == "Oriresnet18":  
		num_block = [2,2,2,2]
		block_channel = [channel,channel*2,channel*4,channel*8]
		model = MyResNet(BasicBlock, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)
	elif which_model == "Oriresnet50":
		num_block = [3, 4, 6, 3]
		block_channel = [channel,channel*2,channel*4,channel*8]
		model = MyResNet(Bottleneck, num_block = num_block, in_channel = block_channel[0], 
			conv_kernel = 3, block_channel = block_channel ,num_classes = num_classes, head = head)

	elif which_model == "modelB":
		model = modelB(channel, num_classes, head)
	return model





def trainClass(train_loader, model, criterion, optimizer, device, loss_name):
	"""one epoch training"""
	epoch_train_loss = 0 
	model.train()  
	for batch, (data, target, _) in enumerate(train_loader):
		# print(target)
		data = data.to(device) 
		target = target.to(device)
		# target2 = target2.to(device)
		optimizer.zero_grad()   
		output = model(data)
		if loss_name == 'fBCE':  
			target = target.to(device, dtype=torch.float)
			output = nn.Sigmoid()(output)
			loss = criterion(output, target, target2)
		elif loss_name == 'BCE':  
			target = target.to(device, dtype=torch.float)
			output = nn.Sigmoid()(output)
			loss = criterion(output, target)
		elif loss_name.startswith('ASL'):
			target = target.to(device, dtype=torch.float)
			loss = criterion(output, target)
		elif loss_name.startswith('BiASL'):
			target = target.to(device, dtype=torch.float)
			loss = criterion(output, target)
		elif loss_name == 'BCElogit':  
			target = target.to(device, dtype=torch.float)
			loss = criterion(output, target)
		elif loss_name == 'CE':
			target = target.to(device, dtype=torch.long)  
			loss = criterion(output, target)
		loss.backward()            
		optimizer.step()
		epoch_train_loss += (loss.item())

	return  model, epoch_train_loss





mask3d = nib.load("./d3_mask.nii.gz").get_fdata()
mask2d = nib.load("./d2_mask.nii.gz").get_fdata()
maskOri = nib.load("./mni_icbm152_t1_tal_nlin_sym_09c_mask.nii.gz").get_fdata()
class DatasetFromNiiOTF(Dataset):    
	def __init__(self, data_path,csv_path, label, sigma, ds):
		self.to_tensor = transforms.ToTensor()
		self.data_info = pd.read_csv(csv_path)
		self.image_arr = np.asarray(self.data_info.iloc[:, 0])
		self.label = label
		self.label_arr = np.asarray(self.data_info[self.label].values)

		self.sigma = sigma
		self.ds = ds
		self.data_path = data_path
		self.data_len = len(self.data_info.index)
	def __len__(self):
		return self.data_len
	def __getitem__(self, index):
		# read single nii
		single_image_path = self.image_arr[index]
		single_image_arrary = nib.load(self.data_path + "/" + single_image_path)
		aff = single_image_arrary.affine
		single_image_arrary = single_image_arrary.get_fdata() 
		# img = nib.Nifti1Image(single_image_arrary, aff)
		# nib.save(img, "/home/wangd2/demCLF_kfold_T1/test/" + single_image_path)
		if self.ds == '2DS':
			mask = mask2d
		elif self.ds == '3DS':
			mask = mask3d
		elif self.ds == '':
			mask = maskOri


		if self.sigma > 0:
			noise = np.random.normal(0, self.sigma , single_image_arrary.shape)
			noise = noise*mask
			single_image_arrary = single_image_arrary + noise
			# img = nib.Nifti1Image(single_image_arrary, aff)
			# nib.save(img, "/home/wangd2/demCLF_kfold_T1/test/N" + single_image_path)


		single_image_arrary = np.expand_dims(single_image_arrary, axis=0).astype(np.float32)
		img_as_tensor = torch.from_numpy(single_image_arrary)

		# get label
		single = self.label_arr[index]
		label_as_tensor = torch.from_numpy(np.array(single))

	
		return (img_as_tensor, label_as_tensor, single_image_path)




