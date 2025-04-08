from calendar import c
from ctypes import c_wchar
import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F 
from PIL import Image
from torchvision.datasets import ImageFolder
from sklearn.utils import shuffle
from transformers import MaskFormerImageProcessor



class PatchImageFolder(ImageFolder):
	def __init__(self, root, patch_size=64, stride=32, mode="train" , augmentation_prob=0.4):
		#super(PatchImageFolder, self).__init__(root)
		"""Initializes image paths and preprocessing module."""
		self.root = root	
		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT_binary/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.patch_size = patch_size
		self.stride = stride
		#self.resize = resize
		print("image count in {} path :{}".format(mode,len(self.image_paths)))
		
	def _extract_patches(self, image, GT):
		# if self.resize:
		# 	image = image.resize(self.resize, Image.ANTIALIAS)
		# img_tensor = T.ToTensor()(image).unsqueeze(0).float()  # Convert to tensor and add batch dimension
		# mute normalization as my mask is binary, in case of non-binary mask, use T.ToTensor()
		# GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).unsqueeze(0).float()	# Convert to tensor and add batch dimension
		img_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float()
		#img_tensor = img_tensor.permute(0, 3, 1, 2)
		GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).float()
		#GT_tensor = GT_tensor.permute(0, 3, 1, 2)
		# GT_tensor = T.ToTensor()(GT).unsqueeze(0).float()
        # Extract patches using unfold
		img_patches = img_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
		img_patches = img_patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)  # Flatten patches 
		GT_patches = GT_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
		GT_patches = GT_patches.contiguous().view(-1, 1, self.patch_size, self.patch_size)
		return img_patches,  GT_patches 
		
	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		#  Read images
		image_path = self.image_paths[index]
		if image_path.endswith('.tif'):
			# for one image
			img_patches, GT_patches = self.read_image(index)
			# for multichannel images
			# img_patches, GT_patches = self.read_multi_image(index)
			
			# Apply transformations
			img_pro_patches = []
			GT_pro_patches = []
			
			# # for fine tuninng
			# img_patch, GT_patch  = img_patches, GT_patches
			# for t in range(5):
			# 	img_pro_patch, GT_pro_patch = self.augment_data(img_patch, GT_patch, t)
			# 	img_pro_patches.append(img_pro_patch)
			# 	GT_pro_patches.append(GT_pro_patch)

			# for normal training
			for img_patch, GT_patch in zip(img_patches, GT_patches):
				transform = self.transformation()				
				# transform the patches with augmenting data			
				for t in range(6):
					img_aug, GT_aug = self.augment_data(img_patch, GT_patch, t)
					img_pro_patch = transform(img_aug)
					GT_pro_patch = transform(GT_aug)
					#img_pro_patch = img_aug
					#GT_pro_patch = GT_aug
					img_pro_patches.append(img_pro_patch)
					GT_pro_patches.append(GT_pro_patch)
				
			# Shuffle the patches
			img_pro_patches, GT_pro_patches = shuffle(img_pro_patches, GT_pro_patches)

			# Convert the lists of patches to tensors
			img_pro = torch.stack(img_pro_patches)
			GT_pro = torch.stack(GT_pro_patches)
		else:
			print("Invalid file format: {}".format(image_path))
		return img_pro, GT_pro

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

	def transformation(self, mode='train',augmentation_prob=0.4, patch_size=256):
		RotationDegree = [0,90,180,270]
		p_transform = random.random()
		hflip = random.random()
		vflip = random.random()
		transform = []
		if (mode == 'train' and p_transform <= augmentation_prob):
			r = random.randint(0,3)
			Rotation = RotationDegree[r]
			transform.append(T.RandomRotation((Rotation,Rotation)))						
			RotationRange = random.randint(-10,10)
			transform.append(T.RandomRotation((RotationRange,RotationRange)))
			if (hflip < 0.4):
				transform.append(F.hflip)
			if (vflip < 0.4):
				transform.append(F.vflip)
			transform.append(T.RandomResizedCrop(size=(patch_size,patch_size), scale=(0.75, 1.5), ratio=(0.75, 1.33)))
	
		transform.append(T.Resize((self.patch_size, self.patch_size)))	
		#transform.append(T.ToTensor())
		#transform.append(T.Normalize(mean=[0.5], std=[0.5]))
		transform = T.Compose(transform)
		return transform

	def augment_data(self, img_patch, GT_patch, t):
		if (t==0):
			img_aug = F.hflip(img_patch)
			GT_aug = F.hflip(GT_patch)
		elif (t==1):
			img_aug = F.vflip(img_patch)
			GT_aug = F.vflip(GT_patch)
		elif (t==2):
			img_aug = F.rotate(img_patch, 90)
			GT_aug = F.rotate(GT_patch, 90)
		elif (t==3):
			img_aug = F.rotate(img_patch, 180)
			GT_aug = F.rotate(GT_patch, 180)
		elif (t==4):
			r1 = np.int32(random.random()*0.25*255)
			r2 = np.int32(random.random()*0.25*255)
			c_height = 255-2*r1
			c_width = 255-2*r2
			img_aug = F.resized_crop(img_patch, r1, r2, c_height, c_width, size=(self.patch_size, self.patch_size))
			GT_aug = F.resized_crop(GT_patch, r1, r2, c_height, c_width, size=(self.patch_size, self.patch_size))
		else:
			img_aug = img_patch
			GT_aug = GT_patch
		return img_aug, GT_aug
	
	def read_image(self, index):
		# Read one image
		n_channels = 1
		n_labels = 1
		image_path = self.image_paths[index]
		filename = image_path.split('/')[-1][:-len(".tif")]
		GT_path = self.GT_paths+"/" + filename + ".tif"
		image = Image.open(image_path)#.convert('RGB')
		GT = Image.open(GT_path)
		GT = np.array(GT)  
		GT[GT > 0] = 1
		GT = Image.fromarray(GT)#.convert('L')

		# additional processing in case of fine tunning
		#processor = MaskFormerImageProcessor(reduce_labels = True, size=(512,512), ignore_index= 255, do_resize=True, do_rescale = True, do_normalize = True)
		# processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
		# inputs = processor(images=image, segmentation_maps=GT, return_tensors="pt")	
		# img_patches = inputs.data['pixel_values']
		# GT_patches = inputs.data['pixel_mask']

		img_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float()
		GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).float()

		# Extract patches using unfold
		img_patches = img_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
		img_patches = img_patches.contiguous().view(-1, n_channels, self.patch_size, self.patch_size)  # Flatten patches 
		GT_patches = GT_tensor.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
		GT_patches = GT_patches.contiguous().view(-1, n_labels, self.patch_size, self.patch_size)

		#img_patches ,GT_patches = img_tensor.unsqueeze(0), GT_tensor.unsqueeze(0)
		return img_patches, GT_patches
	
	def read_multi_image(self, index):
		# Add precceding and following images
		image_path_curr = self.image_paths[index]
		if (index == 0):
			image_path_prec = self.image_paths[index]
			image_path_foll = self.image_paths[index+1]
		elif (index == len(self.image_paths)-1):
			image_path_prec = self.image_paths[index-1]
			image_path_foll = self.image_paths[index]
		else:
			image_path_prec = self.image_paths[index-1]
			image_path_foll = self.image_paths[index+1]
		
		prec_image = Image.open(image_path_prec)
		foll_image = Image.open(image_path_foll)
		curr_image = Image.open(image_path_curr)

		# GT 
		filename = image_path_curr.split('/')[-1][:-len(".tif")]
		GT_path = self.GT_paths+"/" + filename + ".tif"
		GT_curr = Image.open(GT_path)
		GT_curr = np.array(GT_curr)  
		GT_curr[GT_curr > 0] = 1
		GT_curr = Image.fromarray(GT_curr)

		filename = image_path_prec.split('/')[-1][:-len(".tif")]
		GT_path = self.GT_paths+"/" + filename + ".tif"
		GT_prec = Image.open(GT_path)
		GT_prec = np.array(GT_prec)
		GT_prec[GT_prec > 0] = 1
		GT_prec = Image.fromarray(GT_prec)

		filename = image_path_foll.split('/')[-1][:-len(".tif")]
		GT_path = self.GT_paths+"/" + filename + ".tif"
		GT_foll = Image.open(GT_path)
		GT_foll = np.array(GT_foll)
		GT_foll[GT_foll > 0] = 1
		GT_foll = Image.fromarray(GT_foll)

		# add the preceeding and following images as channels
		image = np.stack((prec_image, curr_image, foll_image), axis=-1)
		GT = np.stack((GT_prec, GT_curr, GT_foll), axis=-1)

		img_tensor = torch.from_numpy(np.array(image)).unsqueeze(0).float()
		img_tensor = img_tensor.permute(0, 3, 1, 2)
		GT_tensor = torch.from_numpy(np.array(GT)).unsqueeze(0).float()
		GT_tensor = GT_tensor.permute(0, 3, 1, 2)
        
        # Extract patches using unfold
		img_patches = img_tensor.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
		img_patches = img_patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)  # Flatten patches 
		GT_patches = GT_tensor.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
		GT_patches = GT_patches.contiguous().view(-1, 3, self.patch_size, self.patch_size)
		return img_patches, GT_patches
		
		

def get_my_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = PatchImageFolder(root = image_path, patch_size=256, stride=128, mode=mode , augmentation_prob=augmentation_prob) 
	data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	return data_loader

def det_my_dataset(image_path, patch_size=256, stride=128):
	"""Builds and returns DataSet."""
	
	dataset = PatchImageFolder(root = image_path, patch_size=patch_size, stride=stride) 
	return dataset
