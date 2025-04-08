#import config
from re import I, L
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from PIL import Image, ImageOps
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
#from mynetwork import U_Net, U_Net_plus
from networkom3 import U_Net_plus
import time

import torch

from transformers import pipeline, SwinConfig, SwinModel, AutoImageProcessor, MaskFormerForInstanceSegmentation, Mask2FormerForUniversalSegmentation



def make_prediction(state_dict_path, path):
	
	patch_size= 256   #update QKV attention normalization layer according to the patch size = patch size//32
	stride= 16
	batch_size = 16
	# create the output path
	output_path = path+'mask_om3/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
		 
	# Load the state dictionary
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	state_dict = torch.load(state_dict_path, map_location=torch.device(device))
	
	# Initialize the model
	model = U_Net_plus(img_ch=1,output_ch=1)

	# Load the state dictionary into the model
	model.load_state_dict(state_dict)
	model = model.to(device)

	# set model to evaluation mode
	model.eval()
	
	# get images
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
	output_patches = []
	pHc = 0
	pHf = 0
	pWc = 0
	pWf = 0

	# turn off gradient tracking
	with torch.no_grad():
		for index in range(len(image_paths)):
			if image_paths[index].endswith('.tif'):
				
				# Loading cell channels
				image_path= image_paths[index]
				image = Image.open(image_paths[index])
				# image_rotated = image.rotate(90, Image.NEAREST, expand = 1)
				
				# Padding
				H, W = image.size
				pHc =  int(np.ceil((stride-(H%stride))/2))
				pHf =  int(np.floor((stride-(H%stride))/2))
				pWc =  int(np.ceil((stride-(W%stride))/2))
				pWf =  int(np.floor((stride-(W%stride))/2))
				# Check and swap byte order if necessary
				np_image = np.array(image)
				if np_image.dtype.byteorder not in ('=', '|'):
					np_image = np_image = np_image.byteswap().view(np_image.dtype.newbyteorder())
				img_tensor = torch.from_numpy(np_image).unsqueeze(0).float().to(device)
				# add Normalizer
				#normalizer = nn.LayerNorm([1, W, H]).to(device)
				#img_tensor = normalizer (img_tensor)
				img_tensor = torch.nn.functional.pad(img_tensor, (pHf, pHc, pWf, pWc))
				#img_tensor = torch.nn.functional.pad(img_tensor, (0, pHf+pHc, 0, pWf+pWc))

				# # Mirror across the edges
				# img_tensor[:, :, 0:pHf] = img_tensor[:, :, pHf:2*pHf].flip(2)
				# img_tensor[:, :, pHf+H:] = img_tensor[:, :, pHf+H-pHc:pHf+H].flip(2)
				# img_tensor[:, 0:pWf, :] = img_tensor[:, pWf:2*pWf, :].flip(1)
				# img_tensor[:, pWf+W:, :] = img_tensor[:, pWf+W-pWc:pWf+W, :].flip(1)

				# Extract patches using unfold
				img_patches = img_tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
				img_patches = img_patches.contiguous().view(-1, 1, patch_size, patch_size)  # Flatten patches 
				
				# Handle large batches
				masks = []
				for i in range(0, img_patches.shape[0], batch_size):
					img_batch = img_patches[i:i + batch_size]  # Select the next group of 16
					predMask, _ =  model(img_batch)#.to(device))
					mask = torch.sigmoid(predMask).squeeze(1)
					masks.append(mask)

				masks = torch.cat(masks, dim=0)
				masks = masks.view(1, 1, masks.size(0), patch_size, patch_size)
				masks = masks.permute(0, 1, 3, 4, 2).reshape(1, 1 * patch_size * patch_size, -1)
				output = torch.nn.functional.fold(masks, output_size=(W+pWf+pWc, H+pHf+pHc), kernel_size=patch_size, stride=stride)
				normalization_map = torch.nn.functional.fold(torch.ones_like(masks), output_size=(W+pWf+pWc, H+pHf+pHc), kernel_size=patch_size,stride=stride)
				output /= normalization_map
				output = output.squeeze().squeeze()
				output = output - output.min()
				output = output / output.max()
				#output[output >= 0.3] = 255
				output = output*255
				output = output.byte()
				output_np = output.cpu().numpy()
				output_np = output_np[pWf:pWf + W, pHf:pHf + H]
				#output_np = output_np[0:W, 0:H]
				output_image = Image.fromarray(output_np)
				output_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))

def averag_threshold(path):
	output_path = './dataset/experiments/exp9/avg_th/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))

	for index in range(len(image_paths)):
		if image_paths[index].endswith('.tif'):

			for v in range(-2, 3):
				i = min(min(0, index+v), len(image_paths)-1)
				image = Image.open(image_paths[i])
				image_np = np.array(image)
				image_np += image_np

			avg_image = image_np/5
			output_image = Image.fromarray(avg_image)
			output_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))
  
def normalize_patches(path):
	output_path = path+'normalize_stack/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
	for index in range(len(image_paths)):
		if image_paths[index].endswith('.tif'):
			image = Image.open(image_paths[index])
			W, H = image.size
			ptW = 523
			ptH = 532
			# Padding
			if (H%ptH == 0):
				pHf = 0
				pHc = 0
			else:
				pHf =  int(np.floor((ptH-(H%ptH))/2))
				pHc =  int(np.ceil((ptH-(H%ptH))/2))
			if (W%ptW == 0):
				pWf = 0
				pWc = 0
			else:
				pWf =  int(np.floor((ptW-(W%ptW))/2))
				pWc =  int(np.ceil((ptW-(W%ptW))/2))
			# Check and swap byte order if necessary
			np_image = np.array(image)
			if np_image.dtype.byteorder not in ('=', '|'):
				np_image = np_image = np_image.byteswap().view(np_image.dtype.newbyteorder())
			img_tensor = torch.from_numpy(np_image).unsqueeze(0).float()
			img_tensor = torch.nn.functional.pad(img_tensor, (pWf, pWc,pHf, pHc))
			tH, tW = img_tensor.size(1), img_tensor.size(2)
			nHp, nWp = tH//ptH, tW//ptW
			img_patches = img_tensor.unfold(1, ptH, ptH).unfold(2, ptW, ptW)
			img_patches = img_patches.contiguous().view(-1, 1, ptH, ptW)  # Flatten patches
			# Normalize patches
			for i in range(img_patches.shape[0]):
				img_patches[i] = img_patches[i] - img_patches[i].min()
				img_patches[i] = img_patches[i] / img_patches[i].max()
			# save individual patches
			for i in range(img_patches.shape[0]):
				normalized_image = img_patches[i].squeeze()
				normalized_image = (normalized_image * 255).byte()
				normalized_image_np = normalized_image.cpu().numpy()
				normalized_image = Image.fromarray(normalized_image_np)
				normalized_image.save(os.path.join(output_path, image_paths[index].split('/')[-1][:-len('.tif')] + '_patch_' + str(i) + '.tif'))

			# stich patches
			normalized_image = img_patches.view(1,1,img_patches.size(0), ptH, ptW).permute(0, 1, 3, 4, 2).reshape(1, 1 * ptH * ptW, -1)
			normalized_image = torch.nn.functional.fold(normalized_image, output_size=(tH, tW), kernel_size=(ptH, ptW), stride=(ptH, ptW))
			#normalized_image = img_patches.view(1, 1, nHp, nWp, ptH, ptW).permute(0, 1, 4, 2, 5, 3).reshape(1, 1, tH, tW)
			# save image
			normalized_image = (normalized_image * 255).squeeze()
			normalized_image = normalized_image.byte()
			normalized_image_np = normalized_image.cpu().numpy()
			normalized_image = Image.fromarray(normalized_image_np)
			normalized_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))

def post3d_process( path, th1= 70, th2= 150, search_wnd=8, f_exist =4):
	output_path = path+'postp_stack/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
	for index in range(len(image_paths)):
		if image_paths[index].endswith('.tif'):
			image = Image.open(image_paths[index])
			image_type = np.float32
			image_np = np.array(image).astype(image_type)
			image_np[image_np < th1] = 0
			image_prop = image_np.copy()
			image_np[image_np > th2] = 0
			image_mid = image_np.copy()
			image_high = image_prop - image_mid
			context = [np.array(Image.open(image_paths[x])).astype(image_type) for x in range(max(0, index - search_wnd // 2), min(len(image_paths), index + search_wnd // 2)) if x != index]
			stack_context = np.stack(context).transpose(1, 2, 0) if len(context) > 0 else np.full((*image_prop.shape, 2), 1, dtype=image_mid.dtype)
			context_mul =  stack_context * image_mid[:, :, None]
			output = ((context_mul.sum(axis=2)) / (f_exist*context_mul.shape[2])) + image_high
			# output = output - output.min()
			# output = output / output.max()
			# output = output * 255
			# output = np.clip(output, 0, 255)
			output = output.astype(np.int8)
			output_image = Image.fromarray(output, mode = "L")
			output_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))

def img_mode(path):
	output_path = path+'original_stack/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
	for index in range(len(image_paths)):
		if image_paths[index].endswith('.tif'):
			image = Image.open(image_paths[index])
			image = image.convert("I")  # Convert to 32-bit signed integer pixels
			image_np = np.array(image, dtype=np.uint16)
			image_np = (image_np / 256).astype(np.uint8)  # Convert from 16-bit to 8-bit
			output_image = Image.fromarray(image_np, mode="L")
			output_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))


def hugging_face_inference(path):
 	
	# create the output path
	output_path = path+'HuggingFace_mask/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
		 
	# Load the state dictionary
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Load the state dictionary into the model
	#pipe = pipeline("image-segmentation", model="kiheh85202/yolo")
	#model = pipe.model.to(device)

	processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
	model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")



	# set model to evaluation mode
	model.eval()
	
	# get images
	image_paths = list(map(lambda x: os.path.join(path, x), os.listdir(path)))
	
	# turn off gradient tracking
	with torch.no_grad():
		for index in range(len(image_paths)):
			if image_paths[index].endswith('.tif'):
				
				# Loading cell channels
				image_path= image_paths[index]
				image = Image.open(image_paths[index]).convert("RGB")
				#np_image = np.array(image)
				#img_tensor = torch.from_numpy(np_image).unsqueeze(0).float().to(device)
				#predMask =  model(img_tensor)#.to(device))

				# Semantic Segmentation

				inputimg = processor(image, return_tensors="pt")
				predMask = model(**inputimg)

				#masks_queries_logits  = predMask.masks_queries_logits
				result = processor.post_process_semantic_segmentation(predMask, target_sizes=[image.size[::-1]])[0]
				#predicted_panoptic_map = result["segmentation"]
				
				
				array = result.detach().cpu().numpy()
				array = (array * 255).astype(np.uint8)	
				output_image = Image.fromarray(array)
				output_image.save(os.path.join(output_path, image_paths[index].split('/')[-1]))