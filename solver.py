import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
#from mynetwork import U_Net, U_Net_plus
from networkom3 import U_Net_plus
import csv
import torch.nn as nn
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import threading
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal, QObject
from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerModel, MaskFormerImageProcessor, SegformerForSemanticSegmentation, SegformerImageProcessor
from transformers import Trainer, TrainingArguments 
import evaluate
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class Solver(QObject):
	update_signal = pyqtSignal()
	finished = pyqtSignal()
	def __init__(self, config, train_loader, valid_loader, test_loader):
		super().__init__()
		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		self.id2label = {0: 'background',1: 'sinusoids'}
		self.label2id = {v: k for k, v in self.id2label.items()}

		# Models
		self.model = None
		self.preprocessor= None
		self.processor = None
		self.metric = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		#self.criterion2 = self.diceLoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()
		
		# checker functionality
		self.checker = None
		self.lock = threading.Lock()

		# addtional 
		self.BottleNeck_size = 2048

		# visulaization
		self.fig = Figure(figsize=(12, 12))
		self.canvas = FigureCanvas(self.fig)
		ax = self.fig.add_subplot(111)
		ax.set_title('Training and Validation Loss')
		ax.set_xlabel('Epochs')
		ax.set_ylabel('Loss')
		ax.set_xlim(0, self.num_epochs)
		ax.set_ylim(0, 50)
		ax.legend()
		ax.grid(True)
		self.canvas.draw()
		# Plot lines for training and validation loss, initializing empty
		self.line_train, = ax.plot([], [], label='Training Loss', color='blue', marker='o')
		self.line_val, = ax.plot([], [], label='Validation Loss', color='blue', marker='x')

	

	def set_checker(self, checker):
		self.checker = checker
		self.enable_checker = True

	def diceLoss(self, x, y):
		intersection = torch.sum(x * y) + 1e-7
		union = torch.sum(x) + torch.sum(y) + 1e-7
		return 1 - 2 * intersection / union	
			
	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.model = U_Net(img_ch=3,output_ch=3)
		elif self.model_type == 'MaskFormer':
			#self.preprocessor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
			#self.processor = MaskFormerImageProcessor(reduce_labels = True, size=(512,512), ignore_index= 255, do_resize=False, do_rescale = False, do_normalize = False)
			# use config to intialize the model with random weights
			config_mf = MaskFormerConfig.from_pretrained("facebook/maskformer-swin-base-coco")
			self.model = MaskFormerForInstanceSegmentation(config_mf)
			# Replace random parameters with pre-trained ones
			base_model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-coco")
			self.model.model = base_model		
		elif self.model_type == 'Segformer':
			pretrained_model_name = "nvidia/mit-b0"
			self.model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name,id2label=self.id2label,label2id=self.label2id)
			self.processor = SegformerImageProcessor()
		else:
			self.model = U_Net_plus(img_ch=1, output_ch= 1)
			
		self.optimizer = optim.Adam(list(self.model.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.model.to(self.device)

		# self.print_network(self.model, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.model.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)
		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img
	
	

	def bin_gt(self, x, n):
		for i in range(n-1):
			batch_size, channels, height, width = x.size()
			x = x.view(batch_size, channels, height // 2, 2, width // 2, 2)
			x = x.float()
			x = x.mean(dim=(3, 5))
		return x
			

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))
		
		startTime = time.time()
		lr = self.lr
		best_unet_score = 0.
		best_unet = None
		best_epoch = 0
		best_val_loss = 1000
		best_train_loss = 1000
		array_train_loss = []
		array_val_loss = []
		plt.ion()
		# test data loading 
		# images_patches, GT_patches = self.train_loader.dataset.__getitem__(6)

		for epoch in tqdm(range(self.num_epochs)):

			self.model.train(True)
			epoch_loss = 0
			train_loss = 0
			val_loss = 0
			test_loss = 0
			
			for i, (images_patches, GT_patches) in enumerate(self.train_loader):
				# train for each patch				
				for i in range(images_patches.size(1)):
					images = images_patches[:, i, :, :, :].to(self.device)
					GT = GT_patches[:, i, :, :, :].to(self.device)
						
					# My UNet
					# ------------------
					mask, img_decoder = self.model(images)
					mask_probs = F.sigmoid(mask)
					img_decoder_probs = list(F.sigmoid(x) for x in img_decoder)

					# mask loss
					loss_mask = self.diceLoss(mask_probs.view(mask_probs.size(0),-1),GT.view(GT.size(0),-1))
					train_loss += loss_mask.item()
					
					# loos at decoder stages
					#gt_binned = list(map(lambda x: self.bin_gt(GT, x), [6, 5, 4, 3, 2]))
					#gt_binned = list(map(lambda x: self.bin_gt(GT, x), [5, 4, 3, 2]))
					#loss_decoder = list(self.diceLoss(x.view(x.size(0),-1),y.view(y.size(0),-1)) for x,y in zip (img_decoder_probs, gt_binned))
					# print(loss_mask.item(), list(dloss.item() for dloss in loss_decoder))

					# Total loss
					loss = loss_mask #+ sum(dloss for dloss in loss_decoder)
					epoch_loss += loss_mask.item() #+ sum(dloss.item() for dloss in loss_decoder)
					
					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					# visualization
					#self.checker.set_input_output(images[:,1,:,:].unsqueeze(1), SR_probs[:,1,:,:].unsqueeze(1), GT[:,1,:,:].unsqueeze(1))
					self.checker.set_input_output(images, mask_probs, GT)
					self.update_signal.emit()

			# Print the log info
			print("[INFO] EPOCH: {}/{}".format(epoch+1, self.num_epochs))
			print("Train loss: {:.6f}".format(epoch_loss))
				

			# Decay learning rate
			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr
				print ('Decay learning rate to lr: {}.'.format(lr))
				
				
			#===================================== Validation ====================================#
			self.model.train(False)
			self.model.eval()


			for i, (imageval, GTval) in enumerate(self.valid_loader):
				for i in range(imageval.size(1)):
					images = imageval[:, i, :, :, :].to(self.device)
					GT = GTval[:, i, :, :, :].to(self.device)

					# # # U-Net validation
					mask, _  = self.model(images)
					mask_probs= F.sigmoid(mask)
					loss_val = self.diceLoss(mask_probs.view(mask_probs.size(0),-1),  GT.view(GT.size(0),-1))
					val_loss += loss_val.item()

			print('[Validation] loss: %.4f'%(val_loss))

			# visualization of the training loss and validation loss
			array_train_loss.append(train_loss)
			array_val_loss.append(val_loss)

			# Update the data for each line
			if ((array_train_loss[epoch]<array_train_loss[epoch-1]) and (array_val_loss[epoch]<array_val_loss[epoch-1])):
				self.line_train.set_color('green')
				self.line_val.set_color('green')
			elif ((array_train_loss[epoch]>array_train_loss[epoch-1]) and (array_val_loss[epoch]>array_val_loss[epoch-1])):
				self.line_train.set_color('red')
				self.line_val.set_color('red')
			else:
				self.line_train.set_color('black')
				self.line_val.set_color('blue')
			self.line_train.set_xdata([e for e in range(epoch+1)])
			self.line_train.set_ydata(array_train_loss[:epoch+1])
			self.line_val.set_xdata([e for e in range(epoch+1)])
			self.line_val.set_ydata(array_val_loss[:epoch+1])

			# Redraw the canvas (update the plot)
			self.canvas.draw()
			self.canvas.flush_events()
			plt.pause(0.1)

			# Save Best U-Net model
			if (train_loss < best_train_loss):
				best_train_loss = train_loss
				best_unet = self.model.state_dict()
				unet_best_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-best_train.pkl' %(self.model_type,epoch,epoch_loss,val_loss))
				torch.save(best_unet,unet_best_path)

			if (val_loss < best_val_loss):
				best_val_loss = val_loss
				best_unet = self.model.state_dict()
				unet_best_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-best_val.pkl' %(self.model_type,epoch,epoch_loss,val_loss))
				torch.save(best_unet,unet_best_path)

			if ((self.num_epochs-epoch)<5):
				final_unet = self.model.state_dict()
				unet_final_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-final.pkl' %(self.model_type,epoch,epoch_loss,val_loss))
				torch.save(final_unet,unet_final_path)
		plt.ioff()
		plt.show()
		self.finished.emit()
	
	def fine_tuning(self):
		#====================================== Re-Training ===========================================#
		modelt_path = os.path.join(self.model_path,"fine-tuned-sinusoids")

		startTime = time.time()
		lr = self.lr
		best_unet_score = 0.
		best_unet = None
		best_epoch = 0
		best_loss = 1000
		train_loss = 1000
			
		images_patches, GT_patches = self.train_loader.dataset.__getitem__(6)
		for epoch in tqdm(range(self.num_epochs)):
			self.model.train(True)
			epoch_loss = 0
			val_loss = 0
			test_loss = 0
				
			print("epoch", epoch+1)
				
			for i, (images_patches, GT_patches) in enumerate(self.train_loader):
				for i in range(images_patches.size(1)):
					images = images_patches[:, i, :, :, :].to(self.device).squeeze(1)
					GT = GT_patches[:, i, :, :, :].to(self.device).squeeze(1).floar32()
					
					#inputs = self.processor(images=images, segmentation_maps=GT, return_tensors="pt")
					#output = self.model(**inputs)
					output = self.model(pixel_values=images, mask_labels=GT)
					
					print(self.model)
					loss = output.loss
					
					epoch_loss += loss.item()
					
					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					# visualization
					self.checker.set_input_output(images, output, GT)
					self.update_signal.emit()

			# Print the log info
			print("[INFO] EPOCH: {}/{}".format(epoch+1, self.num_epochs))
			print("Train loss: {:.6f}".format(epoch_loss))
				

			# Decay learning rate
			if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
				lr -= (self.lr / float(self.num_epochs_decay))
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr
				print ('Decay learning rate to lr: {}.'.format(lr))
				
				
			#===================================== Validation ====================================#
			self.model.train(False)
			self.model.eval()
			for i, (imageval, GTval) in enumerate(self.valid_loader):
				for i in range(imageval.size(1)):
					images = imageval[:, i, :, :, :].to(self.device)
					GT = GTval[:, i, :, :, :].to(self.device)
					#inputs = self.processor(images=images, segmentation_maps=GT, return_tensors="pt")
					#output = self.model(**inputs)
					output = self.model(pixel_values=images, mask_labels=GT)
					
					loss_val = output.loss
					val_loss += loss_val.item()

			print('[Validation] loss: %.4f'%(val_loss))

			if (epoch_loss < train_loss):
				train_loss = epoch_loss
				best_model = self.model.state_dict()
				model_best_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-best_train.pkl' %(self.model_type,epoch,epoch_loss,val_loss))
				torch.save(best_model,model_best_path)

			if (val_loss < best_loss):
				best_loss = val_loss
				best_model = self.model.state_dict()
				model_best_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-best_val.pkl' %(self.model_type,epoch,epoch_loss,val_loss))
				torch.save(best_model,model_best_path)

			if ((self.num_epochs-epoch)<<5):
				final_model = self.model.state_dict()
				model_final_path = os.path.join(self.model_path, '%s-%d-%.4f-%.4f-final.pkl' %(self.model_type,epoch,epoch_loss,val_loss))
				torch.save(final_model,model_final_path)
		self.finished.emit()

	def compute_metrics(self, eval_pred):
		self.metric =  evaluate.load("mean_iou")
		with torch.no_grad():
			logits, labels = eval_pred
			logits_tensor = torch.from_numpy(logits)
			# scale the logits to the size of the label
			logits_tensor = nn.functional.interpolate(
				logits_tensor,
				size=labels.shape[-2:],
				mode="bilinear",
				align_corners=False,
			).argmax(dim=1)

			pred_labels = logits_tensor.detach().cpu().numpy()
			metrics = self.metric.compute(
				predictions=pred_labels,
				references=labels,
				num_labels=len(self.id2label),
				ignore_index=0,
				reduce_labels=self.processor.do_reduce_labels,
			)
    
		# add per category metrics as individual key-value pairs
		per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
		per_category_iou = metrics.pop("per_category_iou").tolist()

		metrics.update({f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
		metrics.update({f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)})
    
		return metrics

	def huggingface_trainer(self, train_set, valid_set):
		#====================================== Arguments ===========================================#
		#===========================================================================================#
		model_path = os.path.join(self.model_path,"fine-tuned-sinusoids")
		epochs = 50
		lr = 0.00006
		batch_size = 2
		hub_model_id = "segformer-b0-finetuned-sinusoids"
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
		training_args = TrainingArguments(output_dir=model_path, eval_strategy="epoch") #TrainingArguments()#model_path,learning_rate=lr, num_train_epochs=self.num_epochs, per_device_train_batch_size=2, per_device_eval_batch_size=2)#, save_total_limit=3,	eval_strategy="no", save_strategy="no", save_steps=100, eval_steps=100, logging_steps=10, eval_accumulation_steps=2, load_best_model_at_end=True, push_to_hub=False,)#, hub_model_id=hub_model_id, hub_strategy="end")

		#====================================== Trainer ===========================================#
		self.trainer = Trainer(model=self.model, train_dataset=train_set, eval_dataset=valid_set, compute_metrics=self.compute_metrics)
		self.trainer.train()
		self.model.save_pretrained(model_path)

	