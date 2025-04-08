from math import ceil, sqrt
import tkinter as tk
from tkinter import ttk
import numpy
import torch
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PyQt5.QtCore import pyqtSignal, QObject



class Checker(QObject):
	update_signal = pyqtSignal()
	def __init__(self, solver):
		super().__init__()
		self.solver = solver
		#plt.ion()
		self.fig = Figure(figsize=(12, 12))
		self.canvas = FigureCanvas(self.fig)
		self.axes = None
		self.flag = False
		self.layer = None
		self.hook_layer_index =None
		self.save_output = None
		self.hook_handle = None
		self.hook_grad_handle = None
		self.current_grad_output = None
		self.input = None
		self.output = None
		self.Gt = None
	
	def set_hook_layer(self, layer_index):
		self.hook_layer_index = layer_index
		if self.flag:
			self.update_hook()

	def attach_hook(self):
		if self.solver.model is not None:
			if self.hook_handle is not None:
				self.hook_handle.remove()
			self.save_output = None
			self.layer = dict(self.solver.model.named_modules())[self.hook_layer_index]
			print("Layer updated:", self.layer)
			self.hook_handle = self.layer.register_forward_hook(self.hook_fn)
			print("Hook attached.")

		
	def hook_fn(self, module, input, output):
		with self.solver.lock:
			self.save_output = output.detach()
			self.update_signal.emit()
	
			
	def remove_hook(self):
		if self.hook_handle is not None:
			self.hook_handle.remove()
			self.hook_handle = None
			print("Hook removed.")
	
	def update_hook(self):
		if self.flag:
			self.attach_hook()
		else:
			if self.hook_handle is not None:
				self.remove_hook()


	def set_input_output(self, input, output, Gt):
		if self.flag:
			self.input = input
			self.output = output
			self.Gt = Gt
        

	def visualize_samples(self):
		with self.solver.lock:
			if self.save_output is None:
				print("No activation available. Run a forward pass first.")
				return
			if self.input is None:
				return

			raws = self.save_output.size(0)
			cols = self.save_output.size(1)
			max_channels = min(cols, 8)
			images = self.save_output
			if self.fig is not None:
				self.fig.clear()
			axes = self.fig.subplots(raws, max_channels+3)
			self.axes = numpy.array(axes).reshape(raws, max_channels+3) if raws > 1 else numpy.array([[axes]])

			# visual inputs
			if (self.input == None):
				return
			for i in range(self.input.size(0)):
				image = self.input[i, :, :].squeeze(0).byte().cpu().numpy()
				self.axes[i,0].imshow(image)
				self.axes[i,0].axis('off')

			# visualize the layer
			for i in range(raws):
				for j in range(max_channels):
					image = images[i, j , :, :].byte().cpu().numpy()
					# k = (image.max()-image.max())
					# if k==0:
					# 	image = 0* (image) 
					# else:
					# 	image = 255* (image - image.min()) / k
					image = image*255
					self.axes[i,j+1].clear()
					self.axes[i,j+1].imshow(image)
					self.axes[i,j+1].axis('off')

			# visual output and ground truth
			for i in range(self.output.size(0)):
				out = self.output[i, :, :].squeeze(0)
				out[out >= 0.5] = 255
				out = out.byte().cpu().numpy()
				self.axes[i,max_channels+1].imshow(out)
				self.axes[i,max_channels+1].axis('off')
				gt = self.Gt[i, :, :].squeeze(0).byte().cpu().numpy()
				self.axes[i,max_channels+2].imshow(gt)
				self.axes[i,max_channels+2].axis('off')
			
			self.canvas.draw()


	def visualize_channels(self):
		with self.solver.lock:
			if self.save_output is None:
				print("No activation available. Run a forward pass first.")
				return
			if self.input is None:
				return
		
			img_num = self.save_output.size(1)
			#max_channels = min(cols, 8)
			self.dim = ceil(sqrt(img_num))
			images = self.save_output[1, :, :, :]

			if self.fig is not None:
				self.fig.clear()
			# Create new subplots
			self.dim = max(self.dim, 3)
			axes = self.fig.subplots(self.dim+1, self.dim)
			#self.fig, axes = plt.subplots(self.dim, self.dim, figsize=(12, 12))
			self.axes = numpy.array(axes).reshape(self.dim+1, self.dim) if self.dim > 1 else numpy.array([[axes]])

			# visual inputs, outputs and GT
			if (self.input == None):
				return
			input = self.input[1, :, :].squeeze(0).byte().cpu().numpy()
			self.axes[0,0].imshow(input)
			self.axes[0,0].axis('off')
			out = self.output[1, :, :].squeeze(0)
			out[out >= 0.5] = 255
			out = out.byte().cpu().numpy()
			self.axes[0,1].imshow(out)
			self.axes[0,1].axis('off')
			gt = self.Gt[1, :, :].squeeze(0).byte().cpu().numpy()
			self.axes[0,2].imshow(gt)
			self.axes[0,2].axis('off')
			for j in range(3, self.dim):
				self.axes[0,j].axis('off')

			#print("visualization...")
			for i in range(self.dim):
				for j in range(self.dim):
					if i*self.dim+j < img_num:
						# if (len(images.shape) >> 3):
						# 	images = self.images[:, 1, :, :]
						image = images[i*self.dim+j , :, :].byte().cpu().numpy()
						# k = (image.max()-image.max())
						# if k==0:
						# 	image = 0* (image )
						# else:
						# 	image = 255* (image - image.min()) / k
						image = image*255
						self.axes[i+1,j].clear()
						self.axes[i+1,j].imshow(image)
						self.axes[i+1,j].axis('off')
					else:
						self.axes[i+1,j].axis('off')
			self.canvas.draw()

	def register_for_grad(self):
		if self.layer is None:
			self.layer = dict(self.solver.model.named_modules())[self.hook_layer_index]
		self.hook_grad_handle = self.layer.register_full_backward_hook(self.set_grad)

	def set_grad(self, module, grad_input, grad_output):
		self.current_grad_output = grad_output

	def get_grad(self):
		return self.current_grad_output

	def deregister_for_grad(self):
		if self.hook_grad_handle is not None:
			self.hook_grad_handle.remove()
			self.hook_grad_handle = None

	def get_statis(self):
		with self.solver.lock:
			if self.save_output is None:
				return 'x', 'x', 'x', 'x'
			else:
				data = self.save_output
				data_min = data.min()
				data_max = data.max()
				data_mean = data.mean()
				data_std = data.std()
		return data_min, data_max, data_mean, data_std