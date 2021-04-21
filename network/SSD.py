#!/usr/bin/python3
"""Script for creating basenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import List, Tuple
from utils import box_utils
from collections import namedtuple
from collections import OrderedDict


from .MobileNetV1 import MobileNetV1
from .MobileNetV2 import MobileNetV2, InvertedResidual
from .config import mobilenetv1_ssd_config as config


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import numpy as np
import logging


GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

############################################################################################################
# Definities voor lagen/onderdelen in SSD 
############################################################################################################

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
	"""Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
	Arguments:
		in_channels : number of channels of input
		out_channels : number of channels of output
		kernel_size : kernel size for depthwise convolution
		stride : stride for depthwise convolution
		padding : padding for depthwise convolution
	Returns:
		object of class torch.nn.Sequential
	"""
	return nn.Sequential(
		nn.Conv2d(in_channels=int(in_channels), out_channels=int(in_channels), kernel_size=kernel_size,
			   groups=int(in_channels), stride=stride, padding=padding),
		nn.BatchNorm2d(in_channels),
		nn.ReLU6(),
		nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=1),
	)

class MatchPrior(object):
	"""Matches priors based on the SSD prior config
	Arguments:
		center_form_priors : priors generated based on specs and image size in config file
		center_variance : a float used to change the scale of center
		size_variance : a float used to change the scale of size
		iou_threshold : a float value of thresholf of IOU
	"""
	def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
		self.center_form_priors = center_form_priors
		self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
		self.center_variance = center_variance
		self.size_variance = size_variance
		self.iou_threshold = iou_threshold

	def __call__(self, gt_boxes, gt_labels):
		"""
		Arguments:
			gt_boxes : ground truth boxes
			gt_labels : ground truth labels
		Returns:
			locations of form (batch_size, num_priors, 4) and labels
		"""
		if type(gt_boxes) is np.ndarray:
			gt_boxes = torch.from_numpy(gt_boxes)
		if type(gt_labels) is np.ndarray:
			gt_labels = torch.from_numpy(gt_labels)
		boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
												self.corner_form_priors, self.iou_threshold)
		boxes = box_utils.corner_form_to_center_form(boxes)
		locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
		return locations, labels

def crop_like(x, target):
	"""
	Arguments:
		x : a tensor whose shape has to be cropped
		target : a tensor whose shape has to assert on x
	Returns:
		x having same shape as target
	"""
	if x.size()[2:] == target.size()[2:]:
		return x
	else:
		height = target.size()[2]
		width = target.size()[3]
		crop_h = torch.FloatTensor([x.size()[2]]).sub(height).div(-2)
		crop_w = torch.FloatTensor([x.size()[3]]).sub(width).div(-2)
	# fixed indexing for PyTorch 0.4
	return F.pad(x, [int(crop_w.ceil()[0]), int(crop_w.floor()[0]), int(crop_h.ceil()[0]), int(crop_h.floor()[0])])

class BottleneckLSTMCell(nn.Module):
	""" Creates a LSTM layer cell
	Arguments:
		input_channels : variable used to contain value of number of channels in input
		hidden_channels : variable used to contain value of number of channels in the hidden state of LSTM cell
	"""
	def __init__(self, input_channels, hidden_channels):
		super(BottleneckLSTMCell, self).__init__()

		#assert hidden_channels % 2 == 0

		self.input_channels = int(input_channels)
		self.hidden_channels = int(hidden_channels)
		self.num_features = 4
		self.W = nn.Conv2d(in_channels=self.input_channels, out_channels=self.input_channels, kernel_size=3, groups=self.input_channels, stride=1, padding=1)
		self.Wy  = nn.Conv2d(int(self.input_channels+self.hidden_channels), self.hidden_channels, kernel_size=1)
		self.Wi  = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, groups=self.hidden_channels, bias=False)  
		self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, 1, 0, bias=False)
		self.relu = nn.ReLU6()
		# self.Wci = None
		# self.Wcf = None
		# self.Wco = None
		logging.info("Initializing weights of lstm")
		self._initialize_weights()

	def _initialize_weights(self):
		"""
		Returns:
			initialized weights of the model
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def forward(self, x, h, c): #implemented as mentioned in paper here the only difference is  Wbi, Wbf, Wbc & Wbo are commuted all together in paper
		"""
		Arguments:
			x : input tensor
			h : hidden state tensor
			c : cell state tensor
		Returns:
			output tensor after LSTM cell 
		"""
		x = self.W(x)
		y = torch.cat((x, h),1) #concatenate input and hidden layers
		i = self.Wy(y) #reduce to hidden layer size
		b = self.Wi(i)	#depth wise 3*3
		ci = torch.sigmoid(self.Wbi(b))
		cf = torch.sigmoid(self.Wbf(b))
		cc = cf * c + ci * self.relu(self.Wbc(b))
		co = torch.sigmoid(self.Wbo(b))
		ch = co * self.relu(cc)
		return ch, cc

	def init_hidden(self, batch_size, hidden, shape):
		"""
		Arguments:
			batch_size : an int variable having value of batch size while training
			hidden : an int variable having value of number of channels in hidden state
			shape : an array containing shape of the hidden and cell state 
		Returns:
			cell state and hidden state
		"""
		# if self.Wci is None:
		# 	self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
		# 	self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
		# 	self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
		# else:
		# 	assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
		# 	assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
		return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
				Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda()
				)


class BottleneckLSTM(nn.Module):
	def __init__(self, input_channels, hidden_channels, height, width, batch_size):
		""" Creates Bottleneck LSTM layer
		Arguments:
			input_channels : variable having value of number of channels of input to this layer
			hidden_channels : variable having value of number of channels of hidden state of this layer
			height : an int variable having value of height of the input
			width : an int variable having value of width of the input
			batch_size : an int variable having value of batch_size of the input
		Returns:
			Output tensor of LSTM layer
		"""
		super(BottleneckLSTM, self).__init__()
		self.input_channels = int(input_channels)
		self.hidden_channels = int(hidden_channels)
		self.cell = BottleneckLSTMCell(self.input_channels, self.hidden_channels)
		(h, c) = self.cell.init_hidden(batch_size, hidden=self.hidden_channels, shape=(height, width))
		self.hidden_state = h
		self.cell_state = c

	def forward(self, input):
		new_h, new_c = self.cell(input, self.hidden_state, self.cell_state)
		self.hidden_state = new_h
		self.cell_state = new_c
		return self.hidden_state


###########################################################################################################
# Functies voor creeeren van objectdetectoren: MobileNetV1-SSDLite, MobileNetV2-SSDLite
###########################################################################################################
def mobv1_ssdlite_create(num_classes, alpha=1., is_test=False):
	#Verandert bij het gebruik van LSTM
	alpha_base = alpha	
	alpha_ssd = alpha
	alpha_lstm = alpha

	base_net = MobileNetV1(1001).model 

	source_layer_indexes = [12,14,]

	extras = nn.ModuleList([
		nn.Sequential(	
			nn.Conv2d(in_channels=int(1024*alpha_ssd), out_channels=int(256*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=256*alpha_ssd, out_channels=512*alpha_ssd, kernel_size=3, stride=2, padding=1),
		),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha_ssd), out_channels=int(128*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha_ssd, out_channels=256*alpha_ssd, kernel_size=3, stride=2, padding=1),
		),

		nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha_ssd), out_channels=int(128*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha_ssd, out_channels=256*alpha_ssd, kernel_size=3, stride=2, padding=1),
		),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha_ssd), out_channels=int(128*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha_ssd, out_channels=256*alpha_ssd, kernel_size=3, stride=2, padding=1),
		)
	])

		
	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha_ssd, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha_ssd, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha_ssd, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha_ssd, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha_ssd, out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha_ssd), out_channels=6 * 4, kernel_size=1),
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha_ssd, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha_ssd, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha_ssd, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha_ssd, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha_ssd, out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha_ssd), out_channels=6 * num_classes, kernel_size=1),
	])

	ssd =  SSD(num_classes=num_classes, base_net=base_net, source_layer_indexes=source_layer_indexes, extras=extras, classification_headers=classification_headers, regression_headers=regression_headers, is_test=is_test, config=config)

	return ssd


def mobv2_ssdlite_create(num_classes, alpha=1.0, use_batch_norm=True, is_test=False):
	#Verandert bij het gebruik van LSTM
	alpha_base = alpha	
	alpha_ssd = alpha
	alpha_lstm = alpha	

	base_net = MobileNetV2(width_mult=alpha_base, use_batch_norm=use_batch_norm).features
	

	source_layer_indexes = [GraphPath(14, 'conv', 3), 19,]
	
	extras = nn.ModuleList([
		InvertedResidual(int(1280*alpha_ssd), int(512*alpha_ssd), stride=2, expand_ratio=0.2, use_batch_norm=use_batch_norm),
		InvertedResidual(int(512*alpha_ssd), int(256*alpha_ssd), stride=2, expand_ratio=0.25, use_batch_norm=use_batch_norm),
		InvertedResidual(int(256*alpha_ssd), int(256*alpha_ssd), stride=2, expand_ratio=0.5, use_batch_norm=use_batch_norm),
		InvertedResidual(int(256*alpha_ssd), int(64*alpha_ssd), stride=2, expand_ratio=0.25, use_batch_norm=use_batch_norm)
	])	

	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(576*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(1280*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(256*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(256*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(64*alpha_ssd), out_channels=6 * 4, kernel_size=1)
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(576*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(1280*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(256*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(256*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(64*alpha_ssd), out_channels=6 * num_classes, kernel_size=1)
	])

	ssd = SSD(num_classes=num_classes, base_net=base_net, source_layer_indexes=source_layer_indexes, extras=extras, classification_headers=classification_headers, regression_headers=regression_headers, is_test=is_test, config=config)

	return ssd

###
#Netten met lstm
###
def mobv1_ssdlite_lstm4_create(num_classes, alpha=1., batch_size=None, is_test=False):
	alpha_base = alpha	
	alpha_ssd = 0.5*alpha
	alpha_lstm = 0.25*alpha

	base_net = MobileNetV1(1001).model 

	source_layer_indexes = [12,14,]

	extras = nn.ModuleList([
		BottleneckLSTM(input_channels=int(1024*alpha_lstm), hidden_channels=int(256*alpha_lstm), height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha_ssd), out_channels=int(128*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(128*alpha_ssd), out_channels=int(256*alpha_ssd), kernel_size=3, stride=2, padding=1),
		),

		BottleneckLSTM(input_channels=int(256*alpha_lstm), hidden_channels=int(64*alpha_lstm), height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(64*alpha_ssd), out_channels=int(32*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(32*alpha_ssd), out_channels=int(64*alpha_ssd), kernel_size=3, stride=2, padding=1),
		),

		BottleneckLSTM(input_channels=64*alpha_lstm, hidden_channels=16*alpha_lstm, height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(16*alpha_ssd), out_channels=int(8*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(8*alpha_ssd), out_channels=int(16*alpha_ssd), kernel_size=3, stride=2, padding=1),
		),

		BottleneckLSTM(input_channels=int(16*alpha_lstm), hidden_channels=int(16*alpha_lstm), height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(16*alpha_ssd), out_channels=int(8*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(8*alpha_ssd), out_channels=int(16*alpha_ssd), kernel_size=3, stride=2, padding=1),
		)
	])


		#op alles of enkel op laatste?
	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(256*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(64*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(16*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(16*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(16*alpha_ssd), out_channels=6 * 4, kernel_size=1),
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(256*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(64*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(16*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(16*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(16*alpha_ssd), out_channels=6 * num_classes, kernel_size=1),
	])

	ssd =  SSD(num_classes=num_classes, base_net=base_net, source_layer_indexes=source_layer_indexes, extras=extras, classification_headers=classification_headers, regression_headers=regression_headers, is_test=is_test, config=config)

	return ssd

def mobv2_ssdlite_IR_lstm4_create(num_classes, alpha=1.0, use_batch_norm=True, batch_size=None ,is_test=False):

	alpha_base = alpha	
	alpha_ssd = 0.5*alpha
	alpha_lstm = 0.25*alpha	

	base_net = MobileNetV2(width_mult=alpha_base, use_batch_norm=use_batch_norm).features
	

	source_layer_indexes = [GraphPath(14, 'conv', 3), 19,]
	
	extras = nn.ModuleList([
		BottleneckLSTM(input_channels=int(1280*alpha_lstm), hidden_channels=int(320*alpha_lstm), height=10, width=10, batch_size=batch_size),
		InvertedResidual(int(320*alpha_ssd), int(320*alpha_ssd), stride=2, expand_ratio=0.2, use_batch_norm=use_batch_norm),

		BottleneckLSTM(input_channels=int(320*alpha_lstm), hidden_channels=int(80*alpha_lstm), height=10, width=10, batch_size=batch_size),
		InvertedResidual(int(80*alpha_ssd), int(80*alpha_ssd), stride=2, expand_ratio=0.25, use_batch_norm=use_batch_norm),

		BottleneckLSTM(input_channels=int(80*alpha_lstm), hidden_channels=int(20*alpha_lstm), height=10, width=10, batch_size=batch_size),
		InvertedResidual(int(20*alpha_ssd), int(20*alpha_ssd), stride=2, expand_ratio=0.5, use_batch_norm=use_batch_norm),

		BottleneckLSTM(input_channels=int(20*alpha_lstm), hidden_channels=int(20*alpha_lstm), height=10, width=10, batch_size=batch_size),
		InvertedResidual(int(20*alpha_ssd), int(20*alpha_ssd), stride=2, expand_ratio=0.25, use_batch_norm=use_batch_norm)
	])	

	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(320*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(80*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(20*alpha_ssd), out_channels=6 * 4, kernel_size=1)
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(320*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(80*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(20*alpha_ssd), out_channels=6 * num_classes, kernel_size=1)
	])


	ssd = SSD(num_classes=num_classes, base_net=base_net, source_layer_indexes=source_layer_indexes, extras=extras, classification_headers=classification_headers, regression_headers=regression_headers, is_test=is_test, config=config)

	return ssd

def mobv2_ssdlite_lstm4_create(num_classes, alpha=1.0, use_batch_norm=True, batch_size=None ,is_test=False):

	alpha_base = alpha	
	alpha_ssd = 0.5*alpha
	alpha_lstm = 0.25*alpha	

	base_net = MobileNetV2(width_mult=alpha_base, use_batch_norm=use_batch_norm).features
	

	source_layer_indexes = [GraphPath(14, 'conv', 3), 19,]
	
	extras = nn.ModuleList([
		BottleneckLSTM(input_channels=int(1280*alpha_lstm), hidden_channels=int(320*alpha_lstm), height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(320*alpha_ssd), out_channels=int(160*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(160*alpha_ssd), out_channels=int(320*alpha_ssd), kernel_size=3, stride=2, padding=1),
		),

		BottleneckLSTM(input_channels=int(320*alpha_lstm), hidden_channels=int(80*alpha_lstm), height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(80*alpha_ssd), out_channels=int(40*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(40*alpha_ssd), out_channels=int(80*alpha_ssd), kernel_size=3, stride=2, padding=1),
		),

		BottleneckLSTM(input_channels=80*alpha_lstm, hidden_channels=20*alpha_lstm, height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(20*alpha_ssd), out_channels=int(10*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(10*alpha_ssd), out_channels=int(20*alpha_ssd), kernel_size=3, stride=2, padding=1),
		),

		BottleneckLSTM(input_channels=int(20*alpha_lstm), hidden_channels=int(20*alpha_lstm), height=10, width=10, batch_size=batch_size),
		nn.Sequential(	
			nn.Conv2d(in_channels=int(20*alpha_ssd), out_channels=int(10*alpha_ssd), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=int(10*alpha_ssd), out_channels=int(20*alpha_ssd), kernel_size=3, stride=2, padding=1),
		)
	])	

	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(320*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(80*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(20*alpha_ssd), out_channels=6 * 4, kernel_size=1)
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=int(512*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(320*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(80*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=int(20*alpha_ssd), out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(20*alpha_ssd), out_channels=6 * num_classes, kernel_size=1)
	])


	ssd = SSD(num_classes=num_classes, base_net=base_net, source_layer_indexes=source_layer_indexes, extras=extras, classification_headers=classification_headers, regression_headers=regression_headers, is_test=is_test, config=config)

	return ssd






#########################################################################################################
# Klasse SSD
#########################################################################################################

class SSD(nn.Module):
	def __init__(self, num_classes,  base_net, source_layer_indexes, extras, classification_headers, regression_headers, alpha = 1, is_test=False, config = None, device = None):
		"""
		Arguments:
			num_classes : an int variable having value of total number of classes
			alpha : a float used as width multiplier for channels of model
			is_Test : a bool used to make model ready for testing
			config : a dict containing all the configuration parameters 
		"""
		super(SSD, self).__init__()

		self.num_classes = num_classes
		self.base_net = base_net
		
		# Decoder
		self.source_layer_indexes = source_layer_indexes
		self.extras = extras
		self.classification_headers = classification_headers
		self.regression_headers = regression_headers
		
		self.is_test = is_test
		self.config = config
		
		self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes 
						if isinstance(t, tuple) and not isinstance(t, GraphPath)])
		if device:
			self.device = device
		else:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if is_test:
			self.config = config
			self.priors = config.priors.to(self.device)

		
		logging.info("Initializing weights of SSD.")
		self._initialize_weights()

	def _initialize_weights(self):
		"""
		Returns:
			initialized weights of the model
		"""
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			
	def compute_header(self, i, x):
		"""
		Arguments:
			i : an int used to use particular classification and regression layer
			x : a tensor used as input to layers
		Returns:
			locations and confidences of the predictions
		"""
		confidence = self.classification_headers[i](x)
		confidence = confidence.permute(0, 2, 3, 1).contiguous()
		confidence = confidence.view(confidence.size(0), -1, self.num_classes)

		location = self.regression_headers[i](x)
		location = location.permute(0, 2, 3, 1).contiguous()
		location = location.view(location.size(0), -1, 4)

		return confidence, location

	def forward(self, x):
		"""
		Arguments:
			x : a tensor which is used as input for the model
		Returns:
			confidences and locations of predictions made by model during training
			or
			confidences and boxes of predictions made by model during testing
		"""
		confidences = []
		locations = []
		header_index = 0
		start_layer_index = 0

		for end_layer_index in self.source_layer_indexes:
			#Bepaal end_layer
			if isinstance(end_layer_index, GraphPath):
				path = end_layer_index
				end_layer_index = end_layer_index.s0
				added_layer = None
			elif isinstance(end_layer_index, tuple):
				added_layer = end_layer_index[1]
				end_layer_index = end_layer_index[0]
				path = None
			else:
				added_layer = None
				path = None
			#Doorloop basenet van start tot end layer
			for layer in self.base_net[start_layer_index: end_layer_index]:
				x = layer(x)
				#print("Base_net layer")
				#print(layer)

			if added_layer:
				#print("Added layer")
				y = added_layer(x)
				#print(layer)
			else:
				y = x
			
			#Doorloop end_layer indien er een path is
			if path:
				sub = getattr(self.base_net[end_layer_index], path.name)
				for layer in sub[:path.s1]:
					#print("Path - layer")
					x = layer(x)
					#print(layer)
				y = x
				for layer in sub[path.s1:]:
					x = layer(x)
					#print(layer)
			#End_layer_index verhogen, start_layer_index vervangen
			#print("headers inc")
			end_layer_index += 1
			start_layer_index = end_layer_index
			#Berekenen header en inc header index
			confidence, location = self.compute_header(header_index, y)
			header_index += 1
			confidences.append(confidence)
			locations.append(location)
		
		
		# Doorlopen lagen van basenet, die niet bij in SSD zitten voor predicties (headers) 
		for layer in self.base_net[end_layer_index:]: 
			x = layer(x)
			#print("Laatste basenet laag")
			#print(layer)

		for layer in self.extras:
			x = layer(x)
			#print("Extra laag")
			#print(layer)
			confidence, location = self.compute_header(header_index, x)
			header_index += 1
			confidences.append(confidence)
			locations.append(location)

		confidences = torch.cat(confidences, 1)
		locations = torch.cat(locations, 1)		

		if self.is_test:
			confidences = F.softmax(confidences, dim=2)
			boxes = box_utils.convert_locations_to_boxes(
				locations, self.priors, self.config.center_variance, self.config.size_variance
			)
			boxes = box_utils.center_form_to_corner_form(boxes)
			return confidences, boxes
		else:
			return confidences, locations

