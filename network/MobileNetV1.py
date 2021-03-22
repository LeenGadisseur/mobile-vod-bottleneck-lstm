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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

def conv_bn(inp, oup, stride):
	"""3x3 conv with batchnorm and relu
	Arguments:
		inp : number of channels of input
		oup : number of channels of output
		stride : stride for depthwise convolution
	Returns:
		object of class torch.nn.Sequential
	"""
	return nn.Sequential(
				nn.Conv2d(int(inp), int(oup), 3, stride, 1, bias=False),
				nn.BatchNorm2d(int(oup)),
				nn.ReLU6(inplace=True)
			)


def conv_dw(inp, oup, stride):
	"""Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d having batchnorm and relu layers in between.
	Here kernel size is fixed at 3.
	Arguments:
		inp : number of channels of input
		oup : number of channels of output
		stride : stride for depthwise convolution
	Returns:
		object of class torch.nn.Sequential
	"""
	return nn.Sequential(
				nn.Conv2d(int(inp), int(inp), 3, stride, 1, groups=int(inp), bias=False),
				nn.BatchNorm2d(int(inp)),
				nn.ReLU6(inplace=True),

				nn.Conv2d(int(inp), int(oup), 1, 1, 0, bias=False),
				nn.BatchNorm2d(int(oup)),
				nn.ReLU6(inplace=True),
			)

#########################################################################################
# Klassen: MobileNetV1, Inverted residuals, MobileNetV2, SSD, MobileVOD 
#########################################################################################

class MobileNetV1(nn.Module):
	def __init__(self, num_classes=1024, alpha=1):
		""" torch.nn.module for mobilenetv1 upto conv12
		Arguments:
			num_classes : an int variable having value of total number of classes
			alpha : a float used as width multiplier for channels of model
		"""
		super(MobileNetV1, self).__init__()
		# upto conv 12, 13e lijkt ook nog van mobilenet, fout?
		self.model = nn.Sequential(
			conv_bn(3, 32*alpha, 2),
			conv_dw(32*alpha, 64*alpha, 1),
			conv_dw(64*alpha, 128*alpha, 2),
			conv_dw(128*alpha, 128*alpha, 1),
			conv_dw(128*alpha, 256*alpha, 2),
			conv_dw(256*alpha, 256*alpha, 1),
			conv_dw(256*alpha, 512*alpha, 2),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			conv_dw(512*alpha, 512*alpha, 1),
			#conv_dw(512*alpha, 1024*alpha, 2),#Deze staat bij SSD?
			conv_dw(512*alpha, 1024*alpha, 2)
			conv_dw(1024*alpha,1024*alpha, 1) #to be pruned while adding LSTM layers
			)
		logging.info("Initializing weights of base net")
		self._initialize_weights()
		#self.fc = nn.Linear(1024, num_classes)
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
			
	def forward(self, x):
		"""
		Arguments:
			x : a tensor which is used as input for the model
		Returns:
			a tensor which is output of the model 
		"""
		x = self.model(x)
		return x

