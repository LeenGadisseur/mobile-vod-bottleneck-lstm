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


#########################################################################################
# Definities
#########################################################################################

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
		nn.ReLU6(),
		nn.Conv2d(in_channels=int(in_channels), out_channels=int(out_channels), kernel_size=1),
	)

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
#NEW
def conv_1x1_bn(inp, oup):
	"""1x1 conv with batchnorm and relu
	Arguments:
		inp : number of channels of input
		oup : number of channels of output
		stride : stride for depthwise convolution
	Returns:
		object of class torch.nn.Sequential
	Afkomstig van zelfde github MobileNetv2 (zie link bij MobileNetV2)
	"""
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
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

#NEW
def _make_divisible(v, divisor, min_value=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	:param v:
	:param divisor:
	:param min_value:
	:return:
	"""
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


#################################################################
# Classe MobileNetV2 toevoegen + bijhorende structuren
# https://github.com/d-li14/mobilenetv2.pytorch/blob/1733532bd43743442077326e1efc556d7cfd025d/models/imagenet/mobilenetv2.py#L91
#################################################################
class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expand_ratio)
		self.identity = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(
			# dw
			nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
			nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
			# pw
			nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.ReLU6(inplace=True),
			# dw
			nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.ReLU6(inplace=True),
			# pw-linear
			nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
			nn.BatchNorm2d(oup),
			)
	def forward(self, x):
		if self.identity:
			return x + self.conv(x)
		else:
			return self.conv(x)



class MobileNetV2(nn.Module):
	def __init__(self, num_classes=1000, width_mult=1.):
		super(MobileNetV2, self).__init__()
		# setting of inverted residual blocks
		self.cfgs = [
			# t, c, n, s
			[1,  16, 1, 1],
			[6,  24, 2, 2],
			[6,  32, 3, 2],
			[6,  64, 4, 2],
			[6,  96, 3, 1],
			[6, 160, 3, 2],
			[6, 320, 1, 1],
		]

		# building first layer
		input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
		layers = [conv_bn(3, input_channel, 2)]

		# building inverted residual blocks
		block = InvertedResidual
		for t, c, n, s in self.cfgs:
			output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
		for i in range(n):
			layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
			input_channel = output_channel
		self.features = nn.Sequential(*layers)

		# building last several layers
		output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
		self.conv = conv_1x1_bn(input_channel, output_channel)
		#self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #origineel
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		
		#FC - uitzetten voor SSD gebruik
		#self.classifier = nn.Linear(output_channel, num_classes)

		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = self.conv(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None:
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
#################################################################
