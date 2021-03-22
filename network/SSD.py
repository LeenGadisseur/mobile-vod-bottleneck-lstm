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


def mobv1_ssd_create(num_classes, is_test=False):

	extras = ModuleList([
		nn.Sequential(	
			nn.Conv2d(in_channels=int(1024*alpha), out_channels=int(256*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=256*alpha, out_channels=512*alpha, kernel_size=3, stride=2, padding=1),
		)
		nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)

		nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
		nn.Sequential(	
			nn.Conv2d(in_channels=int(256*alpha), out_channels=int(128*alpha), kernel_size=1),
			nn.ReLU6(inplace=True),
			SeperableConv2d(in_channels=128*alpha, out_channels=256*alpha, kernel_size=3, stride=2, padding=1),
		)
	])

		
	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * 4, kernel_size=1),
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * num_classes, kernel_size=1),
	])

	SSD = SSD(num_classes, extras, classification_headers, regression_headers, is_test=is_test, config=config)

	return SSD


def mobv2_ssdlite_create(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
	extras = ModuleList([
		InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
		InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
		InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
		InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
	])
	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * 4, kernel_size=1),
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * num_classes, kernel_size=1),
	])

	SSD = SSD(num_classes, extras, classification_headers, regression_headers, is_test=is_test, config=config)

	return SSD


def mobv2_ssdlite_create(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
	extras = ModuleList([
		InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
		InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
		InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
		InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
	])
	regression_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * 4, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * 4, kernel_size=1),
	])

	classification_headers = nn.ModuleList([
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=1024*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=512*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		SeperableConv2d(in_channels=256*alpha, out_channels=6 * num_classes, kernel_size=3, padding=1),
		nn.Conv2d(in_channels=int(256*alpha), out_channels=6 * num_classes, kernel_size=1),
	])

	SSD = SSD(num_classes, extras, classification_headers, regression_headers, is_test=is_test, config=config)

	return SSD

class SSD(nn.Module):
	def __init__(self, num_classes, alpha = 1, is_test=False, config = None, device = None):
		"""
		Arguments:
			num_classes : an int variable having value of total number of classes
			alpha : a float used as width multiplier for channels of model
			is_Test : a bool used to make model ready for testing
			config : a dict containing all the configuration parameters 
		"""
		super(SSD, self).__init__()
		# Decoder
		self.extras = extras
		self.classification_headers = classification_headers
		self.regression_headers = regression_headers
		
		self.is_test = is_test
		self.config = config
		self.num_classes = num_classes


		if device:
			self.device = device
		else:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		if is_test:
			self.config = config
			self.priors = config.priors.to(self.device)

		
		logging.info("Initializing weights of SSD")
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
		header_index=0
		
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)
		x = self.conv13(x)
		x = self.conv14(x)
		confidence, location = self.compute_header(header_index, x)
		header_index += 1
		confidences.append(confidence)
		locations.append(location)

		for layer in self.extras:
			x = layer(x)
			confidence, location = self.compute_header(header_index, x)
			header_index += 1
			confidences.append(confidence)
			locations.append(location)

		
		if self.is_test:
			confidences = F.softmax(confidences, dim=2)
			boxes = box_utils.convert_locations_to_boxes(
				locations, self.priors, self.config.center_variance, self.config.size_variance
			)
			boxes = box_utils.center_form_to_corner_form(boxes)
			return confidences, boxes
		else:
			return confidences, locations

