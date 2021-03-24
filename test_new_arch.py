#!/usr/bin/python3
"""

"""
import torch
import network.SSD as net
from network.predictor import Predictor 
from datasets.vid_dataset import ImagenetDataset
from config import mobilenetv1_ssd_config
from utils import box_utils, measurements
from utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import logging
import sys

if __name__=='__main__':
	
	MB1_SSDLite = net.mobv1_ssdlite_create(num_classes = 1024, alpha=1., is_test=False)
	print(MB1_SSDLite)
	MB2_SSDLite = net.mobv2_ssdlite_create(num_classes = 1024, alpha=1., is_test=False)
	print(MB2_SSDLite)
