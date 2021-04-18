#!/usr/bin/python3
"""

"""


import sys
import os
import argparse
import logging
import pathlib
import itertools

import cv2
import numpy as np

import network.SSD as net
from network.predictor import Predictor 
from network.SSD import MatchPrior
from network.multibox_loss import MultiboxLoss


from datasets.EPFL_dataset import EPFLDataset
from dataloaders.data_preprocessing import TrainAugmentation, TestTransform
from config import mobilenetv1_ssd_config as config

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset

from utils import box_utils, measurements
from utils.misc import str2bool, Timer, store_labels




DATASET_PATH_VID = "/media/leen/Acer_500GB_HDD/Imagenet_VID_dataset/ILSVRC/"
DATASET_PATH_VID_MOUNTED ="./mount_dataset/Imagenet_VID_dataset/ILSVRC/"

DATASET_PATH_EPFL = "/media/leen/Acer_500GB_HDD/EPFL/"
DATASET_PATH_EPFL_MOUNTED ="./mount_dataset/EPFL/"
LABEL_PATH_VID_DEFAULT ="./models/vid-model-labels.txt"

#Parser
parser = argparse.ArgumentParser(
	description='Mobile Video Object Detection (Bottleneck LSTM) Training With Pytorch')
parser.add_argument('--net', default="mobv2-ssdl",
					help="The network architecture, it should be of mobv1-ssdl, mobv2-ssdl, mobv1-ssdl-lstm3, mobv2-ssdl-lstm3")
parser.add_argument('--datasets', default = DATASET_PATH_EPFL, help='Dataset directory path')
parser.add_argument('--cache_path', default = DATASET_PATH_EPFL, help='Cache directory path')
parser.add_argument('--freeze_net', action='store_true',
					help="Freeze all the layers except the prediction head.")
parser.add_argument('--width_mult', default=1.0, type=float,
					help='Width Multiplifier')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
					help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
					help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
					help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
					help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
					help='initial learning rate for base net.')
parser.add_argument('--ssd_lr', default=None, type=float,
					help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--pretrained', help='Pre-trained model')
parser.add_argument('--resume', default= 'models/mb2-ssd-lite-mp-0_686.pth', type=str,
					help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
					help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
					help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
					help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=1, type=int,
					help='Batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
					help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
					help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
					help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
					help='Set the debug log output frequency.')
parser.add_argument('--sequence_length', default=10, type=int,
					help='sequence_length of video to unfold')
parser.add_argument('--use_cuda', default=True, type=str2bool,
					help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
					help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
	torch.backends.cudnn.benchmark = True
	logging.info("Use Cuda.")

#Definities
def select_model(args):#predictors hierbijzetten?
	if args.net == 'mobv1-ssdl':
		model = net.mobv1_ssdlite_create(num_classes = 1024, alpha=args.width_mult, is_test=False)

	elif args.net == 'mobv2-ssdl':
		model = net.mobv2_ssdlite_create(num_classes = 21, alpha=args.width_mult, is_test=False)

	elif args.net == 'mobv1-ssdl-lstm3':
		model = net.mobv2_ssdlite_lstm3_create(num_classes = 1024, alpha=args.width_mult, is_test=False)
	
	elif args.net == 'mobv2-ssdl-lstm3':
		model = net.mobv2_ssdlite_lstm3_create(num_classes = 21, alpha=args.width_mult, is_test=False)

	else:
		logging.fatal("The net type is wrong. It should be one of mobv1-ssdl, mobv2-ssdl, mobv1-ssdl-lstm3, mobv2-ssdl-lstm3.")
		parser.print_help(sys.stderr)
		sys.exit(1)  

	return model

def load_EPFL_dataset(args):
	""" Load EPFLDataset.
	Rerturns:
		validation dataset
		training dataset

	"""
	train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
	target_transform = net.MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)

	train_dataset = EPFLDataset(args.datasets, args.cache_path, transform=train_transform, target_transform=target_transform, batch_size=args.batch_size)
	val_dataset = EPFLDataset(args.datasets, args.cache_path, transform=train_transform, target_transform=target_transform, batch_size=args.batch_size, is_val=True)

	return train_dataset, val_dataset

def train(loader, model, criterion, optimizer, device, debug_steps=100, epoch=-1, sequence_length=10):
	""" Train model
	Arguments:
		model : object of MobileVOD class
		loader : validation data loader object
		criterion : Loss function to use
		device : device on which computation is done
		optimizer : optimizer to optimize model
		debug_steps : number of steps after which model needs to debug
		sequence_length : unroll length of model
		epoch : current epoch number
	"""
	model.train(True)
	running_loss = 0.0
	running_regression_loss = 0.0
	running_classification_loss = 0.0
	for i, data in enumerate(loader):
		images, boxes, labels = data
		for image, box, label in zip(images, boxes, labels):
			image = image.to(device)
			box = box.to(device)
			label = label.to(device)

			optimizer.zero_grad()
			confidence, locations = model(image)
			regression_loss, classification_loss = criterion(confidence, locations, label, box)  # TODO CHANGE BOXES
			loss = regression_loss + classification_loss
			loss.backward(retain_graph=True)
			optimizer.step()

			running_loss += loss.item()
			running_regression_loss += regression_loss.item()
			running_classification_loss += classification_loss.item()
		model.detach_hidden()
		if i and i % debug_steps == 0:
			avg_loss = running_loss / (debug_steps*sequence_length)
			avg_reg_loss = running_regression_loss / (debug_steps*sequence_length)
			avg_clf_loss = running_classification_loss / (debug_steps*sequence_length)
			logging.info(
				f"Epoch: {epoch}, Step: {i}, " +
				f"Average Loss: {avg_loss:.4f}, " +
				f"Average Regression Loss {avg_reg_loss:.4f}, " +
				f"Average Classification Loss: {avg_clf_loss:.4f}"
			)
			running_loss = 0.0
			running_regression_loss = 0.0
			running_classification_loss = 0.0
	model.detach_hidden()


def val(loader, model, criterion, device):
	""" Validate model
	Arguments:
		model : object of MobileVOD class
		loader : validation data loader object
		criterion : Loss function to use
		device : device on which computation is done
	Returns:
		loss, regression loss, classification loss
	"""
	model.eval()
	running_loss = 0.0
	running_regression_loss = 0.0
	running_classification_loss = 0.0
	num = 0
	for _, data in enumerate(loader):
		images, boxes, labels = data
		for image, box, label in zip (images, boxes, labels):
			image = image.to(device)
			box = box.to(device)
			label = label.to(device)
			num += 1

			with torch.no_grad():
				confidence, locations = model(image)
				regression_loss, classification_loss = criterion(confidence, locations, label, box)
				loss = regression_loss + classification_loss

			running_loss += loss.item()
			running_regression_loss += regression_loss.item()
			running_classification_loss += classification_loss.item()
		model.detach_hidden()
	return running_loss / num, running_regression_loss / num, running_classification_loss / num

def initialize_model(model):
	""" Loads learned weights from pretrained checkpoint model
	Arguments:
		model : object of SSD
	"""
	if args.pretrained:
		logging.info("Loading weights from pretrained netwok")
		pretrained_net_dict = torch.load(args.pretrained)
		model_dict = net.state_dict()
		# 1. filter out unnecessary keys
		pretrained_dict = {k: v for k, v in pretrained_net_dict.items() if k in model_dict and model_dict[k].shape == pretrained_net_dict[k].shape}
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		net.load_state_dict(model_dict)
	#enkel pretrained gewichten voor base_net
	model.base_net.load_state_dict(base_net_dict)


#Main
if __name__=='__main__':
	timer = Timer()

	logging.info(args)
	#config = config	#config file for priors etc.

	train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
	target_transform = MatchPrior(config.priors, config.center_variance, config.size_variance, 0.5)
	test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

	#Preparing EPFL dataset train + val
	logging.info("Prepare training and val datasets.")
	train_dataset, val_dataset = load_EPFL_dataset(args)

	label_file = os.path.join("models/", "EPFL-labels.txt")
	store_labels(label_file, train_dataset._classes_names)
	num_classes = len(train_dataset._classes_names)

	logging.info(f"Stored labels into file {label_file}.")
	logging.info("Train dataset size: {}".format(len(train_dataset)))
	train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)#Shuffle??? num_workers = threads?

	logging.info("validation dataset size: {}".format(len(val_dataset)))
	val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False) #Shuffle false??

	#Weights
	logging.info("Build network.")
	model = select_model(args)
		#Pretrained weights of resume
	if args.resume is None:
		initialize_model(model)
	else:
		model = select_model(args)
		print("Updating weights from resume model")
		resume_dict = torch.load(args.resume, map_location=lambda storage, loc: storage)
		model.load_state_dict(resume_dict)



	min_loss = -10000.0
	last_epoch = -1

	#Apparte learning rates voor base_net en ssd (extras), anders standaard lr
	base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
	ssd_lr = args.ssd_lr if args.ssd_lr is not None else args.lr

	#Weights vastleggen van pretrained? 
	if args.freeze_net:
		logging.info("Freeze net.")
		for param in model.basenet.parameters():
			param.requires_grad = False
		#Freezing upto new lstm layer, niet nodig want vertrekken van Mobilenetv2 weights?
		#model.extras.requires_grad = False
		#net.pred_decoder.bottleneck_lstm1.requires_grad = False
		#net.pred_decoder.fmaps_1.requires_grad = False
		#net.pred_decoder.bottleneck_lstm2.requires_grad = False
		#net.pred_decoder.fmaps_2.requires_grad = False

	model.to(DEVICE)

	#wat?
	criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
							 center_variance=0.1, size_variance=0.2, device=DEVICE)

	optimizer = torch.optim.RMSprop([{'params': [param for name, param in model.base_net.named_parameters()], 'lr': base_net_lr},
		{'params': [param for name, param in model.extras.named_parameters()], 'lr': ssd_lr},], lr=args.lr, momentum=args.momentum,
								weight_decay=args.weight_decay)
	logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
				 + f"Extra Layers learning rate: {ssd_lr}.")
"""

	# if args.scheduler == 'multi-step':
	# 	logging.info("Uses MultiStepLR scheduler.")
	# 	milestones = [int(v.strip()) for v in args.milestones.split(",")]
	# 	scheduler = MultiStepLR(optimizer, milestones=milestones,
	# 												 gamma=0.1, last_epoch=last_epoch)
	# elif args.scheduler == 'cosine':
	# 	logging.info("Uses CosineAnnealingLR scheduler.")
	# 	scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
	# else:
	# 	logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
	# 	parser.print_help(sys.stderr)
	# 	sys.exit(1)
	output_path = os.path.join(args.checkpoint_folder, f"lstm3")
	if not os.path.exists(output_path):
		os.makedirs(os.path.join(output_path))
	logging.info(f"Start training from epoch {last_epoch + 1}.")
	for epoch in range(last_epoch + 1, args.num_epochs):
		#scheduler.step()
		train(train_loader, net, criterion, optimizer,
			  device=DEVICE, debug_steps=args.debug_steps, epoch=epoch, sequence_length=args.sequence_length)
		
		if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
			val_loss, val_regression_loss, val_classification_loss = val(val_loader, net, criterion, DEVICE)
			logging.info(
				f"Epoch: {epoch}, " +
				f"Validation Loss: {val_loss:.4f}, " +
				f"Validation Regression Loss {val_regression_loss:.4f}, " +
				f"Validation Classification Loss: {val_classification_loss:.4f}"
			)
			model_path = os.path.join(output_path, f"WM-{args.width_mult}-Epoch-{epoch}-Loss-{val_loss}.pth")
			torch.save(net.state_dict(), model_path)
			logging.info(f"Saved model {model_path}")


"""

	
