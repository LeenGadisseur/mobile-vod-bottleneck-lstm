#!/usr/bin/python3
"""
Testfile om verschillende kleinde dingen uit te testen vooralleer te implementeren in andere scripten
"""
import cv2
import torch
import argparse
import pathlib
import numpy as np
import logging
import sys
import network.SSD as net

from network.predictor import Predictor 
from datasets.vid_dataset import ImagenetDataset
from datasets.EPFL_dataset import EPFLDataset
from config import mobilenetv1_ssd_config as config
from utils import box_utils, measurements
from utils.misc import str2bool, Timer
from dataloaders.data_preprocessing import TrainAugmentation, TestTransform

#VIDDATASET PADEN
DATASET_PATH_VID = "/media/leen/Acer_500GB_HDD/Imagenet_VID_dataset/ILSVRC/"
DATASET_PATH_VID_MOUNTED ="./mount_dataset/Imagenet_VID_dataset/ILSVRC/"
LABEL_PATH_VID_DEFAULT ="./models/vid-model-labels.txt"

#EPFLDATASET PADEN
DATASET_PATH_EPFL = "/media/leen/Acer_500GB_HDD/EPFL/"
DATASET_PATH_EPFL_MOUNTED ="./mount_dataset/EPFL/"
DATASET_PATH_EPFL_CACHE = "./datasets/__pycache__/"
LABEL_PATH_EPFL_DEFAULT ="./models/EPFL-labels.txt"



parser = argparse.ArgumentParser(description="MVOD Evaluation on VID dataset")
parser.add_argument('--net', default="mobv2-ssdl",
					help="The network architecture, it should be of mobv1-ssdl, mobv2-ssdl, mobv1-ssdl-lstm3, mobv2-ssdl-lstm3")
parser.add_argument("--trained_model", type=str, default = 'models/mb2-ssd-lite-mp-0_686.pth' )
parser.add_argument("--dataset", type=str, default = DATASET_PATH_EPFL , help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, default = LABEL_PATH_EPFL_DEFAULT, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
#parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
#parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--width_mult', default=1, type=float,
					help='Width Multiplifier for network')
args = parser.parse_args()
#device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
device = torch.device('cpu')
print("Device:", device)
print("Argumenten verwerkt.")

def select_model(args):
	batch_size = 2
	if args.net == 'mobv1-ssdl':
		model = net.mobv1_ssdlite_create(num_classes = 1024, alpha=args.width_mult, is_test=False)

	elif args.net == 'mobv2-ssdl':
		model = net.mobv2_ssdlite_create(num_classes = 21, alpha=args.width_mult, is_test=False)

	elif args.net == 'mobv1-ssdl-lstm4':
		model = net.mobv1_ssdlite_lstm4_create(num_classes = 1024, alpha=args.width_mult, batch_size=batch_size, is_test=False)
	
	elif args.net == 'mobv2-ssdl-lstm4':
		model = net.mobv2_ssdlite_lstm4_create(num_classes = 21, alpha=args.width_mult, batch_size=batch_size, is_test=False)

	elif args.net == 'mobv2-ssdl-IR-lstm4':
		model = net.mobv2_ssdlite_IR_lstm4_create(num_classes = 21, alpha=args.width_mult, batch_size=batch_size, is_test=False)

	else:
		logging.fatal("The net type is wrong. It should be one of mobv1-ssdl, mobv2-ssdl, mobv1-ssdl-lstm4, mobv2-ssdl-lstm4, mobv2-ssdl-IR-lstm4.")
		parser.print_help(sys.stderr)
		sys.exit(1)  

	return model


if __name__=='__main__':
	timer = Timer()
	#img = cv2.imread('dog.jpg')
	#print(img.shape)

	"""Netwerk selectie"""
	model = select_model(args)

	#Base_net gewichten uit halen
	#print(model.base_net)
	#for e in model.extras:
	#	if isinstance(e, net.BottleneckLSTM):
	#		print(type(e))

	print(args.net)
	print(model)

	#for param in model.base_net.parameters():
	#		print("Param: ", param)

	"""Gewichten inladen"""
	pretrained_net_dict = torch.load(args.trained_model,map_location=lambda storage, loc: storage)

	#DICTIONARY ENKEL VOOR MOBILENETV2
	dict_base = {}
	for k,v in pretrained_net_dict.items():
		if'base_net' in k:
			#print(k)
			new_k = k.replace('base_net.','')
			dict_base[new_k] = v
	#model.base_net.load_state_dict(pretrained_net_dict)
	
	#for k,v in dict_base.items():
		#print(k)
			#keys.append(k)

	
			#pretrained_net_dict.fromkeys()
	#		print(v)
	#		print(type(pretrained_net_dict))
	#for name in model.base_net.named_parameters():
		#print("param name:", name)
	model.load_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc: storage))
	

	"""Transforms"""
	#train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
	#target_transform = MatchPrior(config.priors, config.center_variance,config.size_variance, 0.5)


	"""Dataset"""
	train_dataset = EPFLDataset(DATASET_PATH_EPFL, DATASET_PATH_EPFL, batch_size = 2, is_val=False, label_file = LABEL_PATH_EPFL_DEFAULT)
	test_dataset = EPFLDataset(DATASET_PATH_EPFL, DATASET_PATH_EPFL, batch_size = 2, is_val=False, is_test=True, label_file = LABEL_PATH_EPFL_DEFAULT)
	#val_dataset = EPFLDataset(DATASET_PATH_EPFL, DATASET_PATH_EPFL, batch_size = 2, is_val=True, label_file = LABEL_PATH_EPFL_DEFAULT)
	#print(val_dataset)

	dataset = test_dataset
	num_classes = len(dataset._classes_names)
	print("Classes in dataset: ",dataset._classes_names)
	print("Number of classes: ",num_classes)
	print("Dataset lengte:", len(dataset))
	
	"""Predictor"""
	predictor = Predictor(model, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=args.nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=200,
                          sigma=0.5,
                          device=device)

	#boxes, labels, probs = predictor.predict(img)
	#exit(0)

	"""Testing"""
	results = []
	for i in range(len(dataset)):
		if i%10 == 0 and i >=10:
			model.load_state_dict(
			torch.load(args.trained_model,
					   map_location=lambda storage, loc: storage))
			model = model.to(device)
		print("Processing image", i)
		timer.start("Load Image")
		images, boxes, labels = dataset.__getitem__(i) #
		image = images[0]
		#print(images)
		print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
		timer.start("Predict")
		boxes, labels, probs = predictor.predict(image)
		if args.net != 'mobv2-ssdl' or args.net != 'mobv1-ssdl':
			model.detach_hidden()
		print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
		indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
		tmprslt=torch.cat([
			indexes.reshape(-1, 1),
			labels.reshape(-1, 1).float(),
			probs.reshape(-1, 1),
			boxes + 1.0  # matlab's indexes start from 1
		], dim=1)
		if(tmprslt.shape[0] > 0):
			results.append(tmprslt)
		

"""
	seq_counter = 0
	for i in range(len(dataset)):
		print(i)
		
		if i%10 == 0:
			model.load_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc: storage))
			model = model.to(device)
			images, boxes, labels = dataset.__getitem__(seq_counter)
			seq_counter += 1
		print("process image", i)
		timer.start("Load Image")
		#image = images[i%10]#bij val dataset, niet test

		print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
		timer.start("Predict")
		boxes, labels, probs = predictor.predict(image)"""
	
	#cv2.imshow('image',img)
	#cv2.waitKey(1)
	#cv2.destroyAllWindows()

	

	
