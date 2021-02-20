#!/usr/bin/python3
"""
Author: Leen Gadisseur
Description:
Test script voor het inladen van voorbestaande modellen.
"""


import os
import sys
import torch
import torch.nn as nn
import cv2
from network.mvod_basenet import MobileVOD, SSD, MobileNetV1, MatchPrior
from network.predictor import Predictor
from datasets.vid_dataset import VIDDataset

KLASSEN_AANTAL = 31
ROOT_VIDDATASET = '/media/leen/Acer_500GB_HDD/Imagenet_VID_dataset/ILSVRC/'


if __name__ == '__main__':

#################################
#Model inladen van torch 
	#model_torch = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
	#for param_tensor in model_torch.state_dict():
    	#	print(param_tensor, "\t", model_torch.state_dict()[param_tensor].size())
	#model_torch.eval()
	#print(model_torch)


###########################3
#VOD model inladen beschikbaar in github
	#initialeren sturcturen van netten	
	ssd = SSD(KLASSEN_AANTAL)
	#print(ssd)
	mobilenet = MobileNetV1()
	#print(mobilenet)
	net = MobileVOD(mobilenet,ssd)
	#print(net)

	#Inladen state dictionary voor gehele VOD model
	net_params = torch.load('models/MVOD_SSD_mobilenetv1_params.pth',map_location=torch.device('cpu'))
	net.load_state_dict(net_params)
	#for param_tensor in net.state_dict():
		#print(param_tensor, "\t", net.state_dict()[param_tensor].size())


	#Dataset inladen
	VIDDataset = VIDDataset(ROOT_VIDDATASET, is_test= True)
	l = VIDDataset.__len__()
	print("Lengte: ", l)
	for i in range(1,l):
		img = VIDDataset[i]
		w = img.shape[0]
		h = img.shape[1]

		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		pred = Predictor(net, size = img.shape[:2])
		pred.predict(img)

	
	#Predictor gebruiken
	

#################################
#Mobilenet inladen met beschikbaar in github
	#init model
	#model = MobileNetV1()

	#model laden met cpu, CUDA drivers zijn outdated
	#model_params = torch.load('models/MVOD_SSD_mobilenetv1_params.pth',map_location=torch.device('cpu'))
	
	#c
		#print(v)
	#print(type(model_params))

	#model.load_state_dict(model_params)
	#print(model.state_dict())
	
	#print("Model loaded.")
	#print(model)
	
####################################
#basenet en lstm?

#basenet = torch.load('models/basenet/WM-1.0-Epoch-2-Loss-4.629070136970256.pth', map_location=torch.device('cpu'))
	#print("basenet")
	#for k,v in basenet.items():
	#	print(k , "\t")
	#lstm = torch.load('models/lstm1/WM-1.0-Epoch-2-Loss-8.151137044498636.pth',map_location=torch.device('cpu'))
	#print("lstm")
	#for k,v in lstm.items():
	#	print(k , "\t")

	
