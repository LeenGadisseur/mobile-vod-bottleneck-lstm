#!/usr/bin/python3
"""
Leen
Script for running trained network
"""

import os
import sys
import torch
import torch.nn as nn
from network.mvod_basenet import MobileVOD, SSD, MobileNetV1, MatchPrior
from network.predictor import Predictor

KLASSEN_AANTAL = 31


if __name__ == '__main__':

#################################
#Model inladen van torch 
	model_torch = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
	for param_tensor in model_torch.state_dict():
    		print(param_tensor, "\t", model_torch.state_dict()[param_tensor].size())
	model_torch.eval()
	#print(model_torch)


###########################3
#VOD model inladen beschikbaar in github
	#initialeren sturcturen van netten	
	ssd = SSD(KLASSEN_AANTAL)
	print(ssd)
	mobilenet = MobileNetV1()
	print(mobilenet)
	net = MobileVOD(mobilenet,ssd)
	print(net)

	#Inladen state dictionary voor gehele VOD model
	net_params = torch.load('models/MVOD_SSD_mobilenetv1_params.pth',map_location=torch.device('cpu'))
	net.load_state_dict(net_params)
	for param_tensor in net.state_dict():
		print(param_tensor, "\t", net.state_dict()[param_tensor].size())

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

	
