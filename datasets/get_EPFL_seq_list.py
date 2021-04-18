#!/usr/bin/python3
"""Script for creating text file containing sequences of 10 frames of particular video. Here we neglect all the frames where 
there is no object in it as it was done in the official implementation in tensorflow.
Global Variables
----------------
dirs : containing list of all the training dataset folders
dirs_val : containing path to val folder of dataset
dirs_test : containing path to test folder of dataset
"""
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os

dirs = ['Basketball/',
		'Campus/',
		'Laboratory/',
		'Passageway/',
		'Terrace/']
dirs_val = '/media/leen/Acer_500GB_HDD/EPFL/Data/val/'
dirs_test = '/media/leen/Acer_500GB_HDD/EPFL/Data/test/'
	


file_write_obj = open('train_EPFL_seqs_list.txt','w')
print("Writing file: train_EPFL_seqs_list.txt ")
for dir in dirs:
	seqs = np.sort(os.listdir(os.path.join('/media/leen/Acer_500GB_HDD/EPFL/Data/train/'+dir)))
	for seq in seqs:
		seq_path = os.path.join('/media/leen/Acer_500GB_HDD/EPFL/Data/train/',dir,seq)
		print('\t',"Processing: ", seq_path)
		relative_path = dir + seq
		image_list = np.sort(os.listdir(seq_path))
		count = 0
		filtered_image_list = []
		for image in image_list:
			image_id = image.split('.')[0]
			
			#Frames beginnen bij 1, annotaties bij 0 => niet nodig, files zijn aangepast
			image_id = str(int(image_id)).zfill(5)
			anno_file = image_id + '.xml'
			anno_path = os.path.join('/media/leen/Acer_500GB_HDD/EPFL/Annotations/train/',dir,seq,anno_file)
			objects = ET.parse(anno_path).findall("object")
			num_objs = len(objects)
			if num_objs == 0: # discarding images without object
				continue
			else:
				count = count + 1
				filtered_image_list.append(relative_path+'/'+image_id)
		for i in range(0,int(count/10)):
			seqs = ''
			for j in range(0,10):
				seqs = seqs + filtered_image_list[10*i + j] + ','
			seqs = seqs[:-1]
			file_write_obj.writelines(seqs)
			file_write_obj.write('\n')
file_write_obj.close()

print("Writing file: val_EPFL_seqs_list.txt ")
file_write_obj = open('val_EPFL_seqs_list.txt','w')
seq_list = []

with open('val_EPFL_list.txt') as f:
	for line in f:
		seq_list.append(line.rstrip())
for i in range(0,int(len(seq_list)/10)):
	#image_path = seq_list[10*i].split('/')[0]
	#seqs = image_path+'/'+':'
	seqs = ''
	for j in range(0,10):
		seqs = seqs + seq_list[10*i + j] + ','
	seqs = seqs[:-1] 
	file_write_obj.writelines(seqs)
	file_write_obj.write('\n')
file_write_obj.close()

print("Writing file: test_EPFL_seqs_list.txt ")
file_write_obj = open('test_EPFL_seqs_list.txt','w')
for dir in dirs:
	seqs = np.sort(os.listdir(dirs_test + dir))
	for seq in seqs:
		seq_path = os.path.join(dirs_test, dir,seq)
		print('\t',"Processing: ", seq_path)
		image_list = np.sort(os.listdir(seq_path))
		for image in image_list:
			file_write_obj.writelines(dir + seq+'/'+image)
			file_write_obj.write('\n')

file_write_obj.close()
