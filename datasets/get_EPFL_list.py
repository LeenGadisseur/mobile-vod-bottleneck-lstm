#!/usr/bin/python3
"""Script for creating text file containing sequences of all the video frames. Here we neglect all the frames where 
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

#Leen
PATH = '/media/leen/Acer_500GB_HDD/EPFL'

dirs = ['Basketball/',
		'Campus/',
		'Laboratory/',
		'Passageway/',
		'Terrace/']
dirs_val = PATH+'/Data/val/'
dirs_test = PATH+'/Data/test/'


file_write_obj = open('train_EPFL_list.txt','w')
print("Writing file train_EPFL_list.txt :")
for dir in dirs:
	seqs = np.sort(os.listdir(os.path.join(PATH+'/Data/train/'+dir)))
	for seq in seqs:
		#print("seq: ", seq)
		seq_path = os.path.join(PATH+'/Data/train/',dir,seq)
		print('\t',"Processing: ", seq_path)
		relative_path = dir + seq
		image_list = np.sort(os.listdir(seq_path))
		count = 0
		for image in image_list:
			image_id = image.split('.')[0]
			#Frames beginnen bij 1, annotaties bij 0 => niet meer nodig, files zijn
			image_id = str(int(image_id)).zfill(5)
			anno_file = image_id + '.xml'
			anno_path = os.path.join(PATH+'/Annotations/train/',dir,seq,anno_file)
			objects = ET.parse(anno_path).findall("object")
			num_objs = len(objects)
			if num_objs == 0: # discarding images without object
				continue
			else:
				count = count + 1
				file_write_obj.writelines(relative_path+'/'+image_id)
				file_write_obj.write('\n')
file_write_obj.close()

print("Writing file val_EPFL_list.txt :")
file_write_obj = open('val_EPFL_list.txt','w')
for dir in dirs:
	#print('\t', dir)
	seqs = np.sort(os.listdir(dirs_val+dir))
	#print("'\t', Seqs: ",seqs)
	for seq in seqs:
		seq_path = os.path.join(dirs_val,dir,seq)
		print('\t',"Processing: ", seq_path)
		image_list = np.sort(os.listdir(seq_path))
		count = 0
		#print(image_list)
		for image in image_list:
			image_id = image.split('.')[0]
			#Frames beginnen bij 1, annotaties bij 0 => niet meer nodig, files zijn aangepast
			image_id = str(int(image_id)).zfill(5)
			anno_file = image_id + '.xml'
			anno_path = os.path.join(PATH+'/Annotations/val/',dir,seq,anno_file)
			objects = ET.parse(anno_path).findall("object")
			num_objs = len(objects)
			if num_objs == 0:
				continue
			else:
				count = count + 1
				file_write_obj.writelines(dir + seq+'/'+image_id)
				file_write_obj.write('\n')


file_write_obj.close()

print("Writing file test_EPFL_list.txt :")
file_write_obj = open('test_EPFL_list.txt','w')
for dir in dirs:
	#print("Dir:", dir)
	seqs = np.sort(os.listdir(dirs_test + dir))
	#print("Seqs:", seqs)
	for seq in seqs:
		seq_path = os.path.join(dirs_test,dir,seq)
		print('\t',"Processing: ", seq_path)
		#print("seq_path:",seq_path)
		image_list = np.sort(os.listdir(seq_path))
		for image in image_list:
			file_write_obj.writelines(dir + seq+'/'+image)
			file_write_obj.write('\n')

file_write_obj.close()
