#!/usr/bin/python3

"""
Probeersel
Script for creating dataset of EPFL data. Here we have two classes: one for sequencial dataset preparation
and other for normal object localization and classification task.
Classes
----------------
EPFLDataset : class for loading dataset in sequences of 10 consecutive video frames

"""
import pickle
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os


class EPFLDataset:

	def __init__(self, data, root, transform=None, target_transform=None, is_val=False, batch_size=None, label_file=None):
		"""Dataset for EPFL data.
		Args:
			root: the root of the EPFL dataset, the directory contains the following sub-directories:
				Annotations, Data
		"""
		self.data = pathlib.Path(data)
		self.root = pathlib.Path(root)
		self.transform = transform
		self.target_transform = target_transform
		#self.is_test = is_test
		self.is_val = is_val
		if self.is_val:
			image_sets_file = "datasets/val_EPFL_seqs_list.txt"
		else:
			image_sets_file = "datasets/train_EPFL_seqs_list.txt"

		self.seq_list = EPFLDataset._read_image_seq_ids(image_sets_file)
		#print("Seq_list:", self.seq_list)
		rem = len(self.seq_list)%batch_size
		if rem==0:
			self.seq_list =self.seq_list
		else:
			self.seq_list = self.seq_list[:-(rem)]
		#print("Seq_list:", self.seq_list)
		#logging.info("using default Imagenet VID classes.")
		self._classes_names = ['__background__',  # always index 0
					'person']

		#Klassen in de annotatie files
		self._classes_map = ['__background__',  # always index 0
						'"person"']

		self._name_to_class = {self._classes_map[i]: self._classes_names[i] for i in range(len(self._classes_names))}
		
		self._class_to_ind = {class_name: i for i, class_name in enumerate(self._classes_names)}
		self.db = self.gt_roidb() 
		

	def __getitem__(self, index):
		data = self.db[index]
		boxes_seq = data['boxes_seq']
		labels_seq = data['labels_seq']
		images_seq = data['images_seq']
		images = []
		#print("Lengte ", len(images_seq))
		for i in range(0,len(images_seq)):
			#print("image seq",images_seq[i])
			image = self._read_image(images_seq[i])
			boxes = boxes_seq[i]
			labels = labels_seq[i]
			if self.transform:
				image, boxes, labels = self.transform(image, boxes, labels)
			if self.target_transform:
				boxes, labels = self.target_transform(boxes, labels)
			images.append(image)
			boxes_seq[i] = boxes
			labels_seq[i] = labels
		return images, boxes_seq, labels_seq

	def gt_roidb(self):
		"""
		return ground truth image regions database
		:return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
		"""
		if self.is_val:
			cache_file = os.path.join(self.root, 'val_EPFL_seq_gt_db.pkl')
			print("Cache file: ", cache_file)
			print("Cached.")
		else:
			cache_file = os.path.join(self.root, 'train_EPFL_seq_gt_db.pkl')
			print("Cache file: ", cache_file)
			print("Cached.")
		"""if os.path.exists(cache_file):
			with open(cache_file, 'rb') as fid:
				roidb = pickle.load(fid)
			logging.info('gt roidb loaded from {}'.format(cache_file))
			return roidb"""

		#print(self.seq_list)
		print("Loading annos...")
		gt_roidb = [self.load_EPFL_annotation(index) for index in range(0, len(self.seq_list))]
		#print(gt_roidb)#print geeft lege boxes en labels
		with open(cache_file, 'wb') as fid:
			pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
		logging.info('wrote gt roidb to {}'.format(cache_file))

		return gt_roidb

	def load_EPFL_annotation(self, i):
		"""
		for a given sequence index, load images and bounding boxes info from XML file
		:param index: index of a specific image
		:return: record['boxes', 'labels']
		"""
		#print("Loading annos.")
		image_seq = self.seq_list[i]
		image_ids = image_seq.split(',')
		images = []
		boxes_seq = []
		labels_seq = []
		for image_id in image_ids:
			if self.is_val:
				annotation_file = self.data / f"Annotations/val/{image_id}.xml"
				#print("anno file: ",annotation_file )
			else:
				annotation_file = self.data / f"Annotations/train/{image_id}.xml"
				#print("anno file: ",annotation_file )
			objects = ET.parse(annotation_file).findall("object")
			#print("Objects", objects)
			boxes = []
			labels = []
			#print("self._name_to_class: ", self._name_to_class )
			for obj in objects:
				class_name = obj.find('name').text.lower().strip()
				#print("class_name: ", class_name )
				
				# we're only concerned with clases in our list
				if class_name in self._name_to_class:
					#print("added.")
					bbox = obj.find('bndbox')

					# VID dataset format follows Matlab, in which indexes start from 0
					x1 = float(bbox.find('xmin').text) - 1
					y1 = float(bbox.find('ymin').text) - 1
					x2 = float(bbox.find('xmax').text) - 1
					y2 = float(bbox.find('ymax').text) - 1
					boxes.append([x1, y1, x2, y2])
					labels.append(self._class_to_ind[self._name_to_class[class_name]])
		
			image = self.image_path_from_index(image_id)
			boxes  = np.array(boxes, dtype=np.float32)
			labels = np.array(labels, dtype=np.int64)
			
			images.append(image)
			boxes_seq.append(boxes)
			labels_seq.append(labels)
		roi_rec = dict()
		roi_rec['images_seq'] = images
		roi_rec['boxes_seq'] = boxes_seq
		roi_rec['labels_seq'] = labels_seq
		
		#print("Boxes: ", roi_rec['boxes_seq'])
		#print("labels: ", roi_rec['labels_seq'])
		return roi_rec

	def image_path_from_index(self, image_id):
		"""
		given image index, find out full path
		:param index: index of a specific image
		:return: full path of this image
		"""
		if self.is_val:
			image_file = self.data / f"Data/val/{image_id}.JPEG"
		else:
			image_file = self.data / f"Data/train/{image_id}.JPEG"
		return image_file
	

	def __len__(self):
		return len(self.seq_list)

	@staticmethod
	def _read_image_seq_ids(image_sets_file):
		seq_list = []
		with open(image_sets_file) as f:
			for line in f:
				seq_list.append(line.rstrip())
		#print("Image_seqs_file: ",seq_list)
		return seq_list

	def _read_image(self, image_file):
		#print("Image_file: ",image_file)
		image = cv2.imread(str(image_file))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

	

		


