#!/usr/bin/python3
"""Script for implementing Predictor while testing the model
"""
import torch

from utils import box_utils
from dataloaders.data_preprocessing import PredictionTransform
from utils.misc import Timer


class Predictor:
	def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
				 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
		"""Implement Predictor while testing of the model
		Arguments:
			net: an object of net to used for prediction
			size: variable containing size of image as input
			mean: an array variable containing mean value of all the channels of input
			std: variable containing standard deviation of image as input
			nms_method : string variable contaning type of negative mining
			iou_threshold : a float variable containing threshold value of IOU
			filter_threshold : a float variable containing threshold value of filter
			candidate_size: only consider the candidates with the highest scores
			sigma: the parameter in score re-computation. (for soft negative mining)
				scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
			device : variable containing device on which net needs to do computation
		"""
		self.net = net
		self.transform = PredictionTransform(size, mean, std)
		self.iou_threshold = iou_threshold
		self.filter_threshold = filter_threshold
		self.candidate_size = candidate_size
		self.nms_method = nms_method

		self.sigma = sigma
		print("Predictor arg device", device)
		if device:
			self.device = device
		else:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.net.to(self.device)
		self.net.eval()

		self.timer = Timer()

	def predict(self, image, top_k=-1, prob_threshold=None):
		"""Implement Predictor while testing of the model
		Arguments:
			image: image input for predictor
			prob_threshold: threshold for probability
			top_k: keep top_k results. If k <= 0, keep all the results.
		Returns:
			predicted boxes, labels and their probability
		"""
		cpu_device = torch.device("cpu")
		height, width, _ = image.shape
		image = self.transform(image)
		images = image.unsqueeze(0)
		images = images.to(self.device)
		with torch.no_grad():
			self.timer.start()
			scores, boxes = self.net.forward(images)
			print("Inference time: ", self.timer.end())
		boxes = boxes[0]
		scores = scores[0]
		if not prob_threshold:
			prob_threshold = self.filter_threshold
		# this version of nms is slower on GPU, so we move data to CPU.
		boxes = boxes.to(cpu_device)
		scores = scores.to(cpu_device)
		picked_box_probs = []
		picked_labels = []
		for class_index in range(1, scores.size(1)):
			probs = scores[:, class_index]
			mask = probs > prob_threshold
			probs = probs[mask]
			if probs.size(0) == 0:
				continue
			subset_boxes = boxes[mask, :]
			box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
			box_probs = box_utils.nms(box_probs, self.nms_method,
									  score_threshold=prob_threshold,
									  iou_threshold=self.iou_threshold,
									  sigma=self.sigma,
									  top_k=top_k,
									  candidate_size=self.candidate_size)
			picked_box_probs.append(box_probs)
			picked_labels.extend([class_index] * box_probs.size(0))
		if not picked_box_probs:
			return torch.tensor([]), torch.tensor([]), torch.tensor([])
		picked_box_probs = torch.cat(picked_box_probs)
		picked_box_probs[:, 0] *= width
		picked_box_probs[:, 1] *= height
		picked_box_probs[:, 2] *= width
		picked_box_probs[:, 3] *= height
		return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
