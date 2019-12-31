
# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
import numpy as np
import imutils
import cv2
import os
from azureml.core import Model
import argparse


parser = argparse.ArgumentParser(description="Start a keras mask_rcnn horovod model serving")

parser.add_argument('--model_name', dest="model_name", required=False, default='mask_rcnn_lesion_0020.h5')

parser.add_argument('--output_dir', dest="output_dir", required=True)

args = parser.parse_args()


# create output directory if it does not exist

os.makedirs(args.output_dir, exist_ok=True)


LOGS_AND_MODEL_DIR = "./outputs/lesions_logs"
os.makedirs(LOGS_AND_MODEL_DIR, exist_ok=True)

class LesionBoundaryInferenceConfig():
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9


def init():

	global model
	model_path = Model.get_model_path("mask_rcnn_horovod")

	# initialize the inference configuration
	config = LesionBoundaryInferenceConfig()

	# initialize the Mask R-CNN model for inference
	model = modellib.MaskRCNN(mode="inference", config=config,model_dir=LOGS_AND_MODEL_DIR)


	model.load_weights(os.path.join(model_path, 'mask_rcnn_lesion_0020.h5'), by_name=True)


def run(mini_batch):

	print(f'run method start: {__file__}, run({mini_batch})')
	
	for image_file in mini_batch:

		out_filename = os.path.join(args.output_dir, os.path.basename(image_file))

		# load the input image, convert it from BGR to RGB channel
		# ordering, and resize the image
		image = cv2.imread(image_file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = imutils.resize(image, width=1024)
		
		image_mask = np.zeros(image.shape())

		# perform a forward pass of the network to obtain the results
		r = model.detect([image], verbose=1)[0]

		# loop over of the detected object's bounding boxes and
		# masks, drawing each as we go along
		for i in range(0, r["rois"].shape[0]):
			mask = r["masks"][:, :, i]
			for c in range(3):
				image_mask[:, :, c] = np.where(mask == 1,255,0)
				cv2.imwrite(out_filename, image_mask)


