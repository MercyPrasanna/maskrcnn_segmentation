
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
import shutil


# initialize the class names dictionary
CLASS_NAMES = {1: "lesion"}

class LesionBoundaryInferenceConfig(Config):
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)

	NAME = "lesion"
	NUM_CLASSES = len(CLASS_NAMES) + 1
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--output_dir', type=str, dest='output_dir', help='output dir')
ap.add_argument('--dataset_path', dest="dataset_path", required=True)
args = vars(ap.parse_args())



# create output directory if it does not exist

output_dir = args["output_dir"]

print("###############output_dir",output_dir)

os.makedirs(output_dir, exist_ok=True)

data_dir = args["dataset_path"]

input_dir = data_dir + '/' + 'ISIC2018_Task1-2_Training_Input'

groundtruth_dir = data_dir + '/' + 'ISIC2018_Task1_Training_GroundTruth'

file_paths = []

image_list = os.listdir(input_dir)


file_paths = [input_dir + '/' + file_name.rstrip() for file_name in image_list]

LOGS_AND_MODEL_DIR = "./outputs/lesions_logs"
os.makedirs(LOGS_AND_MODEL_DIR, exist_ok=True)

model_path = Model.get_model_path("mask_rcnn_horovod")

# initialize the inference configuration
config = LesionBoundaryInferenceConfig()

# initialize the Mask R-CNN model for inference
model = modellib.MaskRCNN(mode="inference", config=config,model_dir=LOGS_AND_MODEL_DIR)


model.load_weights('mask_rcnn_lesion_0017.h5', by_name=True)

for image_file in file_paths:

	filename = os.path.basename(image_file).split(".")[0]

	out_filename = os.path.join(output_dir, "{}_predicted.png".format(filename))

	segmentation_file = os.path.sep.join([groundtruth_dir,
			"{}_segmentation.png".format(filename)])
	
	out_segmentation_filename = os.path.join(output_dir, os.path.basename(segmentation_file))

	print("filename##################################",filename)
	print("outfilename##################################",out_filename)
	print("segmentationfilename##################################",segmentation_file)


	# load the input image, convert it from BGR to RGB channel
	# ordering, and resize the image
	image = cv2.imread(image_file)
	image, window, scale, padding, crop = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM, min_scale=config.IMAGE_MIN_SCALE, max_dim=config.IMAGE_MAX_DIM, mode=config.IMAGE_RESIZE_MODE)
    
	groundtruth_image = cv2.imread(segmentation_file)
	groundtruth_mask = utils.resize_mask(groundtruth_image, scale, padding, crop)

	#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	#image = imutils.resize(image, width=1024)

	image_mask = np.zeros(image.shape)

	print("#################input image shape",image.shape)
	print("#################ground truth mask shape",groundtruth_mask.shape)
	print("#################predicted mask shape",image_mask.shape)

	# perform a forward pass of the network to obtain the results
	r = model.detect([image], verbose=1)[0]

	# loop over of the detected object's bounding boxes and
	# masks, drawing each as we go along
	for i in range(0, r["rois"].shape[0]):
		mask = r["masks"][:, :, i]
		for c in range(3):
			image_mask[:, :, c] = np.where(mask == 1,255,0)
			cv2.imwrite(out_filename, image_mask)
			cv2.imwrite(out_segmentation_filename, groundtruth_mask)
			shutil.copy(out_filename, "./outputs/")
			shutil.copy(out_segmentation_filename, "./outputs/")
			print("copied file to the outputs")





