# USAGE
# python lesions.py --mode train
# python lesions.py --mode investigate
# python lesions.py --mode predict \
# 	--image isic2018/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg

# import the necessary packages
from imgaug import augmenters as iaa
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
from mrcnn import visualize
import glob
from azureml.core import Run



# dataset object from the run

run = Run.get_context()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
ap.add_argument("-w", "--weights",
    help="optional path to pretrained weights")
ap.add_argument("-m", "--mode",
    help="train or investigate")
args = vars(ap.parse_args())

data_folder = args["data_folder"]
print('Data folder:', data_folder)

# get the file paths on the compute

IMAGES_PATHS = os.path.join(os.path.abspath("."),data_folder, 'ISIC2018_Task1-2_Training_Input')
MASKS_PATHS = os.path.join(os.path.abspath("."),data_folder, 'ISIC2018_Task1_Training_GroundTruth')


print("#################################################")
print(IMAGES_PATHS)
print("#################################################")
print(MASKS_PATHS)

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.8

# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATHS)))

print("#################################################")
print(len(IMAGE_PATHS))

idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
trainIdxs = idxs[:i]
valIdxs = idxs[i:]

# initialize the class names dictionary
CLASS_NAMES = {1: "lesion"}

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "mask_rcnn.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored

LOGS_AND_MODEL_DIR = "./outputs/lesions_logs"
os.makedirs(LOGS_AND_MODEL_DIR, exist_ok=True)

class LesionBoundaryConfig(Config):
	# give the configuration a recognizable name
	NAME = "lesion"

	# set the number of GPUs to use training along with the number of
	# images per GPU (which may have to be tuned depending on how
	# much memory your GPU has)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 2

	# set the number of steps per training epoch and validation cycle
	STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
	VALIDATION_STEPS = len(valIdxs) // (IMAGES_PER_GPU * GPU_COUNT)

	# number of classes (+1 for the background)
	NUM_CLASSES = len(CLASS_NAMES) + 1

class LesionBoundaryInferenceConfig(LesionBoundaryConfig):
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 2
    
	# set the number of steps per training epoch and validation cycle
	STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)
	VALIDATION_STEPS = len(valIdxs) // (IMAGES_PER_GPU * GPU_COUNT)

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9

class LesionBoundaryDataset(utils.Dataset):
	def __init__(self, imagePaths, classNames, width=1024):
		# call the parent constructor
		super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
		self.imagePaths = imagePaths
		self.classNames = classNames
		self.width = width

	def load_lesions(self, idxs):
		# loop over all class names and add each to the 'lesion'
		# dataset
		for (classID, label) in self.classNames.items():
			self.add_class("lesion", classID, label)

		# loop over the image path indexes
		for i in idxs:
			# extract the image filename to serve as the unique
			# image ID
			imagePath = self.imagePaths[i]
			filename = imagePath.split(os.path.sep)[-1]

			# add the image to the dataset
			self.add_image("lesion", image_id=filename,
				path=imagePath)

	def load_image(self, imageID):
		# grab the image path, load it, and convert it from BGR to
		# RGB color channel ordering
		p = self.image_info[imageID]["path"]
		image = cv2.imread(p)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# resize the image, preserving the aspect ratio
		image = imutils.resize(image, width=self.width)

		# return the image
		return image

	def load_mask(self, imageID):
		# grab the image info and derive the full annotation path
		# file path
		info = self.image_info[imageID]
		filename = info["id"].split(".")[0]
		annotPath = os.path.sep.join([MASKS_PATHS,
			"{}_segmentation.png".format(filename)])

		# load the annotation mask and resize it, *making sure* to
		# use nearest neighbor interpolation
		annotMask = cv2.imread(annotPath)
		annotMask = cv2.split(annotMask)[0]
		annotMask = imutils.resize(annotMask, width=self.width,
			inter=cv2.INTER_NEAREST)
		annotMask[annotMask > 0] = 1

		# determine the number of unique class labels in the mask
		classIDs = np.unique(annotMask)

		# the class ID with value '0' is actually the background
		# which we should ignore and remove from the unique set of
		# class identifiers
		classIDs = np.delete(classIDs, [0])

		# allocate memory for our [height, width, num_instances]
		# array where each "instance" effectively has its own
		# "channel" -- since there is only one lesion per image we
		# know the number of instances is equal to 1
		masks = np.zeros((annotMask.shape[0], annotMask.shape[1], 1),
			dtype="uint8")

		# loop over the class IDs
		for (i, classID) in enumerate(classIDs):
			# construct a mask for *only* the current label
			classMask = np.zeros(annotMask.shape, dtype="uint8")
			classMask[annotMask == classID] = 1

			# store the class mask in the masks array
			masks[:, :, i] = classMask

		# return the mask array and class IDs
		return (masks.astype("bool"), classIDs.astype("int32"))

if args["mode"] == "train":
    # load the training dataset
    trainDataset = LesionBoundaryDataset(IMAGE_PATHS, CLASS_NAMES)
    trainDataset.load_lesions(trainIdxs)
    trainDataset.prepare()

    # load the validation dataset
    valDataset = LesionBoundaryDataset(IMAGE_PATHS, CLASS_NAMES)
    valDataset.load_lesions(valIdxs)
    valDataset.prepare()

    # initialize the training configuration
    config = LesionBoundaryConfig()
    config.display()

    # initialize the image augmentation process
    aug = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-10, 10))
    ])

    # initialize the model and load the COCO weights so we can
    # perform fine-tuning
    model = modellib.MaskRCNN(mode="training", config=config,
    model_dir=LOGS_AND_MODEL_DIR)
    model.load_weights(COCO_PATH, by_name=True)




    # unfreeze the body of the network and train *all* layers
    model.train(trainDataset, valDataset, epochs=20,
    layers="all", learning_rate=config.LEARNING_RATE / 10,
    augmentation=aug)

# check to see if we are investigating our images and masks
if args["mode"] == "investigate":
    # load the training dataset
    trainDataset = LesionBoundaryDataset(IMAGE_PATHS, CLASS_NAMES)
    trainDataset.load_lesions(trainIdxs)
    trainDataset.prepare()

    # load the 0-th training image and corresponding masks and
    # class IDs in the masks
    image = trainDataset.load_image(0)
    (masks, classIDs) = trainDataset.load_mask(0)

    # show the image spatial dimensions which is HxWxC
    print("[INFO] image shape: {}".format(image.shape))

    # show the masks shape which should have the same width and
    # height of the images but the third dimension should be
    # equal to the total number of instances in the image itself
    print("[INFO] masks shape: {}".format(masks.shape))

    # show the length of the class IDs list along with the values
    # inside the list -- the length of the list should be equal
    # to the number of instances dimension in the 'masks' array
    print("[INFO] class IDs length: {}".format(len(classIDs)))
    print("[INFO] class IDs: {}".format(classIDs))

    # determine a sample of training image indexes and loop over
    # them
    for i in np.random.choice(trainDataset.image_ids, 3):
        # load the image and masks for the sampled image
        print("[INFO] investigating image index: {}".format(i))
        image = trainDataset.load_image(i)
        (masks, classIDs) = trainDataset.load_mask(i)

        # visualize the masks for the current image
        visualize.display_top_masks(image, masks, classIDs,
            trainDataset.class_names)



