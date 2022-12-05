#!/usr/bin/env python3

import os
import shutil
import logging
import glob
import uuid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #tf is really noisy otherwise
import tensorflow as tf
from tensorflow import keras
import argparse
import cv2 as cv
import numpy as np

class SampledImage:
    tagesschau_class_mapping = {0: "default",
                                1: "description_overlay",
                                2: "name_overlay",
                                3:"logo_overlay"}

    def __init__(self, data, video_filepath, frame_number, frame_ms):
        self.data = data 
        self.video_filepath = video_filepath
        self.frame_number = frame_number
        self.frame_ms = frame_ms
        self.prediction = None

    def as_predictable(self, normalize=True, expand=True, resize_x=256, resize_y=256):
        img = self.data
        if resize_x is not None:
            img = cv.resize(img, (resize_y, resize_x))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if normalize:
            img = keras.utils.normalize(img, axis=1)
        if expand:
            img = np.expand_dims(img, 2)
            img = np.expand_dims(img, 0)
        return img

    def apply_mask(self, show_only_this_class, mask_value=0):
        """
        Resize self.prediction to the size of self.data
        If a pixel on self.prediction has a value listed in mask_classes:
            Zero the corresponding pixel in self.data
        """
        if self.prediction is None:
            raise ValueError("Cannot apply mask without class prediction for sample image")
        #prediction -> (1, 256, 256, 4)
        #upscaled_prediction -> (1080, 1920)
        #data -> (1080, 1920, 3)
        data_dimension_x, data_dimension_y = self.data.shape[1], self.data.shape[0]
        upscaled_prediction = cv.resize(self.prediction, (data_dimension_x, data_dimension_y))
        upscaled_prediction = np.expand_dims(upscaled_prediction, 2) #self.data has a third dimension for RGB, equal shapes required for np.where
        return np.where(upscaled_prediction == show_only_this_class, self.data, 0) 


parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--model-filepath",
                    type=str,
                    default="./tagesschau_unet.h5",
                    help="Path to the file containing the model in h5 format.")
parser.add_argument("-x",
                    "--model-input-size-x",
                    type=int,
                    default=256,
                    help="Input size for classification model (x dimension).")
parser.add_argument("-y",
                    "--model-input-size-y",
                    type=int,
                    default=256,
                    help="Input size for classification model (y dimension).")
parser.add_argument("--temp-dir",
                    type=str,
                    default="./tmp_files",
                    help="Directory to place temporary files in.")
parser.add_argument("--sampling-rate",
                    type=float,
                    default=0.4,
                    help="Rate in frames per second at which to grab frames for classification.")
parser.add_argument("--video-filepath",
                    type=str,
                    default="./tagesschau-vom-04-12-2022-mittagsausgabe.mp4", #TODO deleteme
                    help="Path of video file to classify.")
parser.add_argument("-v",
                    "--verbose",
                    default=False,
                    action="store_true",
                    help="Verbose output")
args = parser.parse_args()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                    format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
log = logging.getLogger()

if not os.access(args.video_filepath, os.R_OK):
  raise FileNotFoundError(f"Failed to read from video file at {args.video_filepath}")

if not os.access(args.model_filepath, os.R_OK):
  raise FileNotFoundError(f"Failed to open classification model at {args.model_filepath}")

try:
  shutil.rmtree(args.temp_dir)
except FileNotFoundError:
  pass

TEMP_DIR = os.path.join(args.temp_dir, str(uuid.uuid4()))
os.makedirs(TEMP_DIR, exist_ok=True)

log.info(f"Classifying video data from {args.video_filepath} with sampling rate of {args.sampling_rate} fps")

#validate POST request, check if its a supported video format

cap = cv.VideoCapture(args.video_filepath)
fps = round(cap.get(cv.CAP_PROP_FPS))
hop = round(fps / args.sampling_rate)

log.info(f"Video has {fps} fps, sampling every {hop}th frame.")

counter = 0
frame_counter = 0
sampled_images = []

while True:
  success, frame = cap.read()
  if success:
    if frame_counter % hop == 0:
      timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
      sampled_images.append(SampledImage(data=frame,
                                         video_filepath=args.video_filepath,
                                         frame_number=frame_counter,
                                         frame_ms=timestamp))
      log.debug(f"Sampled frame {frame_counter} at {timestamp:0.0f} ms")
  else:
    raise Exception(f"CV2 failed to read from video {args.video_filepath}")
  if counter > 300:
    break
  counter += 1
  frame_counter += 1

model = keras.models.load_model(args.model_filepath)

for sampled_image in sampled_images:
  test_img = sampled_image.as_predictable(resize_x=args.model_input_size_x, resize_y=args.model_input_size_y)
  prediction = model.predict(test_img)
  sampled_image.prediction = np.argmax(prediction, axis=3).astype(np.uint8)[0,:,:]


cv.imwrite(f"./masked.jpg", sampled_image.apply_mask(2))
exit()
from matplotlib import pyplot as plt
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title(f"Testing Image")
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Prediction on test image')
plt.imshow(sampled_image.prediction, cmap='jet')
plt.savefig("out.png")
