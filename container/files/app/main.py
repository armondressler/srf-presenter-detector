#!/usr/bin/env python3

import os
import shutil
import logging
import uuid
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #tf is really noisy otherwise
import tensorflow as tf
from tensorflow import keras
import argparse
import cv2 as cv
import pytesseract
import numpy as np

class SampledImage:
    prediction_labels = {"default": 0,
                         "description_overlay": 1,
                         "name_overlay": 2,
                         "logo_overlay": 3}

    def __init__(self, data, video_filepath, frame_number, frame_ms):
        self.data = data 
        self.video_filepath = video_filepath
        self.frame_number = frame_number
        self.frame_ms = frame_ms
        self.prediction = None
        self.text_by_label = {label: "" for label in SampledImage.prediction_labels.keys()}

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

    def recognize_text(self,
                       predict_text_in_labels=["name_overlay", "description_overlay"],
                       prediction_languages=["deu","fra","eng"],
                       prune_chars_regex=r"[\x00-\x1f\x7f-\x9f\|\_]",
                       debug_image_dir=None):
        """
        For each item in predict_text_in_labels:
          Apply a mask on the image, ensuring only pixels of class $item are visible
          Apply ocr on masked picture
          Write text data to self.text_by_label and return
        """
        tesseract_languages = "+".join(prediction_languages)
        for label in predict_text_in_labels:
            if label not in SampledImage.prediction_labels.keys():
                raise ValueError(f"Cannot predict text for unknown label {label}")
            masked_img = self.apply_mask(show_class=SampledImage.prediction_labels[label])
            text = pytesseract.image_to_string(masked_img, lang=tesseract_languages)
            if debug_image_dir:
                img_path = os.path.join(debug_image_dir, f"framenr_{self.frame_number}_framems_{self.frame_ms}.jpg")
            if prune_chars_regex:
                text = re.sub(prune_chars_regex, '', text)
            self.text_by_label[label] = text
        return self.text_by_label


    def apply_mask(self, show_class, mask_value=0, convert_to_black_white=True, black_white_threshold=130, invert_colors=True):
        """
        Resize self.prediction to the size of self.data
        If a pixel on self.prediction has a value not equals show_class:
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
        masked_img = np.where(upscaled_prediction == show_class, self.data, 0) 
        if convert_to_black_white:
            masked_img = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)
            _, masked_img = cv.threshold(masked_img, black_white_threshold, 255, cv.THRESH_BINARY_INV if invert_colors else cv.THRESH_BINARY)
        return masked_img


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
parser.add_argument("--tesseract-filepath",
                    type=str,
                    default="/usr/bin/tesseract",
                    help="Path to tesseract ocr binary.")
parser.add_argument("--tesseract-languages",
                    action="append",
                    default=["deu","eng","fra"],
                    help="Languages to configure tesseract with, see https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html")
parser.add_argument("-v",
                    "--verbose",
                    default=False,
                    action="store_true",
                    help="Verbose output")
args = parser.parse_args()


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
pytesseract.pytesseract.tesseract_cmd = args.tesseract_filepath

logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                    format="%(asctime)s - %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S")
log = logging.getLogger()

if not os.access(args.video_filepath, os.R_OK):
  raise FileNotFoundError(f"Failed to read from video file at {args.video_filepath}")

if not os.access(args.model_filepath, os.R_OK):
  raise FileNotFoundError(f"Failed to open classification model at {args.model_filepath}")

if not os.access(args.tesseract_filepath, os.X_OK):
  raise FileNotFoundError(f"Failed to execute tesseract at {args.tesseract_filepath}")

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
  if counter > 600:
    break
  counter += 1
  frame_counter += 1

model = keras.models.load_model(args.model_filepath)

for sampled_image in sampled_images:
  input_img = sampled_image.as_predictable(resize_x=args.model_input_size_x, resize_y=args.model_input_size_y)
  prediction = model.predict(input_img, verbose=1 if args.verbose else 0)
  sampled_image.prediction = np.argmax(prediction, axis=3).astype(np.uint8)[0,:,:]
  text_by_label = sampled_image.recognize_text()
  log.info(f"Text detected for image at frame {sampled_image.frame_number}: {text_by_label}")

