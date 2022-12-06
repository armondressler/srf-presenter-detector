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
import jellyfish
from webvtt import WebVTT, Caption  #webvtt-py==0.4.6
from webvtt.writers import WebVTTWriter, SRTWriter
import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query, Request, File, UploadFile


class Subtitle:
    #constructs a vtt or srt formatted set of subtitles
    #image gets parsed, self.text_by_label now available
    #combine all labels (name and description itc) into one string
    #check if previous subtitle is similar / equal:
      #extend the previous
    #add subtitle at timestamp frame_ms
        
    def __init__(self, jaro_distance_threshold=0.85):
        self.jaro_distance_threshold = jaro_distance_threshold
        self.elements = []

    def append(self, text, offset_ms, duration_ms, extend_if_similar=True):
        if self.elements:
            if extend_if_similar:
                previous_index = len(self.elements) - 1
                if self.is_similar(text, self.elements[previous_index]["text"]):
                    #If the subtitles are similar theres a good chance that OCR bugged out so we keep the previous subtitle.
                    #The previous subtitle might be buggy but at least we dont change every few secs.
                    log.debug(f"Extending subtitle due to similarity with image at offset {offset_ms} ms")
                    self.extend_duration(index=previous_index, duration_ms=duration_ms)
                    return
        log.debug(f"Adding subtitle for image at offset {offset_ms} ms: {text}")
        self.elements.append({"text": text,
                              "offset_ms": offset_ms,
                              "duration_ms": duration_ms})

    def is_similar(self, text, previous_text):
        return jellyfish.jaro_distance(text, previous_text) > self.jaro_distance_threshold

    def extend_duration(self, index, duration_ms):
        """
        extends the lifetime of the subtitle at index by duration_ms
        """
        self.elements[index]["duration_ms"] += duration_ms

    def convert_from_ms(self, milliseconds):
        seconds, milliseconds = divmod(milliseconds,1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        seconds = seconds + milliseconds/1000
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):04}"

    def generate(self, formatting="vtt"):
        vtt = WebVTT()
        for subtitle in self.elements:
            offset_ms = subtitle.get("offset_ms")
            duration_ms = subtitle.get("duration_ms")
            start_caption_ms = offset_ms
            end_caption_ms = offset_ms + duration_ms
            caption = subtitle.get("text")
            vtt.captions.append(Caption(
                self.convert_from_ms(start_caption_ms),
                self.convert_from_ms(end_caption_ms),
                (caption,))
            )
        if formatting == "vtt":
            return vtt.content
        elif formatting == "srt":
            pass
        else:
            raise ValueError(f"Unknown subtitle format {formatting}")


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
        self.text_by_label = {}

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
                       prune_chars_regex=[r"[\x00-\x1f\x7f-\x9f\|\_]"],
                       masked_images_dir=None):
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
            if label == "description_overlay": #descriptions are black text on grey background, hardcoded here due to project time constraints
                masked_img = self.apply_mask(show_class=SampledImage.prediction_labels[label], black_white_threshold=140, invert_colors=False)
            else:
                masked_img = self.apply_mask(show_class=SampledImage.prediction_labels[label])
            text = pytesseract.image_to_string(masked_img, lang=tesseract_languages)
            for regex in prune_chars_regex:
                text = re.sub(regex, '', text)
            if masked_images_dir:
                debug_img_path = os.path.join(masked_images_dir, f"framenr_{self.frame_number}_framems_{self.frame_ms}_label_{label}.jpg")
                log.debug(f"Writing masked image to {debug_img_path} (Text: {text})")
                cv.imwrite(debug_img_path, masked_img)
            self.text_by_label[label] = text
        return self.text_by_label


    def apply_mask(self, show_class, mask_value=0, convert_to_black_white=True, black_white_threshold=120, invert_colors=True):
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
parser.add_argument("--masked-images-dir",
                    type=str,
                    default="",
                    help="Useful for debuggin, directory to place masked images in.")
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
parser.add_argument("--interactive",
                    default=False,
                    action="store_true",
                    help="Dont launch fastapi server")
parser.add_argument('--port',
                    type=int,
                    default=int(os.environ.get("APP_PORT", 8080)),
                    help="Run api on this port.")
args = parser.parse_args()


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
pytesseract.pytesseract.tesseract_cmd = args.tesseract_filepath
__version__ = "1.0"

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


app = FastAPI(title="SRF Tagesschau Classifier 9000",
              description="Samples image data from tagesschau sessions and adds subtitle cues for text presented in overlays",
              version=__version__,
              contact={"name": "Armon Dressler",
                       "url": "https://github.com/armondressler/srf-presenter-detector",
                       "email": "armon.dressler@stud.hslu.ch"})

@app.get("/version")
async def version():
    return __version__

@app.post("/generate-vtt")
async def generate_vtt(file: UploadFile = File(...)):
    temp_dir = os.path.join(args.temp_dir, str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    video_filepath = os.path.join(temp_dir, file.filename)
    try:
        with open(video_filepath, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    except FileNotFoundError:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    subtitles = process_videofile(filepath=video_filepath, temp_dir=temp_dir)
    return {"VTT": f"{subtitles}"}

def process_videofile(filepath, temp_dir="/tmp", sampling_rate=0.4, tesseract_languages=["deu","fra","eng"], masked_images_dir=None):
    log.info(f"Classifying video data from {filepath} with sampling rate of {args.sampling_rate} fps")
    
    cap = cv.VideoCapture(filepath)
    fps = round(cap.get(cv.CAP_PROP_FPS))
    hop = round(fps / sampling_rate)
    log.info(f"Video has {fps} fps, sampling every {hop}th frame.")
    
    sampled_images = []
    frame_read_success, frame = cap.read()
    frame_counter = 0

    while frame_read_success:
        if frame_counter % hop == 0:
            timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
            sampled_images.append(SampledImage(data=frame,
                                               video_filepath=args.video_filepath,
                                               frame_number=frame_counter,
                                               frame_ms=timestamp))
            log.debug(f"Sampled frame {frame_counter} at {timestamp:0.0f} ms")
        frame_read_success, frame = cap.read()
        frame_counter += 1
    
    model = keras.models.load_model(args.model_filepath)
    subtitle = Subtitle()
    for sampled_image in sampled_images:
        input_img = sampled_image.as_predictable(resize_x=args.model_input_size_x, resize_y=args.model_input_size_y)
        prediction = model.predict(input_img, verbose=1 if args.verbose else 0)
        sampled_image.prediction = np.argmax(prediction, axis=3).astype(np.uint8)[0,:,:]

        text_by_label = sampled_image.recognize_text(prune_chars_regex=[r"^\s*.{,2}\s*$",r"[\x00-\x1f\x7f-\x9f\|\_]"],
                                                     prediction_languages=tesseract_languages,
                                                     masked_images_dir=masked_images_dir)
        name_text = text_by_label.get('name_overlay').strip() + " " if text_by_label.get('name_overlay') else ""
        description_text = f"({text_by_label.get('description_overlay').strip()})" if text_by_label.get('description_overlay') else ""
        subtitle_text = name_text + description_text
        if subtitle_text:
            subtitle.append(subtitle_text, offset_ms=sampled_image.frame_ms, duration_ms=1.0/sampling_rate*1000)
    return subtitle.generate()

if __name__ == "__main__":
    if args.interactive:
        temp_dir = os.path.join(args.temp_dir, str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)
        process_videofile(filepath=args.video_filepath, temp_dir=temp_dir, sampling_rate=args.sampling_rate, masked_images_dir=args.masked_images_dir)
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=args.port, log_level="debug" if args.verbose else "info")
