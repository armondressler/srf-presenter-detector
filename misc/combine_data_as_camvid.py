import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--camvid-directory", action="append", help="Directory containing default (images) and defaultannot (masks) directories")
parser.add_argument("zip_destination_dir", help="Directory to place zipfile in containing the combined images and masks")
args = parser.parse_args()

TMP_DIR = "./camvid_zipme/"
TMP_DIR_IMAGES = os.path.join(TMP_DIR, "images")
TMP_DIR_MASKS = os.path.join(TMP_DIR, "masks")

os.makedirs(TMP_DIR_IMAGES, exist_ok=True)
os.makedirs(TMP_DIR_MASKS, exist_ok=True)

for camvid_directory in args.camvid_directory:
    image_dir = os.path.join(camvid_directory, "default")
    images = [f for f in os.listdir(image_dir) if '.jpg' in f.lower()]
    for image in images:
        shutil.copy(os.path.join(image_dir, image), TMP_DIR_IMAGES)

    mask_dir = os.path.join(camvid_directory, "defaultannot")
    masks = [f for f in os.listdir(mask_dir) if '.png' in f.lower()]
    for mask in masks:
        shutil.copy(os.path.join(mask_dir, mask), TMP_DIR_MASKS)

shutil.make_archive(base_name=os.path.join(args.zip_destination_dir, "tagesschau_images_and_masks"), format="zip", root_dir=TMP_DIR)