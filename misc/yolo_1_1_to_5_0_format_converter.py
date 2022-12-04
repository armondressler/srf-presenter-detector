import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("obj_train_data_path", help="Directory containing jpg and txt files")
args = parser.parse_args()

os.makedirs(os.path.join(args.obj_train_data_path, "../labels"), exist_ok=True)
os.makedirs(os.path.join(args.obj_train_data_path, "../images"), exist_ok=True)


images = [f for f in os.listdir(args.obj_train_data_path) if '.jpg' in f.lower()]

for image in images:
    new_path = os.path.join(args.obj_train_data_path, f"../images/")
    shutil.move(os.path.join(args.obj_train_data_path, image), new_path)


labels = [f for f in os.listdir(args.obj_train_data_path) if '.txt' in f.lower()]

for label in labels:
    new_path = os.path.join(args.obj_train_data_path, f"../labels/")
    shutil.move(os.path.join(args.obj_train_data_path, label), new_path)