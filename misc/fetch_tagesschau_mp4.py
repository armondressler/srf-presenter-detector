#!/usr/bin/env python3

__author__ = "armon.dressler@gmail.com"
__description__ = """Given a URL and a destination dir, fetch the tagesschau in mp4 format and store it in the destination dir. Each session is ~1.1 GBs in size."""

import argparse
import logging
import os
import re

import ffmpeg
import youtube_dl

parser = argparse.ArgumentParser()
parser.add_argument("-u",
                    "--url",
                    type=str,
                    required=True,
                    help="URL to tagesschau show, e.g. https://www.srf.ch/play/tv/tagesschau/video/tagesschau-vom-08-11-2022-mittagsausgabe?urn=urn:srf:video:caf4a877-e935-48f8-84f5-b42800c06fa7")
parser.add_argument("-d",
                    "--download-dir",
                    type=str,
                    default=None,
                    help="Destination dir for video files")
parser.add_argument("-m",
                    "--ffmpeg-location",
                    type=str,
                    default="ffmpeg.exe",
                    help="Location of ffmpeg binary")
parser.add_argument("--video-format",
                    type=str,
                    default="mp4",
                    help="Format to recode video to")
parser.add_argument("--video-sample-rate",
                    type=float,
                    default=0.5,
                    help="Rate at which to capture frames of video which will be placed in separate directory. Rate is defined in pictures per second, e.g. 0.5 meaning 1 image every 2 seconds")
parser.add_argument("-v", 
                    "--verbose",
                    default=False,
                    action="store_true",
                    help="Verbose output")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG) if args.verbose else logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

if not os.path.isabs(args.ffmpeg_location):
    args.ffmpeg_location = os.path.join(os.getcwd(), args.ffmpeg_location)

if not os.path.isfile(args.ffmpeg_location):
    raise FileNotFoundError(f"ffmpeg not found in {args.ffmpeg_location}")

match = re.match(r"^https://www.srf.ch/[a-z0-9-].*/([a-z0-9-].*)\?.*$", args.url)
show_name = match.group(1)
if show_name is None:
    raise ValueError(f"Failed to grab show name from url {args.url}, checking with first group of regex ^https://www.srf.ch/[a-z0-9-].*/([a-z0-9-].*)\?.*$")

if args.download_dir is None:
    args.download_dir = show_name

if not os.path.isdir(args.download_dir):
    log.info(f"Creating download dir {args.download_dir}")
    os.mkdir(args.download_dir)
os.chdir(args.download_dir)
root_dir = os.getcwd()

for directory in ("original","downscaled","images", "labels"):
    try:
        os.mkdir(os.path.join(root_dir, directory))
    except FileExistsError:
        continue
os.chdir(os.path.join(root_dir, "original"))

ydl_opts = {
    "quiet": False if args.verbose else True,
    "recode_video": f"{args.video_format}",
    "outtmpl": f"{show_name}.{args.video_format}"
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    log.info(f"Downloading {args.url} into {args.download_dir}")
    ydl.download([args.url])

os.chdir(os.path.join(root_dir, "downscaled"))
stream = ffmpeg.input(f"{os.path.join(root_dir, 'original')}/{show_name}.{args.video_format}")
stream = ffmpeg.filter(stream, "scale", w="440",h="330",force_original_aspect_ratio="decrease")
stream = ffmpeg.output(stream, f"{show_name}_downscaled.{args.video_format}")
ffmpeg.run(stream, cmd=args.ffmpeg_location)

os.chdir(os.path.join(root_dir, "sample_images"))
stream = ffmpeg.input(f"{os.path.join(root_dir, 'original')}/{show_name}.{args.video_format}")
stream = ffmpeg.filter(stream, "fps", fps=args.video_sample_rate, round="up")
stream = ffmpeg.output(stream, f"{show_name}_%d.jpg")
ffmpeg.run(stream, cmd=args.ffmpeg_location)
