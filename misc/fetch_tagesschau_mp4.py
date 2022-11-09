#!/usr/bin/env python3

__author__ = "armon.dressler@gmail.com"
__description__ = """Given a URL and a destination dir, fetch the tagesschau in mp4 format and store it in the destination dir. Inspired by https://github.com/jmesserli/tagesschau-helper -> Some of the tagesschau sessions don't include a direct link to the underlying mp4. Each session is ~1.1 GBs in size."""

import argparse
import youtube_dl
import ffmpeg


parser = argparse.ArgumentParser()
parser.add_argument("-u",
                    "--url",
                    type=str,
                    required=True,
                    help="URL to tagesschau show, e.g. https://www.srf.ch/play/tv/tagesschau/video/tagesschau-vom-08-11-2022-mittagsausgabe?urn=urn:srf:video:caf4a877-e935-48f8-84f5-b42800c06fa7")
parser.add_argument("-d",
                    "--dest",
                    type=str,
                    default="./tagesschau",
                    help="Destination dir for video files")
parser.add_argument("-v", 
                    "--verbose",
                    action="store_true",
                    help="Verbose output")
args = parser.parse_args()


with youtube_dl.YoutubeDL({}) as ydl:
    ydl.download([args.url])