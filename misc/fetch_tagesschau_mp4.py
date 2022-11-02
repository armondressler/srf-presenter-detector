#!/usr/bin/env python3

__author__ = "armon.dressler@gmail.com"
__description__ = """Given a timerange and a destination dir, fetch every tagesschau in mp4 format in the timerange and store it in the destination dir. Inspired by https://github.com/jmesserli/tagesschau-helper -> Some of the tagesschau sessions don't include a direct link to the underlying mp4. There is a feed at https://www.srf.ch/feed/podcast/hd/ff969c14-c5a7-44ab-ab72-14d4c9e427a9.xml which includes said links however. Each session is ~1.1 GBs in size."""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start-date",
                    action="store",
                    help='ISO 8601 formatted start date to fetch news sessions from. e.g. 2022-09-15'
                    )
parser.add_argument('-v', '--verbose',
                    action='store_true',
                    help='Verbose output'
                    )
args = parser.parse_args()

