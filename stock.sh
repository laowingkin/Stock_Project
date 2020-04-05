#!/bin/bash

cd Stock_Project
#python news_extraction.py "$1"
python tp_extraction.py "$1"
python bb_extraction.py "$1"
python short_extraction.py "$1"

python data_extraction.py "$1" "$2"