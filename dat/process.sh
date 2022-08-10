#!/bin/bash

touch params.csv X_fails.csv Z_fails.csv
python3 final_decoding.py
python3 plot.py
