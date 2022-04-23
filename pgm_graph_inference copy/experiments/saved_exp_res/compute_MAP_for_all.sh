#!/bin/bash

dir=$(pwd)

for entry in *
do
  echo "$entry"
  python compute_MAP_accuracy.py --data_file "$entry"
done
