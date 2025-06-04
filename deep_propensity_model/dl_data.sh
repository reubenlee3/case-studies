#!/bin/bash

# Direct URL to the MovieLens 20M dataset
URL="http://files.grouplens.org/datasets/movielens/ml-20m.zip"

# Destination folder for the dataset
DEST_DIR="ml-20m"

# Create the destination folder if it doesn't exist
mkdir -p $DEST_DIR

# Download the dataset using wget
wget $URL -P $DEST_DIR

# Unzip the dataset
unzip $DEST_DIR/ml-20m.zip -d $DEST_DIR

# Cleanup (optional)
rm $DEST_DIR/ml-20m.zip

echo "Download and extraction complete!"
