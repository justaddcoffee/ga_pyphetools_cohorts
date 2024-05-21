#!/bin/bash

# Define the source directory
SOURCE_DIR="data/synthetic_phenopackets/"
# Define the destination directory
DEST_DIR="data/synthetic_phenopackets_3500"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy all files matching Marfan_syndrome_
find "$SOURCE_DIR" -name 'Marfan_syndrome_*.json' -exec cp {} "$DEST_DIR" \;

# Get a list of all other JSON files excluding Marfan_syndrome_ ones
OTHER_FILES=$(find "$SOURCE_DIR" -name '*.json' | grep -v 'Marfan_syndrome_')

# Select a random sample of 3500 files
RANDOM_SAMPLE=$(echo "$OTHER_FILES" | shuf -n 3500)

# Copy the random sample to the destination directory
for file in $RANDOM_SAMPLE; do
  cp "$file" "$DEST_DIR"
done
