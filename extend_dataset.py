# -*- coding: utf-8 -*-
#

# Imports
import argparse
import os
from random import shuffle
from shutil import copyfile
import numpy as np


# Commands
parser = argparse.ArgumentParser(prog="Extend dataset")

# Command
parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
parser.add_argument("--validation-size", type=int, help="Number of samples in the dataset", required=True)
parser.add_argument("--output", type=str, help="Output directory", required=True)
args = parser.parse_args()

# Output directories
output_training_directory = os.path.join(args.output, "training")
output_validation_directory = os.path.join(args.output, "validation")

# Create training directory
if not os.path.exists(output_training_directory):
    os.mkdir(output_training_directory)
# end if

# Create validation directory
if not os.path.exists(output_validation_directory):
    os.mkdir(output_validation_directory)
# end if

# List of files
samples_files = list()

# Training and validation size
training_length = len(os.listdir(os.path.join(args.dataset, "training"))) / 2
validation_length = len(os.listdir(os.path.join(args.dataset, "validation"))) / 2

# Copy training samples
for file_num in range(1, training_length+1):
    copyfile(os.path.join(args.dataset, "training", "problem-{}.txt".format(file_num)),
             os.path.join(output_training_directory, "problem-{}.txt".format(file_num)))
    copyfile(os.path.join(args.dataset, "training", "problem-{}.truth".format(file_num)),
             os.path.join(output_training_directory, "problem-{}.truth".format(file_num)))
# end for

# Validation files
validation_samples = os.listdir(os.path.join(args.dataset, "validation"))

# Copy training files
for file_num in range(1, validation_length - args.validation_size + 1):
    copy_num = file_num + training_length
    copyfile(os.path.join(args.dataset, "validation", "problem-{}.txt".format(file_num)),
             os.path.join(output_training_directory, "problem-{}.txt".format(copy_num)))
    copyfile(os.path.join(args.dataset, "validation", "problem-{}.truth".format(file_num)),
             os.path.join(output_training_directory, "problem-{}.truth".format(copy_num)))
# end for

# Copy validation files
index = 1
for file_num in np.arange(validation_length - args.validation_size + 1, validation_length + 1):
    copyfile(os.path.join(args.dataset, "validation", "problem-{}.txt".format(file_num)),
             os.path.join(output_validation_directory, "problem-{}.txt".format(index)))
    copyfile(os.path.join(args.dataset, "validation", "problem-{}.truth".format(file_num)),
             os.path.join(output_validation_directory, "problem-{}.truth".format(index)))
    index += 1
# end for
