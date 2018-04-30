# -*- coding: utf-8 -*-
#

# Imports
import argparse
import os
from random import shuffle
from shutil import copyfile
import numpy as np
import codecs
import json
import itertools as it


# Commands
parser = argparse.ArgumentParser(prog="Extend dataset")

# Command
parser.add_argument("--dataset", type=str, help="Dataset path", required=True)
args = parser.parse_args()

# Directories
training_dir = os.path.join(args.dataset, "training")
validation_dir = os.path.join(args.dataset, "validation")

# Number of files
n_files = len(os.listdir(training_dir)) / 2

# Next index
next_index = n_files + 1

# For each file
for file_num in np.arange(1, n_files+1):
    print(u"From file {}".format(os.path.join(training_dir, "problem-{}.txt".format(file_num))))
    # Text
    text = codecs.open(os.path.join(training_dir, "problem-{}.txt".format(file_num)), 'rb', encoding='utf-8').read()

    # Load truth
    truth = json.load(open(os.path.join(training_dir, "problem-{}.truth".format(file_num))))

    # Changed
    changed = truth['changes']

    # If changed
    if changed:
        last_pos = 0
        parts = list()
        for pos in truth['positions']:
            text_part = text[last_pos:pos]
            parts.append(text_part)
            last_pos = pos
        # end for
        parts.append(text[last_pos:])

        # Compute permutations
        perm = list(it.permutations(parts, len(parts)))

        # Create new sample from permutations
        for p in perm:
            if list(p) != parts:
                new_text = u"\n".join(p)
                codecs.open(os.path.join(training_dir, "problem-{}.txt".format(next_index)), 'wb', encoding='utf-8').write(new_text)
                changes_pos = list()
                pos = 0
                for c in p:
                    changes_pos.append(pos + len(c))
                    pos += len(c)
                # end for
                json.dump({'positions': changes_pos[:-1], 'changes': True}, open(os.path.join(training_dir, "problem-{}.truth".format(next_index)), 'wb'))
                print(u"New file {}".format(os.path.join(training_dir, "problem-{}.txt".format(next_index))))
                print(u"New truth {}".format(os.path.join(training_dir, "problem-{}.truth".format(next_index))))
                next_index += 1
            # end if
        # end for

        # Create new sample from parts alone
        for part in parts:
            codecs.open(os.path.join(training_dir, "problem-{}.txt".format(next_index)), 'wb', encoding='utf-8').write(
                part)
            json.dump({'positions': [], 'changes': False},
                      open(os.path.join(training_dir, "problem-{}.truth".format(next_index)), 'wb'))
            print(u"New file {}".format(os.path.join(training_dir, "problem-{}.txt".format(next_index))))
            print(u"New truth {}".format(os.path.join(training_dir, "problem-{}.truth".format(next_index))))
            next_index += 1
    # end if
# end for
