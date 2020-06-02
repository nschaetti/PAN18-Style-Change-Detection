# -*- coding: utf-8 -*-
#

# Imports
import copy
import torch.utils.data
import os
import codecs
import json
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from torchlanguage import models
from tools import functions, settings
from torch.autograd import Variable
from torchlanguage import transforms as ltransforms
import spacy
import itertools
import random
import math
import argparse


def combinations(l1, l2, d):
    comb = list()
    for o1 in l1:
        for o2 in l2:
            if o1 != o2:
                comb.append((o1, o2, d))
            # end if
        # end for
    # end for
    return comb
# end combinations


# Create sample
def create_samples(the_text, the_truth):
    created_samples = list()
    # If changes or not
    if the_truth['changes']:
        parts = list()
        pos = 0
        for position in the_truth['positions']:
            parts.append(the_text[pos:position])
            pos = position
        # end for
        parts.append(the_text[pos:])
        # Same
        for part in parts:
            sentences = nlp(part).sents
            created_samples += combinations(sentences, sentences, 0.0)
        # end for
        # Different
        for p in range(0, len(parts) - 1):
            sentences1 = nlp(parts[p]).sents
            sentences2 = nlp(parts[p + 1]).sents
            created_samples += combinations(sentences1, sentences2, 1.0)
        # end for
    else:
        # For each sentences
        sentences = nlp(the_text).sents
        created_samples = combinations(sentences, sentences, 0.0)
    # end if

    # Shuffle
    random.shuffle(created_samples)
    return created_samples
# end create_samples


# Settings
window_size = 250
n_comp = 10
n_thresholds = 40
n_levels = 50
threshold_list = np.linspace(-0.1, 0.2, n_thresholds)

# Args
parser = argparse.ArgumentParser(u"cnndist sents")
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--voc", type=str, required=True)
parser.add_argument("--dim", type=int, help="Embedding dimension", default=50)
parser.add_argument("--n-gram", type=str, required=True)
parser.add_argument("--n-filters", type=int, help="Number of filters", default=500)
parser.add_argument("--n-linear", type=int, help="Number of linear layer", default=2)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# NLP
nlp = spacy.load('en')

# Data set roots
training_path = os.path.join(args.root, "training")
validation_path = os.path.join(args.root, "validation")

# CNN Distance learning
model = models.CNNCDist(
    window_size=window_size,
    vocab_size=settings.voc_sizes[args.n_gram],
    n_classes=1,
    temporal_division=1,
    out_channels=(args.n_filters, args.n_filters, args.n_filters),
    embedding_dim=args.dim,
    n_linear=args.n_linear
)
if args.cuda:
    model.cuda()
# end if

# Load model and voc
model.load_state_dict(torch.load(open(args.model, 'rb')))
if args.cuda:
    model.cuda()
# end if
voc = torch.load(open(args.voc, 'rb'))

# Eval
model.eval()

if args.n_gram == 'c1':
    transforms = ltransforms.Compose([
        ltransforms.ToLower(),
        ltransforms.Character(),
        ltransforms.ToIndex(start_ix=1, token_to_ix=voc),
        ltransforms.ToLength(length=window_size),
        ltransforms.MaxIndex(max_id=settings.voc_sizes[args.n_gram])
    ])
else:
    transforms = ltransforms.Compose([
        ltransforms.ToLower(),
        ltransforms.Character2Gram(),
        ltransforms.ToIndex(start_ix=1, token_to_ix=voc),
        ltransforms.ToLength(length=window_size),
        ltransforms.MaxIndex(max_id=settings.voc_sizes[args.n_gram])
    ])
# end if

# Validation losses
validation_total = 0
validation_success = np.zeros((n_levels, n_thresholds))
n_files = 0.0
n_same = 0.0
n_diff = 0.0

# Values
same_distance_values = np.array([])
diff_distance_values = np.array([0.5])

# For each file in validation
for file_name in os.listdir(validation_path):
    if ".txt" in file_name:
        # Load text
        text = codecs.open(os.path.join(validation_path, file_name), mode='r', encoding='utf-8').read()

        # Load truth
        json_truth = json.load(open(os.path.join(validation_path, file_name[:-4] + ".truth")))

        # Samples
        samples = create_samples(text, json_truth)

        # List of distance
        distance_list = np.array([])

        # For each validation examples
        for i, (sent1, sent2, label) in enumerate(samples):
            # Inputs
            inputs = torch.cat((transforms(sent1.text), transforms(sent2.text)), dim=1)

            # Truth
            truth = torch.FloatTensor(1).fill_(label)

            # To variable
            inputs, truth = Variable(inputs), Variable(truth)
            if args.cuda:
                inputs, truth = inputs.cuda(), truth.cuda()
            # end if

            # Model
            model_output = model(inputs)

            # Add
            if int(truth[0]) == 0:
                same_distance_values = np.append(same_distance_values, [float(model_output[0])])
            else:
                diff_distance_values = np.append(diff_distance_values, [float(model_output[0])])
            # end if

            # Add to distance
            distance_list = np.append(distance_list, [float(model_output[0])])

            # Total
            validation_total += inputs.size(0)
        # end for

        # For each level and threshold
        for level in range(n_levels):
            for index, threshold in enumerate(threshold_list):
                # Prediction
                if np.sum((distance_list > threshold)) >= level + 1:
                    predicted = 1.0
                else:
                    predicted = 0.0
                # end if

                # Test
                if (predicted == 1.0 and json_truth['changes']) or (predicted == 0.0 and not json_truth['changes']):
                    validation_success[level, index] += 1.0
                # end if
            # end for
        # end for
        if json_truth['changes']:
            n_diff += 1.0
        else:
            n_same += 1.0
        # end if
        n_files += 1
        print u"\r" + str(validation_total),
    # end if
# end for

print(np.average(same_distance_values))
print(np.std(same_distance_values))

print(np.average(diff_distance_values))
print(np.std(diff_distance_values))

# Threshold accuracy
threshold_accuracy = validation_success / n_files * 100.0

# Max accuracy and best threshold
accuracy = np.max(threshold_accuracy)
best_ind = np.unravel_index(np.argmax(threshold_accuracy, axis=None), threshold_accuracy.shape)
best_level = best_ind[0] + 1
best_threshold = threshold_list[best_ind[1]]

print(u"Max accuracy : {}".format(accuracy))
print(u"Best threshold : {}".format(best_threshold))
print(u"Best level : {}".format(best_level))
print(u"Ratio diff : {}".format(n_diff / n_files * 100.0))
print(u"Ratio same : {}".format(n_same / n_files * 100.0))

plt.hist(same_distance_values, 50, alpha=0.5)
plt.hist(diff_distance_values, 50, alpha=0.5)
plt.show()

print(threshold_accuracy)
