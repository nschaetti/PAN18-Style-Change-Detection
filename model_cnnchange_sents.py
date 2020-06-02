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
from torchlanguage import models
from tools import functions, settings
from torch.autograd import Variable
from torchlanguage import transforms as ltransforms
import spacy
import itertools
import random
import math


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
            created_samples += combinations(sentences, sentences, 0)
        # end for
        # Different
        for p in range(0, len(parts) - 1):
            sentences1 = nlp(parts[p]).sents
            sentences2 = nlp(parts[p + 1]).sents
            created_samples += combinations(sentences1, sentences2, 1)
        # end for
    else:
        # For each sentences
        sentences = nlp(the_text).sents
        created_samples = combinations(sentences, sentences, 0)
    # end if

    # Shuffle
    random.shuffle(created_samples)
    return created_samples
# end create_samples


# Settings
window_size = 250

# Argument parser
args = functions.argument_parser_training_model()

# NLP
nlp = spacy.load('en')

# Data set roots
training_path = os.path.join(args.root, "training")
validation_path = os.path.join(args.root, "validation")

if args.n_gram == 'c1':
    transforms = ltransforms.Compose([
        ltransforms.ToLower(),
        ltransforms.Character(),
        ltransforms.ToIndex(start_ix=1),
        ltransforms.ToLength(length=window_size),
        ltransforms.MaxIndex(max_id=settings.voc_sizes[args.n_gram])
    ])
else:
    transforms = ltransforms.Compose([
        ltransforms.ToLower(),
        ltransforms.Character2Gram(),
        ltransforms.ToIndex(start_ix=1),
        ltransforms.ToLength(length=window_size),
        ltransforms.MaxIndex(max_id=settings.voc_sizes[args.n_gram])
    ])
# end if

# Loss function
loss_function = nn.CrossEntropyLoss()

# CNN Distance learning
model = models.CNNCDist(
    window_size=window_size,
    vocab_size=settings.voc_sizes[args.n_gram],
    n_classes=2,
    temporal_division=1,
    out_channels=(args.n_filters, args.n_filters, args.n_filters),
    embedding_dim=args.dim,
    n_linear=args.n_linear
)
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 1000.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

# For each iteration
for epoch in range(args.epoch):
    # Total losses
    training_losses = 0.0
    training_success = 0.0
    training_total = 0.0

    # Train
    model.train()

    # For each file in training
    for file_name in os.listdir(training_path):
        if ".txt" in file_name:
            # Load text
            text = codecs.open(os.path.join(training_path, file_name), mode='r', encoding='utf-8').read()

            # Load truth
            json_truth = json.load(open(os.path.join(training_path, file_name[:-4] + ".truth")))

            # Samples
            samples = create_samples(text, json_truth)

            # Create batches
            for i, (sent1, sent2, label) in enumerate(samples):
                # Inputs
                inputs = torch.cat((transforms(sent1.text), transforms(sent2.text)), dim=1)

                # Truth
                truth = torch.LongTensor(1).fill_(label)

                # Add
                if i == 0:
                    sample_inputs = inputs
                    sample_truths = truth
                else:
                    sample_inputs = torch.cat((sample_inputs, inputs), dim=0)
                    sample_truths = torch.cat((sample_truths, truth), dim=0)
                # end if
            # end for

            # For each batches
            for i in np.arange(0, sample_inputs.size(0), args.batch_size):
                # Inputs
                inputs = sample_inputs[i:i+args.batch_size]

                # Truth
                truth = sample_truths[i:i+args.batch_size]

                # To variable
                inputs, truth = Variable(inputs), Variable(truth)
                if args.cuda:
                    inputs, truth = inputs.cuda(), truth.cuda()
                # end if

                # Set gradient to zero
                model.zero_grad()

                # TRAINING
                model_output = model(inputs)

                # Loss
                loss = loss_function(model_output, truth)

                # Take the max as predicted
                _, predicted = torch.max(model_output.data, 1)

                # Add to correctly classified text
                training_success += (predicted == truth.data).sum()

                # Add
                training_total += inputs.size(0)

                # Loss and step
                training_losses += loss.data[0]
                loss.backward()
                optimizer.step()
            # end for
            print u"\r" + str(training_total),
        # end if
    # end for
    print(u"")

    # Eval
    model.eval()

    # Validation losses
    validation_losses = 0.0
    validation_success = 0.0
    validation_total = 0.0
    same_success = 0.0

    # For each file in validation
    for file_name in os.listdir(validation_path):
        if ".txt" in file_name:
            # Load text
            text = codecs.open(os.path.join(validation_path, file_name), mode='r', encoding='utf-8').read()

            # Load truth
            truth = json.load(open(os.path.join(validation_path, file_name[:-4] + ".truth")))

            # Samples
            samples = create_samples(text, truth)

            # For each validation examples
            for i, (sent1, sent2, label) in enumerate(samples):
                # Inputs
                inputs = torch.cat((transforms(sent1.text), transforms(sent2.text)), dim=1)

                # Truth
                truth = torch.LongTensor(1).fill_(label)

                # To variable
                inputs, truth = Variable(inputs), Variable(truth)
                if args.cuda:
                    inputs, truth = inputs.cuda(), truth.cuda()
                # end if

                # Model
                model_output = model(inputs)

                # Loss
                loss = loss_function(model_output, truth)

                # Take the max as predicted
                _, predicted = torch.max(model_output.data, 1)

                # Add to correctly classified text
                same_success += (truth.data == 0).sum()
                validation_success += (predicted == truth.data).sum()

                # Loss and step
                validation_losses += loss.data[0]
                validation_total += inputs.size(0)
            # end for
            print u"\r" + str(validation_total),
        # end if
    # end for
    print(u"")

    # Accuracies
    training_accuracy = training_success / training_total * 100.0
    validation_accuracy = validation_success / validation_total * 100.0
    same_accuracy = same_success / validation_total * 100.0

    # Show
    print(u"Epoch {}, training loss : {}, training accuracy {}, validation loss {}, validation accuracy {} ({})".format(
        epoch,
        training_losses / training_total,
        training_accuracy,
        validation_losses / validation_total,
        validation_accuracy,
        same_accuracy
    ))

    # Save if better
    if validation_accuracy > best_acc:
        best_acc = validation_accuracy
        print(u"Saving model with best loss {}".format(best_acc))
        torch.save(
            transforms.transforms[2].token_to_ix,
            open(os.path.join(args.output, "cnnchange-voc-sent.pth"), 'wb')
        )
        torch.save(
            model.state_dict(),
            open(os.path.join(args.output, "cnnchange-sent.pth"), 'wb')
        )
    # end if
# end for
