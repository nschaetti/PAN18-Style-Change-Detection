# -*- coding: utf-8 -*-
#

# Imports
import copy
import torch.utils.data
from torch import nn
from torch import optim
import dataset
import numpy as np
from torchlanguage import models
from tools import functions, settings
from torch.autograd import Variable
import math

# Experience parameter
batch_size = 64
n_epoch = 1
validation_ratio = 0.1
training_samples = 5000

# Argument parser
args = functions.argument_parser_training_model()

# Get transforms
transforms = functions.text_transformer_cnn(args.n_gram)

# Style change detection dataset, training set
pan18loader_train = torch.utils.data.DataLoader(
    dataset.SCDPartsDataset(root='./extended/', download=True, transform=transforms, train=True),
    batch_size=1
)

# Style change detection dataset, validation set
pan18loader_valid = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./extended/', download=True, transform=transforms, train=False),
    batch_size=1
)

# Samples
samples = list()
n_samples = 0

# Get training data
for i, data in enumerate(pan18loader_train):
    # Parts and c
    parts, _ = data

    # Add to samples
    samples.append(parts)
    n_samples += 1
# end for

# Batches
batches = list()

# For each training samples
for i in range(training_samples):
    # Batch
    batch = torch.LongTensor(args.batch_size, settings.cnn_window_size * 2).fill_(0)
    batch_truth = torch.zeros(args.batch_size)

    # Batch size
    for b in range(args.batch_size):
        # Pick a random sample
        random_sample = np.random.randint(0, n_samples)

        # The parts
        parts = samples[random_sample]
        n_parts = len(parts)

        # Negative or positive example?
        if n_parts == 1:
            # Pick a random part
            random_part = parts[np.random.randint(0, n_parts)]
            part_length = random_part.size(1)

            # Pick two random starting position
            start_pos1 = np.random.randint(0, part_length - settings.cnn_window_size)
            start_pos2 = np.random.randint(0, part_length - settings.cnn_window_size)

            # Get the two sides
            side1 = random_part[:, start_pos1:start_pos1 + settings.cnn_window_size]
            side2 = random_part[:, start_pos2:start_pos2 + settings.cnn_window_size]

            # Truth
            batch_truth[b] = 0.0
        else:
            # Pick to consecutive random parts
            random_start_part = np.random.randint(0, n_parts-1)
            random_part1 = parts[random_start_part]
            random_part2 = parts[random_start_part+1]

            # Length
            part1_length = random_part1.size(1)
            part2_length = random_part2.size(1)

            # Pick two random starting positions
            start_pos1 = np.random.randint(0, part1_length - settings.cnn_window_size)
            start_pos2 = np.random.randint(0, part2_length - settings.cnn_window_size)

            # Get the two sides
            side1 = random_part1[:, start_pos1:start_pos1 + settings.cnn_window_size]
            side2 = random_part2[:, start_pos2:start_pos2 + settings.cnn_window_size]

            # Truth
            batch_truth[b] = 1.0
        # end if

        # Concatenate
        sides = torch.cat((side1, side2), dim=1)

        # Set
        batch[b] = sides
    # end for

    # Add to batch
    batches.append((batch, batch_truth))
# end for

# Loss function
loss_function = nn.MSELoss()

# Bi-directional Embedding GRU
model = models.CNNCDist(window_size=settings.cnn_window_size, vocab_size=settings.voc_sizes[args.n_gram])
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# For each iteration
for epoch in range(n_epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0

    # For each training examples
    for batch in batches:
        # Inputs and label
        inputs, truth = batch

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

        # Add
        training_loss += loss.data[0]
        training_total += 1.0

        # Backward and step
        loss.backward()
        optimizer.step()
    # end for

    # Counters
    test_diff = 0
    test_total = 0

    # For each test sample
    for i, data in enumerate(pan18loader_valid):
        # Parts and c
        inputs, truth = data

        # Different parts to measure
        parts = list()

        # Get each parts
        for j in torch.arange(0, inputs.size(1) - settings.cnn_window_size, settings.stride):
            # Add
            parts.append(inputs[:, int(j):int(j) + settings.cnn_window_size])
        # end for

        # Number of parts
        n_parts = len(parts)

        # Matrix of distance
        distance_matrix = torch.FloatTensor(n_parts, n_parts).fill_(0.0)

        # For all combination
        for n in range(n_parts):
            for m in range(n_parts):
                # Inputs
                concate = torch.cat((parts[n], parts[m]), dim=1)

                # Variable
                concate = Variable(concate)
                if args.cuda:
                    concate = concate.cuda()
                # end if

                # Model
                model_output = model(concate)

                # Distance
                distance_matrix[n, m] = float(model_output[0])
            # end for
        # end for

        # Max
        max_dist = torch.max(distance_matrix)

        # Diff
        diff = float(truth[0]) - max_dist
        test_diff += diff * diff
        test_total += 1.0
    # end for

    # Show
    print(u"Epoch {}, accuracy : {}".format(epoch, 1.0 / test_total * math.sqrt(test_diff)))
# end for
