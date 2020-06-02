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

# List of threshold
n_thresholds = 40
n_levels = 10
threshold_list = np.linspace(-1.0, 2.0, n_thresholds)

# Argument parser
args = functions.argument_parser_training_model()

# Get transforms
transforms = functions.text_transformer_cnn(args.n_gram)

# Style change detection dataset, training set
pan18loader_train = torch.utils.data.DataLoader(
    dataset.SCDPartsDataset(root='./recalib/', download=True, transform=transforms, train=True),
    batch_size=1
)

# Style change detection dataset, validation set
pan18loader_valid = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./recalib/', download=True, transform=transforms, train=False),
    batch_size=1
)

# Loss function
loss_function = nn.MSELoss()

# CNN Distance learning
model = models.CNNCDist(
    window_size=settings.cnn_window_size,
    vocab_size=settings.voc_sizes[args.n_gram],
    n_classes=1,
    temporal_division=args.temporal_division,
    out_channels=(args.n_filters, args.n_filters, args.n_filters)
)
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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

# Generate training samples
same = True
for i in range(settings.training_samples):
    # Batch
    batch = torch.LongTensor(args.batch_size, settings.cnn_window_size * 2).fill_(0)
    batch_truth = torch.FloatTensor(args.batch_size).fill_(0)

    # Batch size
    b = 0
    while b < args.batch_size:
        # Pick a random sample
        random_sample = np.random.randint(0, n_samples)

        # The parts
        parts = samples[random_sample]
        n_parts = len(parts)

        # Negative or positive example?
        if n_parts == 1 and same:
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

            # Concatenate
            sides = torch.cat((side1, side2), dim=1)

            # Set
            batch[b] = sides

            # Not same next time
            same = False
            b += 1
        elif n_parts > 1 and not same:
            # Pick to consecutive random parts
            random_start_part = np.random.randint(0, n_parts - 1)
            random_part1 = parts[random_start_part]
            random_part2 = parts[random_start_part + 1]

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

            # Concatenate
            sides = torch.cat((side1, side2), dim=1)

            # Set
            batch[b] = sides

            # Not same next time
            same = True
            b += 1
        # end if
    # end while

    # Add to batch
    batches.append((batch, batch_truth))
# end for

# For each iteration
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0

    # For each training examples
    for i, batch in enumerate(batches):
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

    # Validation losses
    validation_success = np.zeros((n_levels, n_thresholds))
    validation_total = 0.0

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

        # List of distances
        ordered_distances = list()
        distance_matrix = np.zeros((n_parts, n_parts))

        # For all combination
        for n in range(n_parts):
            for m in range(n_parts):
                if n != m:
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
                    ordered_distances.append(float(model_output[0]))
                    distance_matrix[n, m] = float(model_output[0])
                # end if
            # end for
        # end for

        # Sort the list of distance and to numpy
        ordered_distances.sort(reverse=True)
        ordered_distances = np.array(ordered_distances)

        # For each level and threshold
        for level in range(n_levels):
            for index, threshold in enumerate(threshold_list):
                # Prediction
                if np.sum((ordered_distances > threshold)) >= level + 1:
                    predicted = 1.0
                else:
                    predicted = 0.0
                # end if

                # Test
                if predicted == float(truth[0]):
                    validation_success[level, index] += 1.0
                # end if
            # end for
        # end for

        # Total
        validation_total += 1.0
    # end for

    # Threshold accuracy
    threshold_accuracy = validation_success / validation_total * 100.0

    # Max accuracy and best threshold
    accuracy = np.max(threshold_accuracy)
    best_ind = np.unravel_index(np.argmax(threshold_accuracy, axis=None), threshold_accuracy.shape)
    best_level = best_ind[0] + 1
    best_threshold = threshold_list[best_ind[1]]

    # Show
    print(u"Epoch {}, training loss : {}, best accuracy {} at level {} with threshold {}".format(
        epoch,
        training_loss / training_total,
        accuracy,
        best_level,
        best_threshold)
    )

    # Save if better
    if accuracy > best_acc:
        best_acc = accuracy
        print(u"Saving model with best accuracy {}".format(best_acc))
        torch.save(
            transforms.transforms[2].token_to_ix,
            open(args.output, 'wb')
        )
        torch.save(
            model.state_dict(),
            open(args.output, 'wb')
        )
    # end if
# end for
