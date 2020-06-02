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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


# List of threshold
n_comp = 10
n_thresholds = 40
n_levels = 10
threshold_list = np.linspace(-1.0, 2.0, n_thresholds)

# Settings
# training_samples = 110000
# test_samples = 11000
training_samples = 10000
cnn_window_size = 1500

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
    window_size=cnn_window_size,
    vocab_size=settings.voc_sizes[args.n_gram],
    n_classes=1,
    temporal_division=args.temporal_division,
    out_channels=(args.n_filters, args.n_filters, args.n_filters),
    embedding_dim=args.dim
)
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

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
# same = True
for i in range(training_samples):
    # Batch
    batch = torch.LongTensor(args.batch_size, cnn_window_size * 2).fill_(0)
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
        if n_parts == 1:
            # Pick a random part
            random_part = parts[np.random.randint(0, n_parts)]
            part_length = random_part.size(1)

            # Pick two random starting position
            if part_length <= cnn_window_size:
                start_pos1 = 0
                start_pos2 = 0
            else:
                start_pos1 = np.random.randint(0, part_length - cnn_window_size)
                start_pos2 = np.random.randint(0, part_length - cnn_window_size)
            # end if

            # Get the two sides
            side1 = random_part[:, start_pos1:start_pos1 + cnn_window_size]
            side2 = random_part[:, start_pos2:start_pos2 + cnn_window_size]

            # Minimum size
            zero_side1 = torch.LongTensor(1, cnn_window_size).fill_(0)
            zero_side2 = torch.LongTensor(1, cnn_window_size).fill_(0)
            zero_side1[0, :side1.size(1)] = side1[0]
            zero_side2[0, :side2.size(1)] = side2[0]

            # Truth
            batch_truth[b] = 0

            # Concatenate
            sides = torch.cat((zero_side1, zero_side2), dim=1)

            # Set
            batch[b] = sides

            # Not same next time
            # same = False
            b += 1
        elif n_parts > 1:
            # Pick to consecutive random parts
            random_start_part = np.random.randint(0, n_parts - 1)
            random_part1 = parts[random_start_part]
            random_part2 = parts[random_start_part + 1]

            # Length
            part1_length = random_part1.size(1)
            part2_length = random_part2.size(1)

            # Pick two random starting positions
            if part1_length <= cnn_window_size:
                start_pos1 = 0
            else:
                start_pos1 = np.random.randint(0, part1_length - cnn_window_size)
            # end if
            if part2_length <= cnn_window_size:
                start_pos2 = 0
            else:
                start_pos2 = np.random.randint(0, part2_length - cnn_window_size)
            # end if

            # Get the two sides
            side1 = random_part1[:, start_pos1:start_pos1 + cnn_window_size]
            side2 = random_part2[:, start_pos2:start_pos2 + cnn_window_size]

            # Minimum size
            zero_side1 = torch.LongTensor(1, cnn_window_size).fill_(0)
            zero_side2 = torch.LongTensor(1, cnn_window_size).fill_(0)
            zero_side1[0, :side1.size(1)] = side1[0]
            zero_side2[0, :side2.size(1)] = side2[0]

            # Truth
            batch_truth[b] = 1

            # Concatenate
            sides = torch.cat((zero_side1, zero_side2), dim=1)

            # Set
            batch[b] = sides

            # Not same next time
            # same = True
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

    # Matrix distance samples
    distance_samples = np.array([])
    distance_truths = list()

    # For each test sample
    for i, data in enumerate(pan18loader_valid):
        # Parts and c
        inputs, truth = data

        # Different parts to measure
        parts = list()

        # Get each parts
        for j in torch.linspace(0, inputs.size(1) - cnn_window_size, n_comp):
            # Add
            parts.append(inputs[:, int(j):int(j) + cnn_window_size])
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
                    try:
                        model_output = model(concate)
                    except ValueError:
                        print(inputs.size())
                        print(truth)
                        print(parts[n])
                        print(parts[m])
                        print(concate)
                        exit()
                    # end try

                    # Distance
                    ordered_distances.append(float(model_output[0]))
                    distance_matrix[n, m] = float(model_output[0])
                # end if
            # end for
        # end for

        # Sort the list of distance and to numpy
        ordered_distances.sort(reverse=True)
        ordered_distances = np.array(ordered_distances)

        # Flatten matrix
        flatten_matrix = distance_matrix.copy()
        flatten_matrix.shape = (1, n_parts * n_parts)

        # Add to samples
        if i == 0:
            distance_samples = flatten_matrix
        else:
            distance_samples = np.vstack((distance_samples, flatten_matrix))
        # end if
        distance_truths.append('same' if float(truth[0]) == 0.0 else 'different')

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

    # Decision tree
    tree_score = np.average(cross_val_score(DecisionTreeClassifier(random_state=0), distance_samples, distance_truths, cv=10) * 100.0)
    print(tree_score)
    # Random forest
    forest_score = np.average(cross_val_score(RandomForestClassifier(random_state=0), distance_samples, distance_truths, cv=10) * 100.0)
    print(forest_score)
    # SVM score
    svm_score = np.average(cross_val_score(svm.SVC(), distance_samples, distance_truths, cv=10) * 100.0)
    print(svm_score)
    # Best classifier
    best_classifier = 'threshold'
    best_accuracy = accuracy
    if tree_score > best_accuracy:
        best_classifier = 'tree'
        best_accuracy = tree_score
    # end if
    if forest_score > best_accuracy:
        best_classifier = 'forest'
        best_accuracy = forest_score
    # end if
    if svm_score > best_accuracy:
        best_classifier = 'svm'
        best_accuracy = svm_score
    # end if

    # Show
    print(u"Epoch {}, training loss : {}, threshold accuracy {} at level {} with threshold {}, best accuracy {} with {}".format(
        epoch,
        training_loss / training_total,
        accuracy,
        best_level,
        best_threshold,
        best_accuracy,
        best_classifier)
    )

    # Save if better
    if best_accuracy > best_acc:
        best_acc = best_accuracy
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
