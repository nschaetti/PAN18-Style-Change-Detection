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

# Argument parser
args = functions.argument_parser_training_model()

# Get transforms
transforms = functions.text_transformer_cnn(args.n_gram)

# Style change detection dataset, training set
pan18loader_train = torch.utils.data.DataLoader(
    dataset.SCDPartsDataset(root='./data/', download=True, transform=transforms, train=True),
    batch_size=1
)

# Style change detection dataset, validation set
pan18loader_valid = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./data/', download=True, transform=transforms, train=False),
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
same = True
for i in range(settings.training_samples):
    # Batch
    batch = torch.LongTensor(args.batch_size, settings.cnn_window_size * 2).fill_(0)
    batch_truth = torch.LongTensor(args.batch_size).fill_(0)

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
            batch_truth[b] = 0

            # Concatenate
            sides = torch.cat((side1, side2), dim=1)

            # Set
            batch[b] = sides
            
            # Not same next time
            same = False
            b += 1
        elif n_parts > 1 and not same:
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
            batch_truth[b] = 1

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

# Training and validation
total_size = len(batches)
validation_size = int(total_size * settings.validation_ratio)
training_size = total_size - validation_size
training_set = batches[:training_size]
validation_set = batches[training_size:]

# Loss function
loss_function = nn.CrossEntropyLoss()

# CNN Distance learning
# model = models.CNNCDist(window_size=settings.cnn_window_size, vocab_size=settings.voc_sizes[args.n_gram], n_classes=2)
model = models.CNNCDist(
    window_size=settings.cnn_window_size,
    vocab_size=settings.voc_sizes[args.n_gram],
    n_classes=2,
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

# For each iteration
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0

    # For each training examples
    for i, batch in enumerate(training_set):
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
    validation_loss = 0.0
    validation_total = 0.0
    success = 0.0
    total = 0.0
    random = 0.0

    # For each validation examples
    for i, batch in enumerate(validation_set):
        # Inputs and label
        inputs, truth = batch

        # To variable
        inputs, truth = Variable(inputs), Variable(truth)
        if args.cuda:
            inputs, truth = inputs.cuda(), truth.cuda()
        # end if

        # TRAINING
        model_output = model(inputs).squeeze(dim=1)

        # Loss
        loss = loss_function(model_output, truth)

        # Take the max as predicted
        _, predicted = torch.max(model_output.data, 1)

        # Add to correctly classified text
        random += truth.data.sum()
        success += (predicted == truth.data).sum()

        # Counter
        total += truth.size(0)

        # Add
        validation_loss += loss.data[0]
        validation_total += 1.0
    # end for

    # Accuracy
    accuracy = success / total * 100.0

    # Show
    print(u"Epoch {}, training loss : {}, validation loss : {}, accuracy : {} ({})".format(epoch, training_loss / training_total, validation_loss / validation_total, accuracy, random / total * 100.0))

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
