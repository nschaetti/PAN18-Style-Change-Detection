# -*- coding: utf-8 -*-
#

# Imports
import torch.utils.data
import dataset
from echotorch.transforms import text
import numpy as np

# Experience parameter
batch_size = 64
n_epoch = 1
window_size = 700
training_set_size = 10
test_set_size = 2
training_samples = training_set_size + test_set_size
stride = 100

# Style change detection dataset, training set
pan18loader_train = torch.utils.data.DataLoader(
    dataset.SCDPartsDataset(root='./data/', download=True, transform=text.Character2Gram(), train=True),
    batch_size=1
)

# Style change detection dataset, validation set
pan18loader_valid = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./data/', download=True, transform=text.Character2Gram(), train=False),
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
    batch = torch.zeros(batch_size, window_size*2)
    batch_truth = torch.zeros(batch_size)

    # Batch size
    for b in range(batch_size):
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
            start_pos1 = np.random.randint(0, part_length - window_size)
            start_pos2 = np.random.randint(0, part_length - window_size)

            # Get the two sides
            side1 = random_part[:, start_pos1:start_pos1 + window_size]
            side2 = random_part[:, start_pos2:start_pos2 + window_size]

            # Truth
            batch_truth[b] = 0
        else:
            # Pick to consecutive random parts
            random_start_part = np.random.randint(0, n_parts-1)
            random_part1 = parts[random_start_part]
            random_part2 = parts[random_start_part+1]

            # Length
            part1_length = random_part1.size(1)
            part2_length = random_part2.size(1)

            # Pick two random starting positions
            start_pos1 = np.random.randint(0, part1_length - window_size)
            start_pos2 = np.random.randint(0, part2_length - window_size)

            # Get the two sides
            side1 = random_part1[:, start_pos1:start_pos1 + window_size]
            side2 = random_part2[:, start_pos2:start_pos2 + window_size]

            # Truth
            batch_truth[b] = 1
        # end if

        # Concatenate
        sides = torch.cat((side1, side2), dim=1)

        # Set
        batch[b] = sides
    # end for

    # Add to batch
    batches.append((batch, batch_truth))
# end for

# Training and test sets
training_set = batches[:training_set_size]
test_set = batches[training_set_size:]

# For each iteration
for epoch in range(n_epoch):
    # For each training examples
    for batch in training_set:
        # Inputs and label
        inputs, label = batch

        # TRAINING
    # end for

    # For each validation sample
    for batch in test_set:
        # Inputs and label
        inputs, label = batch

        # VALIDATE
    # end for

    # Counters
    successes = 0
    total = 0

    # For each test sample
    for i, data in enumerate(pan18loader_valid):
        # Parts and c
        inputs, label = data

        # Different parts to measure
        parts = list()

        # Get each parts
        for j in torch.arange(0, inputs.size(1)-window_size, stride):
            # Add
            parts.append(inputs[:, j:j+window_size])
        # end for

        # Number of parts
        n_parts = len(parts)

        # Similarity matrix
        similarity_matrix = np.zeros((n_parts, n_parts))

        # For all combination
        for n in range(n_parts):
            for m in range(n_parts):
                # Inputs
                concate = torch.cat((parts[n], parts[m]), dim=1)

                # EXEC

                # Set
                similarity_matrix[n, m] = 0
            # end for
        # end for

        # Maximum distance
        max_distance = np.max(similarity_matrix)

        # Predicted class
        threshold = 0.9
        if max_distance > threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        # end if

        # Test
        if label == predicted_class:
            successes += 1.0
        # end if
        total += 1.0
    # end for

    # Show
    print(u"Epoch {}, accuracy : {}".format(epoch, 100.0 * successes / total))
# end for
