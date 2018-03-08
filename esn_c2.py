# -*- coding: utf-8 -*-
#

# Imports
import torch.utils.data
import dataset
from echotorch.transforms import text

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
    dataset.SCDSimpleDataset(root='./data/', download=True, transform=text.Character2Gram(), train=True),
    batch_size=1
)

# Style change detection dataset, validation set
pan18loader_valid = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./data/', download=True, transform=text.Character2Gram(), train=False),
    batch_size=1
)

# Get training data
for i, data in enumerate(pan18loader_train):
    # Inputs and c
    inputs, label = data

    # TRAINING
# end for

# Test loss
successes = 0.0
total = 0.0

# Validation
for i, data in enumerate(pan18loader_valid):
    # Inputs and c
    inputs, label = data

    # EXEC

    # Counter
    total += 1.0
# end for

# Show
print(u"Validation {}".format(
    100.0 * successes / total
))
