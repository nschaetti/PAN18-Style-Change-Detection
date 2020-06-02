# -*- coding: utf-8 -*-
#

# Imports
import torch.utils.data
import dataset
from tools import functions, settings
from torchlanguage import models
from torch.autograd import Variable
from torch import nn
import copy
from torch import optim

# Experience parameter
n_epoch = 1
window_size = 700
training_set_size = 10
test_set_size = 2
training_samples = training_set_size + test_set_size
stride = 100


# Argument parser
args = functions.argument_parser_training_model()

# Get transforms
transforms = functions.text_transformer(args.n_gram, settings.gru_window_size)

# Style change detection dataset, training set
pan18loader_train = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./recalib/', download=True, transform=transforms, train=True),
    batch_size=1
)

# Style change detection dataset, validation set
pan18loader_valid = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./recalib/', download=True, transform=transforms, train=False),
    batch_size=1
)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Bi-directional Embedding GRU
model = models.BiEGRU(window_size=settings.window_size,
                      vocab_size=settings.voc_sizes[args.n_gram],
                      hidden_dim=settings.hidden_dim,
                      n_classes=2,
                      out_channels=(25, 25, 25),
                      embedding_dim=50)
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Init hidden
hidden = model.init_hidden(1)
if args.cuda:
    hidden = hidden.cuda()
# end if

# For each epoch
for epoch in range(n_epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0
    test_loss = 0.0
    test_total = 0.0
    changes = 0.0
    loss = 0.0

    # Get training data
    batch = 0
    for i, data in enumerate(pan18loader_train):
        # Inputs and c
        inputs, label = data

        # To variable
        inputs, label = Variable(inputs.view(-1, settings.window_size)), Variable(label)
        if args.cuda:
            inputs, label = inputs.cuda(), label.cuda()
        # end if

        # Set gradient to zero
        model.zero_grad()

        # TRAINING
        model_output, hidden = model(inputs, hidden)

        # Loss
        sample_loss = loss_function(model_output, label)

        # Add
        if batch == 0:
            loss = sample_loss
        else:
            loss += sample_loss
        # end if
        training_loss += sample_loss.data[0]
        training_total += 1.0
        
        # Backward if last sample
        if batch == args.batch_size - 1:
            # Backward and step
            loss.backward()
            optimizer.step()
        # end if

        # Batch
        batch += 1
        if batch >= args.batch_size:
            batch = 0
        # end if
    # end for

    # Counters
    total = 0.0
    success = 0.0

    # Validation
    for i, data in enumerate(pan18loader_valid):
        # Inputs and c
        inputs, label = data

        # To variable
        inputs, label = Variable(inputs.view(-1, settings.window_size)), Variable(label)
        if args.cuda:
            inputs, label = inputs.cuda(), label.cuda()
        # end if

        # Training
        model_output, hidden = model(inputs, hidden)

        # Loss
        loss = loss_function(model_output, label)

        # Take the max as predicted
        _, predicted = torch.max(model_output.data, 1)

        # Add to correctly classified word
        if predicted == int(label.data[0]):
            success += 1.0
        # end if

        # Counter
        total += 1.0
    # end for

    # Show
    print(u"Epoch {}, training loss {}, test loss {}, validation {}".format(
        epoch,
        training_loss,
        test_loss,
        100.0 * success / total
    ))
# end for
