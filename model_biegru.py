# -*- coding: utf-8 -*-
#

# Imports
import torch.utils.data
import os
from tools import functions, settings
from torchlanguage import models
from torch.autograd import Variable
from torch import nn
import copy
from torch import optim

# Argument parser
args = functions.argument_parser_training_model()

# Get transforms
transforms = functions.text_transformer(args.n_gram, settings.gru_window_size)

# Style change detection dataset, training set
pan18loader_train, pan18loader_valid = functions.load_dataset(transforms, 1, args.root)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Bi-directional Embedding GRU
model = models.BiEGRU(window_size=settings.gru_window_size,
                      vocab_size=settings.voc_sizes[args.n_gram],
                      hidden_dim=settings.hidden_dim,
                      n_classes=2,
                      out_channels=(args.n_filters, args.n_filters, args.n_filters),
                      embedding_dim=args.dim)
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
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0
    test_loss = 0.0
    test_total = 0.0
    changes = 0.0

    # Get training data
    for i, data in enumerate(pan18loader_train):
        # Inputs and c
        inputs, label, _ = data

        # To variable
        inputs, label = Variable(inputs.view(-1, settings.gru_window_size)), Variable(label)
        if args.cuda:
            inputs, label = inputs.cuda(), label.cuda()
        # end if

        # Set gradient to zero
        model.zero_grad()

        # TRAINING
        model_output, hidden = model(inputs, hidden)

        # Loss
        loss = loss_function(model_output, label)

        # Add
        training_loss += loss.data[0]
        training_total += 1.0

        # Backward if last sample
        if i != 25577:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        # end if
        optimizer.step()
        print u"\r" + str(i),
    # end for
    print(u"\n")

    # Counters
    total = 0.0
    success = 0.0

    # Validation
    for i, data in enumerate(pan18loader_valid):
        # Inputs and c
        inputs, label, _ = data

        # To variable
        inputs, label = Variable(inputs.view(-1, settings.gru_window_size)), Variable(label)
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

    # Accuracy
    accuracy = 100.0 * success / total

    # Show
    print(u"Epoch {}, training loss {}, test loss {}, validation {}".format(
        epoch,
        training_loss,
        test_loss,
        accuracy
    ))

    # Save if better
    if accuracy > best_acc:
        best_acc = accuracy
        print(u"Saving model with best accuracy {}".format(best_acc))
        torch.save(
            transforms.transforms[2].token_to_ix,
            open(
                os.path.join(args.output, "biegru.voc.pth"),
                mode='wb'
            )
        )
        torch.save(
            model.state_dict(),
            open(
                os.path.join(args.output, "biegru.pth"),
                mode='wb'
            )
        )
    # end if
# end for
