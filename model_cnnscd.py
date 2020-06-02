# -*- coding: utf-8 -*-
#

# Imports
import os
import copy
import torch.utils.data
from torch import nn
from torch import optim
from torchlanguage import models
from tools import settings
from torch.autograd import Variable
from tools import functions


# Parse arguments
args = functions.argument_parser_training_model()

# CNN text transformer
transforms = functions.text_transformer_cnn(settings.cnn_window_size, args.n_gram, token_to_ix=dict())

# Style change detection dataset, training set
pan18loader_train, pan18loader_valid = functions.load_dataset(transforms, args.batch_size, args.root)

# Loss function
loss_function = nn.CrossEntropyLoss()

# CNN Distance learning
model = models.CNNSCD(
    input_dim=settings.cnn_window_size,
    vocab_size=settings.voc_sizes[args.n_gram],
    embedding_dim=args.dim,
    out_channels=(args.n_filters, args.n_filters, args.n_filters),
    n_linear=args.n_linear,
    linear_size=args.linear_size,
    max_pool_size=args.max_pool_size,
    max_pool_stride=args.max_pool_stride,
    use_dropout=True
)
if args.cuda:
    model.cuda()
# end if
best_model = copy.deepcopy(model.state_dict())
best_acc = 10000

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

# Epoch
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0

    # Train
    model.train()
    n_changes = 0.0
    n_total = 0.0
    # For each training sample
    for i, data in enumerate(pan18loader_train):
        # Parts and c
        inputs, truth, _ = data
        n_changes += (truth == 1).sum()
        n_total += truth.size(0)
        # To variable
        """inputs, truth = Variable(inputs), Variable(truth)
        if args.cuda:
            inputs, truth = inputs.cuda(), truth.cuda()
        # end if

        # Set gradient to zero
        model.zero_grad()

        # Model output
        model_output = model(inputs)

        # Loss
        loss = loss_function(model_output, truth)

        # Add
        training_loss += loss.data[0]
        training_total += 1.0

        # Backward and step
        loss.backward()
        optimizer.step()"""
    # end for
    print(n_changes)
    print(n_total)
    # Eval
    model.eval()

    # Validation losses
    validation_loss = 0.0
    validation_total = 0.0
    success = 0.0
    total = 0.0
    n_changes = 0.0
    n_total = 0.0
    # For each validation sample
    for i, data in enumerate(pan18loader_valid):
        # Parts and c
        inputs, truth, _ = data
        n_changes += (truth == 1).sum()
        n_total += truth.size(0)
        # To variable
        """inputs, truth = Variable(inputs), Variable(truth)
        if args.cuda:
            inputs, truth = inputs.cuda(), truth.cuda()
        # end if

        # Prediction
        model_output = model(inputs)

        # Loss
        loss = loss_function(model_output, truth)

        # Take the max as predicted
        _, predicted = torch.max(model_output.data, 1)

        # Add to correctly classified text
        success += (predicted == truth.data).sum()

        # Counter
        total += truth.size(0)

        # Add
        validation_loss += loss.data[0]
        validation_total += 1.0"""
    # end for
    print(n_changes)
    print(n_total)
    exit()
    # Accuracy
    accuracy = success / total * 100.0

    # Show
    print(u"Epoch {}, training loss : {}, validation loss : {}, accuracy : {}".format(
        epoch,
        training_loss / training_total,
        validation_loss / validation_total,
        accuracy)
    )

    # Save if better
    if validation_loss / validation_total < best_acc:
        best_acc = validation_loss / validation_total
        print(u"Saving model with best accuracy {}".format(best_acc))
        torch.save(
            transforms.transforms[2].token_to_ix,
            open(
                os.path.join(args.output, "cnnscd25.voc.pth"),
                mode='wb'
            )
        )
        torch.save(
            model.state_dict(),
            open(
                os.path.join(args.output, "cnnscd25.pth"),
                mode='wb'
            )
        )
    # end if
# end epoch
