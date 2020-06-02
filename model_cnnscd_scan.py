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
import numpy as np
from torchlanguage import transforms as ltransforms

# Settings
stride = 100
window_size = 3000
security_border = 200

# Parse arguments
args = functions.argument_parser_training_model()

if args.n_gram == 'c1':
    transforms = ltransforms.Compose([
        ltransforms.ToLower(),
        ltransforms.Character(),
        ltransforms.ToIndex(start_ix=1),
        ltransforms.ToLength(length=window_size),
        ltransforms.Reshape((-1)),
        ltransforms.MaxIndex(max_id=settings.voc_sizes[args.n_gram])
    ])
else:
    transforms = ltransforms.Compose([
        ltransforms.ToLower(),
        ltransforms.Character2Gram(),
        ltransforms.ToIndex(start_ix=1),
        ltransforms.ToLength(length=window_size),
        ltransforms.Reshape((-1)),
        ltransforms.MaxIndex(max_id=settings.voc_sizes[args.n_gram])
    ])
# end if

# Style change detection dataset, training set
pan18loader_train, pan18loader_valid = functions.load_dataset(transforms, 1, args.root)

# Loss function
loss_function = nn.CrossEntropyLoss()

# CNN Distance learning
model = models.CNNSCD(
    input_dim=window_size,
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
best_acc = 0.0

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

# Epoch
for epoch in range(args.epoch):
    # Total losses
    training_loss = 0.0
    training_total = 0.0

    # For each training sample
    for i, data in enumerate(pan18loader_train):
        # Parts and c
        inputs, truth, positions = data
        
        # Each possible slide start
        slide_starts = np.arange(0, inputs.size(1) - window_size, stride)
        np.random.shuffle(slide_starts)
        
        # Empty pos
        if slide_starts.shape[0] == 0:
            slide_starts = np.array([0])
        # end if
        
        # For each slide
        for j in slide_starts:
            # To int
            j = int(j)

            # Slide
            slide = inputs[:, j:j+window_size]

            # Truth
            if len(positions) == 0:
                slide_t = 0
            else:
                for k in range(positions[0].size(0)):
                    pos = int(positions[k])
                    if pos > j + security_border and pos < j+window_size-security_border:
                        slide_t = 1
                    # end if
                # end for
            # end if

            # To variable
            slide, slide_truth = Variable(slide), Variable(torch.LongTensor(1).fill_(slide_t))
            if args.cuda:
                slide, slide_truth = slide.cuda(), slide_truth.cuda()
            # end if

            # Set gradient to zero
            model.zero_grad()

            # Model output
            model_output = model(slide)

            # Loss
            loss = loss_function(model_output, slide_truth)

            # Add
            training_loss += loss.data[0]
            training_total += 1.0

            # Backward and step
            loss.backward()
            optimizer.step()
        # end for
        print u"\r" + str(i),
    # end for
    print(u"Training total : {}".format(training_total))

    # Validation losses
    validation_loss = 0.0
    validation_total = 0.0
    success = 0.0
    total = 0.0

    # For each validation sample
    for i, data in enumerate(pan18loader_valid):
        # Parts and c
        inputs, truth, positions = data

        # Each possible slide start
        slide_starts = np.arange(0, inputs.size(1) - window_size, stride)
        np.random.shuffle(slide_starts)

        # Empty pos
        if slide_starts.shape[0] == 0:
            slide_starts = np.array([0])
        # end if

        # For each slide
        for j in slide_starts:
            # Slide
            slide = inputs[:, j:j + window_size]

            # Truth
            if len(positions) == 0:
                slide_t = 0
            else:
                for k in range(positions[0].size(0)):
                    pos = int(positions[k])
                    if pos > j + security_border and pos < j + window_size - security_border:
                        slide_t = 1
                    # end if
                # end for
            # end if

            # To variable
            slide, slide_truth = Variable(slide), Variable(torch.LongTensor(1).fill_(slide_t))
            if args.cuda:
                slide, slide_truth = slide.cuda(), slide_truth.cuda()
            # end if

            # Prediction
            model_output = model(slide)

            # Loss
            loss = loss_function(model_output, slide_truth)

            # Take the max as predicted
            _, predicted = torch.max(model_output.data, 1)

            # Add to correctly classified text
            success += (predicted == slide_truth.data).sum()

            # Counter
            total += 1.0

            # Add
            validation_loss += loss.data[0]
            validation_total += 1.0
        # end for
        print u"\r" + str(i),
    # end for
    print(u"Validation total : {}".format(validation_total))

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
    if accuracy > best_acc:
        best_acc = accuracy
        print(u"Saving model with best accuracy {}".format(best_acc))
        torch.save(
            transforms.transforms[2].token_to_ix,
            open(
                os.path.join(args.output, "cnnscd_slide.voc.pth"),
                mode='wb'
            )
        )
        torch.save(
            model.state_dict(),
            open(
                os.path.join(args.output, "cnnscd_slide.pth"),
                mode='wb'
            )
        )
    # end if
# end epoch
