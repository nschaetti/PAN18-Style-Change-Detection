# -*- coding: utf-8 -*-
#

# Imports
import copy
import torch.utils.data
from torch import nn
from torch import optim
import dataset
from torchlanguage import models
from torchlanguage import transforms as ltransforms
from tools import settings
from torch.autograd import Variable
import argparse


# Settings
window_size = 11400

# Argument parser
parser = argparse.ArgumentParser(description="PAN18 Author Profiling challenge")

# Argument
parser.add_argument("--output", type=str, help="Model output file", default='.')
parser.add_argument("--dim", type=int, help="Embedding dimension", default=300)
parser.add_argument("--n-gram", type=str, help="N-Gram (c1, c2)", default='c1')
parser.add_argument("--n-filters", type=int, help="Number of filters", default=500)
parser.add_argument("--n-linear", type=int, help="Number of linear layer", default=2)
parser.add_argument("--linear-size", type=int, help="Linear layer size", default=1500)
parser.add_argument("--max-pool-size", type=int, help="Max pooling size", default=700)
parser.add_argument("--max-pool-stride", type=int, help="Max pooling stride", default=350)
parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
parser.add_argument("--epoch", type=int, help="Epoch", default=300)
parser.add_argument("--batch-size", type=int, help="Batch size", default=20)
args = parser.parse_args()

# Use CUDA?
args.cuda = not args.no_cuda and torch.cuda.is_available()

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
pan18loader_train = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./extended2/', download=True, transform=transforms, train=True),
    batch_size=args.batch_size
)

# Style change detection dataset, validation set
pan18loader_valid = torch.utils.data.DataLoader(
    dataset.SCDSimpleDataset(root='./extended2/', download=True, transform=transforms, train=False),
    batch_size=args.batch_size
)

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
    max_pool_stride=args.max_pool_stride
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
        inputs, truth, _ = data

        # To variable
        inputs, truth = Variable(inputs), Variable(truth)
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
        optimizer.step()
    # end for

    # Validation losses
    validation_loss = 0.0
    validation_total = 0.0
    success = 0.0
    total = 0.0

    # For each validation sample
    for i, data in enumerate(pan18loader_valid):
        # Parts and c
        inputs, truth, _ = data

        # To variable
        inputs, truth = Variable(inputs), Variable(truth)
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
        validation_total += 1.0
    # end for

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
            (transforms.transforms[2].token_to_ix, model.state_dict()),
            open(args.output, 'wb')
        )
    # end if
# end epoch
