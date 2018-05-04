# -*- coding: utf-8 -*-
#

# Imports
from torchlanguage import transforms as ltransforms
from torchvision import transforms
from torchlanguage import models
import dataset
import argparse
import torch
import settings
import os
import codecs
import json

#################
# Arguments
#################


# Tweet argument parser for training model
def argument_parser_training_model():
    """
    Tweet argument parser
    :return:
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="PAN18 Author Profiling challenge")

    # Argument
    parser.add_argument("--root", type=str, help="Dataset root", default='./root/')
    parser.add_argument("--output", type=str, help="Model output file", default='.')
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=50)
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
    return args
# end argument_parser_training_model


# Execution argument parser
def argument_parser_execution():
    """
    Execution argument parser
    :return:
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="PAN18 Author Profiling main program")

    # Argument
    parser.add_argument("--input-dataset", type=str, help="Input dataset", required=True)
    parser.add_argument("--output-dir", type=str, help="Where to put results", required=True)
    parser.add_argument("--input-run", type=str, help="Input run", required=True)
    parser.add_argument("--model", type=str, help="Image model", required=True)
    parser.add_argument("--n-gram", type=str, help="N-Gram (c1, c2)", default='c1')
    args = parser.parse_args()
    args.cuda = False
    return args
# end argument_parser_execution

#################
# Transformers
#################


# Get text transformer
def text_transformer(n_gram, window_size):
    """
    Get tweet transformer
    :param lang:
    :param n_gram:
    :return:
    """
    if n_gram == 'c1':
        return transforms.Compose([
            ltransforms.ToLower(),
            ltransforms.Character(),
            ltransforms.ToIndex(start_ix=0),
            ltransforms.ToNGram(n=window_size, overlapse=True),
            ltransforms.Reshape((-1, window_size)),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram] - 1)
        ])
    else:
        return transforms.Compose([
            ltransforms.ToLower(),
            ltransforms.Character2Gram(),
            ltransforms.ToIndex(start_ix=0),
            ltransforms.ToNGram(n=window_size, overlapse=True),
            ltransforms.Reshape((-1, window_size)),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram] - 1)
        ])
    # end if
# end tweet_transformer


# Get text transformer
def text_transformer_cnn(window_size, n_gram, token_to_ix):
    """
    Get text transformer for CNNSCD
    :param window_size:
    :param n_gram:
    :return:
    """
    if n_gram == 'c1':
        return ltransforms.Compose([
            ltransforms.ToLower(),
            ltransforms.Character(),
            ltransforms.ToIndex(start_ix=1, token_to_ix=token_to_ix),
            ltransforms.ToLength(length=window_size),
            ltransforms.Reshape((-1)),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram])
        ])
    else:
        return ltransforms.Compose([
            ltransforms.ToLower(),
            ltransforms.Character2Gram(),
            ltransforms.ToIndex(start_ix=1, token_to_ix=token_to_ix),
            ltransforms.ToLength(length=window_size),
            ltransforms.Reshape((-1)),
            ltransforms.MaxIndex(max_id=settings.voc_sizes[n_gram])
        ])
    # end if
# end text_transformer_cnn


#################
# Dataset
#################


# Import data set
def load_dataset(transforms, batch_size, root='./data/'):
    """
    Import data set
    :param transforms:
    :param batch_size:
    :param root
    :return:
    """
    # Style change detection dataset, training set
    pan18loader_train = torch.utils.data.DataLoader(
        dataset.SCDSimpleDataset(root=root, download=True, transform=transforms, train=True),
        batch_size=batch_size
    )

    # Style change detection dataset, validation set
    pan18loader_valid = torch.utils.data.DataLoader(
        dataset.SCDSimpleDataset(root=root, download=True, transform=transforms, train=False),
        batch_size=batch_size
    )
    return pan18loader_train, pan18loader_valid
# end load_dataset


################
# Models
################


# Load models
def load_models(model_type, n_gram, cuda=False):
    """
    Load models
    :param image_model:
    :param cuda:
    :return:
    """
    # Map location
    if not cuda:
        map_location = 'cpu'
    else:
        map_location = None
    # end if

    # Load tweet model
    model, voc = models.cnnscd25(n_gram=n_gram, map_location=map_location)
    if cuda:
        model.cuda()
    else:
        model.cpu()
    # end if

    return model, voc
# end load_models

################
# Results
################


# Save results
def save_result(output, problem_file, predicted):
    """
    Save results
    :param output:
    :param problem_file:
    :param predicted:
    :return:
    """
    # File output
    file_output = os.path.join(output, problem_file)

    # Log
    print(u"Writing result {} for {} to {}".format(predicted, problem_file, file_output))

    # JSON data
    predicted_output = {"changes": predicted}

    # Open
    f = codecs.open(file_output, 'w', encoding='utf-8')

    # Write
    json.dump(predicted_output, f)

    # Close
    f.close()
# end save_result
