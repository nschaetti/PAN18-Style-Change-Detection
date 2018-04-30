# -*- coding: utf-8 -*-
#

# Imports
from torchlanguage import transforms as ltransforms
from torchvision import transforms
import argparse
import dataset
import torch
import settings
import os
import codecs

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
    parser.add_argument("--output", type=str, help="Model output file", default='.')
    parser.add_argument("--dim", type=int, help="Embedding dimension", default=300)
    parser.add_argument("--n-gram", type=str, help="N-Gram (c1, c2)", default='c1')
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
    parser.add_argument("--no-cuda", action='store_true', default=False, help="Enables CUDA training")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=1)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
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

#################
# Dataset
#################


# Import data set
def load_dataset(lang, text_transform, batch_size, val_batch_size):
    """
    Import tweets data set
    :param lang:
    :param text_transform:
    :param batch_size:
    :param val_batch_size:
    :return:
    """
    pass
# end load_dataset


################
# Models
################


# Load models
def load_models(model_file, voc_file, cuda=False):
    """
    Load models
    :param model_file:
    :param cuda:
    :return:
    """
    # Load tweet model
    model = torch.load(open(model_file, 'rb'))
    if cuda:
        model.cuda()
    else:
        model.cpu()
    # end if

    # Load model vocabulary
    voc = torch.load(open(voc_file, 'rb'))

    return model, voc
# end load_models

################
# Results
################


# Save results
def save_result(output, author_id, lang, gender_txt, gender_img, gender_both):
    """
    Save results
    :param output:
    :param author_id:
    :param gender_txt:
    :param gender_img:
    :param gender_both:
    :return:
    """
    # File output
    file_output = os.path.join(output, author_id + ".xml")

    # Log
    print(u"Writing result for {} to {}".format(author_id, file_output))

    # Open
    f = codecs.open(os.path.join(output, author_id + ".xml"), 'w', encoding='utf-8')

    # Write
    f.write(u"<author id=\"{}\" lang=\"{}\" gender_txt=\"{}\" gender_img=\"{}\" gender_comb=\"{}\"/>".
            format(
                    author_id,
                    lang,
                    settings.idx_to_class[int(gender_txt[0])],
                    settings.idx_to_class[int(gender_img[0])],
                    settings.idx_to_class[int(gender_both[0])]
            )
    )

    # Close
    f.close()
# end save_result
