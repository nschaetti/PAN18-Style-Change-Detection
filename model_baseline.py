
import codecs
import os
from nltk.tokenize import word_tokenize
import pickle
import json
import numpy as np

count = dict()
n = 100
window_size = 1600

# Load voc
voc = pickle.load(open("voc.p", 'rb'))

# For each files
for file_name in os.listdir("./data/training"):
    if ".txt" in file_name:
        # Load text
        text = codecs.open(os.path.join("./data/training", file_name), mode='r', encoding='utf-8').read()

        # Load truth
        truth = json.load(open(os.path.join("./data/training", file_name[:-4] + ".truth")))

        # If changes or not
        if truth['changes']:
            # The parts
            parts = list()
        else:
            parts = list()
            text_lenght = len(text)
            # Enough length
            if text_lenght >= 1600 * 2:
                # For each pos
                for pos in np.arange(0, text_lenght - window_size):
                    parts.append()
                # end for
            # end if
        # end
    # end if
# end for

