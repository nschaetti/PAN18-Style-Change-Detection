
import codecs
import os
from nltk.tokenize import word_tokenize
import pickle
import json
import numpy as np
import spacy

nlp = spacy.load('en')
n_comb = 0

stats = np.array([])

# For each files
for file_name in os.listdir("./recalib/training"):
    if ".txt" in file_name:
        # Load text
        text = codecs.open(os.path.join("./recalib/training", file_name), mode='r', encoding='utf-8').read()

        # Load truth
        truth = json.load(open(os.path.join("./recalib/training", file_name[:-4] + ".truth")))

        # Parse
        doc = nlp(text)
        n_sentences = 0
        for sentence in doc.sents:
            if len(str(sentence)) > 10:
                stats = np.append(stats, [len(str(sentence))])
            # end if
        # end for

        """for n in range(1, n_sentences):
            n_comb += n
        # end for"""

        # If changes or not
        """if truth['changes']:
            # For each change
            for position in truth['positions']:
                # print(u"#" + text[position-2:position+2] + u"#")
                pass
            # end for
        # end"""
    # end if
# end for

print(stats.shape)
print(np.max(stats))
print(np.min(stats))
print(np.average(stats))
print(np.std(stats))