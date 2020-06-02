
import codecs
import os
from nltk.tokenize import word_tokenize
import pickle

count = dict()
n = 100

# For each files
for file_name in os.listdir("./data/training"):
    if ".txt" in file_name:
        text = codecs.open(os.path.join("./data/training", file_name), mode='r', encoding='utf-8').read()
        words = word_tokenize(text)
        for word in words:
            try:
                count[word] += 1.0
            except KeyError:
                count[word] = 1.0
            # end try
        # end for
    # end if
# end for

# To tuple list
sorted_words = list()
for word in count.keys():
    sorted_words.append((word, count[word]))
# end for

# Sort
sorted_words.sort(key=lambda tup: tup[1], reverse=True)
sorted_words = sorted_words[:n]

# Word vector
final_words = list()
for word, count in sorted_words:
    final_words.append(word)
# end for
print(final_words)

# Save
pickle.dump(final_words, open("voc.p", 'wb'))
