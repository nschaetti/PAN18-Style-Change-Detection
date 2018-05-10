# -*- coding: utf-8 -*-
#

# Imports
from torchlanguage import transforms as ltransforms
from torchvision import transforms


################
# Settings
################

# Settings
cnn_window_size = 12000
gru_window_size = 20
hidden_dim = 250
voc_sizes = {'c1': 1628, 'c2': 21510}
class_to_idx = {False: 0, True: 1}
idx_to_class = {0: False, 1: True}
