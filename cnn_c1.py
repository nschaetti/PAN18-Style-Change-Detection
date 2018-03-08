# -*- coding: utf-8 -*-
#

# Imports
import torch.utils.data
import dataset
from echotorch.transforms import text

# Style change detection dataset
pan18loader = torch.utils.data.DataLoader(
    dataset.SCDPartsDataset(root='./data/', download=True, transform=text.GloveVector(), train=False)
)

# Get training data
for i, data in enumerate(pan18loader):
    # Parts and c
    parts, c = data

    # Print
    print(u"Parts : {}".format(parts.size()))
    print(u"Class : {}".format(c))
# end for
