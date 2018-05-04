# -*- coding: utf-8 -*-
#

# Imports
from torch.utils.data.dataset import Dataset
import os
import codecs


# TIRA Style change detection dataset
class TIRASCDDataset(Dataset):
    """
    TIRA Style change detection dataset
    """

    # Constructor
    def __init__(self, root='./data', transform=None):
        """
        Constructor
        :param root:
        :param transform:
        """
        # Properties
        self.root = root
        self.transform = transform
        self.samples = list()
        self.changes_char = dict()
        self.truth_files = list()

        # Generate data set
        self._load()
    # end __init__

    ##########################################
    # OVERRIDE
    ##########################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.samples)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Get sample
        text_file = os.path.join(self.root, self.samples[idx])

        # Truth file
        truth_file = self.samples[idx][:-4] + ".truth"
        self.truth_files.append(truth_file)

        # Read text
        sample_text = codecs.open(text_file, 'r', encoding='utf-8').read()

        # Transform
        transformed = self.transform(sample_text)

        return transformed
    # end __getitem__

    ##########################################
    # PRIVATE
    ##########################################

    # Load dataset
    def _load(self):
        """
        Load the dataset
        :return:
        """
        # For each file
        for file_name in os.listdir(self.root):
            # Text file
            if file_name[-4:] == ".txt":
                # Add to samples
                self.samples.append(file_name)
            # end if
        # end for
    # end _load

# end SCDSimpleDataset
