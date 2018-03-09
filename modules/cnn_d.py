# -*- coding: utf-8 -*-
#

import torch
import torch.nn as nn
import torch.nn.functional as F


# A character-based CNN distance
class CNND(nn.Module):
    """
    A character-based CNN
    """

    # Constructor
    def __init__(self, vocab_size, window_size=500, embedding_dim=300, out_channels=(100, 100, 100), kernel_sizes=(3, 4, 5), linear_size=100):
        """
        Constructor
        :param vocab_size:
        :param embedding_dim:
        :param out_channels:
        :param kernel_sizes:
        :param max_pools:
        """
        super(CNND, self).__init__()
        # Properties
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        # Embedding layer
        if embedding_dim > 0:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # end if

        # Conv window 1
        self.conv_w1 = nn.Conv2d(in_channels=1, out_channels=out_channels[0],
                                 kernel_size=(kernel_sizes[0], embedding_dim))

        # Conv window 2
        self.conv_w2 = nn.Conv2d(in_channels=1, out_channels=out_channels[1],
                                 kernel_size=(kernel_sizes[1], embedding_dim))

        # Conv window 3
        self.conv_w3 = nn.Conv2d(in_channels=1, out_channels=out_channels[2],
                                 kernel_size=(kernel_sizes[2], embedding_dim))

        # Max pooling layer
        self.max_pool_w1 = nn.MaxPool1d(kernel_size=window_size - out_channels[0] + 1, stride=0)
        self.max_pool_w2 = nn.MaxPool1d(kernel_size=window_size - out_channels[1] + 1, stride=0)
        self.max_pool_w3 = nn.MaxPool1d(kernel_size=window_size - out_channels[2] + 1, stride=0)

        # Linear layer
        self.linear_size = out_channels[0] + out_channels[1] + out_channels[2]
        self.linear = nn.Linear(self.linear_size, linear_size)
    # end __init__

    # Forward
    def forward(self, x):
        """
        Forward
        :param x:
        :return:
        """
        # Separation
        x1, x2 = x[:, :self.window_size], x[:, :self.window_size:]

        # Author embeddings
        a1 = self.author_embedding(x1)
        a2 = self.author_embedding(x2)

        # Distance
        return torch.sqrt(torch.sum(torch.pow(a1 - a2, 2)))
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

    # Author embedding
    def author_embedding(self, x):
        """
        Author embedding
        :param x:
        :return:
        """
        # Embeddings
        if self.embedding_dim > 0:
            embeds = self.embeddings(x)

            # Add channel dim
            embeds = torch.unsqueeze(embeds, dim=1)
        else:
            # Add channel dim
            embeds = torch.unsqueeze(x, dim=1)
        # end if

        # Conv window
        out_win1 = F.relu(self.conv_w1(embeds))
        out_win2 = F.relu(self.conv_w2(embeds))
        out_win3 = F.relu(self.conv_w3(embeds))

        # Remove last dim
        out_win1 = torch.squeeze(out_win1, dim=3)
        out_win2 = torch.squeeze(out_win2, dim=3)
        out_win3 = torch.squeeze(out_win3, dim=3)

        # Max pooling
        max_win1 = self.max_pool_w1(out_win1)
        max_win2 = self.max_pool_w2(out_win2)
        max_win3 = self.max_pool_w3(out_win3)

        # Concatenate
        out = torch.cat((max_win1, max_win2, max_win3), dim=1)

        # Flatten
        out = out.view(-1, self.linear_size)

        # Linear
        return self.linear(out)
    # end author_embedding

# end CNNC
