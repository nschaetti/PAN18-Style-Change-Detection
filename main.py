#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : main.py
# Description : Main program for execution.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Neuchâtel, Suisse
#
# This file is part of the PAN18 author profiling challenge code.
# The PAN18 author profiling challenge code is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

# Imports
import torch
import dataset
from torch.autograd import Variable
from tools import settings, functions


################################################
# MAIN
################################################

# Parse arguments
args = functions.argument_parser_execution()

# CNN text transformer
transforms = functions.text_transformer_cnn(settings.cnn_window_size, args.n_gram)

# Counters
success = 0.0
total = 0.0

# Load models and voc
model, voc = functions.load_models(
    n_gram=args.n_gram,
    cuda=args.cuda
)

# Style Change Detection dataset
scd_dataset = dataset.TIRASCDDataset(root=args.input_dataset, transform=transforms)
pan18loader = torch.utils.data.DataLoader(scd_dataset, batch_size=1, shuffle=False)

# For the data set
for data in pan18loader:
    # Parts and c
    inputs = data

    # To variable
    inputs = Variable(inputs)
    if args.cuda:
        inputs = inputs.cuda()
    # end if

    # Prediction
    model_output = model(inputs)

    # Take the max as predicted
    _, predicted = torch.max(model_output.data, 1)

    # Write
    functions.save_result(args.output, scd_dataset.truth_files[-1], False if int(predicted[0]) == 0 else True)
# end for
