# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:44:01 2024

@author: Dell-PC
"""
import pandas as pd
from input.features import split_dataset_by_relevant_columns
from input.utilities import  all_cols
from collections import Counter
import os
import models.evaluation.Metrics as m


#import largest training set until season 22-23
filenameTrain = '../../train/22-23.csv'
data = pd.read_csv(filenameTrain)

#split by relevant columns to get only granular matches
data = split_dataset_by_relevant_columns(data, all_cols)[0]

#calculate number of instances of each league
print(Counter(data['Lge']))


#calculate Metrics for an existing csv without rerunning
filenameTrain = '../../models/evaluation/LSTM_2D.csv'
data = pd.read_csv(filenameTrain, sep = ";")


m.MetricsCalculation.evaluateExperiment(data, [], 'LSTM 2D')



