# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:26:39 2024

@author: Dell-PC
"""

import models.evaluation.Metrics as m
import pandas as pd
import ELOXGBoostModel as xgboost
from input.utilities import eval_seasons, betting_odds_cols, market_value_cols, match_stats_cols, all_cols, features_rating_base, features_rating_market_value, features_rating_betting_odds, features_rating_match_stats, features_rating_all, features_test_base, features_test_market_value, features_test_betting_odds, features_test_match_stats, features_test_all

from input.features import (
    split_dataset_by_relevant_columns,
    add_metadata_col,
)

pd.options.mode.chained_assignment = None


# using XGBoost model for experiments, expects features to be used for ratings, features to be used for application to test set and features that are required for match to be included in the training data
def applyModelForExperiment(featuresRating, featuresTest, columnsRequiredTrain, imputationMethod='', handleMissing='carryForward'):

    # dataframe for final test set
    dataTestCombined = pd.DataFrame()

    # input train and test sets
    for season in eval_seasons:
        print("Evaluating season "+season)
        # read data of current season and flag train/test
        filenameTrain = '../../train/' + season+imputationMethod + '.csv'
        filenameTest = '../../test/' + season + '.csv'
        inputTrain = pd.read_csv(filenameTrain)
        inputTrain['type'] = 'train'
        inputTest = pd.read_csv(filenameTest)
        inputTest['type'] = 'test'
        data = pd.concat([inputTrain, inputTest], ignore_index=True)

        lengthBefore = len(data.index)
        # exclude columns not having certain columns (only in training set)
        data = pd.concat([split_dataset_by_relevant_columns(data[data['type'] == 'train'], columnsRequiredTrain)[
                         0], data[data['type'] == 'test']], ignore_index=True)
        lengthAfter = len(data.index)
        data[data['type'] == 'train'].to_csv(
            '../../input/training_data/dataTrain.csv')
        print(str(lengthAfter/lengthBefore*100) +
              " % of matches have been trained on")

        # readd relegation/promotion if dataset has been cut
        if (lengthAfter != lengthBefore):
            data["iso_date"] = pd.to_datetime(data["iso_date"])
            data = add_metadata_col(data)

        # run model and add current prediction set to full prediction set
        dataTestCombined = pd.concat([dataTestCombined, xgboost.ELOXGBoost(
            data, featuresRating, featuresTest, handleMissing=handleMissing)])
    dataTestCombined.to_csv('../../input/training_data/DataTestCombined.csv')

    # evaluate results of experiment
    m.MetricsCalculation.evaluateExperiment(dataTestCombined, [market_value_cols, betting_odds_cols, match_stats_cols,
                                            market_value_cols + betting_odds_cols + match_stats_cols], str(featuresRating)+' '+str(imputationMethod))




# experiment 1
applyModelForExperiment(features_rating_base, features_test_base, [])

# experiments 2_MV - 4_MV
applyModelForExperiment(features_rating_base + features_rating_market_value, features_test_base + features_test_market_value, market_value_cols)
applyModelForExperiment(features_rating_base + features_rating_market_value, features_test_base + features_test_market_value, market_value_cols, imputationMethod = '_simple')
applyModelForExperiment(features_rating_base + features_rating_market_value, features_test_base + features_test_market_value, market_value_cols, imputationMethod = '_knn')

# experiments 2_BO - 4_BO
applyModelForExperiment(features_rating_base + features_rating_betting_odds, features_test_base + features_test_betting_odds, betting_odds_cols)
applyModelForExperiment(features_rating_base + features_rating_betting_odds, features_test_base + features_test_betting_odds, betting_odds_cols, imputationMethod = '_simple')
applyModelForExperiment(features_rating_base + features_rating_betting_odds, features_test_base + features_test_betting_odds, betting_odds_cols, imputationMethod = '_knn')

# experiments 2_MS - 4_MS
applyModelForExperiment(features_rating_base + features_rating_match_stats, features_test_base + features_test_match_stats, match_stats_cols)
applyModelForExperiment(features_rating_base + features_rating_match_stats, features_test_base + features_test_match_stats, match_stats_cols, imputationMethod = '_simple')
applyModelForExperiment(features_rating_base + features_rating_match_stats, features_test_base + features_test_match_stats, match_stats_cols, imputationMethod = '_knn')

# experiments 2_AL - 4_AL
applyModelForExperiment(features_rating_all, features_test_all, all_cols)
applyModelForExperiment(features_rating_all, features_test_all, all_cols, imputationMethod = '_simple')
applyModelForExperiment(features_rating_all, features_test_all, all_cols, imputationMethod = '_knn')


# experiment all features including NANs with version current missing
applyModelForExperiment(features_rating_base + features_rating_market_value, features_test_base + features_test_market_value, [], handleMissing = 'currentMissing')
applyModelForExperiment(features_rating_base + features_rating_betting_odds, features_test_base + features_test_betting_odds, [], handleMissing = 'currentMissing')
applyModelForExperiment(features_rating_base + features_rating_match_stats, features_test_base + features_test_match_stats, [], handleMissing = 'currentMissing')
applyModelForExperiment(features_rating_all, features_test_all, [], handleMissing = 'currentMissing')

# experiment all features including NANs with version history missing
applyModelForExperiment(features_rating_base + features_rating_market_value, features_test_base + features_test_market_value, [], handleMissing = 'historyMissing')
applyModelForExperiment(features_rating_base + features_rating_betting_odds, features_test_base + features_test_betting_odds, [], handleMissing = 'historyMissing')
applyModelForExperiment(features_rating_base + features_rating_match_stats, features_test_base + features_test_match_stats, [], handleMissing='historyMissing')
applyModelForExperiment(features_rating_all, features_test_all, [], handleMissing='historyMissing')
