# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:36:28 2023

@author: Dell-PC
"""

import models.rating_models.AdjustedResultModels as arm
import models.rating_models.RatingModels as rm
import models.probability_models.ProbabilityModels as pm
import models.probability_models.GoalModels as gm
#import models.evaluation.Metrics as m
#import models.imputation.models as imp
import pandas as pd
#from input.utilities import eval_seasons, betting_odds_cols, market_value_cols, match_stats_cols, betting_odds_cols_rm, market_value_cols_rm, match_stats_cols_rm
#from input.features import (
#    split_dataset_by_relevant_columns,
#    add_metadata_col,
#)

pd.options.mode.chained_assignment = None

 
#expects data as an input. Moreover, expects features to be translated to rankings and ones to be used for testing
def ELOXGBoost(data, featuresRating, featuresTest, adjustInPredictionSet = False, handleMissing = 'carryForward'):
    
    #calculate and add ELORatings for different metrics
    if 'piRating' in featuresRating:
        data = rm.piRating().calculateRating(data, 0.7, 0.035, adjustInPredictionSet)
    if 'results' in featuresRating:
        data = rm.ELORating(method = "results").calculateRating(data, 10, 400, 20, 0, 0, adjustInPredictionSet, handleMissing)
    if 'goals' in featuresRating:
        data = rm.ELORating(method = "goals").calculateRating(data, 10, 400, 0, 10, 1.25, adjustInPredictionSet, handleMissing)
    if 'odds' in featuresRating:
        data = arm.AdjustedResultModels().addAdjustedResults(data, {'odds': ['odds', ' ', ' ']})
        data = rm.ELORating(method = "odds").calculateRating(data, 10, 400, 125, 0, 0, adjustInPredictionSet, handleMissing)
    if 'marketvalues' in featuresRating:
        data = arm.AdjustedResultModels().addAdjustedResults(data, {'marketvalue': ['MV', 'home_starter_total', 'away_starter_total']})
        data = rm.ELORating(method = "MV").calculateRating(data, 10, 400, 60, 0, 0, adjustInPredictionSet, handleMissing)
    if 'shots' in featuresRating: 
        data = arm.AdjustedResultModels().addAdjustedResults(data, {'shots': ['S', 'HTS', 'ATS']})
        data = rm.ELORating(method = "S").calculateRating(data, 10, 400, 25, 0, 0, adjustInPredictionSet, handleMissing)
    if 'shotsTarget' in featuresRating: 
        data = arm.AdjustedResultModels().addAdjustedResults(data, {'shotsTarget': ['ST', 'HST', 'AST']})
        data = rm.ELORating(method = "ST").calculateRating(data, 10, 400, 30, 0, 0, adjustInPredictionSet, handleMissing)

    data.to_csv('../../input/training_data/Ratings.csv')
    
    data['avg_home_odds'] = pd.to_numeric(data['avg_home_odds'], errors='coerce')
    data['avg_away_odds'] = pd.to_numeric(data['avg_away_odds'], errors='coerce')
    data['avg_draw_odds'] = pd.to_numeric(data['avg_draw_odds'], errors='coerce')
    
    data = XGBoost(data, featuresTest)
    
    return data[data['type']=='test']


    #calculate predictions based on XGBoost using the specified columns in the dataset
def XGBoost(data, columns):

    data = pm.XGBoost().calculateProbabilities(data, columns)
    data = gm.XGBoost().calculateGoals(data, columns)    
    
    dataTest=data[data['type']=='test']
    dataTest.to_csv('../../input/training_data/dataTest.csv')
    
    return data





