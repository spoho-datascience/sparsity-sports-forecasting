# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:36:28 2023

@author: Dell-PC
"""

import models.evaluation.Metrics as m
import pandas as pd
import numpy as np
import models.rating_models.RatingModels as rm
import models.probability_models.ProbabilityModels as pm
import models.probability_models.GoalModels as gm
from input.utilities import eval_seasons, betting_odds_cols, market_value_cols, match_stats_cols

pd.options.mode.chained_assignment = None


#running a baseline model that uses uniform probabilities and goal numbers
def UNI():

    #dataframe for final test set
    dataTestCombined = pd.DataFrame()
    
    #input train and test sets
    for season in eval_seasons:
        print("Evaluating season "+season)
        #read data of current season and flag train/test
        filenameTrain = '../../train/' + season + '.csv'
        filenameTest = '../../test/' + season + '.csv'
        dataTrain = pd.read_csv(filenameTrain)
        dataTrain['type'] = 'train'
        dataTest = pd.read_csv(filenameTest)
        dataTest['type'] = 'test'
        
        #set uniform numbers
        dataTest['prd_W'] = 1/3
        dataTest['prd_D'] = 1/3
        dataTest['prd_L'] = 1/3
        avgGoals = np.mean(dataTrain['HS'] + dataTrain['AS'])
        dataTest['prd_HS'] = avgGoals/2
        dataTest['prd_AS'] = avgGoals/2
        
        #run model and add current prediction set to full prediction set
        dataTestCombined = pd.concat([dataTestCombined, dataTest])
    dataTestCombined.to_csv('../../input/training_data/DataTestCombined.csv')
    
    #evaluate results of experiment
    m.MetricsCalculation.evaluateExperiment(dataTestCombined, [market_value_cols, betting_odds_cols, match_stats_cols, market_value_cols + betting_odds_cols + match_stats_cols], 'BASELINE UNI')


#running a baseline model that uses frequency probabilities and average home/away goal numbers
def FRQ():

    #dataframe for final test set
    dataTestCombined = pd.DataFrame()
    
    #input train and test sets
    for season in eval_seasons:
        print("Evaluating season "+season)
        #read data of current season and flag train/test
        filenameTrain = '../../train/' + season + '.csv'
        filenameTest = '../../test/' + season + '.csv'
        dataTrain = pd.read_csv(filenameTrain)
        dataTrain['type'] = 'train'
        dataTest = pd.read_csv(filenameTest)
        dataTest['type'] = 'test'
        
        #set uniform numbers
        dataTest['prd_W'] = len(dataTrain['WDL'][dataTrain['WDL'] == 'W'].index)/len(dataTrain.index)
        dataTest['prd_D'] = len(dataTrain['WDL'][dataTrain['WDL'] == 'D'].index)/len(dataTrain.index)
        dataTest['prd_L'] = len(dataTrain['WDL'][dataTrain['WDL'] == 'L'].index)/len(dataTrain.index)
        dataTest['prd_HS'] = np.mean(dataTrain['HS'])
        dataTest['prd_AS'] = np.mean(dataTrain['AS'])
        
        #run model and add current prediction set to full prediction set
        dataTestCombined = pd.concat([dataTestCombined, dataTest])
    dataTestCombined.to_csv('../../input/training_data/DataTestCombined.csv')
    
    #evaluate results of experiment
    m.MetricsCalculation.evaluateExperiment(dataTestCombined, [market_value_cols, betting_odds_cols, match_stats_cols, market_value_cols + betting_odds_cols + match_stats_cols], 'BASELINE FRQ')


#running a baseline model that uses state of the art ELO and regression model
def Elo_Regression_Model(adjustInPredictionSet):

    #dataframe for final test set
    dataTestCombined = pd.DataFrame()
    
    #input train and test sets
    for season in eval_seasons:
        print("Evaluating season "+season)
        #read data of current season and flag train/test
        filenameTrain = '../../train/' + season + '.csv'
        filenameTest = '../../test/' + season + '.csv'
        dataTrain = pd.read_csv(filenameTrain)
        dataTrain['type'] = 'train'
        dataTest = pd.read_csv(filenameTest)
        dataTest['type'] = 'test'
        
        #calculate ELO ratings on full dataset
        data = pd.concat([dataTrain, dataTest])
        data = rm.ELORating(method = "goals").calculateRating(data, 10, 400, 0, 10, 1.25, adjustInPredictionSet)
        dataTrain = data[data['type']=='train']
        dataTest = data[data['type']=='test']
        
        
        #split train dataset to two halfs in order to only use initialised ELOs for training OLR model
        row = int(len(dataTrain.index)/2)
        dataIS = dataTrain.iloc[row:].copy()
        
        #use ordered logistic regression to predict outcome probabilities
        dataTest = pm.OrderedLogisticRegression().calculateProbabilities(dataIS['Rat_H_goals']-dataIS['Rat_A_goals'], dataIS['result'].astype(int), (dataTest['Rat_H_goals']-dataTest['Rat_A_goals']).tolist(), dataTest)
        
        #use linear regression to predict goal expectations
        dataTest['prd_HS'] = gm.LinearRegression().calculateGoals(np.array([dataIS['Rat_H_goals'], dataIS['Rat_A_goals']]).T, dataIS['HS'], np.array([dataTest['Rat_H_goals'], dataTest['Rat_A_goals']]).T)
        dataTest['prd_AS'] = gm.LinearRegression().calculateGoals(np.array([dataIS['Rat_H_goals'], dataIS['Rat_A_goals']]).T, dataIS['AS'], np.array([dataTest['Rat_H_goals'], dataTest['Rat_A_goals']]).T)
 
        
        #run model and add current prediction set to full prediction set
        dataTestCombined = pd.concat([dataTestCombined, dataTest])
    dataTestCombined.to_csv('../../input/training_data/DataTestCombined.csv')
    
    #evaluate results of experiment
    m.MetricsCalculation.evaluateExperiment(dataTestCombined, [market_value_cols, betting_odds_cols, match_stats_cols, market_value_cols + betting_odds_cols + match_stats_cols], 'BASELINE ELO Regression')

#running a baseline model that uses forecasts obtained from betting odds (please note that these are not available for all matches in the test set)    
def BettingOdds():
    
    #dataframe for final test set
    dataTestCombined = pd.DataFrame()
    
    #input and combine test sets
    for season in eval_seasons:
        #read test set of current season
        filenameTest = '../../test/' + season + '.csv'
        dataTest = pd.read_csv(filenameTest)
        #combine data to complete prediction set
        dataTestCombined = pd.concat([dataTestCombined, dataTest])

    #calculate betting odds performance from test set
    #data = pd.read_csv('../../input/training_data/dataTestCombined.csv')
    print(len(dataTestCombined.index))
    dataTestCombined = dataTestCombined[~dataTestCombined['avg_home_odds'].isnull()]
    dataTestCombined = dataTestCombined[~dataTestCombined['avg_draw_odds'].isnull()]
    dataTestCombined = dataTestCombined[~dataTestCombined['avg_away_odds'].isnull()]
    print(len(dataTestCombined.index))
    #no useful goal forecast is obtained from the betting odds
    dataTestCombined['prd_HS']=1
    dataTestCombined['prd_AS']=1
    #use basic normalisation to calculate forecasts
    dataTestCombined['prd_W']=1/dataTestCombined['avg_home_odds']/(1/dataTestCombined['avg_home_odds'] + 1/dataTestCombined['avg_draw_odds'] + 1/dataTestCombined['avg_away_odds'])
    dataTestCombined['prd_D']=1/dataTestCombined['avg_draw_odds']/(1/dataTestCombined['avg_home_odds'] + 1/dataTestCombined['avg_draw_odds'] + 1/dataTestCombined['avg_away_odds'])
    dataTestCombined['prd_L']=1/dataTestCombined['avg_away_odds']/(1/dataTestCombined['avg_home_odds'] + 1/dataTestCombined['avg_draw_odds'] + 1/dataTestCombined['avg_away_odds'])
    m.MetricsCalculation.evaluateExperiment(dataTestCombined, '','Benchmark Betting Odds')


BettingOdds()
#Elo_Regression_Model(adjustInPredictionSet = True)
#Elo_Regression_Model(adjustInPredictionSet = False)
#UNI()
#FRQ()





