# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:05:38 2023

@author: FabianWu
"""


import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from models.evaluation.Metrics import MetricsCalculation

import scripts.train_test_split as tts

from sklearn import linear_model
import xgboost as xgb


#Abstract class defining the interface of saved_networks to forecast exact number of goals.
#Parameters are transferred into expected number of goals for each team
class GoalModel(ABC):

    def __init__(self, modelType):
        self.type = modelType
    
    #calculates the number of goals for both teams in a match
    def calculateGoals(self):
        pass
    
    

#uses linear regression to obtain goals
class LinearRegression(GoalModel):
    
    def __init__(self):
        super().__init__('linearRegression')
    
    def calculateGoals(self, variablesTrain, goalsTrain, variablesTest):
        
        goalsTrainRecoded = goalsTrain.copy()
        
        goalsTrainRecoded = goalsTrainRecoded.replace("W", 2)
        goalsTrainRecoded = goalsTrainRecoded.replace("D", 1)
        goalsTrainRecoded = goalsTrainRecoded.replace("L", 0)
        goalsTrainRecoded = goalsTrainRecoded.astype(int)

        model = linear_model.LinearRegression()
        
        regression = model.fit(variablesTrain, goalsTrain)
        
        return regression.predict(variablesTest)
    
    
#uses XGBoost based on various ELO ratings
class XGBoost(GoalModel):
    
    def __init__(self):
        super().__init__('xgboost')
    
    #expects full dataset and columns to be included in model fitting, returns predictions only for prediction set
    def calculateGoals(self, data, columns):
        dataTrain = data[data['type']=='train']
        dataTest = data[data['type']=='test']
        
        xgbr = xgb.XGBRegressor()
        xgbr.fit(dataTrain[columns], dataTrain['HS'])
        dataTest['prd_HS']=xgbr.predict(dataTest[columns])

        xgbr = xgb.XGBRegressor()        
        xgbr.fit(dataTrain[columns], dataTrain['AS'])
        dataTest['prd_AS']=xgbr.predict(dataTest[columns])
        
        data = pd.concat([dataTrain, dataTest])

        return data
    



class GoalModelTest:   
    

    def testLinearRegression():
        data = pd.read_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\ELORating.csv')
        
        #using half/half train test split
        row = int(len(data.index)/2)
        dataIS = data.iloc[:row].copy()
        dataOOS = data.iloc[row:].copy()
        
        
        goalsHome = LinearRegression().calculateGoals(np.array([dataIS['Rat_H'], dataIS['Rat_A']]).T, dataIS['HS'], np.array([dataOOS['Rat_H'], dataOOS['Rat_A']]).T)
        goalsAway = LinearRegression().calculateGoals(np.array([dataIS['Rat_H'], dataIS['Rat_A']]).T, dataIS['AS'], np.array([dataOOS['Rat_H'], dataOOS['Rat_A']]).T)
        dataOOS['prd_HS']=goalsHome
        dataOOS['prd_AS']=goalsAway

        dataOOS.to_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\LinearRegression.csv')        
        
        print(MetricsCalculation.calculateRMSEFromData(dataOOS))
        

        
        
        data = pd.read_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\piRating.csv')
        #dataIS = data[pd.to_datetime(data['Date'], format = "%Y-%m-%d") < pd.to_datetime("2012-04-12", format="%Y-%m-%d")].copy()

        #using half/half train test split
        row = int(len(data.index)/2)
        dataIS = data.iloc[:row].copy()
        dataOOS = data.iloc[row:].copy()

        #oos data is all data from 2012-04-13 that falls in the prediction time
        #tts.create_date_columns(data, dateFormat="%Y-%m-%d")
        #train, test = tts.return_train_test_dfs(data)
        #dataOOS = pd.DataFrame()
        #for s in range(2012,2023):
        #    dataOOS=pd.concat([dataOOS,test[s]], axis = 0)
       

        goalsHome = LinearRegression().calculateGoals(np.array([dataIS['Rat_H_H'], dataIS['Rat_A_A']]).T, dataIS['HS'], np.array([dataOOS['Rat_H_H'], dataOOS['Rat_A_A']]).T)
        goalsAway = LinearRegression().calculateGoals(np.array([dataIS['Rat_H_H'], dataIS['Rat_A_A']]).T, dataIS['AS'], np.array([dataOOS['Rat_H_H'], dataOOS['Rat_A_A']]).T)
        dataOOS['prd_HS']=goalsHome
        dataOOS['prd_AS']=goalsAway
        dataOOS.to_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\LinearRegression.csv')        
        
        print(MetricsCalculation.calculateRMSEFromData(dataOOS))
        
        
        #not using rating diff, but expected goal diff
        
        #calculate expected goal diff (see 2.3 of Constantinou 2013)
        expectedDiffHome = 10**(abs(data['Rat_H_H'])/3)-1
        expectedDiffHome[data['Rat_H_H'] < 0]= -1 * expectedDiffHome[data['Rat_H_H'] < 0]
        
        expectedDiffAway = 10**(abs(data['Rat_A_A'])/3)-1
        expectedDiffAway[data['Rat_A_A'] < 0]= -1 * expectedDiffAway[data['Rat_A_A'] < 0]
        data['expectedDiff']= (expectedDiffHome-expectedDiffAway)/2
        
        #using half/half train test split
        row = int(len(data.index)/2)
        dataIS = data.iloc[:row].copy()
        dataOOS = data.iloc[row:].copy()
        
        dataIS = data[pd.to_datetime(data['Date'], format = "%Y-%m-%d") < pd.to_datetime("2012-04-12", format="%Y-%m-%d")].copy()

        #oos data is all data from 2012-04-13 that falls in the prediction time
        #tts.create_date_columns(data, dateFormat="%Y-%m-%d")
        #train, test = tts.return_train_test_dfs(data)
        #dataOOS = pd.DataFrame()
        #for s in range(2012,2023):
        #    dataOOS=pd.concat([dataOOS,test[s]], axis = 0)
       

        goalsHome = LinearRegression().calculateGoals(dataIS[['expectedDiff']].to_numpy(), dataIS['HS'], dataOOS[['expectedDiff']].to_numpy())
        goalsAway = LinearRegression().calculateGoals(dataIS[['expectedDiff']].to_numpy(), dataIS['AS'], dataOOS[['expectedDiff']].to_numpy())
        dataOOS['prd_HS']=goalsHome
        dataOOS['prd_AS']=goalsAway
        dataOOS.to_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\LinearRegression.csv')        
        
        print(MetricsCalculation.calculateRMSEFromData(dataOOS))
        
        
        
    
    #testLinearRegression()