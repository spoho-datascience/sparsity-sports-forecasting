
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:15:55 2022

@author: FabianWu
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import poisson
from statsmodels.miscmodels.ordinal_model import OrderedModel
from models.evaluation.Metrics import MetricsCalculation
import xgboost as xgb

import scripts.train_test_split as tts


#Abstract class defining the interface of probability model.
#Parameters are transferred into winning or O/U probabilities 
class ProbabilityModel(ABC):

    def __init__(self, modelType):
        self.type = modelType

    #calculates probabilities
    def calculateProbabilityMatrix(self):
        pass
    
    #expects a probability matrix, i.e. probabilities for each result and calculates summarised probabilities
    def calculateProbabilities(self, probabilities):
        probabilityHome = 0
        probabilityDraw = 0
        probabilityAway = 0
        probabilityOver25 = 0
        probabilityUnder25 = 0
        sum = 0
        
        for i in range(0,11):
            for j in range(0,11):
                if (i>j):
                    probabilityHome += probabilities[i][j]
                elif (i==j):
                    probabilityDraw += probabilities[i][j]
                else:
                    probabilityAway += probabilities[i][j]
                sum += probabilities[i][j]
        
        return [probabilityHome/sum, probabilityDraw/sum, probabilityAway/sum]

    
    
#uses poisson model with possibiliy for being bivariate
class PoissonModel(ProbabilityModel):
    
    def __init__(self):
        super().__init__('poissonModel')
    
    def calculateProbabilityMatrix(self, lambda1, lambda2):
        probabilities=[]
        for i in range(0,11):
            column = []
            for j in range(0,11):
                column.append(poisson.pmf(i, lambda1)*poisson.pmf(j, lambda2))
            probabilities.append(column)
        return probabilities



#uses order logistic regression with one covariate (based on Hvattum & Arntzen, 2010)
class OrderedLogisticRegression(ProbabilityModel):
    
    def __init__(self):
        super().__init__('orderedRegression')
    
    def calculateProbabilities(self, ratingDiffTrain, outcomeTrain, ratingDiffTest, dataTest):

        model = OrderedModel(outcomeTrain, ratingDiffTrain)
        
        regression = model.fit(method='bfgs')
        #regression.summary()
        
        #print(regression.params)
        probabilities = regression.predict(exog=ratingDiffTest)
        dataTest['prd_W']=probabilities[:,2]
        dataTest['prd_D']=probabilities[:,1]
        dataTest['prd_L']=probabilities[:,0]
        
        return dataTest
    
    
    
    
#uses XGBoost based on various ELO ratings
class XGBoost(ProbabilityModel):
    
    def __init__(self):
        super().__init__('xgboost')
    
    #expects full dataset and columns to be included in model fitting, returns predictions only for prediction set
    def calculateProbabilities(self, data, columns):
        dataTrain = data[data['type']=='train']
        dataTest = data[data['type']=='test']

        xgbr = xgb.XGBClassifier()
        xgbr.fit(dataTrain[columns], dataTrain['result'])
        probabilities = xgbr.predict_proba(dataTest[columns])
        
        dataTest['prd_W']=probabilities[:,2]
        dataTest['prd_D']=probabilities[:,1]
        dataTest['prd_L']=probabilities[:,0]
        
        data = pd.concat([dataTrain, dataTest])
        
        return data





class ProbabilityModelTest:   
    
    def testPoissonModel():
        #testing Poisson Model calculation
        #test values are based on calculation in excel.
        
        poissonCalculated = []
        p = PoissonModel()
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(1.0,0.0))[0])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(1.5,1.0))[0])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(1.0,2.0))[0])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(0.0,0.0))[1])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(4.0,1.3))[1])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(3.0,1.7))[1])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(1.2,2.5))[2])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(1.4,6.0))[2])
        poissonCalculated.append(p.calculateProbabilities(p.calculateProbabilityMatrix(1.6,3.0))[2])

        poisson = np.array([0.63212, 0.48795, 0.18259, 1.0, 0.09232, 0.16139, 0.65711, 0.94062, 0.65997])

        errorsPoisson = abs(poisson-poissonCalculated)>0.0001
        print(str(len(errorsPoisson[errorsPoisson == True]))+" errors detected in Poisson model")



    def testOrderedRegression():
        data = pd.read_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\ELORating.csv')
        
        #using half/half train test split
        row = int(len(data.index)/2)
        dataIS = data.iloc[:row].copy()
        dataOOS = data.iloc[row:].copy()
        
        
        #using train test split as specified by the group
        #in-sample is all data until 2012-04-11
        #dataIS = data[pd.to_datetime(data['iso_date'], format = "%Y-%m-%d") < pd.to_datetime("2012-04-12", format="%Y-%m-%d")].copy()

        #oos data is all data from 2012-04-13 that falls in the prediction time
        #tts.create_date_columns(data, dateFormat="%Y-%m-%d")
        #train, test = tts.return_train_test_dfs(data)
        #dataOOS = pd.DataFrame()
        #for s in range(2012,2023):
        #    dataOOS=pd.concat([dataOOS,test[s]], axis = 0)
       

        probabilities = OrderedLogisticRegression().calculateProbabilities(dataIS['Rat_H_results']-dataIS['Rat_A_results'], dataIS['result'], (dataOOS['Rat_H_results']-dataOOS['Rat_A_results']).tolist())
        dataOOS['prd_W']=probabilities[:,2]
        dataOOS['prd_D']=probabilities[:,1]
        dataOOS['prd_L']=probabilities[:,0]
        dataOOS.to_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\OrderedRegression.csv')        
        
        print(MetricsCalculation.calculateRPSFromData(dataOOS))
        
        
        
        data = pd.read_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\piRating.csv')

        dataIS = data[pd.to_datetime(data['iso_date'], format = "%Y-%m-%d") < pd.to_datetime("2012-04-12", format="%Y-%m-%d")].copy()

        #oos data is all data from 2012-04-13 that falls in the prediction time
        tts.create_date_columns(data, dateFormat="%Y-%m-%d")
        train, test = tts.return_train_test_dfs(data)
        dataOOS = pd.DataFrame()
        for s in range(2012,2023):
            dataOOS=pd.concat([dataOOS,test[s]], axis = 0)
       

        probabilities = OrderedLogisticRegression().calculateProbabilities(dataIS['Rat_H_H']-dataIS['Rat_A_A'], dataIS['WDL'], (dataOOS['Rat_H_H']-dataOOS['Rat_A_A']).tolist())
        dataOOS['prd_W']=probabilities[:,2]
        dataOOS['prd_D']=probabilities[:,1]
        dataOOS['prd_L']=probabilities[:,0]
        dataOOS.to_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\OrderedRegression.csv')        
        
        print(MetricsCalculation.calculateRPSFromData(dataOOS))
        
        #not using rating diff, but expected goal diff
        
        #calculate expected goal diff (see 2.3 of Constantinou 2013)
        expectedDiffHome = 10**(abs(data['Rat_H_H'])/3)-1
        expectedDiffHome[data['Rat_H_H'] < 0]= -1 * expectedDiffHome[data['Rat_H_H'] < 0]
        
        expectedDiffAway = 10**(abs(data['Rat_A_A'])/3)-1
        expectedDiffAway[data['Rat_A_A'] < 0]= -1 * expectedDiffAway[data['Rat_A_A'] < 0]
        data['expectedDiff']= (expectedDiffHome-expectedDiffAway)/2
        
        
        
        dataIS = data[pd.to_datetime(data['iso_date'], format = "%Y-%m-%d") < pd.to_datetime("2012-04-12", format="%Y-%m-%d")].copy()

        #oos data is all data from 2012-04-13 that falls in the prediction time
        tts.create_date_columns(data, dateFormat="%Y-%m-%d")
        train, test = tts.return_train_test_dfs(data)
        dataOOS = pd.DataFrame()
        for s in range(2012,2023):
            dataOOS=pd.concat([dataOOS,test[s]], axis = 0)
       

        probabilities = OrderedLogisticRegression().calculateProbabilities(dataIS['expectedDiff'], dataIS['WDL'], (dataOOS['expectedDiff']).tolist())
        dataOOS['prd_W']=probabilities[:,2]
        dataOOS['prd_D']=probabilities[:,1]
        dataOOS['prd_L']=probabilities[:,0]
        dataOOS.to_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\OrderedRegression.csv')        
        
        print(MetricsCalculation.calculateRPSFromData(dataOOS))
        
        
        
        
        
    def testXGBoost():
        data = pd.read_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\ELORating.csv')
        
        #using half/half train test split
        row = int(len(data.index)/2)
        dataIS = data.iloc[:row].copy()
        dataOOS = data.iloc[row:].copy()
        
        
        #using train test split as specified by the group
        #in-sample is all data until 2012-04-11
        #dataIS = data[pd.to_datetime(data['iso_date'], format = "%Y-%m-%d") < pd.to_datetime("2012-04-12", format="%Y-%m-%d")].copy()

        #oos data is all data from 2012-04-13 that falls in the prediction time
        #tts.create_date_columns(data, dateFormat="%Y-%m-%d")
        #train, test = tts.return_train_test_dfs(data)
        #dataOOS = pd.DataFrame()
        #for s in range(2012,2023):
        #    dataOOS=pd.concat([dataOOS,test[s]], axis = 0)
       

        probabilities = XGBoost().calculateProbabilities(dataIS[['Rat_H_ST', 'Rat_A_ST', 'Rat_H_result', 'Rat_A_result', 'Rat_H_odds', 'Rat_A_odds']], dataIS['WDL'], dataOOS[['Rat_H_ST', 'Rat_A_ST', 'Rat_H_result', 'Rat_A_result', 'Rat_H_odds', 'Rat_A_odds']])
        dataOOS['prd_W']=probabilities[:,2]
        dataOOS['prd_D']=probabilities[:,1]
        dataOOS['prd_L']=probabilities[:,0]
        dataOOS.to_csv('C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\XGBoost.csv')        
        
        print(MetricsCalculation.calculateRPSFromData(dataOOS))
        
        
        
        
    #testPoissonModel()
    #testOrderedRegression()
    #testXGBoost()
    