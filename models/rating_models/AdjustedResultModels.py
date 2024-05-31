# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:46:24 2023

@author: Dell-PC
"""


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np



#adds information to the dataset by translating match statistics to adjusted results using XGBoost
class AdjustedResultModels:
    
    def addAdjustedResults(self, data, variables):       
        dataTrain = data[data['type'] == 'train']
        dataTest = data[data['type'] == 'test']
        
        for key in variables:
            var = variables[key]
            dataTrain = AdjustedResultModels.addAdjustedResult(dataTrain, var[0], var[1], var[2])

        #for debugging
        dataTrain.to_csv("C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\DataAdjustedResults_Train.csv")
        pd.concat([dataTrain, dataTest]).to_csv("C:\\Users\\Dell-PC\\Documents\\Projekte\\2023-soccer-prediction\\input\\training_data\\DataAdjustedResults.csv")
        
        return pd.concat([dataTrain, dataTest])


    def addAdjustedResult(data, variable, variableHome, variableAway):
        
        if variable == 'odds':
            overround = 1/data['avg_home_odds'] + 1/data['avg_draw_odds'] + 1/data['avg_away_odds']
            data['pred_odds_home'] = 1/data['avg_home_odds']/overround
            data['pred_odds_draw'] = 1/data['avg_draw_odds']/overround
            data['pred_odds_away'] = 1/data['avg_away_odds']/overround
            
        else:
            #train XGBoost model on the specific variables
            x = data
            y = data['result'].astype(int).tolist()
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50)
            
            #fit and apply to data by 2 fold cross validation
            xgbr = xgb.XGBClassifier()
            x_train_cur = np.array([x_train[variableHome], x_train[variableAway]]).T
            x_test_cur = np.array([x_test[variableHome], x_test[variableAway]]).T
            
            #fit model and predict values
            xgbr.fit(x_test_cur, y_test)
            prediction = xgbr.predict_proba(x_train_cur)
            x_train['pred_'+variable+'_home']=prediction[:,2]
            x_train['pred_'+variable+'_draw']=prediction[:,1]
            x_train['pred_'+variable+'_away']=prediction[:,0]
            
            #fit model and predict values
            xgbr.fit(x_train_cur, y_train)
            prediction = xgbr.predict_proba(x_test_cur)
            x_test['pred_'+variable+'_home']=prediction[:,2]
            x_test['pred_'+variable+'_draw']=prediction[:,1]
            x_test['pred_'+variable+'_away']=prediction[:,0]
            
            data = pd.concat([x_train, x_test], ignore_index=True)

            #set predictions to nan where data is not available
            data['pred_'+variable+'_home'][data[[variableHome, variableAway]].isna().any(axis=1)]=np.nan
            data['pred_'+variable+'_draw'][data[[variableHome, variableAway]].isna().any(axis=1)]=np.nan
            data['pred_'+variable+'_away'][data[[variableHome, variableAway]].isna().any(axis=1)]=np.nan

        return data
        

        

class AdjustedResultModelTest:
    
    def testAdjustedResultModel():
        data = pd.read_csv('../../input/training_data/Dataset.csv')
        variables = {'shotsTarget': ['ST', 'HST', 'AST'], 'position': ['position', 'home_club_position', 'away_club_position'], 'corners': ['C', 'HC', 'AC']}
        data = AdjustedResultModels().addAdjustedResults(data, variables)
        
        

    #testAdjustedResultModel()