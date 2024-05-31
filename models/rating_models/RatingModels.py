# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:20:30 2023

@author: Dell-PC
"""


import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import models.probability_models.ProbabilityModels as pm
import models.evaluation.Metrics as m
import input.training_data.DataCollection as dc
import models.rating_models.AdjustedResultModels as arm
#import models.rating_models.RatingModels as rm


pd.options.mode.chained_assignment = None

#Abstract class defining the interface of rating model.
#rating is calculated based on the dataset
class RatingModel(ABC):

    def __init__(self, modelType):
        self.type = modelType

    #calculates rating for the teams based on the dataset
    def calculateRating(self, data):
        pass

        
    #helper method to update information on the initial ratings to be used for promoted or relegated teams
    def updateRelegationPromotionRatings(self, data, i, season, league, columnName, team, currentRatings, relegationRating, promotionRating):
        if(data[columnName][i]==1):
            #reset relegation ratings in case of new seasons
            if(relegationRating[league]['season']!=season):
                relegationRating[league]['rating']=[] 
            relegationRating[league]['season']=season
            relegationRating[league]['rating'].append(currentRatings[team]['calculated'])  
        elif(data[columnName][i]==-1):
            #reset promotion ratings in case of new seasons
            if(promotionRating[league]['season']!=season):
                 promotionRating[league]['rating']=[] 
            promotionRating[league]['season']=season
            promotionRating[league]['rating'].append(currentRatings[team]['calculated'])  
                    
        
    
    #helper method to initialise ratings to be used for promoted or relegated teams          
    def initialiseRating(self, data, i, columnName, league, team, currentRatings, relegationRating, promotionRating, setNAN = False):
        if(data[columnName][i]==2):
            currentRatings[team]['calculated']=np.mean(promotionRating[league]['rating'])
            if(setNAN):
                currentRatings[team]['actual']=np.nan
            else:
                currentRatings[team]['actual']=np.mean(promotionRating[league]['rating'])
        elif(data[columnName][i]==-2):
            currentRatings[team]['calculated']=np.mean(relegationRating[league]['rating'])
            if(setNAN):
                currentRatings[team]['actual']=np.nan
            else:
                currentRatings[team]['actual']=np.mean(relegationRating[league]['rating'])

    
 
    
#implements ELO rating based on the work of Hvattum&Arntzen (2010)
class ELORating(RatingModel):
    
    def __init__(self, method):
        self.method = method
        super().__init__('ELORating')
        
    #handlingMissing defines how to ract to missing data ('carryForward' = ELO is carried forward, 'currentMissing' = ELO is set to NAN if last information was missing, 'historyMissing' = ELO is set to NAN if all history is missing so far)
    def calculateRating(self, data, c, d, k, k0, lam, adjustInPredictionSet = False, handleMissing = 'carryForward'):
        #sort data by date
        data.sort_values(by=['iso_date'], inplace = True)
        data.reset_index(inplace=True, drop = True)
        
        #distinct list of all teams in the dataset, initialised with rating of 1000/ initialised with league, last match and home/away information
        currentRatings = dict.fromkeys(np.unique(np.concatenate([data['HT'], data['AT']])))
        #if no data is avaiable, the actual rating is not set, but a calculated one exists, the history flag is false until data was available for the first time
        for key in currentRatings:
            currentRatings[key] = {'calculated': 0, 'actual': 0, 'history': False}        
        #distinct lists of all leagues in the dataset
        promotionRating = dict.fromkeys(np.unique(data['Lge']))
        for key in promotionRating:
            promotionRating[key] = {'season': 'NaN', 'rating': [1000,1000]}   
        relegationRating = dict.fromkeys(np.unique(data['Lge']))
        for key in relegationRating:
            relegationRating[key] = {'season': 'NaN', 'rating': [1000,1000]}   

        #Calculation of ELO ratings
        eloRatingsHome=[]
        eloRatingsAway=[]
        
        #iterate over all matches
        for i in range(0,len(data.index)):
            teamHome = data['HT'][i]
            teamAway = data['AT'][i]
            league = data['Lge'][i]
            season = data['Sea'][i]
            
            #update average ratings for promoted or relegated teams
            super().updateRelegationPromotionRatings(data, i, season, league, 'HLC', teamHome, currentRatings, relegationRating, promotionRating)
            super().updateRelegationPromotionRatings(data, i, season, league, 'ALC', teamAway, currentRatings, relegationRating, promotionRating)

            #check if data is available
            dataAvailable = (self.method == "results" or self.method == "goals" or (~np.isnan(data['pred_'+self.method+'_home'][i]) and ~np.isnan(data['pred_'+self.method+'_draw'][i]) and ~np.isnan(data['pred_'+self.method+'_away'][i])))

            #initialise rating if needed 
            super().initialiseRating(data, i, 'HLC', league, teamHome, currentRatings, relegationRating, promotionRating, setNAN = (not dataAvailable and not handleMissing == 'carryForward'))
            super().initialiseRating(data, i, 'ALC', league, teamAway, currentRatings, relegationRating, promotionRating, setNAN = (not dataAvailable and not handleMissing == 'carryForward'))

            #only update actual ELO in train set
            if(data['type'][i]=='train' or adjustInPredictionSet == True):
                #flag histry as existing if it existed before or data is available now
                if(dataAvailable or currentRatings[teamHome]['history'] == True):
                    currentRatings[teamHome]['history'] = True
                if(dataAvailable or currentRatings[teamAway]['history'] == True):
                    currentRatings[teamAway]['history'] = True                     
                #set actual ELO to nan dependent on the handling of missing values
                if(handleMissing == 'historyMissing'):
                    if(currentRatings[teamHome]['history'] == True):
                        currentRatings[teamHome]['actual'] = currentRatings[teamHome]['calculated']
                    if(currentRatings[teamAway]['history'] == True):
                        currentRatings[teamAway]['actual'] = currentRatings[teamAway]['calculated']
                elif(dataAvailable or handleMissing == 'carryForward'):
                    currentRatings[teamHome]['actual']=currentRatings[teamHome]['calculated']
                    currentRatings[teamAway]['actual']=currentRatings[teamAway]['calculated']
                else:
                    currentRatings[teamHome]['actual']=np.nan
                    currentRatings[teamAway]['actual']=np.nan
            
            
            eloHome = currentRatings[teamHome]['actual']
            eloAway = currentRatings[teamAway]['actual']
            eloRatingsHome.append(eloHome)
            eloRatingsAway.append(eloAway)
            
            #calculate expected result
            expHome = 1/(1+c**((eloAway-eloHome)/d))
            expAway = 1-expHome

            
            #obtain actual result
            if(self.method == "results" or self.method == "goals"):
                actHome = 1
                if(data['result'][i] == 1):
                    actHome = 0.5
                elif(data['result'][i] == 0):
                    actHome = 0
            else:
                actHome = data['pred_'+self.method+'_home'][i] + 0.5 * data['pred_'+self.method+'_draw'][i]

            actAway = 1 - actHome            

            #difference between Elo result and Elo goals
            if self.method == "goals":
                adjustmentFactor = k0*(1+np.absolute(data['HS'][i]-data['AS'][i]))**lam
            #adjustedResults changes actual result and adjustment factor
            elif self.method == "adjustedGoals":
                actHome = 1
                if(data['home_score'][i] == data['away_score'][i]):
                    actHome = 0.5
                elif(data['home_score'][i] < data['away_score'][i]):
                    actHome = 0
                actAway = 1 - actHome    
                
                adjustmentFactor = k0*(1+np.absolute(data['home_score'][i]-data['away_score'][i]))**lam
            else:
                adjustmentFactor = k
                   
            
            #calculate new elo rating (only if data is available)
            if(dataAvailable == True): 
                currentRatings[teamHome]['calculated']=eloHome + adjustmentFactor * (actHome-expHome)
                currentRatings[teamAway]['calculated']=eloAway + adjustmentFactor * (actAway-expAway)

        #set final ratings in dataframe    
        data['Rat_H_'+self.method] = eloRatingsHome
        data['Rat_A_'+self.method] = eloRatingsAway
        data['Rat_Diff_'+self.method] = data['Rat_H_'+self.method] - data['Rat_A_'+self.method]
            
        return data
    
    
    #expects parameters and a range of k's, k0's and lam's to be tested
    def optimiseRating(self, data, c, d, k_values, k0_values, lam_values, adjustInPredictionSet = False, putMissingNAN = False):
        bestRPS = 1000
        #optimise over the grid of given values, in most cases only k should vary
        for k in k_values:
            for k0 in k0_values:
                for lam in lam_values:
                    #add ELO rating and do not consider cases where data (and thus ELO) was unavailable
                    dataNew = ELORating.calculateRating(self, data, c, d, k, k0, lam, adjustInPredictionSet, putMissingNAN)
                    dataEval = dataNew[data['type'] == 'train']
                    dataEval = dataEval[~dataEval['Rat_H_'+self.method].isnull()]
                    dataEval = dataEval[~dataEval['Rat_A_'+self.method].isnull()]
                    
                    #extract ELO initialisation by dividing by 3
                    row = int(len(dataEval.index)/3)
                    dataIS = dataEval.iloc[row:2*row].copy()
                    dataOOS = dataEval.iloc[2*row:].copy()
                    
                    dataOOS = pm.OrderedLogisticRegression().calculateProbabilities(dataIS['Rat_H_'+self.method]-dataIS['Rat_A_'+self.method], dataIS['result'].astype(int), (dataOOS['Rat_H_'+self.method]-dataOOS['Rat_A_'+self.method]).tolist(), dataOOS)
                    rps = m.MetricsCalculation.calculateRPSFromData(dataOOS)[1]
                    print('Current k '+str(k)+' ,current k0 '+str(k0)+' ,current lam '+str(lam)+' current RPS '+str(rps))
                    if(rps < bestRPS):
                        bestRPS = rps
                        data = dataNew
                        k_opt = k
                        k0_opt = k0
                        lam_opt = lam
                    print(bestRPS)
        print('optimal k '+str(k_opt)+' ,optimal k0 '+str(k0_opt)+' ,optimal lam '+str(lam_opt)+' best RPS '+str(bestRPS))
        
        #for debugging
        #data.to_csv('../../input/training_data/ELORating.csv')
        return data
                    
    

#implements pi-rating based on the work of Constatinou & Fenton (2013)
class piRating(RatingModel):
    
    def __init__(self):
        super().__init__('pi-rating')
        
    
    def calculateRating(self, data, gamma, lam, adjustInPredictionSet = False):
        #sort data by date
        data.sort_values(by=['iso_date'], inplace = True)
        data.reset_index(inplace =True, drop = True)
        
        
        #distinct list of all teams in the dataset, containing overall, home and away rating
        #team information initialised with league, last match and home/away information
        currentRatingsHome = dict.fromkeys(np.unique(np.concatenate([data['HT'], data['AT']])))
        for key in currentRatingsHome:
            currentRatingsHome[key] = {'calculated': 0, 'actual': 0}
        currentRatingsAway = dict.fromkeys(np.unique(np.concatenate([data['HT'], data['AT']])))
        for key in currentRatingsAway:
            currentRatingsAway[key] = {'calculated': 0, 'actual': 0}
        currentRatingsOverall = dict.fromkeys(np.unique(np.concatenate([data['HT'], data['AT']])))
        for key in currentRatingsOverall:
            currentRatingsOverall[key] = {'calculated': 0, 'actual': 0}
            

        #distinct lists of all leagues in the dataset, dictionaries contain overall, home and away pi-ratings
        promotionRatingHome = dict.fromkeys(np.unique(data['Lge']))
        for key in promotionRatingHome:
            promotionRatingHome[key] = {'season': 'NaN', 'rating': [0,0]}
        promotionRatingAway = dict.fromkeys(np.unique(data['Lge']))
        for key in promotionRatingAway:
            promotionRatingAway[key] = {'season': 'NaN', 'rating': [0,0]}    
        promotionRatingOverall = dict.fromkeys(np.unique(data['Lge']))
        for key in promotionRatingOverall:
            promotionRatingOverall[key] = {'season': 'NaN', 'rating': [0,0]}    

        relegationRatingHome = dict.fromkeys(np.unique(data['Lge']))
        for key in relegationRatingHome:
            relegationRatingHome[key] = {'season': 'NaN', 'rating': [0,0]}
        relegationRatingAway = dict.fromkeys(np.unique(data['Lge']))
        for key in relegationRatingAway:
            relegationRatingAway[key] = {'season': 'NaN', 'rating': [0,0]}
        relegationRatingOverall = dict.fromkeys(np.unique(data['Lge']))
        for key in relegationRatingOverall:
            relegationRatingOverall[key] = {'season': 'NaN', 'rating': [0,0]}


        #Calculation of ELO ratings
        piRatingsHomeTeam_Home=[]
        piRatingsHomeTeam_Away=[]
        piRatingsHomeTeam=[]
        piRatingsAwayTeam_Home=[]
        piRatingsAwayTeam_Away=[]
        piRatingsAwayTeam=[]
        piExpectedDiff=[]
        
        #iterate over all matches
        for i in range(0,len(data.index)):
            teamHome = data['HT'][i]
            teamAway = data['AT'][i]
            league = data['Lge'][i]
            season = data['Sea'][i]
            

            #update average ratings for promoted or relegated teams
            super().updateRelegationPromotionRatings(data, i, season, league, 'HLC', teamHome, currentRatingsHome, relegationRatingHome, promotionRatingHome)
            super().updateRelegationPromotionRatings(data, i, season, league, 'HLC', teamHome, currentRatingsAway, relegationRatingAway, promotionRatingAway)
            super().updateRelegationPromotionRatings(data, i, season, league, 'HLC', teamHome, currentRatingsOverall, relegationRatingOverall, promotionRatingOverall)
            
            super().updateRelegationPromotionRatings(data, i, season, league, 'ALC', teamAway, currentRatingsHome, relegationRatingHome, promotionRatingHome)
            super().updateRelegationPromotionRatings(data, i, season, league, 'ALC', teamAway, currentRatingsAway, relegationRatingAway, promotionRatingAway)
            super().updateRelegationPromotionRatings(data, i, season, league, 'ALC', teamAway, currentRatingsOverall, relegationRatingOverall, promotionRatingOverall)
            

            #initialise ratings if needed
            super().initialiseRating(data, i, 'HLC', league, teamHome, currentRatingsOverall, relegationRatingOverall, promotionRatingOverall)
            super().initialiseRating(data, i, 'HLC', league, teamHome, currentRatingsHome, relegationRatingHome, promotionRatingHome)
            super().initialiseRating(data, i, 'HLC', league, teamHome, currentRatingsAway, relegationRatingAway, promotionRatingAway)
            
            super().initialiseRating(data, i, 'ALC', league, teamAway, currentRatingsOverall, relegationRatingOverall, promotionRatingOverall)
            super().initialiseRating(data, i, 'ALC', league, teamAway, currentRatingsHome, relegationRatingHome, promotionRatingHome)
            super().initialiseRating(data, i, 'ALC', league, teamAway, currentRatingsAway, relegationRatingAway, promotionRatingAway)



            #only update piRating in train set
            if(data['type'][i]=='train' or adjustInPredictionSet == True):
                currentRatingsOverall[teamHome]['actual']=currentRatingsOverall[teamHome]['calculated']
                currentRatingsHome[teamHome]['actual']=currentRatingsHome[teamHome]['calculated']
                currentRatingsAway[teamHome]['actual']=currentRatingsAway[teamHome]['calculated']
                
                currentRatingsOverall[teamAway]['actual']=currentRatingsOverall[teamAway]['calculated']
                currentRatingsHome[teamAway]['actual']=currentRatingsHome[teamAway]['calculated']
                currentRatingsAway[teamAway]['actual']=currentRatingsAway[teamAway]['calculated']


            piHome = currentRatingsOverall[teamHome]['actual']
            piHome_Home = currentRatingsHome[teamHome]['actual']
            piHome_Away = currentRatingsAway[teamHome]['actual']
            piAway = currentRatingsOverall[teamAway]['actual']
            piAway_Home = currentRatingsHome[teamAway]['actual']
            piAway_Away = currentRatingsAway[teamAway]['actual']
            
            piRatingsHomeTeam.append(piHome)
            piRatingsHomeTeam_Home.append(piHome_Home)
            piRatingsHomeTeam_Away.append(piHome_Away)
            piRatingsAwayTeam.append(piAway)
            piRatingsAwayTeam_Home.append(piAway_Home)
            piRatingsAwayTeam_Away.append(piAway_Away)
         
            

            #calculate expected goal diff (see 2.3 of Constantinou 2013)
            expectedDiffHome = 10**(abs(piHome_Home)/3)-1
            if(piHome_Home < 0):
                expectedDiffHome = -1*expectedDiffHome
            expectedDiffAway = 10**(abs(piAway_Away)/3)-1
            if(piAway_Away < 0):
                expectedDiffAway = -1*expectedDiffAway
            expectedDiff = (expectedDiffHome-expectedDiffAway)/2
            
            piExpectedDiff.append(expectedDiff)
            
            
            #only update pi in train set or if the flag is set to true
            if(data['type'][i]=='train' or adjustInPredictionSet == True):
            
             
                #obtain actual goal diff and weighted prediction error (see 2.3 of Constantinou 2013)
                actDiff = data['GD'][i]
                error = abs(actDiff - expectedDiff)
                weightedError = 3 * np.log10(error + 1)
                if (expectedDiff < actDiff):
                    weightedErrorHome = weightedError
                else:
                    weightedErrorHome = -1*weightedError
                if (expectedDiff > actDiff):
                    weightedErrorAway = weightedError
                else:
                    weightedErrorAway = -1*weightedError


                #calculate new pi ratings (see 2.4 of Constantinou 2013)
                currentRatingsHome[teamHome]['calculated'] = piHome_Home + lam * weightedErrorHome
                currentRatingsAway[teamHome]['calculated'] = piHome_Away + gamma * (currentRatingsHome[teamHome]['calculated'] - piHome_Home)
                currentRatingsOverall[teamHome]['calculated'] = (currentRatingsHome[teamHome]['calculated'] + currentRatingsAway[teamHome]['calculated'])/2
            
                currentRatingsAway[teamAway]['calculated'] = piAway_Away + lam * weightedErrorAway
                currentRatingsHome[teamAway]['calculated'] = piAway_Home + gamma * (currentRatingsAway[teamAway]['calculated'] - piAway_Away)
                currentRatingsOverall[teamAway]['calculated'] = (currentRatingsHome[teamAway]['calculated'] + currentRatingsAway[teamAway]['calculated'])/2
        
        
        #set final ratings in dataframe
        data['Rat_Pi_H_H'] = piRatingsHomeTeam_Home
        data['Rat_Pi_H_A'] = piRatingsHomeTeam_Away
        data['Rat_Pi_H'] = piRatingsHomeTeam
        data['Rat_Pi_A_H'] = piRatingsAwayTeam_Home
        data['Rat_Pi_A_A'] = piRatingsAwayTeam_Away
        data['Rat_Pi_A'] = piRatingsAwayTeam
        data['Rat_Pi_Expected_Diff'] = piExpectedDiff
        
        return data
    
    
    
    #expects parameters and a range of k's, k0's and lam's to be tested
    def optimiseRating(self, data, gamma_values, lam_values, adjustInPredictionSet = False):
        bestRPS = 1000
        #optimise over the grid of given values, in most cases only k should vary
        for gamma in gamma_values:
            for lam in lam_values:
                #add pi rating and do not consider cases where data (and thus pi) was unavailable
                data.to_csv('../../input/training_data/DataBeforePi.csv')
                dataNew = piRating.calculateRating(self, data, gamma, lam, adjustInPredictionSet)
                dataNew.to_csv('../../input/training_data/DataAfterPi.csv')
                dataEval = dataNew[data['type'] == 'train']
                dataEval = dataEval[~dataEval['Rat_Pi_H_H'].isnull()]
                dataEval = dataEval[~dataEval['Rat_Pi_H_A'].isnull()]
                dataEval = dataEval[~dataEval['Rat_Pi_H'].isnull()]
                dataEval = dataEval[~dataEval['Rat_Pi_A_H'].isnull()]
                dataEval = dataEval[~dataEval['Rat_Pi_A_A'].isnull()]
                dataEval = dataEval[~dataEval['Rat_Pi_A'].isnull()]

                
                #calculate input for ordered regression
                dataEval['piExpectedDiff']=np.nan
                for i in range(0,len(dataEval.index)):
                    #calculate expected diff for pi rating
                    expectedDiffHome = 10**(abs(dataEval['Rat_Pi_H_H'][i])/3)-1
                    if(dataEval['Rat_Pi_H_H'][i] < 0):
                        expectedDiffHome = -1*expectedDiffHome
                    expectedDiffAway = 10**(abs(dataEval['Rat_Pi_A_A'][i])/3)-1
                    if(dataEval['Rat_Pi_A_A'][i] < 0):
                        expectedDiffAway = -1*expectedDiffAway
                    dataEval['piExpectedDiff'][i] = (expectedDiffHome-expectedDiffAway)/2

                

                #extract pi initialisation by dividing by 3
                row = int(len(dataEval.index)/3)
                dataIS = dataEval.iloc[row:2*row].copy()
                dataOOS = dataEval.iloc[2*row:].copy()
                
                
                dataOOS = pm.OrderedLogisticRegression().calculateProbabilities(dataIS['piExpectedDiff'], dataIS['result'].astype(int), (dataOOS['piExpectedDiff']).tolist(), dataOOS)
                dataOOS.to_csv('../../input/training_data/piRatingPrediction.csv')
                rps = m.MetricsCalculation.calculateRPSFromData(dataOOS)[1]
                print('Current gamma '+str(gamma)+' ,current lam '+str(lam)+' current RPS '+str(rps))
                if(rps < bestRPS):
                    bestRPS = rps
                    data = dataNew
                    gamma_opt = gamma
                    lam_opt = lam
                print(bestRPS)
        print('optimal gamma '+str(gamma_opt)+' ,optimal lam '+str(lam_opt)+' best RPS '+str(bestRPS))
        
        #for debugging
        data.to_csv('../../input/training_data/piRating.csv')
        
        return data
    


class RatingModelTest:   
    
    def testELORating():
        data = pd.read_csv('../../input/training_data/DataAdjustedResults.csv')
        data = data[~data['result'].isnull()]
        data = ELORating(method = "odds").optimiseRating(data, 10, 400, [150], [0], [0])
        data = ELORating(method = "goals").optimiseRating(data, 10, 400, [0], [10], [1])
        data = ELORating(method = "results").optimiseRating(data, 10, 400, [20], [0], [0])
        data.to_csv('../../input/training_data/ELORating.csv')

    def testpiRating():
        data = dc.cleanData(pd.read_csv('../../input/training_data/Trainset_2022_12_12.csv'))
        #data = pd.read_csv('../../input/training_data/Trainset_2022_12_12.csv')
        #data['Date']=pd.to_datetime(data.Date, format="%d/%m/%Y") 
        #data['iso_date']=data['Date']
        
        #add information on train/test
        data['type']='train'
        #data['type'][pd.to_datetime(data['iso_date'], format = "%Y-%m-%d") < pd.to_datetime("2025-07-01", format="%Y-%m-%d")] = 'train'
        #data['type'][pd.to_datetime(data['iso_date'], format = "%Y-%m-%d") >= pd.to_datetime("2025-07-01", format="%Y-%m-%d")] = 'test'
        
        data = piRating().optimiseRating(data, [0.6, 0.7, 0.8], [0.03, 0.035])
        
        data.to_csv('../../input/training_data/piRating.csv')
        
    #optimising the parameters for ELO and pi Rating by simple grid search around optimal parameters from pre-experience 
    def parameterOptimisation():
        #just use train set up to the first test set for optimisation
        filenameTrain = '../../train/10-11.csv'
        data = pd.read_csv(filenameTrain)
        data['type'] = 'train'
        
        #data = ELORating(method = "results").optimiseRating(data, 10, 400, [10, 15, 20, 25], [0], [0], putMissingNAN = True)
        #data = ELORating(method = "goals").optimiseRating(data, 10, 400, [0], [5, 10, 15], [0.75, 1, 1.25, 1.5], putMissingNAN = True)
        data = piRating().optimiseRating(data, [0.65, 0.7, 0.75], [0.03, 0.04, 0.05, 0.06, 0.07])

        #data = arm.AdjustedResultModels().addAdjustedResults(data, {'odds': ['odds', ' ', ' ']})
        #data = ELORating(method = "odds").optimiseRating(data, 10, 400, [75, 125, 175, 225], [0], [0], putMissingNAN = True)
        #data = arm.AdjustedResultModels().addAdjustedResults(data, {'marketvalue': ['MV', 'home_starter_total', 'away_starter_total']})
        #data = ELORating(method = "MV").optimiseRating(data, 10, 400, [40, 50, 60, 70, 80], [0], [0], putMissingNAN = True)
        #data = arm.AdjustedResultModels().addAdjustedResults(data, {'shots': ['S', 'HTS', 'ATS']})
        #data = ELORating(method = "S").optimiseRating(data, 10, 400, [15, 20, 25, 30, 35, 40], [0], [0], putMissingNAN = True)
        #data = arm.AdjustedResultModels().addAdjustedResults(data, {'shotsTarget': ['ST', 'HST', 'AST']})
        #data = ELORating(method = "ST").optimiseRating(data, 10, 400, [15, 20, 25, 30, 35, 40], [0], [0], putMissingNAN = True)

        
    #parameterOptimisation() 


    #testELORating()
    #testpiRating()
