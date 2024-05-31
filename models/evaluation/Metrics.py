# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:24:27 2022

@author: FabianWu
"""

from scipy.stats import bootstrap
import numpy as np
import pandas as pd
import datetime as datetime
from input.features import split_dataset_by_data_intensive_leagues


#implements the metrics (rps, rmse) used in the prediction challenge, and potentially more
class MetricsCalculation():
    
    #input: forecasted probability home/draw; true outcome home/draw; away information is redundant
    #output: average RPS, bootstrap confidence interval and sample size
    def calculateRPS(probHome, probDraw, resultHome, resultDraw, round_by = 10, numberSamples = 1000):
        scores = 0.5*((probHome-resultHome)**2 + (probHome+probDraw-resultHome-resultDraw)**2)
        #if no data is given
        if len(scores)==0:
            return np.nan, np.nan, np.nan
        #if only one entry is given, CI is just set to this value
        elif len(scores)==1:
            upper = scores
            lower = scores
        else:
            bootstrap_ci = bootstrap((scores,), np.mean, n_resamples = numberSamples, confidence_level=0.95, method = 'percentile').confidence_interval
            lower = bootstrap_ci.low
            upper = bootstrap_ci.high
        return np.round(np.mean(scores),decimals = round_by), np.round(lower,decimals = round_by), np.round(upper,decimals = round_by), len(probHome)
    
    
    #uses the predefined table format of the challenge to calculate RPS directly
    def calculateRPSFromData(data, round_by = 10, numberSamples = 1000):
        return MetricsCalculation.calculateRPS(data['prd_W'], data['prd_D'], data['WDL']=="W", data['WDL']=="D", round_by, numberSamples)

        
        
    #input: forecasted goals home/draw; true goals home/draw
    #output: RMSE metric, bootstrap confidence interval and sample size
    def calculateRMSE(predHome, predAway, goalsHome, goalsAway, round_by = 10, rounded = True, numberSamples=1000):
        #round predicted goals as expected in the prediction challenge
        if(rounded == True):
            predHome = predHome.round()
            predAway = predAway.round()
        scores = (goalsHome-predHome)**2 + (goalsAway-predAway)**2
        #if no data is given
        if len(scores)==0:
            return np.nan, np.nan, np.nan
        #if only one entry is given, CI is just set to this value
        elif len(scores)==1:
            upper = np.sqrt(np.mean(scores))
            lower = np.sqrt(np.mean(scores))
        else:
            distribution = np.sqrt(bootstrap((scores,), np.mean, n_resamples = numberSamples, confidence_level=0.95, method = 'percentile').bootstrap_distribution.astype(float))
            lower = np.quantile(distribution, 0.025)
            upper = np.quantile(distribution, 0.975)
        return np.round(np.sqrt(np.mean(scores)), decimals = round_by), np.round(lower, decimals = round_by), np.round(upper, decimals = round_by), len(predHome)
    
    #uses the predefined table format of the challenge to calculate RMSE directly 
    def calculateRMSEFromData(data, round_by = 10, rounded = True, numberSamples = 1000):
        return MetricsCalculation.calculateRMSE(data['prd_HS'], data['prd_AS'], data['HS'], data['AS'], round_by, rounded, numberSamples)
    
    #method to evaluate the results of an experiment, expects test set  and name of the experiment (relevantColumns is deprecated/not used anymore)
    def evaluateExperiment(dataTest, relevantColumns, experimentName):
        with open('results.txt', 'a') as f:
            f.write(experimentName)
            f.write('\n')
            f.write(str(datetime.datetime.now()))
            f.write('\n')
            f.write("RPS")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRPSFromData(dataTest, round_by = 4)))
            f.write(',')
            f.write("RMSE")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRMSEFromData(dataTest, round_by = 4)))
            f.write(',')
            f.write("RMSE unrounded")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRMSEFromData(dataTest, round_by = 4, rounded = False)))
            f.write(',')
            f.write("Separate evaluation for data-intensive leagues")
            f.write(',')
            dataInt, dataNot = split_dataset_by_data_intensive_leagues(dataTest)
            f.write("RPS dataInt")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRPSFromData(dataInt, round_by = 4)))
            f.write(',')
            f.write("RMSE dataInt")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRMSEFromData(dataInt, round_by = 4)))
            f.write(',')
            f.write("RMSE unrounded dataInt")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRMSEFromData(dataInt, round_by = 4, rounded = False)))
            f.write(',')
            f.write("RPS dataNot")
            f.write(',')                
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRPSFromData(dataNot, round_by = 4)))
            f.write(',')
            f.write("RMSE dataNot")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRMSEFromData(dataNot, round_by = 4)))
            f.write(',')
            f.write("RMSE unrounded dataNot")
            f.write(',')
            f.write(','.join(str(v) for v in MetricsCalculation.calculateRMSEFromData(dataNot, round_by = 4, rounded = False)))
            f.write(',')
            f.write('\n')
    
class MetricsTest:   
    
    def evaluationChallenge():
        data = pd.read_csv('../../input/training_data/evaluation_ITS_spoho.csv')
        mean, lower, upper = MetricsCalculation.calculateRPSFromData(data)
        print("RPS ELO")
        print(mean)
        print(lower)
        print(upper)
        mean, lower, upper = MetricsCalculation.calculateRMSEFromData(data)
        print("RMSE ELO")
        print(mean)
        print(lower)
        print(upper)        
        data = pd.read_csv('../../input/training_data/evaluation_deeplearning.csv')
        mean, lower, upper = MetricsCalculation.calculateRPSFromData(data)
        print("RPS DEEP")
        print(mean)
        print(lower)
        print(upper)
        mean, lower, upper = MetricsCalculation.calculateRMSEFromData(data)
        print("RMSE DEEP")
        print(mean)
        print(lower)
        print(upper) 
    
    def testMetrics():
        #testing RPS calculation
        #test values are based on Constantinou, A. C., & Fenton, N. E. (2012). Solving the problem of inadequate scoring rules for assessing probabilistic football forecast saved_networks. Journal of Quantitative Analysis in Sports, 8(1).
        data=pd.DataFrame()
        data['prd_W'] = np.array([1.0,0.9,0.8,0.5,0.35,0.6,0.6,0.6,0.5,0.55])
        data['prd_D'] = np.array([0.0,0.1,0.1,0.25,0.3,0.3,0.3,0.1,0.45,0.1])
        data['WDL'] = np.array(["W","W","W","W","D","D","W","W","W","W"])
        rps = np.array([0.0,0.005,0.025,0.1562,0.1225,0.1850,0.085,0.125,0.1262,0.1625])
        
        rpsCalculated=[]
        for row in range(0,len(data.index)):
            rpsCalculated.append(MetricsCalculation.calculateRPSFromData(data.iloc[[row]], 50000)[0])
        
        errorsRPS = abs(rps-rpsCalculated)>0.0001
        print(str(len(errorsRPS[errorsRPS == True]))+" errors detected in RPS")
        
        #bootstrap test values are based on SPSS calculation
        inaccuraciesRPS = 0
        if abs(MetricsCalculation.calculateRPSFromData(data, 50000)[0] - 0.09925)>0.001:
            inaccuraciesRPS += 1
        if abs(MetricsCalculation.calculateRPSFromData(data, 50000)[1] - 0.05825)>0.001:
            inaccuraciesRPS += 1
        if abs(MetricsCalculation.calculateRPSFromData(data, 50000)[2] - 0.1370)>0.001:
            inaccuraciesRPS += 1
        print(str(inaccuraciesRPS)+" inaccuracies detected in RPS bootstrapping")
            
        
        data=pd.DataFrame()
        data['prd_HS'] = np.array([0.0,1.0,1.5,2.5])
        data['prd_AS'] = np.array([1.0,2.0,0.5,0.5])
        data['HS'] = np.array([0.0,4.0,2.0,0.0])
        data['AS'] = np.array([1.0,0.0,2.0,1.0])
        rmse = np.array([0.0,3.6056,1.5811,2.5495])
        
        rmseCalculated=[]
        for row in range(0,len(data.index)):
            rmseCalculated.append(MetricsCalculation.calculateRMSEFromData(data.iloc[[row]], rounded=False, numberSamples=50000)[0])
        
        errorsRMSE = abs(rmse-rmseCalculated)>0.0001
        print(str(len(errorsRMSE[errorsRMSE == True]))+" errors detected in RMSE")
        
        #bootstrap test values are based on Excel calculation
        inaccuraciesRMSE = 0
        if abs(MetricsCalculation.calculateRMSEFromData(data, 50000)[0] - 2.34520)>0.001:
            inaccuraciesRMSE += 1
        if abs(MetricsCalculation.calculateRMSEFromData(data, 50000)[1] - 1.11803)>0.001:
            inaccuraciesRMSE += 1
        if abs(MetricsCalculation.calculateRMSEFromData(data, 50000)[2] - 3.22102)>0.001:
            inaccuraciesRMSE += 1
        print(str(inaccuraciesRMSE)+" inaccuracies detected in RMSE bootstrapping")
        
        #test for rounded values
        
        data=pd.DataFrame()
        data['prd_HS'] = np.array([0.3,0.8,1.51,2.51])
        data['prd_AS'] = np.array([1.1,1.6,0.51,0.51])
        data['HS'] = np.array([0.0,4.0,2.0,0.0])
        data['AS'] = np.array([1.0,0.0,2.0,1.0])
        rmse = np.array([0.0,3.6056,1.0,3.0])
        
        rmseCalculated=[]
        for row in range(0,len(data.index)):
            rmseCalculated.append(MetricsCalculation.calculateRMSEFromData(data.iloc[[row]], rounded=True, numberSamples=50000)[0])
        
        errorsRMSE = abs(rmse-rmseCalculated)>0.0001
        print(str(len(errorsRMSE[errorsRMSE == True]))+" errors detected in RMSE")
        
    def testOutput():
        data=pd.DataFrame()
        data['prd_W'] = np.array([1.0,0.9,0.8,0.5])
        data['prd_D'] = np.array([0.0,0.1,0.1,0.25])
        data['WDL'] = np.array(["W","W","W","W"])
        data['prd_HS'] = np.array([0.0,1.0,1.5,2.5])
        data['prd_AS'] = np.array([1.0,2.0,0.5,0.5])
        data['HS'] = np.array([0.0,4.0,2.0,0.0])
        data['AS'] = np.array([1.0,0.0,2.0,1.0])
        
        MetricsCalculation.evaluateExperiment(data ,[], "test")
        
    #evaluationChallenge()
    #testMetrics()
    #testOutput()
    
        