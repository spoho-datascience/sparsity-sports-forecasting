import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import scripts.train_test_split as tts
from models.rating_models.RatingModels import ELORating, piRating
from models.evaluation.Metrics import MetricsCalculation
from models.probability_models.PoissonModels import score_grid, plot_score_grid

dat = pd.read_csv('./input/training_data/Trainset_2022_12_12.csv')
tts.create_date_columns(dat)

########### CALCULATE ELO RATING
ratingHome, ratingAway = ELORating(method = "goals").calculateRating(dat, 10, 400, 20, 10, 1)
dat['elo_H']=ratingHome
dat['elo_A']=ratingAway
############################################

########## CALCULATE PI RATING
ratingHomeTeam_Home, ratingHomeTeam_Away, ratingHomeTeam, ratingAwayTeam_Home, ratingAwayTeam_Away, ratingAwayTeam = piRating().calculateRating(
    dat, 0.3, 0.1)
# ratingHome, ratingAway = ELORating(method = "odds").calculateRating(data, 10, 400, 200, 10, 1)
dat['pi_H_H'] = ratingHomeTeam_Home
dat['pi_H_A'] = ratingHomeTeam_Away
dat['pi_H'] = ratingHomeTeam
dat['pi_A_H'] = ratingAwayTeam_Home
dat['pi_A_A'] = ratingAwayTeam_Away
dat['pi_A'] = ratingAwayTeam
#######################################

########## CALCULATE GAP RATING
dat = GAPRating(dat, pi_home='HS', pi_away='AS')

## filter dataset to start with season 10/11
dat = dat[dat['start_year'] > 2010]

# get dicts with train and test sets for each season, respectively
dict_train, dict_test = tts.return_train_test_dfs(dat)

from pygam import PoissonGAM
# TODO: try out including all ratings of GAP in prediction
X_elo = dict_train[2022][['elo_H', 'elo_A']].values
X_gap = dict_train[2022][['home_att_rat', 'away_def_rat', 'home_def_rat', 'away_att_rat']].values

X_gap_home = dict_train[2022][['home_att_rat', 'away_def_rat']].values
X_gap_away = dict_train[2022][['home_def_rat', 'away_att_rat']].values
y_home = dict_train[2022]['HS'].values
y_away = dict_train[2022]['AS'].values

gam_elo_home = PoissonGAM().gridsearch(X_elo, y_home)
gam_elo_away = PoissonGAM().gridsearch(X_elo, y_away)
gam_gap_home = PoissonGAM().gridsearch(X_gap, y_home)
gam_gap_away = PoissonGAM().gridsearch(X_gap, y_away)




########### PREDICT GOALS WITH POISSON AND LINEAR REGRESSION

from statsmodels.miscmodels.ordinal_model import OrderedModel



y_train_outcome = dict_train[2022]['WDL'].replace('W', 2).replace('D', 1).replace('L', 0)

# train Ordered regression to predict outcome
ordered_regression_elo = OrderedModel(y_train_outcome, X_train_elo).fit(method="bfgs")
ordered_regression_gap = OrderedModel(y_train_outcome, pd.concat([X_train_gap_home, X_train_gap_away],
                                                                 axis=1)).fit(method="bfgs")

#################### Predict goals and outcomes in every fold
ls_test = []
for season in np.arange(2011, 2022):
    X_test_elo = dict_test[season][['elo_H', 'elo_A']]
    X_test_gap = dict_test[season][['home_att_rat', 'away_def_rat', 'home_def_rat', 'away_att_rat']].values
    X_test_gap_home = dict_test[season][['home_att_rat', 'away_def_rat']]
    X_test_gap_away = dict_test[season][['away_att_rat', 'home_def_rat']]

    pred_HS_gam_elo = gam_elo_home.predict(X_test_elo)
    pred_AS_gam_elo = gam_elo_away.predict(X_test_elo)

    pred_HS_gam_gap = gam_gap_home.predict(X_test_gap)
    pred_AS_gam_gap = gam_gap_away.predict(X_test_gap)

    test_df['pred_HS_gam_elo'] = pred_HS_poi_elo
    test_df['pred_AS_gam_elo'] = pred_AS_poi_elo
    test_df['pred_HS_gam_gap'] = pred_HS_poi_gap
    test_df['pred_AS_gam_gap'] = pred_AS_poi_gap
    #test_df = pd.concat([test_df, pred_outcome_ordreg_gap, pred_outcome_ordreg_elo], axis=1)

    ls_test.append(test_df)

df_predictions = pd.concat(ls_test)
#########################################################################################
# TODO
df_predictions['RMSE'] =

# TODO: evaluate per season per league
# TODO: GAM for predicting Poisson goals
# TODO: GAM for predicting ordered outcome

############# CALCULATE NUMBER OF GOALS AS THE MOST LIKELY RESULT BASED ON DOUBLE POISSON DISTRIBUTION
# (AND GET PROBABILITIES FOR OUTCOME FROM SAME APPROACH)
from tqdm import tqdm
ls_outcomes = []
for ix, row in tqdm(df_predictions.iterrows(), total=len(df_predictions)):
    outcomes = score_grid(row.pred_HS_poi, row.pred_AS_poi, return_scorematrix=False)
    outcomes.index = ix
    ls_outcomes.append(outcomes)

pred_outcomes = pd.concat(ls_outcomes)
pred_outcomes = pred_outcomes.reset_index(drop=True).set_index(df_predictions.index)

df_predictions = pd.concat([df_predictions, pred_outcomes], axis=1)

# CALCULATE RMSE FOR ALL THREE APPROACHES
rmse_linreg = MetricsCalculation.calculateRMSE(
    df_predictions.pred_HS_lin, df_predictions.pred_AS_lin,
    df_predictions.HS, df_predictions.AS)

rmse_poireg_elo = MetricsCalculation.calculateRMSE(
    df_predictions.pred_HS_poi_elo, df_predictions.pred_AS_poi_elo,
    df_predictions.HS, df_predictions.AS)

rmse_poireg_gap = MetricsCalculation.calculateRMSE(
    df_predictions.pred_HS_poi_gap, df_predictions.pred_AS_poi_gap,
    df_predictions.HS, df_predictions.AS)

rmse_gam_elo = MetricsCalculation.calculateRMSE(
    df_predictions.pred_HS_gam_elo, df_predictions.pred_AS_gam_elo,
    df_predictions.HS, df_predictions.AS)

rmse_gam_gap = MetricsCalculation.calculateRMSE(
    df_predictions.pred_HS_gam_gap, df_predictions.pred_AS_gam_gap,
    df_predictions.HS, df_predictions.AS)

rmse_likelyscore = MetricsCalculation.calculateRMSE(
    df_predictions.likely_home_score, df_predictions.likely_away_score,
    df_predictions.HS, df_predictions.AS)

############ PREDICT MATCH OUTCOME WITH ORDERED REGRESSION
from statsmodels.miscmodels.ordinal_model import OrderedModel
ls_test_ordreg = []
for season in np.arange(2011, 2022):
    X_train = dict_train[season][['elo_H', 'elo_A']]
    y_train = dict_train[season]['WDL'].replace('W', 2).replace('D', 1).replace('L', 0)
    X_test  = dict_test[season][['elo_H', 'elo_A']]
    y_test  = dict_test[season]['WDL'].replace('W', 2).replace('D', 1).replace('L', 0)

    ordered_regression = OrderedModel(y_train, X_train).fit(method="bfgs")

    pred_outcome_ordreg = ordered_regression.predict(X_test)
    pred_outcome_ordreg.columns = ['pred_L', 'pred_D', 'pred_W']


    test_df = dict_test[season].copy()
    test_df = pd.concat([test_df, pred_outcome_ordreg], axis=1)

    ls_test_ordreg.append(test_df)

df_predictions_ordreg = pd.concat(ls_test_ordreg)


####### CALCULATE RPS FOR ORDERED REGRESSION AND DOUBLE POISSON BASED PREDICTION
# Double Poisson
MetricsCalculation.calculateRPS(df_predictions.pred_home, df_predictions.pred_draw,
                                df_predictions.WDL == "W", df_predictions.WDL == "D")
# Ordered Regression
MetricsCalculation.calculateRPS(df_predictions.pred_W_elo, df_predictions.pred_D_elo,
                                df_predictions.WDL == "W", df_predictions.WDL == "D")

# Ordered Regression GAP
MetricsCalculation.calculateRPS(df_predictions.pred_W_gap, df_predictions.pred_D_gap,
                                df_predictions.WDL == "W", df_predictions.WDL == "D")

