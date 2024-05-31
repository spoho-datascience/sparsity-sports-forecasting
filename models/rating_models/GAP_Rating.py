import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# load training data
dat = pd.read_csv('./input/training_data/Trainset_2022_12_12.csv')
# maybe need for melting; check later

# use parameters from Wheatcroft paper
lam = .4
phi1 = .5
phi2 = .6

# TODO: visualize evolution
# TODO: take care of promotion/relegation
# TODO: tune parameters



####################################################################################################
####################################################################################################
def GAPRating(df, pi_home, pi_away,  lam = .4, phi1 = .5, phi2 = .6):
    ratings_dict = {}
    # assign initial ratings of 0
    for t in df.HT.unique():
        ratings_dict[t] = {'att_home': 0,
                       'def_home': 0,
                       'att_away': 0,
                       'def_away': 0
                       }

    for ix, row in tqdm(df.iterrows(), total=len(df)):
        dat.loc[row.name, 'home_att_rat'] = ratings_dict[row.HT]['att_home']
        dat.loc[row.name, 'home_def_rat'] = ratings_dict[row.HT]['def_home']
        dat.loc[row.name, 'away_att_rat'] = ratings_dict[row.AT]['att_away']
        dat.loc[row.name, 'away_def_rat'] = ratings_dict[row.AT]['def_away']

        diff_home_pi = row[pi_home] - np.mean([ratings_dict[row.HT]['att_home'], ratings_dict[row.AT]['def_away']])
        diff_away_pi = row[pi_away] - np.mean([ratings_dict[row.AT]['att_away'], ratings_dict[row.HT]['def_home']])

        ht_update_home_att = lam * phi1 * diff_home_pi
        ht_update_away_att = lam * (1 - phi2) * diff_home_pi
        ht_update_home_def = lam * phi1 * diff_away_pi
        ht_update_away_def = lam * (1-phi1) * diff_away_pi

        at_update_away_att = lam * phi2 * diff_away_pi
        at_update_home_att = lam * (1 - phi2) * diff_away_pi
        at_update_away_def = lam * phi1 * diff_home_pi
        at_update_home_def = lam * (1-phi1) * diff_home_pi

        ratings_dict[row.HT]['att_home'] = max(ratings_dict[row.HT]['att_home'] + ht_update_home_att, 0)
        ratings_dict[row.HT]['def_home'] = max(ratings_dict[row.HT]['def_home'] + ht_update_home_def, 0)
        ratings_dict[row.HT]['att_away'] = max(ratings_dict[row.HT]['att_away'] + ht_update_away_att, 0)
        ratings_dict[row.HT]['def_away'] = max(ratings_dict[row.HT]['def_away'] + ht_update_away_def, 0)

        ratings_dict[row.AT]['att_home'] = max(ratings_dict[row.AT]['att_home'] + at_update_home_att, 0)
        ratings_dict[row.AT]['def_home'] = max(ratings_dict[row.AT]['def_home'] + at_update_home_def, 0)
        ratings_dict[row.AT]['att_away'] = max(ratings_dict[row.AT]['att_away'] + at_update_away_att, 0)
        ratings_dict[row.AT]['def_away'] = max(ratings_dict[row.AT]['def_away'] + at_update_away_def, 0)

    return df