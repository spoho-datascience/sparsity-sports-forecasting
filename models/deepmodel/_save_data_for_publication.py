import os
import pandas as pd
import numpy as np
import collections
from datetime import datetime
from models.deepmodel.utilities import eval_seasons, stats_cols
from models.rating_models import RatingModels
from models.probability_models import ProbabilityModels
from input.training_data import DataCollection
from joblib import Parallel, delayed, cpu_count


# load data
data = pd.read_csv("F:\\work\\data\\2023-soccer-prediction\\training_dataset.csv")
# data = pd.read_csv("../../input/training_data/training_dataset.csv")
data["iso_date"] = pd.to_datetime(data["iso_date"])
data = data.sort_values(by="iso_date")

# preprocessing: remove duplicates in dataset
for uID in data["unique_id"]:
    if len(data[data["unique_id"] == uID]) > 1:
        print(f"Dropping duplicates of match {uID}")
        data = data.drop(data[data["unique_id"] == uID][1:].index)

# preprocessing: remove entries without information
for meta_col in ["unique_id", "iso_date", "Sea", "Lge", "HT", "AT", "WDL"]:
    data = data.drop(data[pd.isna(data[meta_col])].index)

# preprocessing: replace market value of zero with NaN
data[data["home_starter_total"] == 0] = pd.NA
data[data["away_starter_total"] == 0] = pd.NA

# preprocessing: recoding of results
data["result"] = data["WDL"]
data["result"] = data["result"].replace("W", 2)
data["result"] = data["result"].replace("D", 1)
data["result"] = data["result"].replace("L", 0)

# preprocessing: correct doubled team names
data["HT"][
    (data["HT"] == "FC Barcelona") & (data["Lge"] == "ECU1")
] = "FC Barcelona ECU"
data["AT"][
    (data["AT"] == "FC Barcelona") & (data["Lge"] == "ECU1")
] = "FC Barcelona ECU"
data["HT"][(data["HT"] == "Everton") & (data["Lge"] == "CHL1")] = "Everton CHL"
data["AT"][(data["AT"] == "Everton") & (data["Lge"] == "CHL1")] = "Everton CHL"

# preprocessing: use log of market value
data["home_starter_total"] = np.log(data["home_starter_total"])
data["away_starter_total"] = np.log(data["away_starter_total"])

# preprocessing: compute average of available odds
data["avg_home_odds"] = data[["avg_home_odds", "odds_home"]].nanmean(axis=1)
data["avg_draw_odds"] = data[["avg_draw_odds", "odds_draw"]].nanmean(axis=1)
data["avg_away_odds"] = data[["avg_away_odds", "odds_away"]].nanmean(axis=1)

# preprocessing: compute probabilites from betting odds
overround = (
    1 / data["avg_home_odds"] + 1 / data["avg_draw_odds"] + 1 / data["avg_away_odds"]
)
data["pred_odds_home"] = 1 / data["avg_home_odds"] / overround
data["pred_odds_draw"] = 1 / data["avg_draw_odds"] / overround
data["pred_odds_away"] = 1 / data["avg_away_odds"] / overround


# create day-of-year-feature that describes seasonality (metadata)
data["DAY"] = [date.timetuple().tm_yday for date in data["iso_date"]]


# create function to compute rest days (metadata) in parallel
def create_team_specific_links_unique_id_to_restdays(team):
    team_matches = data[(data["HT"] == team) + (data["AT"] == team)]
    team_links_id_to_restdays = {
        unique_id: {} for unique_id in team_matches["unique_id"]
    }
    # compute rest days for remaining matches
    for match_idx in range(len(team_matches)):
        unique_id = team_matches.iloc[match_idx]["unique_id"]
        if match_idx == 0:
            rest_days = -1  # first match ever played by the team in the dataset
        else:
            rest_days = (
                team_matches.iloc[match_idx]["iso_date"]
                - team_matches.iloc[match_idx - 1]["iso_date"]
            ).days
            rest_days = np.clip(rest_days, 0, 30)  # after 30 we expect no advantage

        # assign to home or away depending on role of current team
        if team == team_matches.iloc[match_idx]["HT"]:
            team_links_id_to_restdays[unique_id]["HT"] = rest_days
        elif team == team_matches.iloc[match_idx]["AT"]:
            team_links_id_to_restdays[unique_id]["AT"] = rest_days
        else:
            assert False, "Team not contained in either HT or AT"
    return team_links_id_to_restdays


# create links from ID to restdays for every team
all_teams = np.concatenate((data["HT"].unique(), data["AT"].unique()))
all_links_id_to_restdays = Parallel(n_jobs=cpu_count(), verbose=100)(
    delayed(create_team_specific_links_unique_id_to_restdays)(team)
    for team in all_teams
)


# stack links into complete dict with restdays and add to DataFrame
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


links_id_to_restdays = {}
for links in all_links_id_to_restdays:
    links_id_to_restdays = update(links_id_to_restdays, links)
data["HRD"] = [links_id_to_restdays[unique_id]["HT"] for unique_id in data["unique_id"]]
data["ARD"] = [links_id_to_restdays[unique_id]["AT"] for unique_id in data["unique_id"]]

# create train and test set (similar to prediction challenge)
train_sets = {}  # TRAIN: ALL AVAILABLE MATCHES UNTIL DAY X FOR SEASON
test_sets = {}  # TEST: ALL MATCHES FROM DAY X TO DAY Y FOR SEASON
start_date = (4, 1)  # month and day when the test section starts
end_date = (5, 31)  # month and day when the test section ends

# flag prediction and training intervals to ensure ELO is not updated during prediction
first_year = 2000 + int(eval_seasons[0][-2:])
data["type"] = "train"
data["type"][
    (data["iso_date"].dt.year >= first_year)
    & (data["iso_date"].dt.month > 3)
    & (data["iso_date"].dt.month < 6)
] = "predict"

# set ELO rating with predefined adjustment factor of 150
data = RatingModels.ELORating(method="odds").optimiseRating(
    data, 10, 400, [150], [0], [0]
)
# use data before first eval season to fit OLR model while only using data with rating
dataIS = data[
    (data["iso_date"].dt.year < first_year)
    & (~data["Rat_H_odds"].isna())
    & (~data["Rat_A_odds"].isna())
]
# calculate predicted odds and add to prediction sets (set nan to a rating of 1000)
data["Rat_H_odds"][data["Rat_H_odds"].isna()] = 1000
data["Rat_A_odds"][data["Rat_A_odds"].isna()] = 1000
os.makedirs("../../input/training_data/", exist_ok=True)
data.to_csv("../../input/training_data/DataHenrik.csv")
data = ProbabilityModels.OrderedLogisticRegression().calculateProbabilities(
    dataIS["Rat_H_odds"] - dataIS["Rat_A_odds"],
    dataIS["result"],
    (data["Rat_H_odds"] - data["Rat_A_odds"]).tolist(),
    data,
)
data["est_odds_W"] = 0.95 / data["prd_W"]
data["est_odds_D"] = 0.95 / data["prd_D"]
data["est_odds_L"] = 0.95 / data["prd_L"]
data["est_odds_W"][data["type"] == "train"] = np.nan
data["est_odds_D"][data["type"] == "train"] = np.nan
data["est_odds_L"][data["type"] == "train"] = np.nan

# data.to_csv('../../input/training_data/DataInclPredictedOdds.csv')
# create train test split
for season in eval_seasons:
    season_year = 2000 + int(season[-2:])

    # case for first season -> integrate all available data
    if season == eval_seasons[0]:
        train = data[
            data["iso_date"] < datetime(season_year, start_date[0], start_date[1])
        ]
    else:  # in other seasons, start with end of test set for last season
        train = data[
            data["iso_date"] > datetime(season_year - 1, end_date[0], end_date[1])
        ]
        train = train[
            train["iso_date"] < datetime(season_year, start_date[0], start_date[1])
        ]

    test = data[data["iso_date"] >= datetime(season_year, start_date[0], start_date[1])]
    test = test[test["iso_date"] <= datetime(season_year, end_date[0], end_date[1])]

    train_sets[season] = train
    test_sets[season] = test


# for test sets: define function to approximate stats and market values
def get_approximated_test_sets_for_season(season):
    # define time limits for approximation
    approx_start_date = (3, 1)
    approx_end_date = (3, 31)

    # get test and approximation set
    season_year = 2000 + int(season[-2:])
    train = train_sets[season]
    test = test_sets[season]
    approx = train[
        train["iso_date"]
        < datetime(season_year, approx_start_date[0], approx_end_date[1])
    ]

    # create approximation for market values and match stats for every team in test set
    test_teams = np.concatenate((test["HT"].unique(), test["AT"].unique()))
    approx_per_team = {team: {} for team in test_teams}
    for team in test_teams:
        # market values (starter total)
        home_totals = approx[approx["HT"] == team]["home_starter_total"].values
        away_totals = approx[approx["AT"] == team]["away_starter_total"].values
        total_approx = np.nanmean(np.hstack((home_totals, away_totals)))
        approx_per_team[team]["starter_total"] = total_approx
        # match stats
        for stat in stats_cols:
            home_stats = approx[approx["HT"] == team][stat].values
            away_stats = approx[approx["AT"] == team][stat].values
            stats_approx = np.nanmean(np.hstack((home_stats, away_stats)))
            approx_per_team[team][stat] = stats_approx
    return approx_per_team


# compute approximations in parallel
all_approx_per_team = Parallel(n_jobs=cpu_count(), verbose=100)(
    delayed(get_approximated_test_sets_for_season)(season) for season in test_sets
)

# create estimated feature columns into train (always NaN) and test sets
for n, season in enumerate(train_sets):
    train = train_sets[season]
    train["est_home_starter_total"] = pd.NA
    test["est_away_starter_total"] = pd.NA
    for stat in stats_cols:
        test[f"est_{stat}"] = pd.NA
        test[f"est_{stat}"] = pd.NA

for n, season in enumerate(test_sets):
    test = test_sets[season]
    test["est_home_starter_total"] = [
        all_approx_per_team[n][team]["starter_total"] for team in test["HT"]
    ]
    test["est_away_starter_total"] = [
        all_approx_per_team[n][team]["starter_total"] for team in test["AT"]
    ]
    for stat in stats_cols:
        test[f"est_{stat}"] = [
            all_approx_per_team[n][team][stat] for team in test["HT"]
        ]
        test[f"est_{stat}"] = [
            all_approx_per_team[n][team][stat] for team in test["AT"]
        ]

# only include the following relevant columns in the final dataset
relevant_cols = [
    "HS", "AS", "GD", "WDL", "result",  # labels
    "unique_id", "iso_date", "Sea", "Lge", "HT", "AT", "DAY", "HRD", "ARD"  # meta
    "avg_home_odds", "avg_draw_odds", "avg_away_odds",  # features: betting odds
    "home_starter_total", "away_starter_total",  # features: market values
    "HTS", "ATS", "HST", "AST"  # features: stats
    "est_home_starter_total", "est_away_starter_total"  # estimation: market values
    "est_odds_home", "est_odds_draw", "est_odds_away",  # estimation: betting odds
    "est_HTS", "est_ATS", "est_HST", "est_AST"  # estimation: stats
]
for season in train_sets:
    concise_data = pd.DataFrame(columns=relevant_cols)
    for col in relevant_cols:
        concise_data[col] = train_sets[season][col]
    train_sets[season] = concise_data
for season in test_sets:
    concise_data = pd.DataFrame(columns=relevant_cols)
    for col in relevant_cols:
        concise_data[col] = test_sets[season][col]
    test_sets[season] = concise_data

# save train and test sets to file
train_path = "../../train"
test_path = "../../test"
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
for season in eval_seasons:
    train_sets[season].to_csv(os.path.join(train_path, f"{season}.csv"))
    test_sets[season].to_csv(os.path.join(test_path, f"{season}.csv"))
