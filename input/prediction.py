import os
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed, cpu_count
from models.rating_models import RatingModels
from models.probability_models import ProbabilityModels
from input.utilities import eval_seasons, match_stats_cols


def add_estimated_betting_odds_col(data):
    # set ELO rating with predefined adjustment factor of 150
    data = RatingModels.ELORating(method="odds").optimiseRating(
        data, 10, 400, [150], [0], [0]
    )
    # use data before first eval season to fit OLR model (only using data with rating)
    first_year = 2000 + int(eval_seasons[0][-2:])
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
    data["est_odds_home"] = 0.95 / data["prd_W"]
    data["est_odds_draw"] = 0.95 / data["prd_D"]
    data["est_odds_away"] = 0.95 / data["prd_L"]
    data["est_odds_home"][data["type"] == "train"] = np.nan
    data["est_odds_draw"][data["type"] == "train"] = np.nan
    data["est_odds_away"][data["type"] == "train"] = np.nan

    # fill NaN
    data["est_odds_home"][data["type"] == "predict"] = data["est_odds_home"][
        data["type"] == "predict"
    ].replace(np.nan, 1)
    data["est_odds_draw"][data["type"] == "predict"] = data["est_odds_draw"][
        data["type"] == "predict"
    ].replace(np.nan, 1)
    data["est_odds_away"][data["type"] == "predict"] = data["est_odds_away"][
        data["type"] == "predict"
    ].replace(np.nan, 1)

    return data


def add_estimated_market_value_and_stats_cols(data):
    # define function to approximate stats and market values
    def get_estimated_cols_per_season(season_data, season):
        # define time limits for approximation
        approx_start_date = (3, 1)
        approx_end_date = (3, 31)

        # get test and approximation set
        season_year = 2000 + int(season[-2:])
        train = season_data[season_data["type"] == "train"]
        test = season_data[season_data["type"] == "predict"]
        approx = train[
            train["iso_date"]
            >= datetime(season_year, approx_start_date[0], approx_start_date[1])
        ]
        approx = approx[
            approx["iso_date"]
            <= datetime(season_year, approx_end_date[0], approx_end_date[1])
        ]

        # create approximation for market values match stats for every team in test set
        test_teams = np.concatenate((test["HT"].unique(), test["AT"].unique()))
        approx_per_team = {team: {} for team in test_teams}
        for team in test_teams:
            # market values (starter total)
            home_totals = approx[approx["HT"] == team]["home_starter_total"].values
            away_totals = approx[approx["AT"] == team]["away_starter_total"].values
            total_approx = np.nanmean(np.hstack((home_totals, away_totals)))
            approx_per_team[team]["starter_total"] = total_approx
            # match stats
            home_shots_for = approx[approx["HT"] == team]["HTS"].values
            home_shots_agn = approx[approx["HT"] == team]["ATS"].values
            home_trgts_for = approx[approx["HT"] == team]["HST"].values
            home_trgts_agn = approx[approx["HT"] == team]["AST"].values
            away_shots_for = approx[approx["AT"] == team]["ATS"].values
            away_shots_agn = approx[approx["AT"] == team]["HTS"].values
            away_trgts_for = approx[approx["AT"] == team]["AST"].values
            away_trgts_agn = approx[approx["AT"] == team]["HST"].values

            shots_for = np.nanmean(np.hstack((home_shots_for, away_shots_for)))
            shots_agn = np.nanmean(np.hstack((home_shots_agn, away_shots_agn)))
            trgts_for = np.nanmean(np.hstack((home_trgts_for, away_trgts_for)))
            trgts_agn = np.nanmean(np.hstack((home_trgts_agn, away_trgts_agn)))

            approx_per_team[team]["shots_for"] = shots_for
            approx_per_team[team]["shots_agn"] = shots_agn
            approx_per_team[team]["trgts_for"] = trgts_for
            approx_per_team[team]["trgts_agn"] = trgts_agn

        return approx_per_team

    # split data per season
    data_per_season = []
    for season in eval_seasons:
        data_per_season.append(data[data["Sea"] == season])

    # compute approximations per seasons in parallel
    all_approx_per_team = Parallel(n_jobs=cpu_count(), verbose=100)(
        delayed(get_estimated_cols_per_season)(data_per_season[n], season)
        for n, season in enumerate(eval_seasons)
    )

    # initialize new estimated columns
    data["est_home_starter_total"] = np.nan
    data["est_away_starter_total"] = np.nan
    for stat in match_stats_cols:
        data[f"est_{stat}"] = np.nan

    # add columns with estimated values to test set
    home_starter_totals = {}
    away_starter_totals = {}
    HTS = {}
    HST = {}
    ATS = {}
    AST = {}
    for n, season in enumerate(eval_seasons):
        test = data_per_season[n][data_per_season[n]["type"] == "predict"]

        # market values
        test["est_home_starter_total"] = [
            all_approx_per_team[n][team]["starter_total"] for team in test["HT"]
        ]
        test["est_away_starter_total"] = [
            all_approx_per_team[n][team]["starter_total"] for team in test["AT"]
        ]

        # stats as mean of shots for and shots against
        home_teams = test["HT"].values
        away_teams = test["AT"].values
        test["est_HTS"] = [
            (
                all_approx_per_team[n][home_teams[i]]["shots_for"]
                + all_approx_per_team[n][away_teams[i]]["shots_agn"]
            )
            / 2
            for i in range(len(test))
        ]
        test["est_HST"] = [
            (
                all_approx_per_team[n][home_teams[i]]["trgts_for"]
                + all_approx_per_team[n][away_teams[i]]["trgts_agn"]
            )
            / 2
            for i in range(len(test))
        ]
        test["est_ATS"] = [
            (
                all_approx_per_team[n][away_teams[i]]["shots_for"]
                + all_approx_per_team[n][home_teams[i]]["shots_agn"]
            )
            / 2
            for i in range(len(test))
        ]
        test["est_AST"] = [
            (
                all_approx_per_team[n][away_teams[i]]["trgts_for"]
                + all_approx_per_team[n][home_teams[i]]["trgts_agn"]
            )
            / 2
            for i in range(len(test))
        ]

        # set test columns with NaN to global mean
        test["est_home_starter_total"] = test["est_home_starter_total"].replace(
            np.nan, np.nanmean(data["home_starter_total"])
        )
        test["est_away_starter_total"] = test["est_away_starter_total"].replace(
            np.nan, np.nanmean(data["away_starter_total"])
        )

        test["est_HTS"] = test["HTS"].replace(np.nan, np.nanmean(data["HTS"]))
        test["est_HST"] = test["HST"].replace(np.nan, np.nanmean(data["HST"]))
        test["est_ATS"] = test["ATS"].replace(np.nan, np.nanmean(data["ATS"]))
        test["est_AST"] = test["AST"].replace(np.nan, np.nanmean(data["AST"]))

        # update global container
        unique_ids = test["unique_id"].values
        home_starter_totals.update(
            {
                uID: test["est_home_starter_total"].values[i]
                for i, uID in enumerate(unique_ids)
            }
        )
        away_starter_totals.update(
            {
                uID: test["est_away_starter_total"].values[i]
                for i, uID in enumerate(unique_ids)
            }
        )
        HTS.update({uID: test["est_HTS"].values[i] for i, uID in enumerate(unique_ids)})
        HST.update({uID: test["est_HST"].values[i] for i, uID in enumerate(unique_ids)})
        ATS.update({uID: test["est_ATS"].values[i] for i, uID in enumerate(unique_ids)})
        AST.update({uID: test["est_AST"].values[i] for i, uID in enumerate(unique_ids)})

    # update global Data Frame
    data["est_home_starter_total"] = [
        home_starter_totals[uID] if uID in home_starter_totals else pd.NA
        for uID in data["unique_id"]
    ]
    data["est_away_starter_total"] = [
        away_starter_totals[uID] if uID in away_starter_totals else pd.NA
        for uID in data["unique_id"]
    ]
    data["est_HTS"] = [HTS[uID] if uID in HTS else pd.NA for uID in data["unique_id"]]
    data["est_HST"] = [HST[uID] if uID in HST else pd.NA for uID in data["unique_id"]]
    data["est_ATS"] = [ATS[uID] if uID in ATS else pd.NA for uID in data["unique_id"]]
    data["est_AST"] = [AST[uID] if uID in AST else pd.NA for uID in data["unique_id"]]

    return data
