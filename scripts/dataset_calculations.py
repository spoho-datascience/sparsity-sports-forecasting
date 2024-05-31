import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


path_training = "C:\\Users\\ke6564\\Desktop\\Studium\\Promotion\\PythonProjects\\2023-soccer-prediction\\input\\training_data\\Trainset_2022_12_12.csv"

training_data = pd.read_csv(path_training)

### sanity checks

gd_false = training_data.loc[training_data["GD"] != training_data["HS"] - training_data["AS"]]
wdl_false = training_data.loc[
    (training_data["WDL"] == "W") & (training_data["GD"] <= 0) |
    (training_data["WDL"] == "D") & (training_data["GD"] != 0) |
    (training_data["WDL"] == "L") & (training_data["GD"] >= 0)
    ]

# test lineup permitations

sea = "00-01"
lge = "SCO1"

groups = list(training_data.groupby(["Sea", "Lge"]).groups.keys())

missing_matches = {"sea": [], "lge": [], "teams": [], "exp_games": [], "rec_games": []}
for sea, lge in groups:
    dat = training_data.loc[(training_data["Sea"] == sea) & (training_data["Lge"] == lge)]
    no_games_recorded = len(dat)
    teams = np.unique(dat[["HT", "AT"]].values)
    no_games_expected = len(teams) * (len(teams) - 1)
    if no_games_expected != no_games_recorded:
        missing_matches["sea"].append(sea)
        missing_matches["lge"].append(lge)
        missing_matches["teams"].append(len(teams))
        missing_matches["exp_games"].append(no_games_expected)
        missing_matches["rec_games"].append(no_games_recorded)
missing = pd.DataFrame(missing_matches)




sea, lge = groups[0]

training_df = pd.DataFrame()
for sea, lge in groups:
    sealge = training_data.loc[(training_data["Sea"] == sea) & (training_data["Lge"] == lge)]
    sealge["Date"] = pd.to_datetime(sealge["Date"], format="%d/%m/%Y")
    teams = np.unique(sealge[["HT", "AT"]].values)
    if sealge["Date"].is_monotonic_increasing is False:
        sealge.sort_values("Date", inplace=True, ignore_index=True)


    sealge["pos_point_ht"] = 0
    sealge["pos_point_at"] = 0
    sealge["ach_point_ht"] = 0
    sealge["ach_point_at"] = 0

    sealge_list = sealge.to_dict("records")

    for team in teams:
        possible_points = 0
        achieved_points = 0
        for row in sealge_list:
            if team == row["HT"]:
                possible_points += 3
                row["pos_point_ht"] = possible_points
                if row["WDL"] == "W":
                    achieved_points += 3
                elif row["WDL"] == "D":
                    achieved_points += 1
                row["ach_point_ht"] = achieved_points
            elif team == row["AT"]:
                possible_points += 3
                row["pos_point_at"] = possible_points
                if row["WDL"] == "L":
                    achieved_points += 3
                elif row["WDL"] == "D":
                    achieved_points += 1
                row["ach_point_at"] = achieved_points

    sealge = pd.DataFrame(sealge_list)

    training_df = pd.concat([training_df, sealge])

training_df.to_csv("training_df.csv", index=False)

training_df




season = fbd.loc[fbd["Season"] == "2012"]

teams_fbd = np.unique(fbd[["HomeTeam", "AwayTeam"]].values)
training_teams = list(set(training_data["AT"]))


t_away = [team for team in teams_fbd if team not in training_teams]

x = fbd.loc[fbd["HomeTeam"] == "Sevilla B"]
y = training_data.loc[training_data["Lge"] == "SPA2"]


season_no_team = []
season_with_team = []
for sea, lge in groups:
    sealge = training_data.loc[(training_data["Sea"] == sea) & (training_data["Lge"] == lge)]
    teams = np.unique(sealge[["HT", "AT"]].values)
    matches = [x for x in teams if x in teams_fbd]
    if len(matches) == 0:
        season_no_team.append((sea, lge))
    else:
        season_with_team.append((sea, lge, len(matches)))

a = set([x[1] for x in season_no_team])
np.min([x[2] for x in season_with_team])

