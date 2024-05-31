import os
import json
import pandas as pd
from models.deepmodel.leagues import get_change_of_league_dict, get_teams_in_league_dict
from input.preprocessing import (
    # remove_duplicate_unique_ids,
    remove_rows_with_empty_label_cols,
    remove_rows_with_empty_meta_cols,
    replace_market_value_zero_with_nan,
    rename_duplicate_teams,
    compute_market_value_log,
    # compute_odds_average,
)

# load data
# data_path = os.path.join("input", "training_data")
data_path = os.path.join(os.getcwd(), "\\training_data")
# data = pd.read_csv(os.path.join(data_path, "full_dataset_from_DB.csv"))
data = pd.read_csv("input\\full_dataset_from_DB.csv")

# create date and sort from earliest to latest
data["iso_date"] = pd.to_datetime(data["iso_date"])
data = data.sort_values(by="iso_date")

# create DataFrame with all teams in dataset
teams_data = pd.DataFrame()
teams_data["HT"] = data["HT"]
teams_data["AT"] = data["AT"]

# preprocessing
# data = remove_duplicate_unique_ids(data)
data = rename_duplicate_teams(data)
data = remove_rows_with_empty_label_cols(data)
data = remove_rows_with_empty_meta_cols(data)

# create data structure of teams in league
teams_in_league = get_teams_in_league_dict(data)

# save teams information to file
meta_path = "../../meta"
os.makedirs(meta_path, exist_ok=True)
teams_data.to_csv(os.path.join(meta_path, f"teams.csv"))

# create dict of teams changing leagues between seasons (due to relegation or promotion)
meta_path = "../../meta"
os.makedirs(meta_path, exist_ok=True)
with open(os.path.join(meta_path, "teams_in_league.json"), "w+") as f:
    json.dump(teams_in_league, f)

# change_of_league_dict = get_change_of_league_dict(data)
# test compare change of league dict with cols
# new_col_dict = {}
# # loop over seasons
# for season in data["Sea"].unique():
#     # get leagues of all teams that played the last season and reset playing date
#     new_col_dict[season] = {league: [] for league in data["Lge"].unique()}
# for i, match in data.iterrows():
#     if match["HLC"] in [2, -2]:
#         new_col_dict[match["Sea"]][match["Lge"]].append(match["HT"])
#     if match["ALC"] in [2, -2]:
#         new_col_dict[match["Sea"]][match["Lge"]].append(match["AT"])
#
# new_change_dict = {season: {}for season in data["Sea"].unique()}
# for season in change_of_league_dict:
#     for league in change_of_league_dict[season]:
#         new_change_dict[season].update({
#                 league: [
#                     team
#                     for team in change_of_league_dict[season][league]
#                     if change_of_league_dict[season][league][team] == False
#                 ]
#         })
#
#
# for season in change_of_league_dict:
#     for league in change_of_league_dict[season]:
#         if any([team not in new_col_dict[season][league] for team in new_change_dict[season][league]]):
#             print(f"Mismatch for {season} and {league}")
#             print("Henrik")
#             print(new_change_dict[season][league])
#             print("Fabian")
#             print(new_col_dict[season][league])
#             print("\n")
