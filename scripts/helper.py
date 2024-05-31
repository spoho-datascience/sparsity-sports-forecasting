import editdistance
from typing import Tuple
import warnings


import numpy as np
import pandas as pd


def get_team_map(
        season_train: pd.DataFrame,
        season_ext: pd.DataFrame,
        ht_ext="HomeTeam",
        at_ext="AwayTeam"
) -> Tuple:
    """
    Creates a team map for one season and league between an external data source and the
    training data. A team map is a dictionary with the team names in the training data
    as keys and the team name in the external data as the values.
    """

    if any(len(s) == 0 for s in (season_train, season_ext)):
        warnings.warn("This Season is empty!")
        return {}, "Empty"
    if len(season_train) != len(season_ext):
        warnings.warn("Length of DataFrames are not equal!")


    teams_fbd = list(np.unique(season_ext[[ht_ext, at_ext]].values))
    teams_train = list(np.unique(season_train[["HT", "AT"]].values))

    team_map = {
        train_name: train_name for train_name in teams_train if train_name in teams_fbd
    }

    # season_train["ht_fbd"] = season_train["HT"].replace({team: team_map.get(team, None) for team in teams_train})
    # season_train["at_fbd"] = season_train["AT"].replace({team: team_map.get(team, None) for team in teams_train})
    season_train['Date'] = pd.to_datetime(season_train['Date']).dt.normalize()
    season_ext['Date'] = pd.to_datetime(season_ext['Date']).dt.normalize()
    season_train_list = season_train.to_dict(orient="records")
    season_fbd_list = season_ext.to_dict(orient="records")

    converged = "No matches"

    for match in season_train_list:
        ht = team_map.get(match["HT"], None)
        at = team_map.get(match["AT"], None)
        date = match["Date"]
        # ht = match["ht_fbd"]
        # at = match["at_fbd"]

        if any(t for t in (ht, at)):
            converged = "Partially"
            if ht is not None:
                ats = [x[at_ext] for x in season_fbd_list if (x[ht_ext] == ht) & (x["Date"] == date)]
                if len(ats) > 0:
                    if ats[0] not in team_map.values():
                        team_map[match["AT"]] = ats[0]
                if len(ats) > 1:
                    warnings.warn(f"Team {ht} played twice on date {date}! Tchuligom..")
            else:
                hts = [x[ht_ext] for x in season_fbd_list if (x[at_ext] == at) & (x["Date"] == date)]
                if len(hts) > 0:
                    if hts[0] not in team_map.values():
                        team_map[match["HT"]] = hts[0]
                if len(hts) > 1:
                    warnings.warn(f"Team {at} played twice on date {date}! Tchuligom..")

            if len(team_map) == len(teams_train):
                converged = "Fully"
                break

    return team_map, converged

def get_team_map_by_distance(
    training_teams: list,
    external_teams: list
):
    """
    Approximates the team map between two lists of teams by string edit distance.
    """
    team_map = {}
    for training_team_name in training_teams:
        closest = None
        closest_distance = float("inf")
        for external_team_name in external_teams:
            distance = editdistance.eval(training_team_name, external_team_name)
            if distance < closest_distance:
                closest = external_team_name
                closest_distance = distance
        team_map[training_team_name] = closest
    return team_map