import pandas as pd
import numpy as np
import os
import torch
import json


def get_league_dict(season_data):
    last_season_teams = np.concatenate(
        (
            season_data["HT"].unique(),
            season_data["AT"].unique(),
        )
    )
    last_season_leagues = {}
    for team in last_season_teams:
        lgs = pd.concat(
            (
                season_data[season_data["HT"] == team],
                season_data[season_data["AT"] == team],
            )
        )["Lge"].unique()
        league = lgs[0]
        last_season_leagues[team] = league

    return last_season_leagues


def get_matchday_ids(data):
    # initialize container
    season_matchday_ids = {season: {} for season in data["Sea"].unique()}

    # loop over seasons
    for season in data["Sea"].unique():
        # retrieve season data
        season_data = data[data["Sea"] == season]

        # loop over leagues
        season_matchday_ids[season] = {
            league: {} for league in season_data["Lge"].unique()
        }
        for league in season_data["Lge"].unique():
            league_data = season_data[season_data["Lge"] == league]
            league_teams = pd.concat((league_data["HT"], league_data["AT"]))
            matchups = np.zeros((len(league_data), 2), dtype=object)
            matchups[:, 0] = league_data["HT"].values
            matchups[:, 1] = league_data["AT"].values
            matchday_ids = {}
            num_teams_played = {team: 0 for team in league_teams.unique()}
            for idx in range(len(league_data)):
                num_teams_played[matchups[idx, 0]] += 1
                num_teams_played[matchups[idx, 1]] += 1
                matchday = np.maximum(
                    num_teams_played[matchups[idx, 0]],
                    num_teams_played[matchups[idx, 1]],
                )
                if matchday in matchday_ids:
                    matchday_ids[matchday].append(league_data["unique_id"].iloc[idx])
                else:
                    matchday_ids[matchday] = [league_data["unique_id"].iloc[idx]]

            season_matchday_ids[season][league] = matchday_ids

    # concatenate different seasons to create monotonically ascending match days
    global_league_matchday_counters = {}
    global_matchday_ids = {league: {} for league in data["Lge"].unique()}
    for season in season_matchday_ids:
        for league in season_matchday_ids[season]:
            # initialize league matchday counter if it does not exist
            if league not in global_league_matchday_counters:
                global_league_matchday_counters[league] = 0

            # append matchdays to global container
            for matchday in season_matchday_ids[season][league]:
                global_matchday_ids[league].update(
                    {
                        global_league_matchday_counters[league]: season_matchday_ids[
                            season
                        ][league][matchday]
                    }
                )
                global_league_matchday_counters[league] += 1

    return global_matchday_ids


def get_change_of_league_dict(data):
    change_of_league_dict = {}

    # loop over seasons
    season_data = None
    last_season_leagues = None
    for season in data["Sea"].unique():
        # get leagues of all teams that played the last season and reset playing date
        if season_data is not None:
            last_season_data = season_data
            last_season_leagues = get_league_dict(last_season_data)
            change_of_league_dict[season] = {}

        # retrieve season train and test data
        season_data = data[data["Sea"] == season]
        if last_season_leagues is not None:
            curr_season_leagues = get_league_dict(season_data)
            for team in curr_season_leagues:
                # find relegating/promoting teams and reset hidden/cell states
                if team not in last_season_leagues:  # new team this season
                    stay_league = False
                elif curr_season_leagues[team] != last_season_leagues[team]:
                    stay_league = False
                else:  # same league
                    stay_league = True

                team_league = curr_season_leagues[team]
                if team_league in change_of_league_dict[season]:
                    change_of_league_dict[season][team_league][team] = stay_league
                else:
                    change_of_league_dict[season][team_league] = {team: stay_league}

    # remove entries with no change teams
    for season in change_of_league_dict:
        for league in change_of_league_dict[season]:
            all_changes = [
                change_of_league_dict[season][team]
                for team in change_of_league_dict[season]
            ]
            if not any(all_changes):
                change_of_league_dict[season].pop(league)

    return change_of_league_dict


def update_team_states_for_season(season, team_states, team_dic):
    # get data structure for teams promoting/relegating before season
    meta_path = "F:\\work\\data\\2023-soccer-prediction\\meta"
    with open(os.path.join(meta_path, "change_of_league_dict.json"), "r") as file:
        change_of_league_dict = json.load(file)

    if season != "00-01":  # skip first season
        # find promoting/relegating teams
        for league in change_of_league_dict[season]:
            league_hidden = torch.vstack(
                (
                    [
                        team_states["hidden"][team_dic[team]]
                        for team in change_of_league_dict[season][league]
                    ]
                )
            )
            league_cell = torch.vstack(
                (
                    [
                        team_states["cell"][team_dic[team]]
                        for team in change_of_league_dict[season][league]
                    ]
                )
            )
            # set cell and hidden states of new teams to league average
            for team in change_of_league_dict[season][league]:
                if change_of_league_dict[season][league][team]:
                    team_idx = team_dic[team]
                    team_states["hidden"][team_idx] = torch.mean(league_hidden, dim=0)
                    team_states["cell"][team_idx] = torch.mean(league_cell, dim=0)

    return team_states


def get_teams_in_league_dict(data):
    # intialize dict
    teams_in_league = {}

    # loop over seasons
    for season in data["Sea"].unique():
        teams_in_league[season] = {}
        season_data = data[data["Sea"] == season]
        for league in season_data["Lge"].unique():
            league_data = season_data[season_data["Lge"] == league]

            league_teams = list(set(np.hstack(
                (
                    league_data["HT"].values,
                    league_data["AT"].values,
                )
            )))

            teams_in_league[season][league] = league_teams

    return teams_in_league


def check_and_update_promoting_and_relegating_team_states(
    team_states, batch_matches, team_dic
):
    """Checks for a batch of data rows (matches) if teams have been promoted/relegated
    and resets their team states if this is the case.
    """
    # initialize values
    league_mean_states = {}
    meta_path = "F:\\work\\data\\2023-soccer-prediction\\meta"
    with open(os.path.join(meta_path, "teams_in_league.json"), "r") as file:
        teams_in_league = json.load(file)

    # check for which leagues computation of mean team states is relevant
    for _, match in batch_matches.iterrows():
        # check if any team in the match promoted/relegated
        if match["HLC"] in [2, -2] or match["ALC"] in [2, -2]:
            league = match["Lge"]
            season = match["Sea"]
            if league not in league_mean_states:  # check if mean is already computed
                league_hidden = torch.vstack(
                    (
                        [
                            team_states["hidden"][team_dic[team]]
                            for team in teams_in_league[season][league]
                        ]
                    )
                )
                league_cell = torch.vstack(
                    (
                        [
                            team_states["cell"][team_dic[team]]
                            for team in teams_in_league[season][league]
                        ]
                    )
                )
                league_mean_states[league] = {
                    "cell": league_cell,
                    "hidden": league_hidden,
                }

    # update cell and hidden states of relegating/promoting teams to league mean
    for _, match in batch_matches.iterrows():
        league = match["Lge"]
        if league not in league_mean_states:
            continue

        # home update
        if match["HLC"] in [2, -2]:
            team = match["HT"]
            team_idx = team_dic[team]
            team_states["hidden"][team_idx] = torch.mean(
                league_mean_states[league]["hidden"], dim=0
            )
            team_states["cell"][team_idx] = torch.mean(
                league_mean_states[league]["cell"], dim=0
            )

        # away update
        if match["ALC"] in [2, -2]:
            league = match["Lge"]
            team = match["AT"]
            team_idx = team_dic[team]
            team_states["hidden"][team_idx] = torch.mean(
                league_mean_states[league]["hidden"], dim=0
            )
            team_states["cell"][team_idx] = torch.mean(
                league_mean_states[league]["cell"], dim=0
            )

    return team_states
