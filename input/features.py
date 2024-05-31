import collections
import numpy as np
from joblib import Parallel, delayed, cpu_count
from input.utilities import eval_seasons
import pandas as pd


def add_data_type_col(data):
    # flag prediction and training intervals to prevent ELO update during testing
    first_year = 2000 + int(eval_seasons[0][-2:])
    data["type"] = "train"
    data["type"][
        (data["iso_date"].dt.year >= first_year)
        & (data["iso_date"].dt.month > 3)
        & (data["iso_date"].dt.month < 6)
        ] = "predict"
    return data

def add_results_integer_col(data):
    data["result"] = data["WDL"]
    data["result"] = data["result"].replace("W", 2)
    data["result"] = data["result"].replace("D", 1)
    data["result"] = data["result"].replace("L", 0)
    return data


def add_probabilities_from_betting_odds_col(data):
    overround = (
        1 / data["avg_home_odds"]
        + 1 / data["avg_draw_odds"]
        + 1 / data["avg_away_odds"]
    )
    data["pred_odds_home"] = 1 / data["avg_home_odds"] / overround
    data["pred_odds_draw"] = 1 / data["avg_draw_odds"] / overround
    data["pred_odds_away"] = 1 / data["avg_away_odds"] / overround
    return data

#splits dataset to two subdataset, where the first contains all relevant columns and the second misses at least one of them
def split_dataset_by_relevant_columns(data, relevantColumns):
    dataExtracted = pd.DataFrame()
    for col in relevantColumns:
        dataExtracted = pd.concat([dataExtracted, data[np.isnan(data[col])]], ignore_index = True)
        data = data[~np.isnan(data[col])]
    return data, dataExtracted


#splits dataset to two subdataset, where the first contains data-intensive leagues and the last contains non-data-intensive leagues
def split_dataset_by_data_intensive_leagues(data):
    relevantLeagues = ['ENG1', 'ENG2', 'ENG3', 'ENG4', 'ENG5', 'FRA1', 'FRA2', 'SPA1', 'SPA2', 'ITA1', 'ITA1', 'GER1', 'GER2', 'SCO1', 'HOL1', 'POR1', 'BEL1', 'GRE1']
    dataIntensive = pd.DataFrame()
    for league in relevantLeagues:
        dataIntensive = pd.concat([dataIntensive, data.loc[data['Lge'] == league]], ignore_index = True)
        data = data.loc[data['Lge']!=league]
    return dataIntensive, data


def add_day_of_year_col(data):
    data["DAY"] = [date.timetuple().tm_yday for date in data["iso_date"]]
    return data


def add_metadata_col(data):
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def create_team_specific_links_unique_id_to_metadata(team):
        team_matches = data[(data["HT"] == team) + (data["AT"] == team)]
        team_links_id_to_metadata = {
            unique_id: {} for unique_id in team_matches["unique_id"]
        }
        unique_id = 0
        # compute rest days for remaining matches
        for match_idx in range(len(team_matches)):
            last_unique_id = unique_id
            unique_id = team_matches.iloc[match_idx]["unique_id"]
            leagueChange = 0
            leagueChangeLast = 0
            #if a match is the last match of a team in the dataset, it is flagged as relegated
            if match_idx == len(team_matches)-1:
                leagueChange = -1
                
            if match_idx == 0:
                rest_days = -1  # first match ever played by the team in the dataset
                leagueChange = 2
            else:
                rest_days = (
                    team_matches.iloc[match_idx]["iso_date"]
                    - team_matches.iloc[match_idx - 1]["iso_date"]
                ).days
                #identify relegation or promotion (1=last match before promotion, 2 = first match after promotion, -1 = last match before relegation, -2 = first match after relegation
                leagueDiff = int(team_matches.iloc[match_idx - 1]["Lge"][-1]) - int(team_matches.iloc[match_idx]["Lge"][-1])
                #if a team disappeared for more than a year we expect it has been in a lower league
                if(rest_days > 365):
                    leagueChange = 2
                    leagueChangeLast = -1
                elif(leagueDiff >= 1):
                    leagueChange = 2
                    leagueChangeLast = 1
                elif(leagueDiff <= -1):
                    leagueChange = -2
                    leagueChangeLast = -1
                rest_days = np.clip(rest_days, 0, 30)  # after 30 we expect no advantage

            # assign to home or away depending on role of current team
            if team == team_matches.iloc[match_idx]["HT"]:
                team_links_id_to_metadata[unique_id]["HT"] = rest_days
                team_links_id_to_metadata[unique_id]["HLC"] = leagueChange
            elif team == team_matches.iloc[match_idx]["AT"]:
                team_links_id_to_metadata[unique_id]["AT"] = rest_days
                team_links_id_to_metadata[unique_id]["ALC"] = leagueChange
            else:
                assert False, "Team not contained in either HT or AT"
            #assign to home or away team depending on role of team in last match
            if(match_idx > 0 and team == team_matches.iloc[match_idx - 1]["HT"] and leagueChangeLast != 0):
                team_links_id_to_metadata[last_unique_id]["HLC"] = leagueChangeLast
            if(match_idx > 0 and team == team_matches.iloc[match_idx - 1]["AT"] and leagueChangeLast != 0):
                team_links_id_to_metadata[last_unique_id]["ALC"] = leagueChangeLast
                
        return team_links_id_to_metadata

    # create links from ID to metadata (restdays, promotion, relegation) for every team
    all_teams = np.concatenate((data["HT"].unique(), data["AT"].unique()))
    all_links_id_to_metadata = Parallel(n_jobs=cpu_count(), verbose=100)(
        delayed(create_team_specific_links_unique_id_to_metadata)(team)
        for team in all_teams
    )

    # stack links into complete dict with rest days
    links_id_to_metadata = {}
    for links in all_links_id_to_metadata:
        links_id_to_metadata = update(links_id_to_metadata, links)

    # add rest day column to dataframe
    data["HRD"] = [
        links_id_to_metadata[unique_id]["HT"] for unique_id in data["unique_id"]
    ]
    data["ARD"] = [
        links_id_to_metadata[unique_id]["AT"] for unique_id in data["unique_id"]
    ]
    data["HLC"] = [
        links_id_to_metadata[unique_id]["HLC"] for unique_id in data["unique_id"]
    ]
    data["ALC"] = [
        links_id_to_metadata[unique_id]["ALC"] for unique_id in data["unique_id"]
    ]

    return data
