import pandas as pd
import numpy as np
from input.utilities import label_cols, meta_cols


def remove_duplicate_unique_ids(data):
    for uID in data["unique_id"]:
        if len(data[data["unique_id"] == uID]) > 1:
            print(f"Dropping duplicates of match {uID}")
            data = data.drop(data[data["unique_id"] == uID][1:].index)
    return data


def rename_duplicate_teams(data):
    data["HT"][
        (data["HT"] == "FC Barcelona") & (data["Lge"] == "ECU1")
    ] = "FC Barcelona ECU"
    data["AT"][
        (data["AT"] == "FC Barcelona") & (data["Lge"] == "ECU1")
    ] = "FC Barcelona ECU"
    data["HT"][(data["HT"] == "Everton") & (data["Lge"] == "CHL1")] = "Everton CHL"
    data["AT"][(data["AT"] == "Everton") & (data["Lge"] == "CHL1")] = "Everton CHL"
    return data


def remove_rows_with_empty_label_cols(data):
    for meta_col in label_cols:
        if meta_col in data:
            data = data.drop(data[pd.isna(data[meta_col])].index)
    return data


def remove_rows_with_empty_meta_cols(data):
    for meta_col in meta_cols:
        if meta_col in data:
            data = data.drop(data[pd.isna(data[meta_col])].index)
    return data


def replace_market_value_zero_with_nan(data):
    data["home_starter_total"] = [
        total if total > 0 else pd.NA for total in data["home_starter_total"]
    ]
    data["away_starter_total"] = [
        total if total > 0 else pd.NA for total in data["away_starter_total"]
    ]
    return data


def compute_market_value_log(data):
    data["home_starter_total"] = np.log(data["home_starter_total"])
    data["away_starter_total"] = np.log(data["away_starter_total"])
    return data


def compute_odds_average(data):
    data["avg_home_odds"] = data[["avg_home_odds", "odds_home"]].mean(
        axis=1, skipna=True
    )
    data["avg_draw_odds"] = data[["avg_draw_odds", "odds_draw"]].mean(
        axis=1, skipna=True
    )
    data["avg_away_odds"] = data[["avg_away_odds", "odds_away"]].mean(
        axis=1, skipna=True
    )
    return data
