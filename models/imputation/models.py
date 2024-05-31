import math
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

from models.imputation.simple import BaseImputation

numerical_variables = {
    "HT": ["HS", "home_starter_total", "HTS", "HST", "avg_home_odds"],
    "AT": ["AS", "away_starter_total", "ATS", "AST", "avg_away_odds"],
}

rename_map = {
    "HS": "S",
    "AS": "S",
    "avg_home_odds": "odds",
    "avg_away_odds": "odds",
    "HT": "team",
    "AT": "team",
    "home_starter_total": "starter_total",
    "away_starter_total": "starter_total",
    "HTS": "TS",
    "HST": "ST",
    "HF": "F",
    "HY": "Y",
    "HR": "R",
    "ATS": "TS",
    "AST": "ST",
    "AF": "F",
    "AY": "Y",
    "AR": "R",
}


class KNNImputation(BaseImputation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _input_values(self, df):
        knn_imputer = KNNImputer(n_neighbors=201, weights="distance")
        knn_imputer.set_output(transform="pandas")
        categorical_variables = df[["Sea", "Lge", "WDL"]]
        categorical_encoding = pd.get_dummies(categorical_variables, drop_first=True)
        full_df = pd.concat([df, categorical_encoding], axis=1)
        full_df.drop(
            ["Sea", "Lge", "iso_date", "unique_id", "HT", "AT", "WDL"],
            axis=1,
            inplace=True,
        )
        df.loc[:, self.numerical_variables + self.betting_odds] = (
            knn_imputer.fit_transform(full_df).loc[
                :, self.numerical_variables + self.betting_odds
            ]
        )


class RollingAverage(BaseImputation):

    def __init__(self, **kwargs):
        self.impute_to_new_cols = kwargs.get("impute_to_new_cols", True)
        self.window_size = kwargs.get("window_size", 5)
        super().__init__(**kwargs)

    def _input_values(self, df):
        seasons = df.Sea.unique()
        leagues = df.Lge.unique()
        for season in seasons:
            season_mask = df["Sea"] == season
            for league in leagues:
                league_mask = df["Lge"] == league
                season_and_league_mask = season_mask & league_mask
                filtered_df = df[season_and_league_mask]
                stats_by_team = self._get_stats_by_team(filtered_df)
                new_df = filtered_df.merge(
                    stats_by_team,
                    left_on=["unique_id", "HT"],
                    right_on=["unique_id", "team"],
                    how="left",
                    suffixes=(None, "_HT"),
                )
                df.loc[
                    season_and_league_mask,
                    [
                        f"RM_{numerical_variable}"
                        for numerical_variable in numerical_variables["HT"]
                    ],
                ] = new_df.loc[
                    :,
                    [
                        f"RM_{rename_map[numerical_variable]}"
                        for numerical_variable in numerical_variables["HT"]
                    ],
                ].values
                new_df = filtered_df.merge(
                    stats_by_team,
                    left_on=["unique_id", "AT"],
                    right_on=["unique_id", "team"],
                    how="left",
                    suffixes=(None, "_AT"),
                )
                df.loc[
                    season_and_league_mask,
                    [
                        f"RM_{numerical_variable}"
                        for numerical_variable in numerical_variables["AT"]
                    ],
                ] = new_df.loc[
                    :,
                    [
                        f"RM_{rename_map[numerical_variable]}"
                        for numerical_variable in numerical_variables["AT"]
                    ],
                ].values

    def _get_stats_by_team(self, df):
        home_team_df = df.loc[
            :, ["iso_date", "unique_id", "HT"] + numerical_variables["HT"]
        ].rename(columns=rename_map)
        away_team_df = df.loc[
            :, ["iso_date", "unique_id", "AT"] + numerical_variables["AT"]
        ].rename(columns=rename_map)
        stats_df = pd.concat([home_team_df, away_team_df]).sort_values(by="iso_date")
        for numerical_variable in numerical_variables["HT"]:
            new_var = rename_map[numerical_variable]
            stats_df[f"RM_{new_var}"] = stats_df.groupby("team")[new_var].transform(
                lambda x: x.rolling(
                    self.window_size,
                    min_periods=int(self.window_size / 2),
                    closed="left",
                ).mean()
            )
        #stats_df.to_csv("stats_df.csv")
        return stats_df
