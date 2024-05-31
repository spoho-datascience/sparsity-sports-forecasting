import math

import numpy as np

NUMERICAL = [
    "home_starter_total",
    "away_starter_total",  # features: market values
    "HTS",
    "ATS",
    "HST",
    "AST",  # "HF", "AF", "HY", "AY", "HR", "AR"  # features: stats
    "est_home_starter_total",
    "est_away_starter_total",
    "est_odds_home",
    "est_odds_draw",
    "est_odds_away",
    "est_HTS",
    "est_ATS",
    "est_HST",
    "est_AST",  # features: estimated
]

BETTING = ["avg_home_odds", "avg_draw_odds", "avg_away_odds"]  # features: betting odds


class BaseImputation:
    def __init__(self, **kwargs):
        self.numerical_variables = kwargs.get("numerical_variables", NUMERICAL)
        self.betting_odds = kwargs.get("betting_odds", BETTING)
        pass

    def impute(self, df, in_place=False):
        if in_place:
            return_df = df
        else:
            return_df = df.copy()
        return_df["home_starter_total"] = return_df["home_starter_total"].replace(
            0, np.nan
        )
        return_df["away_starter_total"] = return_df["away_starter_total"].replace(
            0, np.nan
        )
        self._input_values(return_df)
        return None if in_place else return_df

    def _input_values(self, df):
        print("Method not implemented")


class SimpleImputation(BaseImputation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _input_values(self, df):
        # TODO check for Sea and Lge attributes to be present in DF.
        seasons = df.Sea.unique()
        leagues = df.Lge.unique()
        for season in seasons:
            season_mask = df["Sea"] == season
            for league in leagues:
                league_mask = df["Lge"] == league
                season_and_league_mask = season_mask & league_mask
                for numerical_variable in self.numerical_variables:
                    mean_value = df[season_and_league_mask][numerical_variable].mean()
                    if math.isnan(mean_value):
                        mean_value = df[league_mask][numerical_variable].mean()
                        if math.isnan(mean_value):
                            mean_value = df[season_mask][numerical_variable].mean()
                        else:
                            mean_value = df[numerical_variable].mean()
                    df.loc[
                        season_and_league_mask & (df[numerical_variable].isnull()),
                        numerical_variable,
                    ] = mean_value
                for odds_variable in self.betting_odds:
                    probability_array = 0.95 / df[season_and_league_mask][odds_variable]
                    if all(probability_array.isnull()):
                        probability_array = 0.95 / df[league_mask][odds_variable]
                        if all(probability_array.isnull()):
                            probability_array = 0.95 / df[season_mask][odds_variable]
                        else:
                            probability_array = 0.95 / df[odds_variable]
                    df.loc[
                        season_and_league_mask & (df[odds_variable].isnull()),
                        odds_variable,
                    ] = (
                        0.95 / probability_array.mean()
                    )
