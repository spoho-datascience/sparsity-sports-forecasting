eval_seasons = [
    # "01-02",
    # "02-03",
    # "03-04",
    # "04-05",
    # "05-06",
    # "06-07",
    # "07-08",
    # "08-09",
    # "09-10",
    "10-11",
    "11-12",
    "12-13",
    "13-14",
    "14-15",
    "15-16",
    "16-17",
    "17-18",
    "18-19",
    # "19-20",
    "20-21",
    "21-22",
    "22-23",
]

all_seasons = [
    "01-02",
    "02-03",
    "03-04",
    "04-05",
    "05-06",
    "06-07",
    "07-08",
    "08-09",
    "09-10",
    "10-11",
    "11-12",
    "12-13",
    "13-14",
    "14-15",
    "15-16",
    "16-17",
    "17-18",
    "18-19",
    "19-20",
    "20-21",
    "21-22",
    "22-23",
]

label_cols = ["WDL", "HS", "AS", "GD", "result"]

meta_cols = ["unique_id", "iso_date", "Sea", "Lge", "HT", "AT", "DAY", "HRD", "ARD", "HLC", "ALC"]

betting_odds_cols = ["avg_home_odds", "avg_draw_odds", "avg_away_odds"]

market_value_cols = ["home_starter_total", "away_starter_total"]

match_stats_cols = ["HTS", "ATS", "HST", "AST"]  # "HF", "AF", "HY", "AY", "HR", "AR"]

all_cols = market_value_cols + betting_odds_cols + match_stats_cols

est_market_value_cols = ["est_home_starter_total", "est_away_starter_total"]

est_betting_odds_cols = ["est_odds_home", "est_odds_draw", "est_odds_away"]

est_stats_cols = ["est_HTS", "est_ATS", "est_HST", "est_AST"]

relevant_cols = (
    label_cols
    + meta_cols
    + betting_odds_cols
    + market_value_cols
    + match_stats_cols
    + est_betting_odds_cols
    + est_market_value_cols
    + est_stats_cols
)

#definition of ratings used in xgboost model
features_rating_base = ['results', 'goals', 'piRating']
features_rating_market_value = ['marketvalues']
features_rating_betting_odds = ['odds']
features_rating_match_stats = ['shots', 'shotsTarget']
features_rating_all = features_rating_base + features_rating_market_value + features_rating_betting_odds + features_rating_match_stats
#definition of features used in xgboost model
features_test_base = ['Rat_Diff_results', 'Rat_Diff_goals', 'Rat_Pi_Expected_Diff', 'DAY', 'HRD', 'ARD']
features_test_market_value = ['Rat_Diff_MV']
features_test_betting_odds = ['Rat_Diff_odds']
features_test_match_stats = ['Rat_Diff_S', 'Rat_Diff_ST']
features_test_all = features_test_base + features_test_market_value + features_test_betting_odds + features_test_match_stats


# relevant_cols = [
#     "HS", "AS", "GD", "WDL", "result",  # labels
#     "unique_id", "iso_date", "Sea", "Lge", "HT", "AT", "DAY", "HRD", "ARD"  # meta
#     "avg_home_odds", "avg_draw_odds", "avg_away_odds",  # features: betting odds
#     "home_starter_total", "away_starter_total",  # features: market values
#     "HTS", "ATS", "HST", "AST"  # features: stats
#     "est_home_starter_total", "est_away_starter_total"  # estimation: market values
#     "est_odds_home", "est_odds_draw", "est_odds_away",  # estimation: betting odds
#     "est_HTS", "est_ATS", "est_HST", "est_AST"  # estimation: stats
# ]
