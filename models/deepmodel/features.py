import pandas as pd
import torch


def create_metadata_feature_vector(
    data_row,
    home_index,
    away_index,
    result_by_team,
    prediction_by_team,
):
    """Creates a feature vector with meta information which is used as input for the
    LSTM network.

    Parameters
    __________
    data_row: pd.Series
        One row of the complete dataset.
    home_index: int
        Index to identify the home team.
    away_index: int
        Index to identify the away team.
    outcomes_by_team: List of torch.Tensor of shape (3,)
        List of labels describing the outcome of the previous match of all teams.
    predictions_by_team: List of torch.Tensor of shape (3,)
        List of label predictions for the previous match of all teams.
    hidden_by_team: List of torch.Tensor
        List of team-specific hidden states.

    Returns
    _______
    feature_vector: np.array
        Feature vector containing the previous team outcome (one hot encoded label from
        the perspective of the given team, e.g., [1, 0, 0] for team win), the previous
        team outcome prediction (probabilities for team win, draw, and loss), the
        previous hidden and cell states as well as the rest days of the team and the
        three betting odds (for team win, draw, or loss) in the form:

            feats = [y(t-1), p(t-1), h(t-1), c(t-1), rest days, betting odds]
    """

    # get predictions and outcomes at time step t-1
    pred_home = prediction_by_team[home_index].squeeze(0)
    pred_away = prediction_by_team[away_index].squeeze(0)
    result_home = result_by_team[home_index].squeeze(0)
    result_away = result_by_team[away_index].squeeze(0)

    # get day-of-year and rest days meta features
    day_of_year = torch.tensor(data_row["DAY"]).unsqueeze(0)
    home_rest_days = torch.tensor(data_row["HRD"]).unsqueeze(0)
    away_rest_days = torch.tensor(data_row["ARD"]).unsqueeze(0)

    # create feature vectors for home and away team
    meta_features = torch.cat(
        (
            result_home,
            pred_home,
            result_away,
            pred_away,
            day_of_year,
            home_rest_days,
            away_rest_days,
        ),
    )

    return meta_features


def create_bettingodds_feature_vector(
    data_row,
    home_index,
    away_index,
    result_by_team,
    prediction_by_team,
    mode
):
    """Creates a feature vector with meta and market value information which
    is used as input for the LSTM network.

       Parameters
       __________
       data_row: pd.Series
           One row of the complete dataset.
       home_index: int
           Index to identify the home team.
       away_index: int
           Index to identify the away team.
       outcomes_by_team: List of torch.Tensor of shape (3,)
           List of labels describing the outcome of the previous match of all teams.
       predictions_by_team: List of torch.Tensor of shape (3,)
           List of label predictions for the previous match of all teams.
       hidden_by_team: List of torch.Tensor
           List of team-specific hidden states.
       mode: str
           Whether the feature vectors are created for training or test case.

       Returns
       _______
       feature_vector: np.array
           Feature vector containing the previous team outcome (one hot encoded label from
           the perspective of the given team, e.g., [1, 0, 0] for team win), the previous
           team outcome prediction (probabilities for team win, draw, and loss), the
           day of the year when the match was played, the rest days, and the total
           market value of starters for the home and away team
    """

    # get predictions and outcomes at time step t-1
    pred_home = prediction_by_team[home_index].squeeze(0)
    pred_away = prediction_by_team[away_index].squeeze(0)
    result_home = result_by_team[home_index].squeeze(0)
    result_away = result_by_team[away_index].squeeze(0)

    # get day-of-year and rest days meta features
    day_of_year = torch.tensor(data_row["DAY"]).unsqueeze(0)
    home_rest_days = torch.tensor(data_row["HRD"]).unsqueeze(0)
    away_rest_days = torch.tensor(data_row["ARD"]).unsqueeze(0)

    # setup column names for train or test
    if mode == "train":
        home_col = "avg_home_odds"
        away_col = "avg_away_odds"
        draw_col = "avg_draw_odds"
    elif mode == "test":
        home_col = "est_odds_home"
        away_col = "est_odds_away"
        draw_col = "est_odds_draw"
    else:
        home_col, away_col, draw_col = None, None, None

    # get betting odds features
    home_odds = torch.tensor(data_row[home_col]).unsqueeze(0)
    draw_odds = torch.tensor(data_row[draw_col]).unsqueeze(0)
    away_odds = torch.tensor(data_row[away_col]).unsqueeze(0)

    # create feature vectors for home and away team
    odds_features = torch.cat(
        (
            result_home,
            pred_home,
            result_away,
            pred_away,
            day_of_year,
            home_rest_days,
            away_rest_days,
            home_odds,
            draw_odds,
            away_odds,
        ),
    )

    return odds_features


def create_marketvalue_feature_vector(
    data_row,
    home_index,
    away_index,
    result_by_team,
    prediction_by_team,
    mode
):
    """Creates a feature vector with meta and market value information which
    is used as input for the LSTM network.

       Parameters
       __________
       data_row: pd.Series
           One row of the complete dataset.
       home_index: int
           Index to identify the home team.
       away_index: int
           Index to identify the away team.
       outcomes_by_team: List of torch.Tensor of shape (3,)
           List of labels describing the outcome of the previous match of all teams.
       predictions_by_team: List of torch.Tensor of shape (3,)
           List of label predictions for the previous match of all teams.
       hidden_by_team: List of torch.Tensor
           List of team-specific hidden states.
       mode: str
           Whether the feature vectors are created for training or test case.

       Returns
       _______
       feature_vector: np.array
           Feature vector containing the previous team outcome (one hot encoded label from
           the perspective of the given team, e.g., [1, 0, 0] for team win), the previous
           team outcome prediction (probabilities for team win, draw, and loss), the
           day of the year when the match was played, the rest days, and the total
           market value of starters for the home and away team
    """
    # get predictions and outcomes at time step t-1
    pred_home = prediction_by_team[home_index].squeeze(0)
    pred_away = prediction_by_team[away_index].squeeze(0)
    result_home = result_by_team[home_index].squeeze(0)
    result_away = result_by_team[away_index].squeeze(0)

    # get day-of-year and rest days meta features
    day_of_year = torch.tensor(data_row["DAY"]).unsqueeze(0)
    home_rest_days = torch.tensor(data_row["HRD"]).unsqueeze(0)
    away_rest_days = torch.tensor(data_row["ARD"]).unsqueeze(0)

    # setup column names for train or test
    if mode == "train":
        hmv_col = "home_starter_total"
        amv_col = "away_starter_total"
    elif mode == "test":
        hmv_col = "est_home_starter_total"
        amv_col = "est_away_starter_total"
    else:
        hmv_col, amv_col = None, None

    # get market value features
    home_starter_total = torch.tensor(data_row[hmv_col]).unsqueeze(0)
    away_starter_total = torch.tensor(data_row[amv_col]).unsqueeze(0)

    # create feature vectors for home and away team
    value_features = torch.cat(
        (
            result_home,
            pred_home,
            result_away,
            pred_away,
            day_of_year,
            home_rest_days,
            away_rest_days,
            home_starter_total,
            away_starter_total,
        ),
    )
    if any(torch.isnan(value_features)):
        print("WTF")
    return value_features


def create_matchstats_feature_vector(
    data_row,
    home_index,
    away_index,
    result_by_team,
    prediction_by_team,
    mode
):
    """Creates a feature vector with meta and market value information which
    is used as input for the LSTM network.

       Parameters
       __________
       data_row: pd.Series
           One row of the complete dataset.
       home_index: int
           Index to identify the home team.
       away_index: int
           Index to identify the away team.
       outcomes_by_team: List of torch.Tensor of shape (3,)
           List of labels describing the outcome of the previous match of all teams.
       predictions_by_team: List of torch.Tensor of shape (3,)
           List of label predictions for the previous match of all teams.
       hidden_by_team: List of torch.Tensor
           List of team-specific hidden states.
       mode: str
           Whether the feature vectors are created for training or test case.

       Returns
       _______
       feature_vector: np.array
           Feature vector containing the previous team outcome (one hot encoded label from
           the perspective of the given team, e.g., [1, 0, 0] for team win), the previous
           team outcome prediction (probabilities for team win, draw, and loss), the
           day of the year when the match was played, the rest days, and the total
           market value of starters for the home and away team
    """

    # get predictions and outcomes at time step t-1
    pred_home = prediction_by_team[home_index].squeeze(0)
    pred_away = prediction_by_team[away_index].squeeze(0)
    result_home = result_by_team[home_index].squeeze(0)
    result_away = result_by_team[away_index].squeeze(0)

    # get day-of-year and rest days meta features
    day_of_year = torch.tensor(data_row["DAY"]).unsqueeze(0)
    home_rest_days = torch.tensor(data_row["HRD"]).unsqueeze(0)
    away_rest_days = torch.tensor(data_row["ARD"]).unsqueeze(0)

    # setup column names for train or test
    if mode == "train":
        hts_col = "HTS"
        ats_col = "ATS"
        hst_col = "HST"
        ast_col = "AST"
    elif mode == "test":
        hts_col = "est_HTS"
        ats_col = "est_ATS"
        hst_col = "est_HST"
        ast_col = "est_AST"
    else:
        hts_col = None
        ats_col = None
        hst_col = None
        ast_col = None

    # get stats features
    hts = torch.tensor(data_row[hts_col]).unsqueeze(0)
    ats = torch.tensor(data_row[ats_col]).unsqueeze(0)
    hst = torch.tensor(data_row[hst_col]).unsqueeze(0)
    ast = torch.tensor(data_row[ast_col]).unsqueeze(0)

    # create feature vectors for home and away team
    stats_features = torch.cat(
        (
            result_home,
            pred_home,
            result_away,
            pred_away,
            day_of_year,
            home_rest_days,
            away_rest_days,
            hts,
            ats,
            hst,
            ast,
        ),
    )

    return stats_features


def create_all_features_vector(
    data_row,
    home_index,
    away_index,
    result_by_team,
    prediction_by_team,
    mode
):
    """Creates a feature vector with meta, betting odds, market value, and match stat
     information which is used as input for the LSTM network.

       Parameters
       __________
       data_row: pd.Series
           One row of the complete dataset.
       home_index: int
           Index to identify the home team.
       away_index: int
           Index to identify the away team.
       outcomes_by_team: List of torch.Tensor of shape (3,)
           List of labels describing the outcome of the previous match of all teams.
       predictions_by_team: List of torch.Tensor of shape (3,)
           List of label predictions for the previous match of all teams.
       hidden_by_team: List of torch.Tensor
           List of team-specific hidden states.
       mode: str
           Whether the feature vectors are created for training or test case.

       Returns
       _______
       feature_vector: np.array
           Feature vector containing the previous team outcome (one hot encoded label from
           the perspective of the given team, e.g., [1, 0, 0] for team win), the previous
           team outcome prediction (probabilities for team win, draw, and loss), the
           day of the year when the match was played, the rest days, and the total
           market value of starters for the home and away team
    """

    # get predictions and outcomes at time step t-1
    pred_home = prediction_by_team[home_index].squeeze(0)
    pred_away = prediction_by_team[away_index].squeeze(0)
    result_home = result_by_team[home_index].squeeze(0)
    result_away = result_by_team[away_index].squeeze(0)

    # get day-of-year and rest days meta features
    day_of_year = torch.tensor(data_row["DAY"]).unsqueeze(0)
    home_rest_days = torch.tensor(data_row["HRD"]).unsqueeze(0)
    away_rest_days = torch.tensor(data_row["ARD"]).unsqueeze(0)

    # setup column names for train or test
    if mode == "train":
        home_col = "avg_home_odds"
        away_col = "avg_away_odds"
        draw_col = "avg_draw_odds"
        hmv_col = "home_starter_total"
        amv_col = "away_starter_total"
        hts_col = "HTS"
        ats_col = "ATS"
        hst_col = "HST"
        ast_col = "AST"
    elif mode == "test":
        home_col = "est_odds_home"
        away_col = "est_odds_away"
        draw_col = "est_odds_draw"
        hmv_col = "est_home_starter_total"
        amv_col = "est_away_starter_total"
        hts_col = "est_HTS"
        ats_col = "est_ATS"
        hst_col = "est_HST"
        ast_col = "est_AST"

    else:
        home_col = None
        away_col = None
        draw_col = None
        hmv_col = None
        amv_col = None
        hts_col = None
        ats_col = None
        hst_col = None
        ast_col = None

    # get betting odds features
    home_odds = torch.tensor(data_row[home_col]).unsqueeze(0)
    draw_odds = torch.tensor(data_row[draw_col]).unsqueeze(0)
    away_odds = torch.tensor(data_row[away_col]).unsqueeze(0)

    # get market value features
    home_starter_total = torch.tensor(data_row[hmv_col]).unsqueeze(0)
    away_starter_total = torch.tensor(data_row[amv_col]).unsqueeze(0)

    # get stats features
    hts = torch.tensor(data_row[hts_col]).unsqueeze(0)
    ats = torch.tensor(data_row[ats_col]).unsqueeze(0)
    hst = torch.tensor(data_row[hst_col]).unsqueeze(0)
    ast = torch.tensor(data_row[ast_col]).unsqueeze(0)

    # create feature vectors for home and away team
    stats_features = torch.cat(
        (
            result_home,
            pred_home,
            result_away,
            pred_away,
            day_of_year,
            home_rest_days,
            away_rest_days,
            home_odds,
            draw_odds,
            away_odds,
            home_starter_total,
            away_starter_total,
            hts,
            ats,
            hst,
            ast,
        ),
    )

    return stats_features

# retrieve betting odds
# odds_home = data_row["avg_home_odds"]
# odds_draw = data_row["avg_draw_odds"]
# odds_away = data_row["avg_away_odds"]
# odds_home_perspective = torch.ones((1, 3))  # TODO: value for missing odds?
# odds_away_perspective = torch.ones((1, 3))
# if not any(pd.isna([odds_home, odds_draw, odds_away])):
#     odds_home_perspective[0, 0] = odds_home
#     odds_away_perspective[0, 0] = odds_away
#     odds_home_perspective[0, 1] = odds_draw
#     odds_away_perspective[0, 1] = odds_draw
#     odds_home_perspective[0, 2] = odds_away
#     odds_away_perspective[0, 2] = odds_home
