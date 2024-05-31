import torch


def get_labels_dict(data, CRITERION_TYPE):
    """

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame with all matches.
    CRITERION_TYPE: int
        1 for prediction of scores and 2 for prediction of match outcomes.

    Returns
    -------
    dict_labels: dict[int, dict[str, torch.Tensor]]
        Ground-truth labels of every match, key is given by unique_id and value is
        either a 3D tensor with one hot encoding of WDL from perspective of the home and
        away teams or a 2D tensor with score of the team, score of the opponent.
    """
    dict_labels = {}

    for idx, data_row in data.iterrows():

        if CRITERION_TYPE == 1:
            # transform result string, format: (this team's score, rival's score)
            label_home = torch.zeros((1, 2))
            label_away = torch.zeros((1, 2))

            label_home[0, 0] = int(data_row["HS"])
            label_home[0, 1] = int(data_row["AS"])
            label_away[0, 1] = int(data_row["HS"])
            label_away[0, 0] = int(data_row["AS"])
        else:
            label_home = torch.zeros((1, 3))
            label_away = torch.zeros((1, 3))
            if data_row["WDL"] == "W":
                label_home[0, 0] = 1
                label_away[0, 2] = 1
            elif data_row["WDL"] == "D":
                label_home[0, 1] = 1
                label_away[0, 1] = 1
            else:
                label_home[0, 2] = 1
                label_away[0, 0] = 1

        dict_labels[data_row["unique_id"]] = {"home": label_home,
                                              "away": label_away}

    return dict_labels
