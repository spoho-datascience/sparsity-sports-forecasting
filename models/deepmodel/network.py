import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn import init
from torch.optim.lr_scheduler import StepLR
from models.deepmodel.utilities import hparams
from models.deepmodel.labels import get_labels_dict
from models.deepmodel.features import (
    create_metadata_feature_vector,
    create_bettingodds_feature_vector,
    create_marketvalue_feature_vector,
    create_matchstats_feature_vector,
    create_all_features_vector,
)
from models.deepmodel.leagues import (
    get_matchday_ids,
    check_and_update_promoting_and_relegating_team_states,
)
from input.utilities import eval_seasons
from models.evaluation.Metrics import MetricsCalculation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# network saving to file
def save_network_to_file(network, epoch, network_path):
    torch.save(
        network.state_dict(),
        os.path.join(network_path, f"lstm_{epoch + 1}"),
    )


def save_predictions_to_file(predictions, epoch, network_path):
    df = pd.DataFrame()
    df["unique_id"] = list(predictions.keys())
    df["prd_W"] = [predictions[unique_id][0].cpu() for unique_id in predictions]
    df["prd_D"] = [predictions[unique_id][1].cpu() for unique_id in predictions]
    df["prd_L"] = [predictions[unique_id][2].cpu() for unique_id in predictions]

    df.to_csv(os.path.join(network_path, f"predictions_{epoch + 1}.csv"))


# networks creation
def create_deep_model_classes(TASK_TYPE, EXPERIMENT):
    input_size = 4 * 2 + 3 if TASK_TYPE == 1 else 4 * 3 + 3
    if EXPERIMENT == "exp2a" or EXPERIMENT == "exp3a" or EXPERIMENT == "exp4a":
        input_size += 2  # home_starter_total, away_starter_total
    elif EXPERIMENT == "exp2b" or EXPERIMENT == "exp3b" or EXPERIMENT == "exp4b":
        input_size += 3  # home, away, draw odds
    elif EXPERIMENT == "exp2c" or EXPERIMENT == "exp3c" or EXPERIMENT == "exp4c":
        input_size += 4  # shots and shots on target for home and away
    elif EXPERIMENT == "exp2d" or EXPERIMENT == "exp3d" or EXPERIMENT == "exp4d":
        input_size += 2 + 3 + 4  # all

    network = PredLayer(
        input_size,
        hparams["hidden_size"],
        hparams["use_hidden"],
        hparams["dropout"],
        TASK_TYPE,
    ).to(device)

    if hparams["optim"] == "SGD":
        optimizer = torch.optim.SGD(network.parameters(), lr=hparams["lr"])
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=hparams["lr"])
    scheduler = None
    if hparams["lr_decay"]:
        scheduler = StepLR(optimizer, step_size=1, gamma=hparams["lr_decay"])

    return network, optimizer, scheduler


# network training & testing
def train_network_for_season(
    network, train, season, team_states, team_dic, optimizer, TASK_TYPE, EXPERIMENT
):
    print(
        f"Start Training LSTM for task {'goals' if TASK_TYPE == 1 else 'wdl'} "
        f"on device {device}"
    )

    # set networks to training mode
    network.train()

    # initialize containers for predictions and results
    if TASK_TYPE == 1:
        prediction_by_team = torch.zeros((len(team_states["cell"]), 2))
        result_by_team = torch.zeros((len(team_states["cell"]), 2))
    else:
        prediction_by_team = torch.zeros((len(team_states["cell"]), 3))
        result_by_team = torch.zeros((len(team_states["cell"]), 3))

    # create labels for train and test data
    labels_train = get_labels_dict(train, TASK_TYPE)

    # get match day specification needed for batch creation
    train_matchday_ids = get_matchday_ids(train)

    # loop over training data
    list_train_loss = []
    max_matchdays = np.max(
        np.hstack(
            [list(train_matchday_ids[league].keys()) for league in train_matchday_ids]
        )
    )

    for matchday in range(1, max_matchdays + 1):
        # create batch
        batch_ids = np.hstack(
            [
                train_matchday_ids[league][matchday]
                for league in train_matchday_ids
                if matchday in train_matchday_ids[league]
            ]
        )
        batch_matches = pd.DataFrame()
        for batch_id in batch_ids:
            batch_matches = pd.concat(  # len(batch_matches["unique_id"].value_counts())
                (
                    batch_matches,
                    train.loc[train["unique_id"] == batch_id],
                )
            )  # batch_matches.unique_id.value_counts()

        # update cell and hidden states of relegating/promoting teams to league mean
        team_states = check_and_update_promoting_and_relegating_team_states(
            team_states, batch_matches, team_dic
        )

        # find home and away teams
        home_inds = [team_dic[team] for team in batch_matches["HT"]]
        away_inds = [team_dic[team] for team in batch_matches["AT"]]

        # zero the gradients
        optimizer.zero_grad()

        # retrieve hidden states and cell states at current batch
        hidden_home = torch.vstack(
            ([team_states["hidden"][home_idx].unsqueeze(0) for home_idx in home_inds])
        )
        cell_home = torch.vstack(
            ([team_states["cell"][home_idx].unsqueeze(0) for home_idx in home_inds])
        )
        hidden_away = torch.vstack(
            ([team_states["hidden"][away_idx].unsqueeze(0) for away_idx in away_inds])
        )
        cell_away = torch.vstack(
            ([team_states["cell"][away_idx].unsqueeze(0) for away_idx in away_inds])
        )

        # create feature vectors for different experiments
        if EXPERIMENT == "exp1":
            input_meta = torch.vstack(
                (
                    [
                        create_metadata_feature_vector(
                            batch_matches.iloc[idx],
                            home_inds[idx],
                            away_inds[idx],
                            result_by_team,
                            prediction_by_team,
                        )
                        for idx in range(len(home_inds))
                    ]
                )
            )
            net_input = input_meta
        elif EXPERIMENT == "exp2a" or EXPERIMENT == "exp3a" or EXPERIMENT == "exp4a":
            input_value = torch.vstack(
                (
                    [
                        create_marketvalue_feature_vector(
                            batch_matches.iloc[idx],
                            home_inds[idx],
                            away_inds[idx],
                            result_by_team,
                            prediction_by_team,
                            mode="train",
                        )
                        for idx in range(len(home_inds))
                    ]
                )
            )
            net_input = input_value
        elif EXPERIMENT == "exp2b" or EXPERIMENT == "exp3b" or EXPERIMENT == "exp4b":
            input_odds = torch.vstack(
                (
                    [
                        create_bettingodds_feature_vector(
                            batch_matches.iloc[idx],
                            home_inds[idx],
                            away_inds[idx],
                            result_by_team,
                            prediction_by_team,
                            mode="train",
                        )
                        for idx in range(len(home_inds))
                    ]
                )
            )
            net_input = input_odds
        elif EXPERIMENT == "exp2c" or EXPERIMENT == "exp3c" or EXPERIMENT == "exp4c":
            input_stats = torch.vstack(
                (
                    [
                        create_matchstats_feature_vector(
                            batch_matches.iloc[idx],
                            home_inds[idx],
                            away_inds[idx],
                            result_by_team,
                            prediction_by_team,
                            mode="train",
                        )
                        for idx in range(len(home_inds))
                    ]
                )
            )
            net_input = input_stats
        elif EXPERIMENT == "exp2d" or EXPERIMENT == "exp3d" or EXPERIMENT == "exp4d":
            input_all = torch.vstack(
                (
                    [
                        create_all_features_vector(
                            batch_matches.iloc[idx],
                            home_inds[idx],
                            away_inds[idx],
                            result_by_team,
                            prediction_by_team,
                            mode="train",
                        )
                        for idx in range(len(home_inds))
                    ]
                )
            )
            net_input = input_all
        else:
            net_input = None

        # transport input to GPU
        net_input = net_input.float().to(device)

        # concatenate hidden vectors and cell states of teams
        hidden_match = torch.cat((hidden_home, hidden_away), dim=1).to(device)
        cell_match = torch.cat((cell_home, cell_away), dim=1).to(device)

        # DEBUG: remove batch logic and iterate over one match at a time
        # predictions_match = (
        #     torch.zeros((len(batch_ids), 2))
        #     if TASK_TYPE == 1
        #     else torch.zeros((len(batch_ids), 3))
        # )
        # lstm_out = (
        #     torch.zeros((len(batch_ids), 2 * hparams["hidden_size"])),
        #     torch.zeros((len(batch_ids), 2 * hparams["hidden_size"])),
        # )
        # for batch_idx, uID in enumerate(batch_ids):
        #     match_input = net_input[batch_idx, :].reshape(1, net_input.shape[1])
        #     hidden_single = hidden_match[batch_idx, :].reshape(1, hidden_match.shape[1])
        #     cell_single = cell_match[batch_idx, :].reshape(1, cell_match.shape[1])

        # compute predictions and hidden states at time step t
        predictions_match, lstm_out = network(net_input, hidden_match, cell_match)
        # predictions_single, single_out = network(
        #     match_input, hidden_single, cell_single
        # )

        # retrieve labels
        labels_home = torch.vstack(
            ([labels_train[uID]["home"] for uID in batch_ids])
        ).to(device)
        labels_away = torch.vstack(
            ([labels_train[uID]["away"] for uID in batch_ids])
        ).to(device)
        # label_home = labels_train[uID]["home"].to(device)

        # compute loss and update networks
        if TASK_TYPE == 1:
            # loss of RMSE
            loss = criterion_task1(predictions_match, labels_home)
            # loss = criterion_task1(predictions_single, label_home)
        else:
            # loss of RPS
            loss = criterion_task2(predictions_match, labels_home)
            # loss = criterion_task2(predictions_single, label_home)
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        optimizer.step()
        list_train_loss.append(loss.item())
        # predictions_match[batch_idx, :] = predictions_single
        # lstm_out[0][batch_idx, :] = single_out[0]
        # lstm_out[1][batch_idx, :] = single_out[1]

        # retrieve labels
        labels_home = torch.vstack(
            ([labels_train[uID]["home"] for uID in batch_ids])
        ).to(device)
        labels_away = torch.vstack(
            ([labels_train[uID]["away"] for uID in batch_ids])
        ).to(device)

        # update hidden and cell states, predictions, results, and play dates
        for idx in range(len(home_inds)):
            home_index = home_inds[idx]
            away_index = away_inds[idx]

            team_states["hidden"][home_index] = lstm_out[0][
                idx, : hparams["hidden_size"]
            ].detach()
            team_states["cell"][home_index] = lstm_out[1][
                idx, : hparams["hidden_size"]
            ].detach()
            team_states["hidden"][away_index] = lstm_out[0][
                idx, hparams["hidden_size"] :
            ].detach()
            team_states["cell"][away_index] = lstm_out[1][
                idx, hparams["hidden_size"] :
            ].detach()
            prediction_by_team[home_index] = predictions_match[idx].detach()
            prediction_by_team[away_index] = torch.flip(
                predictions_match[idx], [0]
            ).detach()
            result_by_team[home_index] = labels_home[idx]
            result_by_team[away_index] = labels_away[idx]

        # display train performance
        status = (
            f"Season {season} / {eval_seasons[-1]} | "
            f"Matchday {matchday + 1} / {max_matchdays + 1} "
            f"({np.round(100 * (matchday + 1) / (max_matchdays + 1), 2)}%) | "
            f"Avg. Epoch Loss: {np.round(np.mean(list_train_loss), 6)}"
        )
        print(status, end="\r", flush=True)

    return network, result_by_team, prediction_by_team


def get_network_predictions_for_season(
    network,
    test,
    team_states,
    team_dic,
    result_by_team,
    prediction_by_team,
    TASK_TYPE,
    EXPERIMENT,
):
    # set network to evaluation mode
    network.eval()

    # loop over evaluation data
    season_predictions = {}
    with torch.no_grad():
        for idx in range(len(test)):
            # retrieve index of home and away team and unique_id
            unique_id = test["unique_id"].iloc[idx]
            home_index = team_dic[test["HT"].iloc[idx]]
            away_index = team_dic[test["AT"].iloc[idx]]

            # get hidden states and cell states at time step t-1
            hidden_home = team_states["hidden"][home_index].unsqueeze(0)
            cell_home = team_states["cell"][home_index].unsqueeze(0)
            hidden_away = team_states["hidden"][away_index].unsqueeze(0)
            cell_away = team_states["cell"][away_index].unsqueeze(0)

            # create feature vectors for different experiments
            if EXPERIMENT == "exp1":
                input_meta = create_metadata_feature_vector(
                    test.iloc[idx],
                    home_index,
                    away_index,
                    result_by_team,
                    prediction_by_team,
                )
                net_input = input_meta
            elif (
                EXPERIMENT == "exp2a" or EXPERIMENT == "exp3a" or EXPERIMENT == "exp4a"
            ):
                input_value = create_marketvalue_feature_vector(
                    test.iloc[idx],
                    home_index,
                    away_index,
                    result_by_team,
                    prediction_by_team,
                    mode="test",
                )
                net_input = input_value
            elif (
                EXPERIMENT == "exp2b" or EXPERIMENT == "exp3b" or EXPERIMENT == "exp4b"
            ):
                input_odds = create_bettingodds_feature_vector(
                    test.iloc[idx],
                    home_index,
                    away_index,
                    result_by_team,
                    prediction_by_team,
                    mode="test",
                )
                net_input = input_odds
            elif (
                EXPERIMENT == "exp2c" or EXPERIMENT == "exp3c" or EXPERIMENT == "exp4c"
            ):
                input_stats = create_matchstats_feature_vector(
                    test.iloc[idx],
                    home_index,
                    away_index,
                    result_by_team,
                    prediction_by_team,
                    mode="test",
                )
                net_input = input_stats

            elif (
                EXPERIMENT == "exp2d" or EXPERIMENT == "exp3d" or EXPERIMENT == "exp4d"
            ):
                input_all = create_all_features_vector(
                    test.iloc[idx],
                    home_index,
                    away_index,
                    result_by_team,
                    prediction_by_team,
                    mode="test",
                )
                net_input = input_all
            else:
                net_input = None

            # send network input to GPU
            net_input = net_input.float().to(device).reshape(1, len(net_input))

            # concatenate hidden vectors and cell states of teams
            hidden_match = torch.cat((hidden_home, hidden_away), dim=1)
            cell_match = torch.cat((cell_home, cell_away), dim=1)
            hidden_match = hidden_match.to(device)
            cell_match = cell_match.to(device)

            # compute predictions and hidden states at time step t
            prediction_match, lstm_out = network(net_input, hidden_match, cell_match)

            # update hidden and cell states, predictions, and play dates
            team_states["hidden"][home_index] = lstm_out[0][
                :,  hparams["hidden_size"]
            ].detach()
            team_states["cell"][home_index] = lstm_out[1][
                :, : hparams["hidden_size"]
            ].detach()
            team_states["hidden"][away_index] = lstm_out[0][
                :, hparams["hidden_size"] :
            ].detach()
            team_states["cell"][away_index] = lstm_out[1][
                :, hparams["hidden_size"] :
            ].detach()
            prediction_by_team[home_index] = prediction_match.squeeze(0).detach()
            prediction_by_team[away_index] = (
                torch.flip(prediction_match, [0]).squeeze(0).detach()
            )

            # update results as approximated from network prediction
            if TASK_TYPE == 1:
                goals_pred_home = torch.vstack(
                    (
                        torch.round(prediction_match[:, 0]),
                        torch.round(prediction_match[:, 1]),
                    )
                )
                goals_pred_away = torch.vstack(
                    (
                        torch.round(prediction_match[:, 1]),
                        torch.round(prediction_match[:, 0]),
                    )
                )
                result_by_team[home_index] = goals_pred_home.squeeze(1)
                result_by_team[away_index] = goals_pred_away.squeeze(1)
            else:
                wdl_pred_home = torch.vstack(
                    (
                        torch.round(prediction_match[:, 0]),
                        torch.round(prediction_match[:, 1]),
                        torch.round(prediction_match[:, 2]),
                    )
                )
                wdl_pred_away = torch.vstack(
                    (
                        torch.round(prediction_match[:, 2]),
                        torch.round(prediction_match[:, 1]),
                        torch.round(prediction_match[:, 0]),
                    )
                )
                result_by_team[home_index] = wdl_pred_home.squeeze(1)
                result_by_team[away_index] = wdl_pred_away.squeeze(1)

            # update network predictions
            season_predictions[unique_id] = prediction_match

    return season_predictions


# BUILDING BLOCKS
# input: features (output(t-1), prediction(t-1), hidden(t-1), cell(t-1), rest days)
# output: team-specific hidden(t) and cell(t) of size hparams["hidden_size"]
# initialize pred_network
# input: team-specific hidden(t-1) of size hparams["hidden_size"] and features
# output: score of (This team, Rival team)
class PredLayer(nn.Module):
    def __init__(self, input_size, hidden_size, use_hidden, dropout, CRITERION_TYPE):
        super(PredLayer, self).__init__()
        self.use_hidden = use_hidden

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        if CRITERION_TYPE == 1:
            self.lstm = nn.LSTMCell(input_size, 2 * self.hidden_size)
            self.fc = torch.nn.Linear(2 * self.hidden_size, 2)
            self.act = torch.nn.ReLU()
        else:
            self.lstm = nn.LSTMCell(input_size, 2 * self.hidden_size)
            self.fc = torch.nn.Linear(2 * self.hidden_size, 3)
            self.act = torch.nn.Softmax(dim=1)

        # initialize weights of LSTM
        init.xavier_uniform_(self.lstm.weight_hh)
        init.xavier_uniform_(self.lstm.weight_ih)

    def forward(self, x, h, c):
        h, c = self.lstm(x, (h, c))
        if self.use_hidden:
            lstm_out = h
        else:
            lstm_out = c
        relu = self.relu(lstm_out)
        relu_output = self.fc(relu)
        if self.dropout:
            relu_output = self.dropout(relu_output)
        output = self.act(relu_output)
        return output, (h, c)


# LOSS FUNCTIONS
def criterion_task1(pred, label):  # MSE
    return torch.sum((pred - label) ** 2) / label.shape[0]


def criterion_task2(pred, label):  # RPS
    rps = 0
    for b in range(pred.shape[0]):
        # rps += 0.5 * (
        #     (pred[b, 0] - label[b, 0]) ** 2
        #     + (pred[b, 0] + pred[b, 1] - label[b, 0] - label[b, 1]) ** 2
        # )
        for i in range(1, pred.shape[1]):
            rps += torch.sum(pred[b, :i] - label[b, :i]) ** 2 / (
                pred.shape[1] - 1
            )
        # curr_rps, _, _, _ = MetricsCalculation.calculateRPS(
        #     pred[b, 0].cpu().detach().numpy().reshape(1, 1),
        #     pred[b, 1].cpu().detach().numpy().reshape(1, 1),
        #     label[b, 0].cpu().detach().numpy().reshape(1, 1),
        #     label[b, 1].cpu().detach().numpy().reshape(1, 1),
        # )
        # curr_rps = torch.tensor(curr_rps, requires_grad=True)
        # rps += curr_rps
    return rps / pred.shape[0]  # mean over batch
