from models.deepmodel.labels import get_labels_dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from models.deepmodel.utilities import hparams
from models.evaluation.Metrics import MetricsCalculation


def log_hparams_to_tensorboard(epoch_results, logging_path, TASK_TYPE):
    # get best performing epoch
    best_metric_result = np.min(epoch_results)
    best_epoch = np.argmin(epoch_results)

    # log hyperparameter performance of last epoch to tensorboard
    with SummaryWriter(logging_path) as w:
        if TASK_TYPE == 1:
            metric_dict = {
                "hparam/goals_rmse_best": best_metric_result,
                "hparam/best_epoch": best_epoch,
            }
        else:
            metric_dict = {
                "hparam/wdl_arps_best": best_metric_result,
                "hparam/wdl_best_epoch": best_epoch,
            }
        w.add_hparams(hparams, metric_dict)


def log_performance_to_tensorboard(
    test, network_predictions, season, epoch, logging_path, TASK_TYPE
):
    # create labels for train and test data
    labels_dict = get_labels_dict(test, TASK_TYPE)

    # get predictions and labels from home perspective
    predictions = (
        np.full((len(network_predictions), 2), np.nan)
        if TASK_TYPE == 1
        else np.full((len(network_predictions), 3), np.nan)
    )
    labels = (
        np.full((len(network_predictions), 2), np.nan)
        if TASK_TYPE == 1
        else np.full((len(network_predictions), 3), np.nan)
    )
    for n, unique_id in enumerate(test["unique_id"]):
        predictions[n] = network_predictions[unique_id].cpu().detach()
        labels[n] = labels_dict[unique_id]["home"]

    # log epoch performance to tensorboard
    with SummaryWriter(logging_path) as w:
        if TASK_TYPE == 1:
            total_goals_prediction = np.round(np.sum(predictions, axis=1))
            total_goals_labels = np.sum(labels, axis=1)
            labels_rounded = np.round(labels)

            # create confusion matrix plot
            confusion = confusion_matrix(
                total_goals_labels, total_goals_prediction, normalize="true"
            )
            fig = plt.figure(figsize=(10, 7))
            sns.heatmap(confusion, annot=True)
            plt.tight_layout()
            w.add_figure(f"eval_goals_cm_{season}", fig, epoch)

            # compute metrics
            acc = np.sum(total_goals_prediction == total_goals_labels) / len(
                total_goals_labels
            )
            # rmse = np.mean(mean_squared_error(labels, predictions, squared=False))
            rmse, conf_lower, conf_upper, _ = MetricsCalculation.calculateRMSE(
                predictions[:, 0],
                predictions[:, 1],
                labels[:, 0],
                labels[:, 1],
            )
            rounded_rmse, _, _, _ = MetricsCalculation.calculateRMSE(
                predictions[:, 0],
                predictions[:, 1],
                labels_rounded[:, 0],
                labels_rounded[:, 1],
            )
            w.add_scalar(f"goals_acc_{season}", acc, epoch)
            w.add_scalar(f"goals_rmse_{season}", rmse, epoch)
            w.add_scalar(f"goals_conf_lower{season}", conf_lower, epoch)
            w.add_scalar(f"goals_conf_upper{season}", conf_upper, epoch)
            w.add_scalar(f"goals_rounded_rmse_{season}", rounded_rmse, epoch)
        else:
            # get predictions and translate to str in {"W", "D", "L"}
            outcome_dict = {0: "W", 1: "D", 2: "L"}
            prediction_outcomes = np.array(
                [outcome_dict[int(np.argmax(prediction))] for prediction in predictions]
            )
            real_outcomes = np.array(
                [outcome_dict[int(np.argmax(label))] for label in labels]
            )

            # create confusion matrix plot
            confusion = confusion_matrix(
                real_outcomes, prediction_outcomes, normalize="true"
            )
            df_cm = pd.DataFrame(
                confusion,
                index=[i for i in ["W", "D", "L"]],
                columns=[i for i in ["W_pred", "D_pred", "L_pred"]],
            )
            fig = plt.figure(figsize=(10, 7))
            sns.heatmap(df_cm, annot=True)
            plt.tight_layout()
            w.add_figure(f"wdl_cm_total", fig, epoch)

            # compute metrics
            acc = np.sum(prediction_outcomes == real_outcomes) / len(real_outcomes)
            rps, conf_lower, conf_upper, _ = MetricsCalculation.calculateRPS(
                predictions[:, 0],
                predictions[:, 1],
                labels[:, 0],
                labels[:, 1],
            )
            # mean_rps = np.mean(
            #     [
            #         criterion_task2(
            #             torch.Tensor(predictions[i]).unsqueeze(0),
            #             torch.Tensor(labels[i]).unsqueeze(0),
            #         )
            #         for i in range(len(labels))
            #     ]
            # )
            w.add_scalar(f"wdl_acc_{season}", acc, epoch)
            w.add_scalar(f"wdl_arps_{season}", rps, epoch)
            w.add_scalar(f"wdl_conf_lower{season}", conf_lower, epoch)
            w.add_scalar(f"wdl_conf_upper{season}", conf_upper, epoch)

    # return performance in main metric
    if TASK_TYPE == 1:
        return rmse
    else:
        return rps


def print_team_individual_wdl_performance(test, network_predictions, TASK_TYPE):
    outcome_dict = {0: "W", 1: "D", 2: "L"}

    # create labels for train and test data
    labels_dict = get_labels_dict(test, TASK_TYPE)

    # get predictions and labels from home perspective
    predictions = (
        np.full((len(network_predictions), 2), np.nan)
        if TASK_TYPE == 1
        else np.full((len(network_predictions), 3), np.nan)
    )
    labels = (
        np.full((len(network_predictions), 2), np.nan)
        if TASK_TYPE == 1
        else np.full((len(network_predictions), 3), np.nan)
    )
    for n, uID in enumerate(test["unique_id"]):
        predictions[n] = network_predictions[uID].cpu().detach()
        labels[n] = labels_dict[uID]["home"]

    # assess individual matches
    for team in [
        "Bayern Munich",
        "Real Madrid",
        "Torino",
        "Vissel Kobe",
        "Club Leon",
    ]:
        team_away_matches = test[test["AT"] == team]
        team_predictions = []
        team_labels = []
        pred_list = []
        result_list = []
        for n, uID in enumerate(test["unique_id"]):
            if uID in team_away_matches["unique_id"].values:
                # print(
                #     f"{team}:"
                #     f"{team_home_matches[team_home_matches['unique_id'] == uID]['AT'].values[0]} "
                #     f"{team_home_matches[team_home_matches['unique_id'] == uID]['HS'].values[0]}:"
                #     f"{team_home_matches[team_home_matches['unique_id'] == uID]['AS'].values[0]}"
                # )
                # print(network_predictions[uID])
                team_predictions.append(predictions[n])
                team_labels.append(labels[n])
                pred_wdl = outcome_dict[int(np.argmax(predictions[n]))]
                result_wdl = outcome_dict[int(np.argmax(labels[n]))]
                pred_list.append(pred_wdl)
                result_list.append(result_wdl)
        team_predictions = np.array(team_predictions)
        team_labels = np.array(team_labels)
        team_rps, _, _, _ = MetricsCalculation.calculateRPS(
            team_predictions[:, 0],
            team_predictions[:, 1],
            team_labels[:, 0],
            team_labels[:, 1],
        )
        print("\n")
        print(
            f"{team}: "
            f"PRD: {np.unique(pred_list, return_counts=True)[0]} {np.unique(pred_list, return_counts=True)[1]}, "
            f"RES: {np.unique(result_list, return_counts=True)[0]} {np.unique(result_list, return_counts=True)[1]}, "
            f"RPS: {team_rps}"
        )
        print("\n\n")
