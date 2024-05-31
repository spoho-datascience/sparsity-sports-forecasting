import os
import time
from input.utilities import (
    market_value_cols,
    betting_odds_cols,
    match_stats_cols,
    est_market_value_cols,
    est_betting_odds_cols,
    est_stats_cols,
)

hparams = {
    "use_hidden": True,  # whether to use hidden or cell state for prediction
    "hidden_size": 500,  # size of *one* team-specific hidden vector, doubles in network
    "num_epochs": 5,  # training epochs
    "optim": "ADAM",  # SGD or ADAM
    "lr": 0.001,  # learning rate used for backprop
    "lr_decay": 0.99,  # if a number is given, lr decay by this factor every epoch
    "dropout": 0.1,
}


def get_logging_path(EXPERIMENT, TASK_TYPE):
    # experiment
    logging_path = os.path.join(f"./tensorboard/{EXPERIMENT}")

    # task
    if TASK_TYPE == 1:
        logging_path = os.path.join(logging_path, "goals")
    else:
        logging_path = os.path.join(logging_path, "wdl")

    # date
    logging_path = os.path.join(
        logging_path,
        time.strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(logging_path, exist_ok=True)

    return logging_path


def get_network_path(EXPERIMENT, TASK_TYPE):
    # experiment
    save_path = os.path.join(f"./saved_networks/{EXPERIMENT}")

    # task
    if TASK_TYPE == 1:
        save_path = os.path.join(save_path, "goals")
    else:
        save_path = os.path.join(save_path, "wdl")

    # date
    save_path = os.path.join(
        save_path,
        time.strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(save_path, exist_ok=True)

    return save_path


def get_necessary_experiment_cols(exp):
    """Returns the columns where entries are necessary in order"""
    if exp == "exp2a" or exp == "exp3a" or exp == "exp4a":
        return market_value_cols + est_market_value_cols
    elif exp == "exp2b" or exp == "exp3b" or exp == "exp4b":
        return betting_odds_cols + est_betting_odds_cols
    elif exp == "exp2c" or exp == "exp3c" or exp == "exp4c":
        return match_stats_cols + est_stats_cols
    elif exp == "exp2d" or exp == "exp3d" or exp == "exp4d":
        return market_value_cols + betting_odds_cols + match_stats_cols + est_market_value_cols + est_betting_odds_cols + est_stats_cols


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
