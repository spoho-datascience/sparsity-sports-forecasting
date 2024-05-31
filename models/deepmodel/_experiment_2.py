import os

import pandas as pd
import numpy as np
from datetime import datetime
from models.deepmodel.network import (
    create_deep_model_classes,
    train_network_for_season,
    get_network_predictions_for_season,
    save_network_to_file,
    save_predictions_to_file
)
from models.deepmodel.teams import initialize_team_specific_variables
from input.utilities import eval_seasons
from models.deepmodel.utilities import get_logging_path, get_network_path
from models.deepmodel.tboard import (
    log_performance_to_tensorboard,
    log_hparams_to_tensorboard,
    print_team_individual_wdl_performance,
)
from input.features import split_dataset_by_relevant_columns
from models.deepmodel.utilities import get_necessary_experiment_cols, hparams
from input.utilities import label_cols, meta_cols

# choose experiment and task
EXPERIMENT = "exp2d"  # exp2a: values, exp2b: odds, exp2c: stats, exp2d: all
TASK_TYPE = 2  # 1: goals prediction; 2: outcome prediction
necessary_cols = get_necessary_experiment_cols(EXPERIMENT)

# setup paths
meta_path = "F:\\work\\data\\2023-soccer-prediction\\meta"
train_path = "F:\\work\\data\\2023-soccer-prediction\\train"
test_path = "F:\\work\\data\\2023-soccer-prediction\\test"
logging_path = get_logging_path(EXPERIMENT, TASK_TYPE)
network_path = get_network_path(EXPERIMENT, TASK_TYPE)

# setup deep network
network, optimizer, scheduler = create_deep_model_classes(TASK_TYPE, EXPERIMENT)

# setup team specific variables
teams_data = pd.read_csv(os.path.join(meta_path, f"teams.csv"))
teams = np.concatenate((teams_data["HT"].unique(), teams_data["AT"].unique()))
team_dic, team_states = initialize_team_specific_variables(teams)

# EXPERIMENT 2: Leave out matches with missing data in columns
epoch_results = []
print(f"Model Training for experiment {EXPERIMENT} with hparams {hparams}")
for epoch in range(hparams["num_epochs"]):
    print(f"Epoch: {epoch}")
    all_seasons_prd = {}
    all_seasons_test = pd.DataFrame()
    for s, season in enumerate(eval_seasons):
        train = pd.read_csv(os.path.join(train_path, f"{season}.csv"))

        # cut matches that were already used for training previous season
        if s > 0:
            train["iso_date"] = pd.to_datetime(train["iso_date"])
            prev_season_year = 2000 + int(eval_seasons[s - 1][-2:])
            start_date = (4, 1)  # month and day when the test section starts (incl.)
            train = train[
                train["iso_date"]
                >= datetime(prev_season_year, start_date[0], start_date[1])
            ]

        # cut all but the relevant columns for Experiment 2
        keep_cols = label_cols + meta_cols + necessary_cols
        train = train[keep_cols]

        # cut train rows with missing entries in necessary columns for Experiment 2
        train_necessary_cols = [col for col in necessary_cols if "est" not in col]
        train = split_dataset_by_relevant_columns(train, train_necessary_cols)[0]

        # train model
        network, result_by_team, prediction_by_team = train_network_for_season(
            network=network,
            train=train,
            season=season,
            team_states=team_states,
            team_dic=team_dic,
            optimizer=optimizer,
            TASK_TYPE=TASK_TYPE,
            EXPERIMENT=EXPERIMENT,
        )

        # update lr scheduler
        if scheduler is not None:
            scheduler.step()

        # save network and predictions at current epoch
        save_network_to_file(network, epoch, network_path)

        if epoch == hparams["num_epochs"] - 1:
            test = pd.read_csv(os.path.join(test_path, f"{season}.csv"))
            test = test[keep_cols]

            # create test predictions from model
            season_predictions = get_network_predictions_for_season(
                network=network,
                test=test,
                team_states=team_states,
                team_dic=team_dic,
                result_by_team=result_by_team,
                prediction_by_team=prediction_by_team,
                TASK_TYPE=TASK_TYPE,
                EXPERIMENT=EXPERIMENT,
            )

            # append season results to containers
            all_seasons_test = pd.concat((all_seasons_test, test))
            all_seasons_prd.update(season_predictions)

            # log season evaluation metrics to tensorboard
            # log_performance_to_tensorboard(
            #     test, season_predictions, season, epoch, logging_path, TASK_TYPE
            # )

            # log overall evaluation metrics to tensorboard and add metric to container
            epoch_result = log_performance_to_tensorboard(
                all_seasons_test, all_seasons_prd, "total", epoch, logging_path, TASK_TYPE
            )
            epoch_results.append(epoch_result)

        # print performance for specific teams and save predictions for DEBUG
        #if TASK_TYPE == 2:
            #print_team_individual_wdl_performance(
            #    all_seasons_test, all_seasons_prd, TASK_TYPE
            #)
        #    save_predictions_to_file(all_seasons_prd, epoch, network_path)

# after training: log hyperparameters along with the best epoch performance
log_hparams_to_tensorboard(epoch_results, logging_path, TASK_TYPE)

df = pd.DataFrame()
df["unique_id"] = list(all_seasons_prd.keys())
df["prd_D"] = [all_seasons_prd[unique_id][0, 1].cpu().detach().numpy() for unique_id in all_seasons_prd]
df["prd_W"] = [all_seasons_prd[unique_id][0, 0].cpu().detach().numpy() for unique_id in all_seasons_prd]
df["prd_L"] = [all_seasons_prd[unique_id][0, 2].cpu().detach().numpy() for unique_id in all_seasons_prd]
