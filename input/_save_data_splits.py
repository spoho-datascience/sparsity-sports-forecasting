# saves "training_dataset.csv" into local season splits "train.csv" and "test.csv"
import os
import numpy as np
import pandas as pd
import models.imputation.models as knn
import models.imputation.simple as simple
from datetime import datetime
from input.preprocessing import (
    # remove_duplicate_unique_ids,
    remove_rows_with_empty_label_cols,
    remove_rows_with_empty_meta_cols,
    replace_market_value_zero_with_nan,
    rename_duplicate_teams,
    compute_market_value_log,
    # compute_odds_average,
)
from input.features import (
    add_data_type_col,
    add_results_integer_col,
    add_probabilities_from_betting_odds_col,
    add_day_of_year_col,
    add_metadata_col,
)
from input.prediction import (
    add_estimated_market_value_and_stats_cols,
    add_estimated_betting_odds_col,
)
from input.utilities import relevant_cols, eval_seasons


# load data
# data_path = os.path.join("input", "training_data")
data_path = os.path.join(os.getcwd(), '\\training_data')
# data = pd.read_csv(os.path.join(data_path, "full_dataset_from_DB.csv"))
data = pd.read_csv("input\\full_dataset_from_DB.csv")

# create date and sort from earliest to latest
data["iso_date"] = pd.to_datetime(data["iso_date"])
data = data.sort_values(by="iso_date")

# preprocessing
# data = remove_duplicate_unique_ids(data)
data = rename_duplicate_teams(data)
data = remove_rows_with_empty_label_cols(data)
data = remove_rows_with_empty_meta_cols(data)
# data = replace_market_value_zero_with_nan(data) redundant because logarithm(0) = NA
data = compute_market_value_log(data)
# data = compute_odds_average(data)

# add feature columns
data = add_data_type_col(data)
data = add_results_integer_col(data)
data = add_probabilities_from_betting_odds_col(data)
data = add_day_of_year_col(data)
data = add_metadata_col(data)

# add estimated feature columns (Exp 2 & 3)
data = add_estimated_market_value_and_stats_cols(data)
data = add_estimated_betting_odds_col(data)

# remove non-relevant columns from dataset
concise_data = pd.DataFrame(columns=relevant_cols)
for col in relevant_cols:
    concise_data[col] = data[col]
data = concise_data
    
# initialize train/test splits with start and end date (simulate prediction challenge)
train_sets = {}  # train: all available matches until day x for season
train_sets_simple = {}  # train sets, but with data imputed by simple imputation
# train_sets_knn = {}  # train sets, but with data imputed by knn imputation
test_sets = {}  # test: all matches from day x to day y for season
start_date = (4, 1)  # month and day when the test section starts (incl.)
end_date = (5, 31)  # month and day when the test section ends (incl.)
for season in eval_seasons:
    season_year = 2000 + int(season[-2:])
    # integrate all available data until the challenge and impute on this data
    train = data[
        data["iso_date"] < datetime(season_year, start_date[0], start_date[1])
    ]
    train_sets[season] = train
    train_sets_simple[season] = simple.SimpleImputation().impute(train)
    train = train.fillna(np.nan)
    # train_sets_knn[season] = knn.KNNImputation().impute(train)
    
    test = data[data["iso_date"] >= datetime(season_year, start_date[0], start_date[1])]
    test = test[test["iso_date"] <= datetime(season_year, end_date[0], end_date[1])]
    test_sets[season] = test

# save train and test sets to file
train_path = "../../train"
test_path = "../../test"
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
for season in eval_seasons:
    train_sets[season].to_csv(os.path.join(train_path, f"{season}.csv"))
    train_sets_simple[season].to_csv(os.path.join(train_path, f"{season}"+"_simple"+".csv"))
    # train_sets_knn[season].to_csv(os.path.join(train_path, f"{season}"+"_knn"+".csv"))
    test_sets[season].to_csv(os.path.join(test_path, f"{season}.csv"))
