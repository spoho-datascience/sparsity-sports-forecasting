import pandas as pd
from utils import mongo_db

success_in_connection, conn, conn_info = mongo_db.open_mongo_secure_connection()
if success_in_connection:
    print('Yup you are connected')
    print(conn_info)

variables = [
    #"avg_home_odds", "avg_draw_odds", "avg_away_odds",  # features: betting odds
    "home_starter_total", "away_starter_total",  # features: market values
    "HTS", "ATS", "HST", "AST", "HF", "AF", "HY", "AY", "HR", "AR"  # features: stats
]

join_tables = {
    'fbd': {
        'collection': 'footballdata_matches',
        'columns': [
            "avg_home_odds", "avg_draw_odds", "avg_away_odds",
            "HTS", "ATS", "HST", "AST", "HF", "AF", "HY", "AY", "HR", "AR"
        ]
    },
    'tfmkt_metrics': {
        'collection': 'tfmkt_metrics',
        'columns': ["home_starter_total", "away_starter_total"]
    }
}

from utils import dataset

cursor_data = dataset.get_raw_data(
    mongo_conn=conn,
    training_filter={"Sea": {"$in": ["15-16", "16-17", "17-18", "18-19", "20-21"]}},
    extra_sources=['fbd', 'tfmkt_metrics'],
    columns=dataset.TRAINING_COLUMNS + \
        [f'fbd.{c}' for c in join_tables['fbd']['columns']] + \
        [f'tfmkt_metrics.{c}' for c in join_tables['tfmkt_metrics']['columns']],
    flatten=True
)

data_df = pd.DataFrame(list(cursor_data))

def get_summary(df):
    summary = {}
    summary_variables = [v for v in df.columns if v not in ['unique_id', 'iso_date', 'HT', 'AT']]
    for variable in summary_variables:
        if variable in ['Sea', 'Lge', 'WDL', 'HS', 'AS']:
            variable_info = df[variable].value_counts(normalize=True).to_dict()
        else:
            variable_info = df[variable].describe().to_dict()
        for value, statistic in variable_info.items():
                summary.setdefault(variable, {})[value] = statistic
    return summary

import plotly.graph_objects as go
from plotly.subplots import make_subplots

list_of_figures_categorical = {}
for v in variables:
    fig = make_subplots(rows=4, cols=1)
    fig.update_layout(
        height=800
    )
    null_values = data_df[v].isnull()
    if any(null_values):
        not_null_summary = get_summary(data_df[null_values])
        null_summary = get_summary(data_df[~null_values])
        for row, variable in enumerate(['Sea', 'WDL', 'HS', 'AS']):
            x = list(not_null_summary.get(variable, {}).keys())
            for k in null_summary.get(variable, {}).keys():
                if k not in x:
                    x.append(k)
            fig.append_trace(
                go.Bar(
                    name="Missing values",
                    x=sorted(x),
                    y=[null_summary.get(variable, {}).get(pointer, 0) for pointer in sorted(x)],
                    marker_color="#EF553B",
                    legendgroup='Missing values',
                    showlegend=False
                    
                ),
                row=row+1, col=1
            )
            fig.append_trace(
                go.Bar(
                    name="Values",
                    x=sorted(x),
                    y=[not_null_summary.get(variable, {}).get(pointer, 0) for pointer in sorted(x)],
                    marker_color="#00CC96",
                    legendgroup='Values',
                    showlegend=False
                ),
                row=row+1, col=1
            )
    fig.update_layout(title=v, xaxis_type='category')
    list_of_figures_categorical[v] = fig

import math

# Draft of simple imputer with season and league means

vs = ['home_starter_total']
window = None
seasons = data_df.Sea.unique()
leagues = data_df.Lge.unique()
for season in seasons:
    season_mask = data_df['Sea'] == season
    for league in leagues:
        league_mask = data_df['Lge'] == league
        season_and_league_mask = season_mask & league_mask
        for v in variables:
            mean_value = data_df[season_and_league_mask][v].mean()
            if math.isnan(mean_value):
                mean_value = data_df[league_mask][v].mean()
                if math.isnan(mean_value):
                    mean_value = data_df[season_mask][v].mean()
                else:
                    mean_value = data_df[v].mean()
            print(mean_value)
            data_df.loc[season_and_league_mask & (data_df[v].isnull()), v] = mean_value

from sklearn.impute import KNNImputer

knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
categorical_variables = data_df[['Sea', 'Lge']]
categorical_encoding = pd.get_dummies(categorical_variables, drop_first=True)
full_df = pd.concat([data_df, categorical_encoding], axis=1)
full_df.loc[:, variables] = knn_imputer.fit_transform(full_df.loc[:, variables])

         
        
        
         









