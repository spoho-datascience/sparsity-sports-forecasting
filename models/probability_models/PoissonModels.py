from scipy.stats import poisson, skellam
import numpy as np
import pandas as pd

def score_grid(lambda1, lambda2, max_goals=9, return_scorematrix=True):
    """
    Takes two scoring rates lambda1 and lambda2, calculates the probability of outcomes for
    all combinations of goals from 0 to max_goals and returns this score grid as pandas df,
    if return_scorematrix==True. Based on score grid, the probabilities of the three possible
    match outcomes as well as the most likely result are calculated and returned as pandas df.
    All calculations are based on a Double Poisson model, i.e. independence between scoring rates
    is assumed.
    :param lambda1: scoring rate of home team
    :param lambda2: scoring rate of away team
    :param max_goals: maximum number of possible goals considered in score grid
    :param return_scorematrix: Boolean, indicates whether score grid should be returned or only outcome prediction
    :return: pandas df with predictions for outcome, or, if return_scorematrix==True, tuple of two pandas df
    """
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in [lambda1, lambda2]]

    score_matrix = pd.DataFrame(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

    pred_df = pd.DataFrame({'pred_home' : [np.sum(np.tril(score_matrix, -1))],
                            'pred_draw' : [np.sum(np.diag(score_matrix))],
                            'pred_away' : [np.sum(np.triu(score_matrix, 1))],
                            })

    # get the most probable result
    largest_value_index = score_matrix.values.argmax()
    row_index, col_index = divmod(largest_value_index, len(score_matrix.columns))

    pred_df['likely_home_score'] = row_index
    pred_df['likely_away_score'] = col_index

    if return_scorematrix == True:
        return score_matrix, pred_df
    else:
        return pred_df


def plot_score_grid(lambda1, lambda2):
    """
    takes two scoring rates, calculates the probabilities of results based on score_grid function,
    and plots the score grid with color coding for probability of respective outcome
    :param lambda1: scoring rate of home team
    :param lambda2: scoring rate of away team
    :return: None
    """
    df, dict_prob = score_grid(lambda1, lambda2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    plt.ylabel('Home Goals')
    plt.xlabel('Away Goals')
    plt.xticks(np.arange(0, 10, 1))
    plt.yticks(np.arange(0, 10, 1))
    c = ax.pcolor(df, edgecolors='k', linewidths=4)

    fig.tight_layout()
    plt.show()