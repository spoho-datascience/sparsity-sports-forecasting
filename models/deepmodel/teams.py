import torch
from models.deepmodel.utilities import hparams


def initialize_team_specific_variables(teams):
    num_teams = len(teams)
    team_dic = {}
    for idx in range(num_teams):
        team_dic[teams[idx]] = idx
    team_states = {}
    hidden_by_team = torch.randn((num_teams, hparams["hidden_size"]))
    cell_by_team = torch.randn((num_teams, hparams["hidden_size"]))
    team_states["hidden"] = hidden_by_team
    team_states["cell"] = cell_by_team

    return team_dic, team_states
