### DIXON COLES MODEL

# rho correction
def rho_correction(x, y, lambda1, lambda2, rho):
    if x==0 and y==0:
        return 1- (lambda1 * lambda2 * rho)
    elif x==0 and y==1:
        return 1 + (lambda1 * rho)
    elif x==1 and y==0:
        return 1 + (lambda2 * rho)
    elif x==1 and y==1:
        return 1 - rho
    else:
        return 1.0

# likelihood function for ELO
# alpha: coefficient for elo home rating; beta: coefficient for elo away rating
def dc_log_like(x, y, elo_H, elo_A, alpha_x, beta_x, alpha_y, beta_y, rho):
    lambda1, lambda2 = np.exp(alpha_x * elo_H + beta_y * elo_A), np.exp(alpha_y * elo_H + beta_x * elo_A)
    return (np.log(rho_correction(x, y, lambda1, lambda2, rho)) +
            np.log(poisson.pmf(x, lambda1)) + np.log(poisson.pmf(y, lambda2)))


def solve_parameters(df, home_colname='HT', away_colname='AT', debug = False, init_vals=None, options={'disp': True, 'maxiter':100},
                     constraints = [{'type':'eq', 'fun': lambda x: sum(x[:20])-20}] , **kwargs):
    home_teams = np.sort(df[home_colname].unique())
    # check for no weirdness in df
    away_teams = np.sort(df[away_colname].unique())
    if not np.array_equal(home_teams, away_teams):
        raise ValueError("Something's not right")
    n_teams = len(home_teams)
    if init_vals is None:
        # random initialisation of model parameters
        init_vals = np.array([.0015, -.0015, -.0015, .0015, 0])

    def dc_log_like(x, y, elo_H, elo_A, alpha_x, beta_x, alpha_y, beta_y, rho):
        lambda1, lambda2 = np.exp(alpha_x * elo_H + beta_y * elo_A), np.exp(alpha_y * elo_H + beta_x * elo_A)
        return (np.log(rho_correction(x, y, lambda1, lambda2, rho)) +
                np.log(poisson.pmf(x, lambda1)) + np.log(poisson.pmf(y, lambda2)))

    def estimate_paramters(params, home_colname=home_colname, away_colname=away_colname):
        #rho = params[-1]
        log_like = [dc_log_like(row.HS, row.AS, row.elo_H, row.elo_A, init_vals[0], init_vals[1],
                                init_vals[2], init_vals[3], init_vals[4]) for row in df.itertuples()]
        return -sum(log_like)
    opt_output = minimize(estimate_paramters, init_vals, options=options,  **kwargs) #constraints = constraints # removed
    if debug:
        # sort of hacky way to investigate the output of the optimisation process
        return opt_output
    else:
        return dict(zip(["alpha_x", "beta_x", "alpha_y", "beta_y", "rho"],
                        opt_output.x))


t0 = time.time()
params = solve_parameters(dat)
t1 = time.time()

total = t1-t0 # 172 seconds


############ PREDICT GOALS AND MATCH OUTCOME WITH DIXON-COLES
def dixon_coles_simulate_match(params_dict, homeTeam, awayTeam, max_goals=10):
    team_avgs = [np.exp(params_dict['attack_'+homeTeam] + params_dict['defence_'+awayTeam] + params_dict['home_adv']),
                 np.exp(params_dict['defence_'+homeTeam] + params_dict['attack_'+awayTeam])]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in team_avgs]
    output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
    correction_matrix = np.array([[rho_correction(home_goals, away_goals, team_avgs[0],
                                                   team_avgs[1], params['rho']) for away_goals in range(2)]
                                   for home_goals in range(2)])
    output_matrix[:2,:2] = output_matrix[:2,:2] * correction_matrix
    return output_matrix
