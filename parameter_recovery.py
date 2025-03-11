import numpy as np
from tqdm import tqdm
from ctypes import ArgumentError
from model_simulation import simulate_experiment
from parameter_fitting import param_fit_grid_search_parallel


def param_recovery(
    num_trials,
    T,
    x_0,
    to_recover_g,
    to_recover_c,
    to_recover_alpha,
    to_recover_gamma,
    g_values,
    c_values,
    alpha_values,
    gamma_values,
    sigma=0.2,
):
    # make sure all parameters to recover have the same number of values
    num_true_param_set = len(to_recover_g)
    if (
        len(to_recover_c) != num_true_param_set
        or len(to_recover_alpha) != num_true_param_set
        or len(to_recover_gamma) != num_true_param_set
    ):
        raise ArgumentError(
            "you have to give the same amount of values for all parameters to recover (g, c, alpha, gamma)"
        )

    # Create arrays for logging true and fitted parameters
    true_g_log = np.zeros(num_true_param_set)
    true_c_log = np.zeros(num_true_param_set)
    true_alpha_log = np.zeros(num_true_param_set)
    true_gamma_log = np.zeros(num_true_param_set)

    fitted_g_log = np.zeros(num_true_param_set)
    fitted_c_log = np.zeros(num_true_param_set)
    fitted_alpha_log = np.zeros(num_true_param_set)
    fitted_gamma_log = np.zeros(num_true_param_set)
    LL_log = np.zeros(num_true_param_set)

    # For each run, simulate data with randomly sampled parameters and attempt to fit those data
    for run in tqdm(range(num_true_param_set), total=num_true_param_set, desc="run", leave=True, position=0):

        # Randomly sample model parameters from which to sample surrogate data
        true_g = to_recover_g[run]
        true_c = to_recover_c[run]
        true_alpha = to_recover_alpha[run]
        true_gamma = to_recover_gamma[run]

        print(true_alpha, true_gamma)
        # Log those parameters
        true_g_log[run] = true_g
        true_c_log[run] = true_c
        true_alpha_log[run] = true_alpha
        true_gamma_log[run] = true_gamma

        # Simulate surrogate data
        df = simulate_experiment(num_trials, T, x_0, true_g, true_c, true_alpha, true_gamma, sigma)
        # best_g, best_c, best_alpha, best_gamma, best_LL, LL_matrix = param_fit_grid_search(df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma,pbar_start_pos=1)
        best_g, best_c, best_alpha, best_gamma, best_LL, LL_matrix = param_fit_grid_search_parallel(
            df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma
        )

        # Store the parameter combination yielding the highest log likelihood
        fitted_g_log[run] = best_g
        fitted_c_log[run] = best_c
        fitted_alpha_log[run] = best_alpha
        fitted_gamma_log[run] = best_gamma
        LL_log[run] = best_LL

    return fitted_g_log, fitted_c_log, fitted_alpha_log, fitted_gamma_log, LL_log
