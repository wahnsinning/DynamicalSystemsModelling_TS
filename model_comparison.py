import numpy as np
from parameter_fitting import param_fit_grid_search_parallel


def bic(k, n, ll):
    return -2 * ll + k * np.log(n)


def fit_parameters(data_cleaned, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma, tau_P=1):
    # Fit the base model (alpha and gamma fixed to 0)
    best_params_base, best_LL_base, LL_matrix_base = param_fit_grid_search_parallel(
        data_cleaned, T, x_0, g_values, c_values, np.zeros(1), np.zeros(1), sigma, tau_P
    )
    best_g_base, best_c_base, best_alpha_base, best_gamma_base = best_params_base

    # Fit the persistence model (alpha fixed to 0)
    best_params_P, best_LL_P, LL_matrix_P = param_fit_grid_search_parallel(
        data_cleaned, T, x_0, g_values, c_values, np.zeros(1), gamma_values, sigma, tau_P
    )
    best_g_P, best_c_P, best_alpha_P, best_gamma_P = best_params_P

    # Fit the full model with feedback (gamma fixed to 0)
    best_params_F, best_LL_F, LL_matrix_F = param_fit_grid_search_parallel(
        data_cleaned, T, x_0, g_values, c_values, alpha_values, np.zeros(1), sigma, tau_P
    )
    best_g_F, best_c_F, best_alpha_F, best_gamma_F = best_params_F

    # Fit the full model with perseverance and feedback (all parameters free to vary)
    best_params_PF, best_LL_PF, LL_matrix_PF = param_fit_grid_search_parallel(
        data_cleaned, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma, tau_P
    )
    best_g_PF, best_c_PF, best_alpha_PF, best_gamma_PF = best_params_PF

    return {
        "best_LL_base": best_LL_base,
        "best_c_base": best_c_base,
        "best_g_base": best_g_base,
        "best_alpha_base": best_alpha_base,
        "best_gamma_base": best_gamma_base,
        "best_LL_P": best_LL_P,
        "best_c_P": best_c_P,
        "best_g_P": best_g_P,
        "best_alpha_P": best_alpha_P,
        "best_gamma_P": best_gamma_P,
        "best_LL_F": best_LL_F,
        "best_c_F": best_c_F,
        "best_g_F": best_g_F,
        "best_alpha_F": best_alpha_F,
        "best_gamma_F": best_gamma_F,
        "best_LL_PF": best_LL_PF,
        "best_c_PF": best_c_PF,
        "best_g_PF": best_g_PF,
        "best_alpha_PF": best_alpha_PF,
        "best_gamma_PF": best_gamma_PF,
        "n": len(data_cleaned),  # Number of trials (data points)
    }


def calculate_bic(results):
    n = results["n"]

    # Base model BIC
    k_base = 2  # c and g are free parameters
    BIC_base = bic(k_base, n, results["best_LL_base"])

    # Persistence model BIC
    k_P = 3  # c, g, alpha, gamma are free parameters
    BIC_P = bic(k_P, n, results["best_LL_P"])

    # Feedback model BIC
    k_F = 3  # c, g, alpha, gamma are free parameters
    BIC_F = bic(k_F, n, results["best_LL_F"])

    # Perseverance Feedback model BIC
    k_PF = 4  # c, g, alpha, gamma are free parameters
    BIC_PF = bic(k_PF, n, results["best_LL_PF"])

    return BIC_base, BIC_P, BIC_F, BIC_PF
