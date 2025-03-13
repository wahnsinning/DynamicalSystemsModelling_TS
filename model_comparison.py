import numpy as np
from parameter_fitting import param_fit_grid_search_parallel

def bic(k, n, ll):
    return -2 * ll + k * np.log(n)

def fit_parameters(data_cleaned, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma):
    # Fit the persistence model (gamma also free to vary)
    best_g_P, best_c_P, best_alpha_P, best_gamma_P, best_LL_P, LL_matrix_P = param_fit_grid_search_parallel(
        data_cleaned, T, x_0, g_values, c_values, np.zeros(1), gamma_values, sigma)
    
    # Fit the full model with strategy feedback (all parameters free to vary)
    best_g_PF, best_c_PF, best_alpha_PF, best_gamma_PF, best_LL_PF, LL_matrix_PF = param_fit_grid_search_parallel(
        data_cleaned, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma)
    
    # Fit the base model (alpha and gamma fixed to 0)
    best_g_base, best_c_base, best_alpha_base, best_gamma_base, best_LL_base, LL_matrix_base = param_fit_grid_search_parallel(
        data_cleaned, T, x_0, g_values, c_values, np.zeros(1), np.zeros(1), sigma)
    
    return {
        "best_LL_base": best_LL_base,
        "best_c_base": best_c_base,
        "best_g_base": best_g_base,
        "best_alpha_base": 0,  # alpha is fixed to 0 in the base model
        "best_gamma_base": 0,  # gamma is fixed to 0 in the base model
        "best_LL_P": best_LL_P,
        "best_c_P": best_c_P,
        "best_g_P": best_g_P,
        "best_alpha_P": 0,  # gamma is fixed to 0 in the persistence model
        "best_gamma_P": best_gamma_P,
        "best_LL_PF": best_LL_PF,
        "best_c_PF": best_c_PF,
        "best_g_PF": best_g_PF,
        "best_alpha_PF": best_alpha_PF,
        "best_gamma_PF": best_gamma_PF,
        "n": len(data_cleaned)  # Number of trials (data points)
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
    k_PF = 4  # c, g, alpha, gamma are free parameters
    BIC_PF = bic(k_PF, n, results["best_LL_PF"])

    return BIC_base, BIC_P, BIC_PF