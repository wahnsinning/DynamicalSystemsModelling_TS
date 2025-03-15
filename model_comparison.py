import numpy as np
from parameter_fitting import param_fit_grid_search_parallel


def BIC(LL, k, n):
    """Compute the Bayesian Information Criterion

    Inputs
    -------
    LL: log likelihood
    k: number of parameters
    n: number of data points
    """
    return -2 * LL + k * np.log(n)


def model_fit(
    data,
    T,
    x_0,
    g_values,
    c_values,
    alpha_values,
    gamma_values,
    sigma,
    tau_P=1,
    model_names=["base", "perseverance", "feedback", "PF"],
):
    n = len(data)
    print("fitting data to models: ", model_names)
    print("number of data points: ", n)
    results = {}
    if "base" in model_names:
        # Fit the base model (alpha and gamma fixed to 0)
        best_params_base, best_LL_base, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, np.zeros(1), np.zeros(1), sigma, tau_P
        )
        best_g_base, best_c_base, best_alpha_base, best_gamma_base = best_params_base
        k_base = 2  # c and g are free parameters
        BIC_base = BIC(best_LL_base, k_base, n)
        results["base"] = {
            "g": best_g_base,
            "c": best_c_base,
            "alpha": best_alpha_base,
            "gamma": best_gamma_base,
            "LL": best_LL_base,
            "BIC": BIC_base,
        }

    if "perseverance" in model_names:
        # Fit the persistence model (alpha fixed to 0)
        best_params_P, best_LL_P, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, np.zeros(1), gamma_values, sigma, tau_P
        )
        best_g_P, best_c_P, best_alpha_P, best_gamma_P = best_params_P
        k_P = 3  # c, g, gamma are free parameters
        BIC_P = BIC(best_LL_P, k_P, n)
        results["perseverance"] = {
            "g": best_g_P,
            "c": best_c_P,
            "alpha": best_alpha_P,
            "gamma": best_gamma_P,
            "LL": best_LL_P,
            "BIC": BIC_P,
        }

    if "feedback" in model_names:
        # Fit the full model with feedback (gamma fixed to 0)
        best_params_F, best_LL_F, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, alpha_values, np.zeros(1), sigma, tau_P
        )
        best_g_F, best_c_F, best_alpha_F, best_gamma_F = best_params_F
        k_F = 3  # c, g, alpha are free parameters
        BIC_F = BIC(best_LL_F, k_F, n)
        results["feedback"] = {
            "g": best_g_F,
            "c": best_c_F,
            "alpha": best_alpha_F,
            "gamma": best_gamma_F,
            "LL": best_LL_F,
            "BIC": BIC_F,
        }

    if "PF" in model_names:
        # Fit the full model with perseverance and feedback (all parameters free to vary)
        best_params_PF, best_LL_PF, _ = param_fit_grid_search_parallel(
            data, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma, tau_P
        )
        best_g_PF, best_c_PF, best_alpha_PF, best_gamma_PF = best_params_PF
        k_PF = 4  # c, g, alpha, gamma are free parameters
        BIC_PF = BIC(best_LL_PF, k_PF, n)
        results["PF"] = {
            "g": best_g_PF,
            "c": best_c_PF,
            "alpha": best_alpha_PF,
            "gamma": best_gamma_PF,
            "LL": best_LL_PF,
            "BIC": BIC_PF,
        }

    return results
