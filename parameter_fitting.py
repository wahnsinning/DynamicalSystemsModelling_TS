# defining the functions for parameter fitting
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import itertools
from tqdm.notebook import tqdm
from model import simulate_dynamics
from model_simulation import compute_choice_probabilities


# defining a function for computing the log likelihoods of the parameters given the data
def compute_log_likelihood(df, T, x_0, g, c, alpha, gamma, sigma):
    """
    This function computes the log likelihood for a given set of model parameters

    Input arguments:
      df: data frame
      c: parameter used to compute the softmax parameter beta
    """
    # initialize log likelihood
    LL = 0
    choices = df["choice_adjusted"].values
    tasks = df["task"].values
    responses = df["response"].values
    num_trials = len(df)
    num_sample_points_per_trial = 100

    for trial_number in range(num_trials):
        if trial_number == 0:
            feedback = 0
        else:
            feedback = 1 if responses[trial_number] == 1 else -1

        # when computing the likelihood, we don't need to make a choice
        # instead, we observe the choice from the data
        choice = choices[trial_number]

        # get input from df for current trial
        if tasks[trial_number] == "letter":
            input = np.array([1, 0])
        else:
            input = np.array([0, 1])

        _, x1_values, x2_values, P_values = simulate_dynamics(
            T, x_0, g, alpha, gamma, input, feedback, sigma, num_sample_points=num_sample_points_per_trial
        )  # run model i.e. solve OED

        # update x_0
        x_0 = np.array([x1_values[-1], x2_values[-1], P_values[-1]])

        activities_n_final = x_0[:2]  # take x1 and x2 final activities

        choice_probabilities = compute_choice_probabilities(activities_n_final, trial_number, c)

        LL = LL + np.log(choice_probabilities[choice])

    return LL


# defining a function for grid search of the log likelihods of the paramerers given the data
def param_fit_grid_search(df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma=0.2, pbar_start_pos=0):
    num_c, num_g, num_alpha, num_gamma = len(c_values), len(g_values), len(alpha_values), len(gamma_values)
    LL_matrix = np.zeros(shape=(num_g, num_c, num_alpha, num_gamma))
    best_LL = -np.inf
    best_c = 0
    best_g = 0
    best_alpha = 0
    best_gamma = 0

    # loop through each parameter combination (grid search)
    for g_idx, g in tqdm(enumerate(c_values), total=num_c, desc="c", leave=False, position=pbar_start_pos):
        for c_idx, c in tqdm(enumerate(g_values), total=num_g, desc="g", leave=False, position=pbar_start_pos + 1):
            for alpha_idx, alpha in tqdm(
                enumerate(alpha_values), total=num_alpha, desc="alpha", leave=False, position=pbar_start_pos + 2
            ):
                for gamma_idx, gamma in tqdm(
                    enumerate(gamma_values), total=num_gamma, desc="gamma", leave=False, position=pbar_start_pos + 3
                ):
                    # compute log likelihood
                    LL = compute_log_likelihood(df, T, x_0, g, c, alpha, gamma, sigma)

                    # store log likelihood in the data matrix
                    LL_matrix[g_idx, c_idx, alpha_idx, gamma_idx] = LL

                    if LL > best_LL:
                        best_g = g
                        best_c = c
                        best_alpha = alpha
                        best_gamma = gamma
                        best_LL = LL

    return best_g, best_c, best_alpha, best_gamma, best_LL, LL_matrix


# Function to create 2D heatmaps
def plot_heatmap(matrix, y_values, x_values, y_label, x_label, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    img = ax.imshow(
        matrix,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
    )
    # ax.set_xticks(np.linspace(x_values[0], x_values[-1], num=len(x_values)))
    # ax.set_yticks(np.linspace(y_values[0], y_values[-1], num=len(y_values)))
    # ax.set_xticklabels([round(val, 2) for val in x_values])
    # ax.set_yticklabels([round(val, 2) for val in y_values])

    plt.colorbar(img, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if ax is None:
        plt.show()

    return ax


# region parallelized parameter fitting
def param_fit_grid_search_parallel(df, T, x_0, g_values, c_values, alpha_values, gamma_values, sigma=0.2, n_jobs=-1):
    """
    Efficient parallelized parameter fitting by splitting work **without storing all combinations**.
    """
    # Ensure parameters are numpy arrays
    g_values = np.array(g_values)
    c_values = np.array(c_values)
    alpha_values = np.array(alpha_values)
    gamma_values = np.array(gamma_values)

    param_sets = (g_values, c_values, alpha_values, gamma_values)

    # Total number of combinations
    total_combinations = len(g_values) * len(c_values) * len(alpha_values) * len(gamma_values)

    def process_chunk(start_idx, end_idx, param_sets):
        """
        Worker function to process a range of parameter combinations lazily.
        """
        g_values, c_values, alpha_values, gamma_values = param_sets
        results = []

        # Generate only the necessary combinations within this range
        total_combinations = itertools.product(g_values, c_values, alpha_values, gamma_values)

        for idx, (g, c, alpha, gamma) in enumerate(total_combinations):
            if start_idx <= idx < end_idx:
                LL = compute_log_likelihood(df, T, x_0, g, c, alpha, gamma, sigma)
                results.append((LL, g, c, alpha, gamma))

        return results

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = min(cpu_count(), total_combinations)  # Don't use more cores than needed

    # Compute range for each worker
    chunk_size = total_combinations // n_jobs
    ranges = [(i * chunk_size, (i + 1) * chunk_size if i < n_jobs - 1 else total_combinations) for i in range(n_jobs)]

    # Parallel execution: each worker processes a *disjoint range* of combinations
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(start, end, param_sets)
        for start, end in tqdm(ranges, desc="Parallel Grid Search", leave=False)
    )

    # Flatten results
    results = [item for sublist in results for item in sublist]

    # Initialize matrix
    LL_matrix = np.zeros((len(g_values), len(c_values), len(alpha_values), len(gamma_values)))
    best_LL = -np.inf
    best_params = None

    # Populate matrix
    for LL, g, c, alpha, gamma in results:
        g_idx = np.where(g_values == g)[0][0]
        c_idx = np.where(c_values == c)[0][0]
        alpha_idx = np.where(alpha_values == alpha)[0][0]
        gamma_idx = np.where(gamma_values == gamma)[0][0]

        LL_matrix[g_idx, c_idx, alpha_idx, gamma_idx] = LL

        if LL > best_LL:
            best_LL = LL
            best_params = (g, c, alpha, gamma)

    return *best_params, best_LL, LL_matrix
