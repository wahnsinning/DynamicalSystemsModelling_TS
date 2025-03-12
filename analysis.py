import pandas as pd
import matplotlib.pyplot as plt


def calculate_accuracy_per_condition(df):
    """
    Calculates the average accuracy for switch and repeat trials.
    """
    # filter the data for switch and repeat trials
    switch_trials = df[df["trial_type"] == "switch"]
    repeat_trials = df[df["trial_type"] == "repeat"]

    # sum over response values for each condition
    sum_accuracy_switch = switch_trials["response"].sum()
    sum_accuracy_repeat = repeat_trials["response"].sum()

    # get the number of switch and repeat trials
    count_switch = len(switch_trials)
    count_repeat = len(repeat_trials)

    # calculate the average accuracy of each condition
    average_accuracy_switch = sum_accuracy_switch / count_switch if count_switch > 0 else 0
    average_accuracy_repeat = sum_accuracy_repeat / count_repeat if count_repeat > 0 else 0

    return {"average_accuracy_switch": average_accuracy_switch, "average_accuracy_repeat": average_accuracy_repeat}


def plot_accuracy_per_condition(df, ax=None):
    """
    Calculates and plots the average accuracy for switch and repeat trials.

    If an Axes object is provided via the 'ax' parameter, the plot is drawn on that Axes.
    Otherwise, a new figure and Axes are created.
    """
    # Calculate average accuracy using the calculate_accuracy_per_condition function
    results = calculate_accuracy_per_condition(df)

    # Extract results for plotting
    trial_types = ["Switch", "Repeat"]
    average_accuracies = [results["average_accuracy_switch"], results["average_accuracy_repeat"]]

    # Create new axes if none is provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True

    # Create the bar plot on the provided (or new) axis
    ax.bar(trial_types, average_accuracies, color=["blue", "green"])
    ax.set_xlabel("Trial Type")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Average Accuracy for Switch and Repeat Trials")
    ax.set_ylim(0, 1)

    # Add values on top of the bars
    for i, value in enumerate(average_accuracies):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom")

    # If a new figure was created, display the plot
    if created_fig:
        plt.show()


def analyze_accuracy_by_previous_repeats(df, df_name):
    """
    Analyzes and plots accuracy based on the number of preceding consecutive repeat trials.

    """
    # Initialize a counter for consecutive repeat trials
    df["prev_repeat_count"] = 0
    prev_count = 0

    # Iterate through the DataFrame to count previous consecutive repeats
    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "repeat":
            df.loc[i, "prev_repeat_count"] = prev_count
            prev_count += 1
        else:
            prev_count = 0  # Reset count if it's a switch trial

    # Calculate accuracy for each previous repeat count
    accuracy_by_prev_repeats = (
        df[df["trial_type"] == "repeat"].groupby("prev_repeat_count")["response"].mean().reset_index()
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        accuracy_by_prev_repeats["prev_repeat_count"],
        accuracy_by_prev_repeats["response"],
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.xlabel("Number of Previous Consecutive Repeat Trials")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Number of Previous Consecutive Repeat Trials\n({df_name})")
    plt.grid(True)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.show()


def analyze_accuracy_by_previous_switches(df, df_name):
    """
    Analyzes and plots accuracy based on the number of preceding consecutive switch trials.

    Parameters:
    df (pd.DataFrame): A DataFrame containing 'trial_type' and 'response'.
    df_name (str): Name of the DataFrame for labeling the plot.
    """
    # Initialize a counter for consecutive switch trials
    df["prev_switch_count"] = 0
    prev_count = 0

    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "switch":
            df.loc[i, "prev_switch_count"] = prev_count
            prev_count += 1
        else:
            prev_count = 0  # Reset count if it's a repeat trial

    # Calculate accuracy for each previous switch count
    accuracy_by_prev_switches = (
        df[df["trial_type"] == "switch"].groupby("prev_switch_count")["response"].mean().reset_index()
    )

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(
        accuracy_by_prev_switches["prev_switch_count"],
        accuracy_by_prev_switches["response"],
        marker="o",
        linestyle="-",
        color="r",
    )
    plt.xlabel("Number of Previous Consecutive Switch Trials")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs. Number of Previous Consecutive Switch Trials\n({df_name})")  # add name of DataFrames
    plt.grid(True)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.show()


def analyze_switch_accuracy_after_repeats(df, df_name):
    """
    Analyzes and plots accuracy of switch trials based on the number of preceding consecutive repeat trials.

    Parameters:
    df (pd.DataFrame): A DataFrame containing 'trial_type' and 'response'.
    df_name (str): Name of the DataFrame for labeling the plot.
    """
    # Initialize a counter for consecutive repeat trials before a switch trial
    df["prev_repeats_before_switch"] = 0
    prev_repeat_count = 0

    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "repeat":
            prev_repeat_count += 1  # Zähle consecutive repeats hoch
        else:  # Falls es ein switch Trial ist
            df.loc[i, "prev_repeats_before_switch"] = prev_repeat_count  # safe anzahl of repeats
            prev_repeat_count = 0  # Reset, cause switch follows

    # calculates Accuracy of switch Trials in dependent on previous repeat Trials
    accuracy_by_prev_repeats = (
        df[df["trial_type"] == "switch"].groupby("prev_repeats_before_switch")["response"].mean().reset_index()
    )

    # Plot der Ergebnisse
    plt.figure(figsize=(10, 6))
    plt.plot(
        accuracy_by_prev_repeats["prev_repeats_before_switch"],
        accuracy_by_prev_repeats["response"],
        marker="o",
        linestyle="-",
        color="g",
    )
    plt.xlabel("Number of Previous Consecutive Repeat Trials")
    plt.ylabel("Accuracy of Following Switch Trial")
    plt.title(f"Accuracy of Switch Trials vs. Previous Repeat Trials\n({df_name})")  # add DataFrame-Name
    plt.grid(True)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.show()


def analyze_repeat_accuracy_after_switches(df, df_name):
    """
    Analyzes and plots accuracy of repeat trials based on the number of preceding consecutive switch trials.

    Parameters:
    df (pd.DataFrame): A DataFrame containing 'trial_type' and 'response'.
    df_name (str): Name of the DataFrame for labeling the plot.
    """
    # Initialize a counter for consecutive switch trials before a repeat trial
    df["prev_switches_before_repeat"] = 0
    prev_switch_count = 0

    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "switch":
            prev_switch_count += 1  # count consecutive switch Trials up
        else:  # if it is repeat Trial
            df.loc[i, "prev_switches_before_repeat"] = prev_switch_count  # safe number of previous switches
            prev_switch_count = 0  # Reset, cause now follows a repeat

    # calculate Accuracy of repeat Trials in dependent on previous switch Trials
    accuracy_by_prev_switches = (
        df[df["trial_type"] == "repeat"].groupby("prev_switches_before_repeat")["response"].mean().reset_index()
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        accuracy_by_prev_switches["prev_switches_before_repeat"],
        accuracy_by_prev_switches["response"],
        marker="o",
        linestyle="-",
        color="purple",
    )
    plt.xlabel("Number of Previous Consecutive Switch Trials")
    plt.ylabel("Accuracy of Following Repeat Trial")
    plt.title(f"Accuracy of Repeat Trials vs. Previous Switch Trials\n({df_name})")  # add DataFrame-Name
    plt.grid(True)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.show()
