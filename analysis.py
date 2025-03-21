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
        plt.draw()


import matplotlib.pyplot as plt


def analyze_accuracy_by_previous_repeats(df, title="", ax=None):
    """
    Analyzes and plots accuracy based on the number of preceding consecutive repeat trials.
    """
    df["prev_repeat_count"] = 0
    prev_count = 0

    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "repeat":
            df.loc[i, "prev_repeat_count"] = prev_count
            prev_count += 1
        else:
            prev_count = 0

    accuracy_by_prev_repeats = (
        df[df["trial_type"] == "repeat"].groupby("prev_repeat_count")["response"].mean().reset_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        accuracy_by_prev_repeats["prev_repeat_count"],
        accuracy_by_prev_repeats["response"],
        marker="o",
        linestyle="-",
        color="b",
    )
    ax.set_xlabel("Consecutive Repeat")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True)
    ax.set_ylim(0, 1)

    if ax is None:
        plt.draw()


def analyze_accuracy_by_previous_switches(df, title="", ax=None):
    """
    Analyzes and plots accuracy based on the number of preceding consecutive switch trials.
    """
    df["prev_switch_count"] = 0
    prev_count = 0

    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "switch":
            df.loc[i, "prev_switch_count"] = prev_count
            prev_count += 1
        else:
            prev_count = 0

    accuracy_by_prev_switches = (
        df[df["trial_type"] == "switch"].groupby("prev_switch_count")["response"].mean().reset_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        accuracy_by_prev_switches["prev_switch_count"],
        accuracy_by_prev_switches["response"],
        marker="o",
        linestyle="-",
        color="r",
    )
    ax.set_xlabel("Consecutive Switch")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True)
    ax.set_ylim(0, 1)

    if ax is None:
        plt.draw()


def analyze_switch_accuracy_after_repeats(df, title="", ax=None):
    """
    Analyzes and plots accuracy of switch trials based on the number of preceding consecutive repeat trials.
    """
    df["prev_repeats_before_switch"] = 0
    prev_repeat_count = 0

    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "repeat":
            prev_repeat_count += 1
        else:
            df.loc[i, "prev_repeats_before_switch"] = prev_repeat_count
            prev_repeat_count = 0

    accuracy_by_prev_repeats = (
        df[df["trial_type"] == "switch"].groupby("prev_repeats_before_switch")["response"].mean().reset_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        accuracy_by_prev_repeats["prev_repeats_before_switch"],
        accuracy_by_prev_repeats["response"],
        marker="o",
        linestyle="-",
        color="g",
    )
    ax.set_xlabel("Consecutive Repeat")
    ax.set_ylabel("Accuracy Following Switch")
    ax.set_title(title)
    ax.grid(True)
    ax.set_ylim(0, 1)

    if ax is None:
        plt.draw()


def analyze_repeat_accuracy_after_switches(df, title="", ax=None):
    """
    Analyzes and plots accuracy of repeat trials based on the number of preceding consecutive switch trials.
    """
    df["prev_switches_before_repeat"] = 0
    prev_switch_count = 0

    for i in range(len(df)):
        if df.loc[i, "trial_type"] == "switch":
            prev_switch_count += 1
        else:
            df.loc[i, "prev_switches_before_repeat"] = prev_switch_count
            prev_switch_count = 0

    accuracy_by_prev_switches = (
        df[df["trial_type"] == "repeat"].groupby("prev_switches_before_repeat")["response"].mean().reset_index()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        accuracy_by_prev_switches["prev_switches_before_repeat"],
        accuracy_by_prev_switches["response"],
        marker="o",
        linestyle="-",
        color="purple",
    )
    ax.set_xlabel("Consecutive Switch")
    ax.set_ylabel("Accuracy Following Repeat")
    ax.set_title(title)
    ax.grid(True)
    ax.set_ylim(0, 1)

    if ax is None:
        plt.draw()
