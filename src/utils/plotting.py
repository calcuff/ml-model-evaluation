import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Plot mean accuracies for a set of k values
def plot_k_val_results(k_choices, k_vals, metric:str, title:str, image_name:str, sub_title:str=None):
    mean_vals = []
    std_devs = []
    for k in k_choices:
        # Calculate mean for each k value
        mean_vals.append(np.mean(k_vals[k]))
        # Std deviation for each k value
        std_devs = np.std(k_vals[k])
    
    # Make scatter plot
    plt.scatter(k_choices, mean_vals)
    # Error bars for standard deviation of each k
    plt.errorbar(k_choices, mean_vals, yerr=std_devs)

    # Formatting
    title = title
    if sub_title is not None:
        title += f'\n{sub_title}'
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel(metric)
    # Save for later
    plt.savefig(image_name, bbox_inches='tight')
    # Display
    plt.show()
    

def results_to_csv(k_values, k_accuracies, k_f1s, filename):
    results = []
    for k in k_values:
        acc_mean = np.mean(k_accuracies[k])
        f1_mean = np.mean(k_f1s[k])
        results.append({"k": k, "mean_accuracy": acc_mean, "mean_f1": f1_mean})

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv(filename, index=False)
    
def decision_tree_rsults_to_csv(stop_criteria_accuracies, stop_criteria_f1s, results_file):
    results = []
    for criterion in stop_criteria_accuracies.keys():
        accs = stop_criteria_accuracies[criterion]
        f1s = stop_criteria_f1s[criterion]
        results.append({
            "stop_criterion": criterion,
            "mean_accuracy": np.mean(accs),
            "mean_f1": np.mean(f1s)
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    
def plot_dt_results(results_file, dataset:str, image_name):
    df = pd.read_csv(results_file)
    
    # Melt for long-format bar plot
    plot_df = df.melt(id_vars="stop_criterion", value_vars=["mean_accuracy", "mean_f1"],
                    var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=plot_df, x="Score", y="stop_criterion", hue="Metric", palette="Set2")

    # Add value labels
    for p in ax.patches:
        score = p.get_width()
        ax.text(score + 0.005, p.get_y() + p.get_height() / 2, f"{score:.2f}", va="center")

    plt.title(f"Decision Tree Performance by Stop Criterion\n{dataset} dataset")
    plt.xlabel("Score")
    plt.ylabel("Stop Criterion")
    plt.tight_layout()
    plt.savefig(image_name)
    plt.show()
    
def plot_nn_results(results_file, dataset:str, image_name):
    df = pd.read_csv(results_file)
    df["config"] = df["hidden_dims"].astype(str) + " | lr=" + df["learning_rate"].astype(str) + " | reg=" + df["regularization"].astype(str)

    # Get top 10 configs by F1
    top = df.sort_values("mean_f1", ascending=False).head(10)

    # Melt for grouped barplot
    plot_df = top[["config", "mean_accuracy", "mean_f1"]].melt(id_vars="config", var_name="Metric", value_name="Score")

    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_df,
        x="Score",
        y="config",
        hue="Metric",
        palette="Set2"
    )

    # Add value labels
    for p in ax.patches:
        width = p.get_width()
        ax.text(
            width + 0.005,
            p.get_y() + p.get_height() / 2,
            f"{width:.3f}",
            va="center"
        )

    plt.title(f"Top 10 Neural Network Configurations by F1 Score \n{dataset} dataset")
    plt.xlabel("Score")
    plt.ylabel("Model Configuration")
    plt.legend(title="Metric")
    plt.tight_layout()

    plt.savefig(image_name)
    plt.show()