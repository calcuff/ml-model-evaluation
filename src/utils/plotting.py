import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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