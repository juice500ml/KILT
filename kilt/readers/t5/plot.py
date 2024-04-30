import json
import matplotlib.pyplot as plt
import argparse

PATH = "kilt/readers/t5/f1"

def plot_k_vs_macro_f1(file_path):
    # Extract the dataset name from the filename
    dataset_name = file_path.split('/')[-1].split('_')[0]

    # Load the data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Prepare lists to store the values of k and corresponding Macro F1 scores
    ks = []
    macro_f1_scores = []
    
    # Iterate over possible k values, assuming they are labeled as 'k=0' to 'k=10'
    for k in range(6):
        key = f'k={k}'
        if key in data:
            ks.append(k)
            macro_f1_scores.append(data[key]['macro_f1'])
        else:
            print(f"Warning: '{key}' not found in the data.")
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(ks, macro_f1_scores, marker='o', linestyle='-', color='b')
    plt.title(f'{dataset_name}')
    plt.xlabel('k')
    plt.ylabel('Macro F1 Score')
    plt.grid(True)
    plt.xticks(ks)  # Ensure ticks for every k value present
    # plt.show()
    plt.savefig(f"{PATH}/{dataset_name}_f1.png")

# Example usage:
# plot_k_vs_macro_f1('path_to_your_file.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, help='Name of the dataset'
    )
    args = parser.parse_args()
    plot_k_vs_macro_f1(f'{PATH}/{args.dataset}_results.json')
