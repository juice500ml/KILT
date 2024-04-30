import json
import matplotlib.pyplot as plt
import argparse

PATH = "kilt/readers/t5/f1/"

def plot_k_vs_macro_f1(file_path, dataset_name):
    # Extract the dataset name from the filename
    # dataset_name = file_path.split('/')[-1].split('_')[0]

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

def plot_k_vs_macro_f1_multiple(fixed, M_top_k, M_adaptive_k, dataset):
    # Load the data from the JSON file
    with open(fixed, 'r') as file:
        fixed_data = json.load(file)
    with open(M_top_k, 'r') as file:
        M_top_k_data = json.load(file)
    with open(M_adaptive_k, 'r') as file:
        M_adaptive_k_data = json.load(file)
    
    # Prepare lists to store the values of k and corresponding Macro F1 scores
    ks = []
    fixed_macro_f1_scores = []
    M_top_k_macro_f1_scores = []
    M_adaptive_k_macro_f1_scores = []
    
    # Iterate over possible k values, assuming they are labeled as 'k=0' to 'k=10'
    for k in range(6):
        key = f'k={k}'
        if key in fixed_data:
            ks.append(k)
            if k == 0:
                M_top_k_macro_f1_scores.append(fixed_data[key]['macro_f1'])
                M_adaptive_k_macro_f1_scores.append(fixed_data[key]['macro_f1'])
            else:
                M_top_k_macro_f1_scores.append(M_top_k_data[key]['macro_f1'])
                M_adaptive_k_macro_f1_scores.append(M_adaptive_k_data[key]['macro_f1'])

            fixed_macro_f1_scores.append(fixed_data[key]['macro_f1'])
            
        else:
            print(f"Warning: '{key}' not found in the data.")
    
    # if dataset == 'fever':
    #     print('ks: ', ks)
    #     print('fixed: ', fixed_macro_f1_scores)
    #     print('M_top-k: ', M_top_k_macro_f1_scores)
    #     print('M_adaptive-k: ', M_adaptive_k_macro_f1_scores)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(ks, fixed_macro_f1_scores, marker='o', linestyle='-', color='b', label='Fixed')
    plt.plot(ks, M_top_k_macro_f1_scores, marker='o', linestyle='-', color='y', label='M_Top-k')
    plt.plot(ks, M_adaptive_k_macro_f1_scores, marker='o', linestyle='-', color='g', label='M_Adaptive-k')
    plt.title(f'{dataset}')
    plt.xlabel('k')
    plt.ylabel('Macro F1 Score')
    plt.grid(True)
    plt.xticks(ks)  # Ensure ticks for every k value present
    plt.legend()
    # plt.show()
    plt.savefig(f"{PATH}/{dataset}_f1.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, help='Name of the dataset'
    )
    args = parser.parse_args()

    fixed = f'{PATH}/fixed/{args.dataset}_results.json'
    M_top_k = f'{PATH}/adaptive/{args.dataset}_results.json'
    M_adaptive_k = f'{PATH}/adaptive_new/{args.dataset}_results.json'

    # plot_k_vs_macro_f1(f'{PATH}/{args.dataset}_results.json', args.dataset)
    plot_k_vs_macro_f1_multiple(fixed, M_top_k, M_adaptive_k, args.dataset)
