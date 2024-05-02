import json
import matplotlib.pyplot as plt
import argparse

PATH = "kilt/readers/t5/all/"

def map_metrics(dataset_name):
    if dataset_name == 'fever' or dataset_name == 'trex' or dataset_name == 'zeroshot':
        metric = 'accuracy'
    elif dataset_name == 'nq' or dataset_name == 'hotpotqa' or dataset_name == 'triviaqa':
        metric = 'em'
    elif dataset_name == 'eli5':
        metric = 'rougel'
    elif dataset_name == 'wow':
        metric = 'f1'

    return metric

def plot_k_vs_macro_f1_multiple(fixed, M_adaptive_k, dataset, retriever):
    # Load the data from the JSON file
    with open(fixed, 'r') as file:
        fixed_data = json.load(file)
    # with open(M_top_k, 'r') as file:
    #     M_top_k_data = json.load(file)
    with open(M_adaptive_k, 'r') as file:
        M_adaptive_k_data = json.load(file)

    metric = map_metrics(dataset)
    
    # Prepare lists to store the values of k and corresponding Macro F1 scores
    ks = []
    fixed_metric_scores = []
    M_adaptive_k_metric_scores = []
    
    # Iterate over possible k values, assuming they are labeled as 'k=0' to 'k=10'
    for k in range(6):
        key = f'k={k}'
        if key in fixed_data:
            ks.append(k)
            if k == 0:
                M_adaptive_k_metric_scores.append(fixed_data[key]['downstream'][metric])
            else:
                M_adaptive_k_metric_scores.append(M_adaptive_k_data[key]['downstream'][metric])

            fixed_metric_scores.append(fixed_data[key]['downstream'][metric])
            
        else:
            print(f"Warning: '{key}' not found in the data.")
    
    # if dataset == 'fever':
    #     print('ks: ', ks)
    #     print('fixed: ', fixed_macro_f1_scores)
    #     print('M_top-k: ', M_top_k_macro_f1_scores)
    #     print('M_adaptive-k: ', M_adaptive_k_macro_f1_scores)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(ks, fixed_metric_scores, marker='o', linestyle='-', color='blue', label='Top-k')
    plt.plot(ks, M_adaptive_k_metric_scores, marker='o', linestyle='-', color='orange', label='Adaptive-k')
    plt.title(f'{dataset} with {retriever}')
    plt.xlabel('k')
    if metric == 'em': 
        plt.ylabel('Exact Match')
    plt.ylabel(f'{metric.capitalize()}')
    plt.grid(True)
    plt.xticks(ks)  # Ensure ticks for every k value present
    plt.legend()
    # plt.show()
    plt.savefig(f"{PATH}/{dataset}_{metric}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, help='Name of the dataset'
    )
    parser.add_argument(
        '--retriever', type=str, help='Name of the retriever'
    )
    args = parser.parse_args()

    fixed = f'{PATH}/fixed/{args.dataset}_results.json'
    # M_top_k = f'{PATH}/adaptive/{args.dataset}_results.json'
    M_adaptive_k = f'{PATH}/adaptive/{args.dataset}_results.json'

    # plot_k_vs_macro_f1(f'{PATH}/{args.dataset}_results.json', args.dataset)
    plot_k_vs_macro_f1_multiple(fixed, M_adaptive_k, args.dataset, args.retriever)
