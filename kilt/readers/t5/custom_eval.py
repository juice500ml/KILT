from typing import List
from collections import Counter
import numpy as np
import csv
import argparse
import re
import os
import json


def load_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        data = [line.strip() for line in file.readlines()]

    return data

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return " ".join(
            [word for word in text.split() if word not in ["a", "an", "the"]]
        )

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punct(text):
        return "".join([char for char in text if char.isalnum() or char == " "])

    # for each word, turn <something> into something
    def remove_wrapper(text):
        return re.sub(r"<([^>]+)>", r"\1", text)

    def extract_binary(text):
        if text.startswith("yes") or text.startswith("no"):
            return text.split()[0]
        return text

    return white_space_fix(extract_binary(remove_articles(remove_punct(remove_wrapper(s.lower())))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, recall

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate(gold_answers, predictions):
    exact_match = 0
    total = len(gold_answers)
    f1_scores = []
    recalls = []

    assert len(gold_answers) == len(predictions)

    for ground_truth, prediction in zip(gold_answers, predictions):
        exact_match += int(exact_match_score(prediction, ground_truth))
        f1, recall = f1_score(prediction, ground_truth)
        f1_scores.append(f1)
        recalls.append(recall)

    exact_match = 100.0 * exact_match / total
    macro_f1 = np.mean(f1_scores) * 100.0
    answer_recall = np.mean(recalls) * 100.0

    return {
        "exact_match": exact_match,
        "macro_f1": macro_f1,
        "answer_recall": answer_recall,
    }

def plot_results(k, f1, path, dataset):
    import matplotlib.pyplot as plt

    # Plot the F1 scores against k
    plt.plot(k, f1, label='Macro F1')
    plt.scatter(k, f1, color='green')  # Mark each data point

    # Setting the labels and title
    plt.xlabel("k")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 vs k")

    # Ensure x-axis ticks show only the integer values of k
    plt.xticks(k)

    # Save the figure
    plt.savefig(f"{path}/{dataset}.png")
    plt.close()  # Close the plot to free up memory


def main(args):
    PATH = "kilt/readers/t5/f1"
    file_path = f"{PATH}/{args.dataset}_results.json"

    gold_answers = load_file(args.gold_answers)
    predictions = load_file(args.predictions)

    results = evaluate(gold_answers, predictions)

    print("Macro F1: {:.2f}".format(results['macro_f1']))
    print("Answer Recall: {:.2f}".format(results['answer_recall']))
    print("Exact Match: {:.2f}".format(results['exact_match']))

    data = {}

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        # create the file if it doesn't exist
        with open(file_path, "w") as file:
            json.dump({}, file)

    data[f"k={args.k}"] = results

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_answers",
        type=str,
        required=True,
        help="Path to the file containing the gold answers.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the file containing the model predictions.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fever",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of top context passages to include.",
    )

    args = parser.parse_args()
    main(args)
