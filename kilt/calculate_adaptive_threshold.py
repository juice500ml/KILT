import argparse
from typing import List

from kilt import kilt_utils


def get_adaptive_threshold(list_of_scores: List[List[float]], k: float):
    """
    Parameters
    list_of_scores:
        Example. Q1's retrieved docs' scores: [4, 3, 1], Q2's scores: [3, 3, 2]
        then, it should be [ [4, 3, 1], [3, 3, 2] ]
        Fyi, it doesn't need to be sorted
    k: hyperparameter (float)

    Returns
    threshold (float): it has to 
    """
    num_questions = len(list_of_scores)
    scores = [s for ss in list_of_scores for s in ss]
    scores = sorted(scores, reverse=True)
    thr = scores[int(num_questions * k) - 1]
    return thr


def get_adaptive_threshold_from_jsonl(guess_file: str, k: float):
    guess_dataset = kilt_utils.load_data(guess_file)
    list_of_scores = []
    for guess_item in guess_dataset:
        list_of_scores.append([float(i["score"]) for i in guess_item["output"][0]["provenance"]])
    return get_adaptive_threshold(list_of_scores, k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--guess", help="Guess KILT file", type=str)
    parser.add_argument("--k", help="Hyperparameter k", type=float)

    args = parser.parse_args()
    thr = get_adaptive_threshold_from_jsonl(args.guess, args.k)
    print(f"Threshold for {args.k}: {thr}")
