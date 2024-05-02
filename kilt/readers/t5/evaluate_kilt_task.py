# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import os
from pathlib import Path
import json

import torch
# from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from finetune import Seq2seqTransformer
from calculate_adaptive_threshold import get_adaptive_threshold_from_jsonl


SEED = 42
torch.manual_seed(SEED)

MAP = {
    'eli5': 24,
    'zeroshot': 7,
    'fever': 11,
    'triviaqa': 18,
    'wow': 21,
    'trex': 7,
    'hotpotqa': 20,
    'nq': 10
}


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# def generate_answers(lns, output_file_path, model, tokenizer, batch_size, device):
#     output_file = Path(output_file_path).open("w")

#     model.to(device)

#     # update config with specific params
#     task_specific_params = model.config.task_specific_params
#     if task_specific_params is not None:
#         model.config.update(task_specific_params.get("nq", {}))

#     for batch in tqdm(list(chunks(lns, batch_size))):
#         batch = [(model.config.prefix or '') + text for text in batch]

#         dct = tokenizer.batch_encode_plus(
#             batch, max_length=512, return_tensors="pt", pad_to_max_length=True
#         )
#         input_ids = dct["input_ids"].to(device)
#         attention_mask = dct["attention_mask"].to(device)

#         answers = model.generate(input_ids=input_ids, attention_mask=attention_mask)
#         dec = [
#             tokenizer.decode(
#                 g, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )
#             for g in answers
#         ]

#         for hypothesis in dec:
#             output_file.write(hypothesis + "\n")
#             output_file.flush()


def generate_answers(questions, output_file_path, model, tokenizer, batch_size, device, contexts, top_k, M, task, threshold):
    model.config.max_length = 512
    output_file = Path(output_file_path).open("w")

    model.to(device)

    # update config with specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get("nq", {}))

    for i, batch_questions in enumerate(tqdm(list(chunks(questions, batch_size)))):
        batch_contexts = contexts[i * batch_size:(i + 1) * batch_size]
        batch_with_context = []

        for question, context_passages in zip(batch_questions, batch_contexts):
            task_prefix = f'{task}: ' if task else ''
            combined_input = task_prefix + question

            if threshold is None:
                k = top_k
            else:
                k = len([1 for item in context_passages if float(item["score"]) >= threshold])

            M = M // k if k > 0 else 0

            # Combine question with top-k context passages
            if k > 0:
                # context_texts = []
                for idx, p in enumerate(context_passages[:k]):
                    try:
                        # p_tokens = tokenizer.encode(f' context {idx+1}: {p["text"]}', add_special_tokens=False)
                        text = p["text"]
                    except KeyError:
                        # p_tokens = tokenizer.encode(f' context {idx+1}: {p["texts"]}', add_special_tokens=False)
                        text = p["texts"]
                    # context_tokens.extend(p_tokens[:M])
                    truncated_tokens = tokenizer.encode(text, add_special_tokens=False)[:M]
                    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    # context_texts.append(f'context {idx+1}: {truncated_text}')
                    combined_input += f' [SEP] context {idx+1}: {truncated_text}'
                
                # context = tokenizer.decode(context_tokens, skip_special_tokens=True)
                # context = ' '.join(context_texts)
                
                if i == 0 and batch_questions.index(question) == 0:
                    print(f"\nCombined input: {combined_input}")
                    
                # combined_input = f'question: {question} {context}'

            encoded_length = len(tokenizer.encode(combined_input))

            # print(f"\nEncoded length: {encoded_length}")
            # print(f"\nModel max length: {model.config.max_length}")

            if encoded_length > model.config.max_length:
                print(f"k = {k}, Encoded {encoded_length} Truncating input as it exceeds the max length for question: {question}")

            # We still add the combined input for processing, it will be truncated later during encoding
            batch_with_context.append(combined_input)

        # Encoding with truncation to max length
        dct = tokenizer.batch_encode_plus(
            batch_with_context, max_length=model.config.max_length, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = dct["input_ids"].to(device)
        attention_mask = dct["attention_mask"].to(device)

        # Generate answers
        answers = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        
        # Decode generated answers
        dec = [
            tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for g in answers
        ]

        # Write to output file
        for hypothesis in dec:
            output_file.write(hypothesis + "\n")
            output_file.flush()

    output_file.close()


def calculate_rouge(output_lns, reference_lns, score_path):
    score_file = Path(score_path).open("w")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    score_file.write(
        "ROUGE_1: \n{} \n\n ROUGE_2: \n{} \n\n ROUGE_L: \n{} \n\n".format(
            result["rouge1"], result["rouge2"], result["rougeL"]
        )
    )


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        help="T5 model size, either 't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'. Defaults to 't5-base'.",
        default="t5-base",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="like nqa/test_articles_questions.txt",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="where to save summaries",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="where to save the model",
    )
    # parser.add_argument(
    #     "reference_path", type=str, help="like nqa/test_reference_answers.txt"
    # )
    # parser.add_argument(
    #     "score_path",
    #     type=str,
    #     help="where to save the rouge score",
    # )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        required=False,
        help="batch size: how many to summarize at a time",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        type=bool,
        help="Whether to force the execution on CPU.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='t5-base',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--dataset",
        default='fever'
    )
    parser.add_argument(
        "--data_dir",
        default='data/dev'
    )

    parser.add_argument(
        "--context_path",
        type=str,
        help="Path to the JSONL file containing context passages.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of top context passages to include.",
    )
    parser.add_argument(
        "--adaptive_k",
        action="store_true",
        help="Turn on adaptive k thresholding"
    )

    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )

    source_lns = [x.rstrip() for x in open(args.input_path).readlines()]

    
    contexts = []
    with open(args.context_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            contexts.append(data['output'][0]['provenance'])  # Assuming this is the correct key

    # Check if contexts list is valid
    if not contexts or len(contexts) != len(source_lns):
        raise ValueError("The number of contexts must match the number of questions.")


    sq2sq = Seq2seqTransformer(args)
    checkpoints = list(
        sorted(
            glob.glob(
                os.path.join(args.output_dir, f"{args.dataset}.ckpt"), recursive=True
            )
        )
    )

    # print(checkpoints)

    model = sq2sq.load_from_checkpoint(checkpoints[-1]).model

    tokenizer = sq2sq.tokenizer

    L = tokenizer.model_max_length
    # print(L)
    L_bar = MAP[args.dataset]

    if args.top_k == 0:
        M = 0
    else:
        # M = (L - L_bar) // args.top_k
        M = (L - L_bar)

    if args.dataset == 'fever':
        task = "Fact Checking"
    elif args.dataset == 'nq' or args.dataset == 'triviaqa' or args.dataset == 'hotpotqa' or args.dataset == 'eli5':
        task = "Question Answering"
    elif args.dataset == 'trex' or args.dataset == 'zeroshot':
        task = "Relation Extraction"
    elif args.dataset == 'wow':
        task = "Dialogue"

    threshold = None
    if args.adaptive_k:
        threshold = get_adaptive_threshold_from_jsonl(args.context_path, args.top_k)
    generate_answers(
        source_lns, args.output_path, model, tokenizer, args.batch_size, args.device, contexts, args.top_k, M, task, threshold
    )
    # output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
    # reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

    # calculate_rouge(output_lns, reference_lns, args.score_path)


if __name__ == "__main__":
    run_generate()
