"""
Adapted to work for GPT-J from https://github.com/EleutherAI/knowledge-neurons/blob/main/pararel_evaluate.py
"""

from knowledge_neurons import (
    KnowledgeNeurons,
    pararel_expanded,
    pararel
)
from transformers import AutoTokenizer, GPTJForCausalLM
import random
import torch
import torch.nn.functional as F
from functools import lru_cache
import fire
from tqdm import tqdm
import json
import os 
import numpy as np
from collections import defaultdict

from utils import make_device_map, set_seed, timeit
from data import trex


def main(batch_size=1,
         steps=20,
         checkpoint_fn='/mnt/data/checkpoints/pytorch_model.bin',
         results_dir='gpt_j_neurons_pararel',
         adaptive_threshold=0.3,
         p=0.3,
         seed=42,
         ):
    """
    Identify the Knowledge Neurons of GPT-J
    """

    set_seed(seed)

    # Loading PareRel Dataset
    PARAREL = pararel_expanded(autoregressive=True)
    INDICES = list(range(len(PARAREL)))
    KEYS = list(PARAREL.keys())

    with timeit():
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        print('Found', len(devices), 'devices')
        device_map = make_device_map(num_layers=28, num_devices=len(devices))

        print('Loading tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

        print('Loading model checkpoint to state dict...')
        state_dict = torch.load(checkpoint_fn, map_location='cpu')
        print('Creating model with state dict...')
        model = GPTJForCausalLM.from_pretrained(None, state_dict=state_dict, config="EleutherAI/gpt-j-6B")
        print('Parallelizing model onto GPUs...')
        model.parallelize(device_map)
        model.eval()

    quit()

    kn = KnowledgeNeurons(model, tokenizer, model_type='gpt-j', device='cuda:0')

    RESULTS = {}
    NEURONS = {}

    @lru_cache(maxsize=None)
    def get_neurons(_uuid):
        PROMPTS, GROUND_TRUTH, RELATION_NAME = (
            PARAREL[_uuid]["sentences"],
            PARAREL[_uuid]["obj_label"],
            PARAREL[_uuid]["relation_name"],
        )
        print('Getting refined neurons', PROMPTS)
        neurons = kn.get_refined_neurons(
            prompts=PROMPTS,
            ground_truth=GROUND_TRUTH.lower(),
            p=p,
            batch_size=batch_size,
            steps=steps,
            coarse_adaptive_threshold=adaptive_threshold,
            quiet=False,
        )
        return neurons, PARAREL[_uuid]

    def get_unrelated_fact(KEYS, uuid):
        n_keys = len(KEYS)
        while True:
            random_uuid = KEYS[random.randint(0, n_keys - 1)]
            if random_uuid == uuid:
                continue
            return random_uuid

    # go through each item in the PARAREL dataset, get the refined neurons, save them, and evaluate the results when suppressing the
    # refined neurons vs. unrelated neurons.
    print('Finding neurons...')
    for i, idx in enumerate(INDICES):
        print(i, '...')
        uuid = KEYS[idx]
        neurons, data = get_neurons(uuid)  # get refined neurons
        unrelated_uuid = get_unrelated_fact(
            KEYS, uuid
        )  # get a uuid for an unrelated fact / relation
        unrelated_neurons, unrelated_data = get_neurons(
            unrelated_uuid
        )  # get the unrelated neurons

        # initialize a results dict
        results_this_uuid = {
            "suppression": {
                "related": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(data["sentences"]),
                },
                "unrelated": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(unrelated_data["sentences"]),
                }},
            "enhancement": {
                "related": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(data["sentences"]),
                },
                "unrelated": {
                    "pct_change": [],
                    "correct_before": [],
                    "correct_after": [],
                    "n_prompts": len(unrelated_data["sentences"]),
                }},
        }

        for PROMPT in data["sentences"]:
            print('    ', PROMPT)
            gt = data["obj_label"].lower()
            # really should be using a different for the suppression, but the authors didn't make their bing dataset available
            suppression_results, _ = kn.suppress_knowledge(PROMPT, gt, neurons, quiet=True)
            enhancement_results, _ = kn.enhance_knowledge(PROMPT, gt, neurons, quiet=True)

            # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
            suppression_prob_diff = (suppression_results["after"]["gt_prob"] - suppression_results["before"]["gt_prob"]) / suppression_results["before"]["gt_prob"]
            results_this_uuid["suppression"]["related"]["pct_change"].append(suppression_prob_diff)

            enhancement_prob_diff = (enhancement_results["after"]["gt_prob"] - enhancement_results["before"]["gt_prob"]) / enhancement_results["before"]["gt_prob"]
            results_this_uuid["enhancement"]["related"]["pct_change"].append(enhancement_prob_diff)

            # check whether the answer was correct before/after suppression
            results_this_uuid["suppression"]["related"]["correct_before"].append(
                suppression_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["suppression"]["related"]["correct_after"].append(
                suppression_results["after"]["argmax_completion"] == gt
            )

            results_this_uuid["enhancement"]["related"]["correct_before"].append(
                enhancement_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["enhancement"]["related"]["correct_after"].append(
                enhancement_results["after"]["argmax_completion"] == gt
            )

        for PROMPT in unrelated_data["sentences"]:
            # do the same but with unrelated facts

            gt = unrelated_data["obj_label"].lower()

            unrelated_suppression_results, _ = kn.suppress_knowledge(
                PROMPT, gt, neurons, quiet=True
            )
            unrelated_enhancement_results, _ = kn.suppress_knowledge(
                PROMPT, gt, neurons, quiet=True
            )

            # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
            suppression_prob_diff = (unrelated_suppression_results["after"]["gt_prob"] - unrelated_suppression_results["before"]["gt_prob"]) / unrelated_suppression_results["before"]["gt_prob"]
            results_this_uuid["suppression"]["unrelated"]["pct_change"].append(suppression_prob_diff)
            enhancement_prob_diff = (unrelated_enhancement_results["after"]["gt_prob"] - unrelated_enhancement_results["before"]["gt_prob"]) / unrelated_enhancement_results["before"]["gt_prob"]
            results_this_uuid["enhancement"]["unrelated"]["pct_change"].append(enhancement_prob_diff)

            # check whether the answer was correct before/after suppression
            results_this_uuid["suppression"]["unrelated"]["correct_before"].append(
                unrelated_suppression_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["suppression"]["unrelated"]["correct_after"].append(
                unrelated_suppression_results["after"]["argmax_completion"] == gt
            )

            results_this_uuid["enhancement"]["unrelated"]["correct_before"].append(
                unrelated_enhancement_results["before"]["argmax_completion"] == gt
            )
            results_this_uuid["enhancement"]["unrelated"]["correct_after"].append(
                unrelated_enhancement_results["after"]["argmax_completion"] == gt
            )

        results_this_uuid["n_refined_neurons"] = len(neurons)
        results_this_uuid["n_unrelated_neurons"] = len(unrelated_neurons)
        results_this_uuid["relation_name"] = data["relation_name"]
        RESULTS[uuid] = results_this_uuid
        NEURONS[uuid] = neurons

    # save results + neurons to json file
    with open(results_dir / f"gpt_j_pararel_neurons.json", "w") as f:
        json.dump(NEURONS, f, indent=4)
    with open(results_dir / f"gpt_j_pararel_results.json", "w") as f:
        json.dump(RESULTS, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
