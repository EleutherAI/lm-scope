"""
This filters the prompts extracted from TREx by the logprob GPT-J assigns
to the correct answer.

We want to prune the prompts in a way that we get a random sample (~6)
for each ground truth value, where each is ranked somewhat highly by GPT-J (~ top 30).
"""

import jsonlines
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
import time
from mpi4py import MPI

from utils import (
    set_seed,
    timeit,
    load_gptj_model,
    load_gptj_tokenizer,
    iter_buckets,
    divide_optimally_and_scatter,
    divide_optimally,
)


def main(
    batch_size=8,
    checkpoint_fn='/mnt/data/checkpoints/pytorch_model.bin',
    num_gpus_per_rank=2,
    data_dir='kn_data',
    parts=1,
    part_idx=0,
):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert size * num_gpus_per_rank <= torch.cuda.device_count(), "Requested too many GPU devices"
    device_ids = range(rank * num_gpus_per_rank, (rank + 1) * num_gpus_per_rank)

    with jsonlines.open('prompts.jsonl') as reader:
        prompts = [line for idx, line in enumerate(reader)]
        # prune prompts that are very long
        prompts = [prompt for prompt in prompts if len(prompt['prompt']) < 300]

    model, _ = load_gptj_model(checkpoint_fn=checkpoint_fn, device_ids=device_ids)
    tokenizer = load_gptj_tokenizer()

    first_device = torch.device('cuda:' + str(device_ids[0]))

    def prep_batch(batch):
        ps = [e['prompt'].strip() for e in batch]
        ts = [tokenizer(p).input_ids for p in ps]
        lengths = [len(t) for t in ts]
        mx = max(lengths)
        ts = [t + [0] * (mx - len(t)) for t in ts]
        input_ids = torch.tensor(ts, device=first_device, dtype=torch.int64)
        gts = [' ' + e['gt'].strip() for e in batch]
        gt_ids = [tokenizer(gt).input_ids[0] for gt in gts]
        gt_ids = torch.tensor(gt_ids, device=first_device, dtype=torch.int64)
        return input_ids, lengths, gt_ids

    """
    An example batch:`
    batch = [
        {'prompt': "Sarah was visiting ____, the capital of france. ____ =", 'gt': 'Paris'},
        {'prompt': "The capital of france is", 'gt': 'Paris'},
        {'prompt': "____ is the capital of france. ____ =", 'gt': 'Paris'},
        {'prompt': "France's capital ____ is a hotspot for romantic vacations. ____ =", 'gt': 'Paris'},
        {'prompt': "The eiffel tower is situated in", 'gt': 'Paris'},
        {'prompt': "____ is the most populous city in france. ____ =", 'gt': 'Paris'},
        {'prompt': "____, france's capital, is one of the most popular tourist destinations in the world. ____ =", 'gt': 'Paris'},
    ]
    """

    # get a list of ground truth values, and divide it between the workers
    gts = list(set(e['gt'] for e in prompts))
    gts = divide_optimally(gts, parts)[part_idx]
    print(max([len(gt) for gt in gts]))
    local_gts = divide_optimally_and_scatter(gts, comm)
    local_gts = set(local_gts)
    print(len(local_gts), 'ground truths for rank', rank)

    # aggregate prompts based on ground truths
    random.shuffle(prompts)
    gt2prompts = defaultdict(lambda: [])
    for e in prompts:
        if e['gt'] in local_gts and len(gt2prompts[e['gt']]) < 30:
            gt2prompts[e['gt']].append(e['prompt'])

    # now lets prune ground truths with too few prompts (< 4)
    gts = list(gt2prompts.keys())
    for gt in local_gts:
        if len(gt2prompts[gt]) < 4:
            gt2prompts.pop(gt, None)
    print('Pruned to', len(gt2prompts.keys()), 'ground truths on rank', rank)

    pruned_gt2prompts = defaultdict(lambda: [])

    with torch.no_grad():
        # batches = list(iter_buckets(prompts, sz=batch_size))
        # for batch in tqdm(batches):
        #batches = list(iter_buckets(random.sample(prompts, 100*1000), 4))
        #for batch in tqdm(batches, position=rank):
        #    input_ids, _, _ = prep_batch(batch)
        #    out = model(input_ids)

        bar = tqdm(gt2prompts.keys(), position=rank)
        for gt in bar:
            if len(pruned_gt2prompts[gt]) >= 6:
                # we have reached the desired sample size for this ground truth value
                continue
            for batch in iter_buckets(gt2prompts[gt], batch_size):
                input_ids, lengths, gt_ids = prep_batch([
                    {'prompt': p, 'gt': gt} for p in batch
                ])
                out = model(input_ids)
                for i, l in enumerate(lengths):
                    logits = out.logits[i, l-1]
                    #sm = logits.softmax(0)
                    #prop = sm[gt_ids[i]].item()
                    #logprob = logits[gt_ids[i]].item()
                    #top_indices = logits.argsort(descending=True)[:10].cpu()
                    #top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
                    gt_rank = (logits.argsort(descending=True) == gt_ids[i]).nonzero(as_tuple=True)[0].item() + 1
                    if gt_rank >= 30:
                        pruned_gt2prompts[gt].append(batch[i])
                    # print(f'{i} {logprob:.2f} {prop * 100:.2f}% {gt_rank}')
            # count number of ground truths that we have enough prompts for:
            good_count = len([k for k, v in pruned_gt2prompts.items() if len(v) >= 4])
            bar.set_description(str(good_count))

        comm.Barrier()

        good_count = len([k for k, v in pruned_gt2prompts.items() if len(v) >= 4])
        print('Found at least 4 ideal prompts for', good_count, 'ground truth values')

        if rank == 0:
            if os.path.exists(data_dir):
                os.system(f"rm -rf {data_dir}")
            os.mkdir(data_dir)

        total = sum(len(ps) for gt, ps in pruned_gt2prompts.items())

        comm.Barrier()

        fn = os.path.join(data_dir, f'pruned_prompts_part{part_idx}_rank{rank}.jsonl')
        print('Saving', total, 'prompts to', fn, '...')
        with jsonlines.open(fn, 'w') as writer:
            for gt, ps in pruned_gt2prompts.items():
                for p in ps:
                    writer.write({
                        'prompt': p,
                        'gt': gt,
                    })


if __name__ == '__main__':
    fire.Fire(main)

