import gc
from tqdm import tqdm
from functools import partial
import fire
import torch
import os
from transformers import AutoTokenizer, GPTJForCausalLM
from multiprocessing import Process, Queue
import time
from collections import defaultdict
import os
import json
from tqdm import tqdm
import jsonlines
import pickle
import numpy as np
from functools import cmp_to_key
from multiprocessing import Pool, Queue


def get_range_of_file(fn):
    # neuron<start>-<end>.pickle.<part>
    a, b = list(map(int, fn[7:fn.find('.')].split('-')))
    return a, b


def get_index_of_file(fn):
    return int(fn.split('.')[-1])


def get_raw_data_files_in_order(data_dir):
    def _cmp_files(a, b):
        if get_range_of_file(a)[0] < get_range_of_file(b)[0]:
            return -1
        elif get_range_of_file(a)[0] > get_range_of_file(b)[0]:
            return 1
        elif get_index_of_file(a) < get_index_of_file(b):
            return -1
        elif get_index_of_file(a) > get_index_of_file(b):
            return 1
        else:
            return 0
    
    fns = os.listdir(data_dir)
    fns = [fn for fn in fns if 'pickle' in fn]
    fns = sorted(fns, key=cmp_to_key(_cmp_files))
    return fns


def iter_neuron_records(with_pbar=True):
    data_dir = 'data/raw'
    for fn in tqdm(get_raw_data_files_in_order(data_dir)):
        with open(os.path.join(data_dir, fn), 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break


def iter_hidden_neurons():
    for l in range(28):
        for f in range(4096*4):
            yield l, f


def round_all(v, decimals=2):
    return [round(e, 2) for e in v]


def round_all_2d(vv, decimals=2):
    return [round_all(v, decimals=decimals) for v in vv]


def generate_example_index(args):
    idx, example, index_path, tokenizer = args
    attn = np.array(example['attentions'])
    # layer, batch, head, seq, seq
    attentions = {}
    for l in range(28):
        for h in range(16):
            seq = attn.shape[-1]
            for s1 in range(seq):
                for s2 in range(seq):
                    v = attn[l, 0, h, s1, s2]
                    if v > 0.1:
                        attentions[f'{l}:{h}:{s1}:{s2}'] = round(v, 2)

    with open(os.path.join(index_path, f'example-{idx:05}.json'), 'w') as f:
        f.write(json.dumps({
            'example': example['text'],
            'hidden': [{'l': e['l'], 'f': e['f'], 'a': round_all(e['a'])} for e in example['hidden']],
            'logits': example['logits'],
            'tokens': [tokenizer.decode([t]) for t in tokenizer(example['text'])['input_ids']],
            'attentions': attentions,
        }))


def process(q, f):
    while True:
        args = q.get()
        if args is None:
            break
        f(args)


def main(index_path='index',
         data_path='data/raw',
         np=1):

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    if not os.path.exists(index_path):
        os.makedirs(index_path, exist_ok=True)

    print(f'Generating example-level indices into the {index_path} folder...')
    q = Queue(maxsize=np)
    p = Pool(np, initializer=process, initargs=(q, generate_example_index))
    for idx, record in enumerate(iter_neuron_records()):
        if idx > 100:
            break
        q.put((idx, record, index_path, tokenizer))
    for _ in range(np):
        q.put(None)

    print(f'Generating neuron-level indices into the {index_path} folder...')
    neuron_to_example_indices = defaultdict(list)
    for idx, record in enumerate(iter_neuron_records()):
        for neuron in record['hidden']:
            mx = max(neuron['a'])
            if mx >= 2:
                neuron_to_example_indices[(neuron['l'], neuron['f'])].append({ 'a': neuron['a'], 'exampleIdx': idx })

    for k in tqdm(neuron_to_example_indices.keys()):
        neuron_to_example_indices[k] = list(sorted(neuron_to_example_indices[k], key=lambda e: max(e['a']), reverse=True))

    for (l, f) in tqdm(list(iter_hidden_neurons())):
        if (l, f) in neuron_to_example_indices:
            with open(os.path.join(index_path, f'neuron-{l}-{f}.json'), 'w') as file:
                file.write(json.dumps(neuron_to_example_indices[(l, f)]))


if __name__ == '__main__':
    fire.Fire(main)

