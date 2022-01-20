from functools import lru_cache
from collections import defaultdict
from multiprocessing import Pool, Queue
import time
import os
from azure.storage.blob import BlobServiceClient
import random
import torch
import numpy as np
from transformers import AutoTokenizer, GPTJForCausalLM


CHECKPOINT_NAMES = [
    'step_500',
    'step_38500',
    'step_78500',
    'step_118500',
    'step_158500',
    'step_198500',
    'step_238500',
    'step_278500',
]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_raw_directory(dataset_name: str, checkpoint_name: str) -> str:
    """ Format: raw/<dataset_name>/<checkpoint_name> """
    return os.path.join('raw', dataset_name, checkpoint_name)


def get_index_directory(dataset_name: str, checkpoint_name: str) -> str:
    """ Format: index/<dataset_name>/<checkpoint_name> """
    return os.path.join('index', dataset_name, checkpoint_name)


def get_checkpoint_blob_path(checkpoint_name: str):
    # subject to change
    return checkpoint_name + '.pt'


@lru_cache(maxsize=None)
def az_make_blob_client(blob_path: str):
    account_name = os.environ['AZ_ACCOUNT_NAME']
    account_key = os.environ['AZ_ACCOUNT_KEY']
    container_name = os.environ['AZ_CONTAINER_NAME']
    account_url = f'https://{account_name}.blob.core.windows.net'
    blob_service_client = BlobServiceClient(account_url=account_url, credential=account_key)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
    return blob_client


def az_upload_file(local_fn: str, remote_fn: str):
    client = az_make_blob_client(remote_fn)
    with open(local_fn, 'rb') as data:
        client.upload_blob(data, overwrite=True)


def az_download_file(local_fn: str, remote_fn: str, max_concurrency=8):
    client = az_make_blob_client(remote_fn)
    with open(local_fn, 'wb') as f:
        f.write(client.download_blob(max_concurrency=max_concurrency, validate_content=True).readall())


def download_checkpoint(checkpoint_name: str):
    assert checkpoint_name in CHECKPOINT_NAMES
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    remote_fn = get_checkpoint_blob_path(checkpoint_name)
    local_fn = os.path.join('checkpoints', checkpoint_name)
    az_download_file(local_fn=local_fn, remote_fn=remote_fn)
    return local_fn


@lru_cache(int(1e6))
def cached_decode_tok(tokenizer, tok):
    """
    Much time is spent decoding single tokens, so cache it.
    """
    return tokenizer.decode([tok])


class StopWatch:
    def __init__(self):
        self.timers = dict()
        self.times = dict()
    def start(self, name):
        self.timers[name] = time.time()
    def stop(self, name):
        st = self.timers[name]
        en = time.time()
        self.times[name] = en-st
    def show_last_times(self):
        for k, v in self.times.items():
            print(k.ljust(10) + ':', f'{v:.5f}')


def partition(n, parts):
    return [n // parts + (1 if i < n % parts else 0) for i in range(parts)]


def prefix_sums(v):
    s = 0
    ret = []
    for e in v:
        ret.append(s)
        s += e
    return ret


def divide_optimally(v, parts):
    sizes = partition(len(v), parts)
    offsets = prefix_sums(sizes)
    buckets = []
    for size, offset in zip(sizes, offsets):
        s = v[offset:offset+size]
        buckets.append(s)
    return buckets


def divide_optimally_and_scatter(v, comm):
    if comm.Get_rank() == 0:
        v = divide_optimally(v, comm.Get_size())
    else:
        v = None
    v = comm.scatter(v, root=0)
    return v


def make_device_map(num_layers, device_ids):
    layer_distribution = partition(num_layers, len(device_ids))
    device_map = {}
    c = 0
    for i, l in enumerate(layer_distribution):
        device_map[device_ids[i]] = list(range(c, c+l))
        c += l
    return device_map


def round_all(v, decimals=2):
    return [round(e, 2) for e in v]


def round_all_2d(vv, decimals=2):
    return [round_all(v, decimals=decimals) for v in vv]


def pmap(f, gen, np):
    def _worker_process(q, f):
        while True:
            args = q_in.get()
            if args is None:
                break
            f(args)

    q = Queue(maxsize=np)
    p = Pool(np, initializer=_worker_process, initargs=(q, f))
    for e in gen:
        q.put(e)
    for _ in range(np):
        q.put(None)


def iter_buckets(v, sz):
    bucket = []
    for e in v:
        bucket.append(e)
        if len(bucket) == sz:
            yield bucket
            bucket = []
    if len(bucket) > 0:
        yield bucket


def iter_hidden_neurons():
    for l in range(28):
        for f in range(4096*4):
            yield l, f

class timeit:
    def __enter__(self):
        self.st = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        en = time.time()
        print(en-self.st)


def load_gptj_model(
    device_ids=None,
    checkpoint_fn=None,
):

    if device_ids is None:
        devices_ids = range(torch.cuda.device_count())
    device_map = make_device_map(num_layers=28, device_ids=device_ids)
    print('device map:', device_map)

    if checkpoint_fn is None:
        print('Loading model...')
        model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    else:
        print(f'Loading model from checkpoint {checkpoint_fn}...')
        model = GPTJForCausalLM.from_pretrained(checkpoint_fn, config="EleutherAI/gpt-j-6B")
    print('Parallelizing model onto GPUs...')
    model.parallelize(device_map)
    model.eval()

    return model, device_map


def load_gptj_tokenizer(pad_token=None):
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", pad_token=pad_token)
    return tokenizer


