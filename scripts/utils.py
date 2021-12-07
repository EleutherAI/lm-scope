from functools import lru_cache
from collections import defaultdict
from multiprocessing import Pool, Queue
import time


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


def make_device_map(num_layers, num_devices):
    device_ids = list(range(num_devices))
    num_layers_per_device = num_layers // len(device_ids)
    layer_distribution = [num_layers_per_device + (1 if i < num_layers % len(device_ids) else 0) for i in device_ids]
    device_map = {}
    c = 0
    for i, l in enumerate(layer_distribution):
        device_map[i] = list(range(c, c+l))
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
    p = Pool(sz)
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

