from functools import lru_cache
from collections import defaultdict
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
