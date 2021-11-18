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
        for k, v in self.times:
            print(k.ljust(10) + ':', f'{v[-1]:.5f}')

