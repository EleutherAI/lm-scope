from tqdm import tqdm
import jsonlines
import os
import sys


class CombinedDataset:
    def __init__(self, data_dir, limit=-1):
        print('cleaning & loading data...')

        self.examples = []
        self.data_dir = data_dir
        self.fns = [
            os.path.join(data_dir, 'wikipedia-first-lines.jsonl'),
            #os.path.join(data_dir, 'wikipedia-random-sentences.jsonl'),
            #os.path.join(data_dir, 'lama.jsonl'),
        ]

        for fn in self.fns:
            print(f'Loading {fn}...')
            with jsonlines.open(fn, mode='r') as reader:
                for example in reader:
                    if len(self) == limit:
                        break
                    self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for e in self.examples:
            yield e

