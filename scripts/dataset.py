from tqdm import tqdm
import jsonlines
import os
import sys


class CombinedDataset:
    def __init__(self, data_dir, offset=0, limit=-1):
        print('Loading data...')

        self.examples = []
        self.data_dir = data_dir
        self.fns = [
            os.path.join(data_dir, 'UniversalDependencies-sentences.jsonl'),
            #os.path.join(data_dir, 'wikipedia-first-lines.jsonl'),
            #os.path.join(data_dir, 'wikipedia-random-sentences.jsonl'),
        ]

        o = 0
        for fn in self.fns:
            if len(self) == limit:
                break
            print(f'Loading {fn}...')
            with jsonlines.open(fn, mode='r') as reader:
                for example in reader:
                    if o < offset:
                        o += 1
                        continue
                    if len(self) == limit:
                        break
                    self.examples.append(example)

        print('Dataset has', len(self.examples), 'examples')

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for e in self.examples:
            yield e

