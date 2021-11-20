from datasets import list_datasets, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import jsonlines
import os
import sys


class CombinedDataset:
    def __init__(self, tokenizer, limit=-1):
        print('cleaning & loading data...')

        self.tokenizer = tokenizer
        self.lama_dataset = load_dataset('lama', 'google_re')

        self.examples = []

        self.wikipedia_fn = './data/wikipedia.jsonl'

        # TODO: download wikipedia dataset automatically
        if not os.path.exists(self.wikipedia_fn):
            print('You must first download the preprocessed Wikipedia dataset')
            sys.exit(1)

        print('Loading Wikipedia dataset (this may take a few minutes)')
        with jsonlines.open(self.wikipedia_fn, mode='r') as reader:
            for example in reader:
                if len(self) == limit:
                    break
                self.examples.append(example)

        print('Loading LAMA dataset')
        with jsonlines.open(self.wikipedia_fn, mode='w') as writer:
            for record in self.lama_dataset['train']:
                text = record['masked_sentence']
                text = text.replace('[MASK]', record['obj_label'])
                example = { 'text': text, 'source': 'google_re' }
                self.examples.append(example)
                writer.write(example)
                if len(self) == limit:
                    break

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for e in self.examples:
            yield e


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    d = CombinedDataset(tokenizer)
    print(len(d))

