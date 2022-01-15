from tqdm import tqdm
import jsonlines
import os
import sys

from lm_eval import tasks, evaluator
from lm_eval.utils import join_iters


class CombinedDataset:
    def __init__(self, offset=0, limit=-1):
        print('Loading data...')
        self.examples = self._load_examples()
        print('Dataset has', len(self.examples), 'examples')

    def _load_examples(self):
        examples = []
        skip = [
            'math_asdiv'  # bug when loading (should report to lm-eval upstream)
        ]
        for idx, (tname, Task) in enumerate(tasks.TASK_REGISTRY.items()):
            if tname in skip:
                continue
            task = Task()
            iters = []
            if task.has_validation_docs():
                iters.append(task.validation_docs())
            if task.has_test_docs():
                iters.append(task.test_docs())
            if task.has_training_docs():
                iters.append(task.training_docs())
            docs = join_iters(iters)
            for i, doc in enumerate(docs):
                example = {
                    "prompt": task.doc_to_text(doc),
                    "target": task.doc_to_target(doc),
                    "source": tname,
                }
                examples.append(example)
            print('examples:', len(examples))
        return examples

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for e in self.examples:
            yield e

if __name__ == '__main__':
    CombinedDataset()