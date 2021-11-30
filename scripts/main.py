import gc
from tqdm import tqdm
from functools import partial
import fire
import torch
import os

from dataset import CombinedDataset
from model import ModelWatcher
from pickler import BatchPickler


def main(max_num_tokens: int = 30,
         top_k: int = 5,
         activation_threshold: int = 3,
         dataset_offset: int = 0,
         dataset_limit: int = 999999999,
         data_dir: str = '/mnt/data',
         checkpoint_fn: str = '/mnt/data/checkpoints/pytorch_model.bin',
         # how many bytes to pickle into a single file before compressing it and starting a new file
         file_size_goal: int = 128 * 1024 * 1024,
         verbose: bool = True):

    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    print(f'Found {len(devices)} CUDA devices.')

    dataset = CombinedDataset(data_dir, offset=dataset_offset, limit=dataset_limit)
    watcher = ModelWatcher(checkpoint_fn=checkpoint_fn, max_num_tokens=max_num_tokens, top_k=top_k, activation_threshold=activation_threshold)
    output_dir = os.path.join(data_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    base_neuron_fn = os.path.join(output_dir, f"neurons{dataset_offset}-{dataset_offset+dataset_limit}.pickle")
    pickler = BatchPickler(base_neuron_fn, file_size_goal, lambda *args: None)

    try:
        with torch.no_grad():
            for idx, row in enumerate(tqdm(dataset)):

                if idx == dataset_limit:
                    break

                text = row['text']
                source = row['source']

                record = watcher.forward(text)
                record['source'] = source

                watcher.timer.start('pickle')
                pickler.dump(record)
                watcher.timer.stop('pickle')

                if verbose:
                    watcher.timer.show_last_times()

    finally:
        pickler.close()


if __name__ == '__main__':
    fire.Fire(main)

