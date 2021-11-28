import gc
from tqdm import tqdm
from functools import partial
import fire

from dataset import CombinedDataset
from model import ModelWatcher
from extract import extract_logit_lens, extract_neuron_values
from utils import StopWatch
from pickler import BatchPickler


def main(max_num_tokens: int = 2048,
         top_k: int = 5,
         activation_threshold: int = 3,
         # stop after this many records in the dataset have been processed
         dataset_limit: int = 999999999,
         # how many bytes to pickle into a single file before compressing it and starting a new file
         file_size_goal: int = 128 * 1024 * 1024,
         verbose: bool = True):

    devices = [torch.cuda(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    print(f'Found {len(devices)} CUDA devices.')

    dataset = CombinedDataset()
    # TODO: multiple checkpoints
    watcher = ModelWatcher()
    pickler = BatchPickler("output/neurons.pickle", file_size_goal, compress_upload)
    timer = StopWatch()

    try:
        with torch.no_grad():
            for idx, row in enumerate(tqdm(dataset)):

                if idx == dataset_limit:
                    break

                text = row['text']
                source = row['source']

                watcher.clear_hidden_states()
                # gc.collect()

                timer.start('forward')
                output = watcher.forward(text)
                timer.stop('forward')

                timer.start('hidden')
                hidden = watcher.extract_neuron_values(activation_threshold)
                timer.stop('hidden')

                timer.start('logits')
                logits = watcher.extract_logit_lens(k=top_k)
                timer.stop('logits')

                timer.start('attn')
                attentions = watcher.extract_attentions()
                timer.stop('attn')

                record = {
                    'text': text,
                    'source': source,
                    'tokens': context,
                    'hidden': hidden,
                    'logits': logits,
                    'attentions': attentions,
                }
                timer.start('pickle')
                pickler.dump(record)
                timer.stop('pickle')

                if verbose:
                    timer.show_last_times()

    finally:
        pickler.close()


if __name__ == '__main__':
    fire.Fire(main)

