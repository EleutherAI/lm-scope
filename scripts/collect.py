import gc
from tqdm import tqdm
from functools import partial
import fire
import torch
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, GPTJForCausalLM
from multiprocessing import Process, Queue
from azure.storage.blob import BlockBlobService
import time
import pickle

from dataset import CombinedDataset
from watcher import ModelWatcher
from pickler import BatchPickler
from utils import make_device_map, iter_hidden_neurons, cached_decode_tok


def upload_neuron_file(local_fn):
    blob_service_client = BlockBlobService(account_name=os.environ['AZ_ACCOUNT_NAME'], account_key=os.environ['AZ_ACCOUNT_KEY'])
    blob_path = 'raw2/' + local_fn.split('/')[-1]
    print('Uploading', local_fn, 'to blob path', blob_path)
    blob_service_client.create_blob_from_path(os.environ['AZ_CONTAINER_NAME'], blob_path, local_fn)


def run_upload_test():
    with open('test.txt', 'w') as f:
        f.write('Hello, world!')
    upload_neuron_file('test.txt')
    os.remove('test.txt')
    print('success')


#def uploader():
#    while True:
#        for fn in os.listdir('output'):
#            upload_file(local_fn)
#            os.remove(local_fn)
#        time.sleep(5)


def get_knowledge_neuron_identifiers(model, device_map, top_k=5):
    """
    For every neuron in each hidden MLP block, this will activate the neuron,
    pass it through the second MLP layer, then perform the logit lens trick on
    the result.
    """

    def _get_device(l):
        for k, v in device_map.items():
            if l in v:
                return torch.device('cuda:' + str(k))
        raise ValueError('Invalid layer idx' + str(l))

    mp = {}
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    with torch.no_grad():
        for (l, f) in tqdm(list(iter_hidden_neurons())):
            x = torch.zeros([1, 1, 4096*4], device=_get_device(l))
            x[0, 0, f] = 10
            x = model.transformer.h[l].mlp.fc_out(x)
            x = model.transformer.h[l].mlp.dropout(x)
            x = x.to(torch.device('cuda:0'))
            x = model.lm_head(x)[0]
            values, indices = torch.topk(x, k=top_k)
            norm_values = F.softmax(values, dim=-1)
            top = []
            for tok, prob in zip(indices[0], norm_values[0]):
                tok = tok.item()
                prob = prob.item()
                top.append({
                    'tok': cached_decode_tok(tokenizer, tok),
                    'prob': round(prob, 4),
                })
            mp[f'{l}:{f}'] = top
    return mp


def worker(rank,
           max_num_tokens: int = 30,
           top_k: int = 5,
           activation_threshold: int = 3,
           dataset_offset: int = 0,
           dataset_limit: int = 999999999,
           data_dir: str = '/mnt/data',
           checkpoint_fn: str = '/mnt/data/checkpoints/pytorch_model.bin',
           # how many bytes to pickle into a single file before compressing it and starting a new file
           file_size_goal: int = 128 * 1024 * 1024,
           show_times: bool = False):

    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    print(f'Found {len(devices)} CUDA devices.')

    device_map = make_device_map(num_layers=28, num_devices=len(devices))

    print('Loading dataset...')
    dataset = CombinedDataset(data_dir, offset=dataset_offset, limit=dataset_limit)

    print('Loading model...')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    state_dict = torch.load(checkpoint_fn)
    model = GPTJForCausalLM.from_pretrained(None, state_dict=state_dict, config="EleutherAI/gpt-j-6B")
    print(f'Parallelizing on {len(device_map)} GPUs...')
    model.parallelize(device_map)
    model.eval()
    print('done')

    watcher = ModelWatcher(model=model, max_num_tokens=max_num_tokens, top_k=top_k, activation_threshold=activation_threshold)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    base_neuron_fn = os.path.join(output_dir, f"neurons{dataset_offset}-{dataset_offset+dataset_limit}.pickle")
    pickler = BatchPickler(base_neuron_fn, file_size_goal, upload_neuron_file)

    try:
        with torch.no_grad():

            if rank == 0:
                print('Collecting knowledge neuron identifiers (only rank 0)')
                identifiers = get_knowledge_neuron_identifiers(model, device_map)
                print('Saving identifiers')
                fn = os.path.join(output_dir, 'kn-id.pickle')
                with open(fn, 'wb') as f:
                    pickle.dump(identifiers, f)
                upload_neuron_file(fn)

            print('Processing all examples..')
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

                if show_times:
                    watcher.timer.show_last_times()

    finally:
        pickler.close()


def main(max_num_tokens: int = 2048,
         top_k: int = 5,
         activation_threshold: int = 3,
         dataset_offset: int = 0,
         dataset_limit: int = 999999999,
         data_dir: str = '/mnt/data',
         checkpoint_fn: str = '/mnt/data/checkpoints/pytorch_model.bin',
         # how many bytes to pickle into a single file before compressing it and starting a new file
         file_size_goal: int = 128 * 1024 * 1024,
         show_times: bool = False,
         num_workers: int = 1):

    run_upload_test()

    num_rows_per_worker = dataset_limit // num_workers
    row_distribution = [num_rows_per_worker + (1 if i < dataset_limit % num_workers else 0) for i in range(num_workers)]
    ps = []
    offset = dataset_offset
    for proc, num_rows in enumerate(row_distribution):
        print('Starting process', proc)
        p = Process(target=worker, args=(
            proc,
            max_num_tokens,
            top_k,
            activation_threshold,
            offset,
            num_rows,
            data_dir,
            checkpoint_fn,
            file_size_goal,
            show_times,
        ))
        p.start()
        ps.append(p)
        offset += num_rows

    #p = Process(target=uploader)
    #ps.append(p)

    for p in ps:
        p.join()

    print('done :)')
    

if __name__ == '__main__':
    fire.Fire(main)

