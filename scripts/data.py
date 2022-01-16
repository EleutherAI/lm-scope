from transformers import AutoTokenizer, GPTJForCausalLM
import random
import torch
import torch.nn.functional as F
from functools import lru_cache
import fire
from tqdm import tqdm
import json
import os 
import numpy as np
from collections import defaultdict
import sys

from utils import make_device_map, set_seed


def trex(datasets_dir='datasets'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    trex_fn = os.path.join(datasets_dir, 'TREx.zip')
    if not os.path.exists(trex_fn):
        print('TODO: TREx downloader')
        sys.exit(0)
