from knowledge_neurons import (
    KnowledgeNeurons,
    pararel_expanded,
    pararel
)
from transformers import AutoTokenizer, GPTJForCausalLM
import random
import torch
import torch.nn.functional as F
from utils import make_device_map


#PARAREL = pararel_expanded()
#for k in random.sample(list(PARAREL.keys()), 10):
#    print(PARAREL[k]['sentences'])
#quit()

# setup model, tokenizer + kn class
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
device_map = make_device_map(num_layers=28, num_devices=len(devices))
state_dict = torch.load('/mnt/data/checkpoints/pytorch_model.bin', map_location='cpu')
model = GPTJForCausalLM.from_pretrained(None, state_dict=state_dict, config="EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model.parallelize(device_map)
model.eval()
kn = KnowledgeNeurons(model, tokenizer, model_type='gpt-j', device='cuda:0')

GROUND_TRUTH = "Paris"
BATCH_SIZE = 1
STEPS = 20

ENG_TEXTS = [
    "Sarah was visiting the capital of france,",
    "The capital of france is",
    "France's capital, a hotspot for romantic vacations, is",
    "The eiffel tower is situated in",
    "The most populous city in france is",
    "One of the most popular tourist destinations in the world is France's capital,",
]

P = 0.5 # sharing percentage

refined_neurons_eng = kn.get_refined_neurons(
    ENG_TEXTS,
    GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
)

print(refined_neurons_eng)
