import torch
from transformers import AutoTokenizer, GPTJForCausalLM
import torch.nn.functional as F

from utils import cached_decode_tok


class ModelWatcher:
    def __init__(self):

        num_layers = 28
        device_ids = list(range(torch.cuda.device_count()))
        num_layers_per_device = num_layers // len(device_ids)
        device_map = { i: list(range(i * num_layers_per_device : (i + 1) * num_layers_per_device)) for i in device_ids }
        self.devices = [torch.device(f'cuda:{i}') for i in device_ids]

        print('Loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", low_cpu_mem_usage=True)
        print(f'Parallelizing on {len(device_ids)} GPUs...')
        self.model.parallelize(device_map)
        self.model.eval()

        print('Setting up handlers...')
        self.module_names_to_track_for_activations = \
            [f'transformer.h.{i}.mlp.fc_out' for i in range(self.model.config.n_layer)]
        self.module_names_to_track_for_logits = \
            [f'transformer.h.{i}' for i in range(self.model.config.n_layer)]

        self.hidden_states = dict()

        self.handles = []
        cnt = 0
        for name, m in self.model.named_modules():
            if name in self.module_names_to_track_for_activations or name in self.module_names_to_track_for_logits:
                cnt += 1
                handle = m.register_forward_hook(partial(self._save_hidden_states, name))
                self.handles.append(handle)

    def _save_hidden_states(self, name, module, input, output):
        # we care about input for activations, and output for logit lens
        hidden_states[name] = { 'input': input[0], 'output': output }

    def clear_hidden_states(self):
        self.hidden_states = dict()

    def extract_neuron_values(self, threshold):
        """
        For each MLP, determine which neurons fire at any point during the entire sequence,
        unless it only fires on the first token (which we will just assume is noise).

        The output is a list of dicts resembling individual neurons with fields:
            l: the layer of the neuron
            f: the index of the neuron in the feature dimension
            a: a list of activations equal to the length of the sequence

        """

        values = []
        uniq = set()
        neurons = []
        high_activations = []
        for name in self.module_names_to_track_for_activations:
            h = self.hidden_states[name]['input'][0]
            high_activations.append((h > threshold).nonzero())
            neurons.append(h)

        for layer_idx, (neurons, (_, feature_idx)) in enumerate(zip(neurons, high_activations)):
            feature_idx = feature_idx.item()
            if (layer_idx, feature_idx) in uniq:
                # we already have it
                continue
            uniq.add((layer_idx, feature_idx))
            values.append({
                'l': layer_idx,
                'f': feature_idx,
                'a': neurons[layer_idx, :, feature_idx].reshape([neurons.shape[1]]).tolist(),
            })
        return values

    def extract_logit_lens(self, k=5):
        """
        Extract the output logits for each layer (including the final layer)

        Returns a nested list structure of shape [n_layers, n_seq, k]
        where each element is a dict containing:
            tok: the predicted token
            prob: the probability given to this token (from softmax of logits)

        Note: The sum of the final dimension probabilities will be very close to 1.
        """

        per_layer_tokens = []
        for module_name in self.module_names_to_track_for_logits:
            h2 = self.hidden_states[module_name]['output'][0]  # x, present
            with torch.no_grad():
                layer_logits = self.model.lm_head(h2.to(self.devices[0])).detach()[0]
            seq = layer_logits.shape[0]
            values, indices = torch.topk(layer_logits, k=k)
            norm_values = F.softmax(values, dim=-1)
            indices = indices.cpu()
            norm_values = norm_values.cpu()
            top_in_sequence = []
            for i in range(seq):
                top_tokens = []
                for tok, prob in zip(indices[i], norm_values[i]):
                    tok = tok.item()
                    prob = prob.item()
                    top_tokens.append({
                        'tok': cached_decode_tok(self.tokenizer, tok),
                        'prob': prob,
                    })
                top_in_sequence.append(top_tokens)
            per_layer_tokens.append(top_in_sequence)
        return per_layer_tokens

    def extract_attentions(self):
        scores = self.hidden_states['attentions']
        return torch.stack([a.to(main_device) for a in scores]).tolist()

    def extract_neuron_desc(self):
        # TODO
        return None

    def forward(self, text):
        inputs = tokenizer(text, return_tensors="pt")
        context = inputs["input_ids"][0][:max_num_tokens]
        context = context.to(self.devices[0])
        output = model(context, return_dict=True, output_attentions=True)
        self.hidden_states['attentions'] = output.attentions

