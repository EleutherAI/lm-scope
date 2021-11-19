from datasets import list_datasets, load_dataset


def clean_text(text):
    original = text

    # we probably won't need more than 1k characters
    text = text[:1000]

    # remove parenthesized portions
    k1 = 0
    k2 = 0
    k3 = 0
    new_text = ''
    for i in range(len(text)):
        if text[i] == '(':
            k1 += 1
        elif text[i] == ')':
            k1 -= 1
        elif text[i] == '[':
            k2 += 1
        elif text[i] == ']':
            k2 -= 1
        elif text[i] == '{':
            k3 += 1
        elif text[i] == '}':
            k3 -= 1
        else:
            if k1 == 0 and k2 == 0 and k3 == 0:
                new_text += text[i]
    text = new_text

    # fix strange punctuation
    text = text.replace(' .', '.')
    text = text.replace(' ,', ',')
    text = text.replace(' ; ', '')

    # put everything on one line
    text = ' '.join(text.split('\n'))

    # clean up white space
    text = ' '.join(text.split()).strip()

    # possible degenerate cases
    if len(text) < 5:
        return

    # only take the first few tokens
    num_tokens = 30
    text = text[:num_tokens * 10]
    tokens = tokenizer.encode(text)
    tokens = tokens[:num_tokens]
    text = tokenizer.decode(tokens)
    return text


class CombinedDataset:
    def __init__(self, limit):
        print('cleaning & loading data...')

        self.wikipedia_dataset = load_dataset('wikipedia', '20200501.en')
        self.lama_dataset = load_dataset('lama', 'google_re')

        self.examples = []

        # factual knowledge probing with Wikipedia first sentences
        for idx, record in enumerate(wikipedia_dataset['train']):
            text = record['text']
            text = clean_text(text)
            self.examples.append({ 'text': text, 'source': 'wikipedia' })
            if len(self.examples) == limit:
                break

        # factual knowledge probing with LAMA
        for record in lama_dataset['train']:
            text = record['masked_sentence']
            text = text.replace('[MASK]', record['obj_label'])
            text = clean_text(text)
            self.examples.append({ 'text': text, 'source': 'google_re' })
            if len(self.examples) == limit:
                break

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for e in self.examples:
            yield e

if __name__ == '__main__':
    d = CombinedDataset(float('inf'))
    print(len(d))

