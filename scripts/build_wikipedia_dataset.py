from datasets import list_datasets, load_dataset
import jsonlines


def clean(self, text):
    original = text

    # we probably won't need more than 200 characters
    text = text[:200]

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


self.wikipedia_dataset = load_dataset('wikipedia', '20200501.en')

with jsonlines.open('./data/wikipedia.jsonl', mode='w') as writer:
    for idx, record in enumerate(tqdm(self.wikipedia_dataset['train'])):
        if idx >= len(self.examples_wikipedia):
            writer.write({
                'text': clean(record['text']),
                'source': 'wikipedia',
            })

