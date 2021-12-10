import { useEffect, useRef } from 'react';


export function usePrevious<T>(value: T) {
    const ref = useRef<T>();
    useEffect(() => {
        ref.current = value;
    }, [value]);
    return ref.current;
}

export function getBaseIndexPath(): string {
    if (process.env.NODE_ENV === 'development') {
        return 'http://localhost:3001/data/index';
    } else {
        return '/index';
    }
}

export type NeuronActivation = {
    l: number;
    f: number;
    a: number[];
}

export type Logit = {
    tok: string;
    prob: number;
}

export type Example = {
    example: string;
    tokens: string[];
    hidden: NeuronActivation[];
    logits: Logit[][][];  // [layer][seq][k]
    attentions: Map<string, number>;
}

export type Neuron = {
    high: {a: number[], exampleIdx: number},
    low: {a: number[], exampleIdx: number},
    kn: {
        [k: string]: Logit[]
    }
}

export type Prompt = {
    text: string;
    source: string;
    meta: any;
    tokens: string[];
}

export type Dataset = Prompt[];

export async function getNeuronData(exampleIdx: number): Promise<Example> {
    const n = exampleIdx.toString().padStart(5, '0');
    const res = await fetch(`${getBaseIndexPath()}/example-${n}.json`);
    const j = await res.json();
    return j;
}

export async function getNeuronMatches(l: number, f: number): Promise<Neuron> {
    const res = await fetch(`${getBaseIndexPath()}/neuron-${l}-${f}.json`);
    const j = await res.json();
    return j;
}

export async function getDataset(name: 'ud' | 'wikipedia-fl'): Promise<Dataset> {
    let url;
    if (name === 'ud') {
        url = `${getBaseIndexPath()}/ud.json`;
    } else if (name === 'wikipedia-fl') {
        url = `${getBaseIndexPath()}/wikipedia-fl.json`;
    } else {
        throw new Error(`Unknown dataset: ${name}`);
    }
    const res = await fetch(url);
    const dataset = await res.json();
    return dataset;
}

export function updateQueryParameter(key: string, value: string | null) {
    const params = new URLSearchParams(window.location.search);
    if (value === null) {
        params.delete(key);
    } else {
        params.set(key, value);
    }
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState(null, '', newUrl);
}

export function filterResults(dataset: Dataset, query: string): number[] {
    const q = query.toLowerCase();
    const results = dataset
        .map((e, idx): [Prompt, number] => [e, idx])
        .filter(([e, idx]) => e.text.toLowerCase().indexOf(q.toLowerCase()) > -1)
        .sort(([a, aIdx], [b, bIdx]) => a.text.toLowerCase().indexOf(q.toLowerCase()) - b.text.toLowerCase().indexOf(q.toLowerCase()))
        .map(([d, idx]) => idx);
    return results;
}
