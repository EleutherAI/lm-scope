import React, { useEffect, useMemo, useState } from "react";
import {
    Dataset,
    filterResults,
    getDataset,
    updateQueryParameter
} from './utils';
import Split from 'react-split'
import { debounce } from 'debounce';


function Viewer() {
    const [query, setQuery] = useState('');
    const [dataset, setDataset] = useState<Dataset | null>(null);
    const [results, setResults] = useState<number[]>([]);
    const [selectedExample, setSelectedExample] = useState<number>(-1);
    // const [selectedToken, setSelectedToken] = useState<number>(0);
    const selectedToken = 0;

    const debouncedUpdateResults = useMemo(() => debounce(() => {
        if (dataset) {
            setResults(filterResults(dataset, query));
        } else {
            setResults([]);
        }
    }, 200), [dataset, query]);

    useEffect(() => {
        (async () => {
            const data = await getDataset('ud');
            console.log(data);
            setDataset(data);
        })();
    }, []);

    useEffect(() => {
        debouncedUpdateResults();
    }, [debouncedUpdateResults]);

    const onUpdateExample = (selectedExample: number) => {
        updateQueryParameter('example', selectedExample.toString());
        setSelectedExample(selectedExample);
    };

    const onUpdateQuery = (query: string) => {
        if (query.length === 0) {
            updateQueryParameter('query', null);
        } else {
            updateQueryParameter('query', query);
        }
        setQuery(query);
    };

    return <>
        <Split
            sizes={[25, 75]}
            minSize={300}
            expandToMin={true}
            gutterSize={5}
            gutterAlign="center"
            snapOffset={30}
            dragInterval={1}
            direction="horizontal"
            cursor="col-resize"
            gutterStyle={(dimension, gutterSize, index) => ({
                backgroundColor: 'white',
                backgroundImage: "url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAeCAYAAADkftS9AAAAIklEQVQoU2M4c+bMfxAGAgYYmwGrIIiDjrELjpo5aiZeMwF+yNnOs5KSvgAAAABJRU5ErkJggg==')",
                backgroundRepeat: 'no-repeat',
                backgroundPosition: '50%',
                border: '1px solid #eee',
                width: gutterSize.toString() + 'px',
                borderRadius: '4px',
                cursor: 'col-resize',
            })}
            style={{
                display: 'flex',
                flexDirection: 'row',
                height: '100%',
            }}
        >
            <div
                style={{
                    flex: 1,
                    display: 'flex',
                    flexDirection: 'column',
                }}
            >
                <div
                    style={{
                        display: 'flex',
                        flexDirection: 'column',
                    }}
                >
                    <input
                        value={query}
                        placeholder='Search for a prompt...'
                        onChange={e => onUpdateQuery(e.target.value)}
                        style={{
                            flex: 1,
                            padding: 5,
                            margin: 5,
                        }}
                    />
                </div>
                <div
                    style={{
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                    }}
                >
                    <p
                        style={{ marginLeft: 5 }}
                    >{results.length.toString()} results</p>
                    <div
                        style={{
                            flex: 1,
                            position: 'relative',
                        }}
                    >
                        <div
                            style={{
                                width: '100%',
                                height: '100%',
                                overflow: 'scroll',
                                position: 'absolute',
                            }}
                        >
                            {results.slice(0, 50).map(res => {
                                return <div
                                    key={res}
                                    style={{
                                        margin: '5px',
                                        padding: '5px',
                                        border: '1px solid #eee',
                                        borderRadius: '4px',
                                        backgroundColor: selectedExample === res ? 'lightblue' : undefined,
                                        cursor: 'pointer',
                                    }}
                                    onClick={() => onUpdateExample(res)}
                                >
                                    {dataset && dataset[res].tokens.map((token, tokenIdx) => {
                                        return <span
                                            key={tokenIdx}
                                            style={{
                                                textDecoration: selectedToken === tokenIdx ? 'underline' : undefined,
                                                fontStyle: selectedToken === tokenIdx ? 'italic' : undefined,
                                            }}
                                        >
                                            {token}
                                        </span>;
                                    })}
                                </div>;
                            })}
                        </div>
                    </div>
                </div>
            </div>
            <div>
                right panel
            </div>
        </Split>
    </>;
}

export default Viewer;