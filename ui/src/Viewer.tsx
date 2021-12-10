import React, { useEffect, useMemo, useState } from "react";
import {
    Dataset,
    filterResults,
    getDataset,
    updateQueryParameter,
    getDataForExample,
    ExampleData,
} from './utils';
import Split from 'react-split'
import { debounce } from 'debounce';
import LogitAttnView from "./LogitAttnView";
import SearchPanel from "./SearchPanel";


function Viewer() {
    const [query, setQuery] = useState('');
    const [dataset, setDataset] = useState<Dataset | null>(null);
    const [results, setResults] = useState<number[]>([]);
    const [selectedExample, setSelectedExample] = useState<number>(-1);
    // const [selectedToken, setSelectedToken] = useState<number>(0);
    const [dataForExample, setDataForExample] = useState<ExampleData | null>(null);
    const [headIdx, setHeadIdx] = useState<number>(0);
    const [hoveringCell, setHoveringCell] = useState<{ layer: number, seq: number } | null>(null);
    const [loading, setLoading] = useState(false);
    const selectedToken = hoveringCell ? hoveringCell.seq : -1;

    const debouncedUpdateResults = useMemo(() => debounce(() => {
        if (dataset) {
            setResults(filterResults(dataset, query));
        } else {
            setResults([]);
        }
    }, 200), [dataset, query]);

    useEffect(() => {
        const query = new URLSearchParams(window.location.search).get('query');
        if (query) {
            setQuery(query);
        }

        const example = new URLSearchParams(window.location.search).get('example');
        if (example) {
            const i = parseInt(example);
            if (!isNaN(i)) {
                setSelectedExample(i);
            }
        }
    }, []);

    useEffect(() => {
        (async () => {
            setLoading(true);
            const data = await getDataset('ud');
            setDataset(data);
            setLoading(false);
        })();
    }, []);

    useEffect(() => {
        debouncedUpdateResults();
    }, [debouncedUpdateResults]);

    useEffect(() => {
        (async () => {
            if (selectedExample > -1 && dataset) {
                setLoading(true);
                const data = await getDataForExample(selectedExample);
                setDataForExample(data);
                setLoading(false);
            } else {
                setDataForExample(null);
            }
        })();
    }, [selectedExample, dataset]);

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
            minSize={100}
            expandToMin={true}
            gutterSize={5}
            gutterAlign="center"
            snapOffset={0}
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
            <SearchPanel
                dataset={dataset}
                query={query}
                results={results}
                selectedExample={selectedExample}
                selectedToken={selectedToken}
                onUpdateQuery={onUpdateQuery}
                onUpdateExample={onUpdateExample}
            />
            <LogitAttnView
                dataset={dataset}
                exampleIdx={selectedExample}
                dataForExample={dataForExample}
                loading={loading}
                headIdx={headIdx}
                hoveringCell={hoveringCell}
                updateHeadIdx={setHeadIdx}
                updateHoveringCell={setHoveringCell}
            />
        </Split>
    </>;
}

export default Viewer;