import React, { useEffect, useRef } from "react";
import ReactTooltip from "react-tooltip";
import {
    Dataset,
    ExampleData,
    range,
    rgba,
} from "./utils";


const COLORS = [
    [255, 17, 0],
    [255, 149, 0],
    [255, 238, 0],
    [77, 255, 0],
    [0, 255, 242],
    [0, 255, 98],
    [0, 153, 255],
    [0, 34, 255],
    [128, 0, 255],
    [221, 0, 255],
    [255, 0, 170],
    [128, 128, 128],
    [166, 102, 0],
    [173, 0, 0],
    [128, 128, 0],
    [174, 255, 0],
];

interface LogitAttnViewProps {
    dataset: Dataset | null;
    exampleIdx: number;
    dataForExample: ExampleData | null;
    loading: boolean;
    headIdx: number;
    updateHeadIdx: (idx: number) => void;
    hoveringCell: { layer: number, seq: number } | null;
    updateHoveringCell: (cell: { layer: number, seq: number } | null) => void;
    hideFirstAttn: boolean;
    updateHideFirstAttn: (hide: boolean) => void;
}

export default function LogitAttnView({
    dataset,
    exampleIdx,
    dataForExample,
    loading,
    headIdx,
    updateHeadIdx,
    hoveringCell,
    updateHoveringCell,
    hideFirstAttn,
    updateHideFirstAttn,
}: LogitAttnViewProps) {

    useEffect(() => {
        const onKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'ArrowLeft') {
                updateHeadIdx(Math.max(0, headIdx - 1));
            } else if (e.key === 'ArrowRight') {
                updateHeadIdx(Math.min(15, headIdx + 1));
            }
        }
        window.addEventListener('keydown', onKeyDown);
        return () => {
            window.removeEventListener('keydown', onKeyDown);
        }
    }, [headIdx, updateHeadIdx]);

    if (loading || !dataset) {
        return <div>Loading...</div>
    }

    if (!dataForExample || exampleIdx < 0) {
        return <div>
            <p
                style={{
                    color: 'grey',
                    textAlign: 'center',
                    paddingTop: '100px',
                }}
            >
                Select an example to visualize activations.
            </p>
        </div>;
    }

    const example = dataset[exampleIdx];

    const renderInputTokenRow = () => {
        return <div
            style={{
                display: 'flex',
                flexDirection: 'row',
                paddingLeft: '13px',
                paddingRight: '13px',
            }}
        >
            {example.tokens.map((tok, seqIdx) => {
                const hovering = hoveringCell && hoveringCell.layer === -1 && hoveringCell.seq === seqIdx;
                return <div
                    key={seqIdx}
                    style={{
                        flex: 1,
                        outline: '1px solid white',
                        backgroundColor: '#eee',
                        textAlign: 'center',
                        fontSize: '9px',
                        overflow: 'hidden',
                        fontWeight: 'bold',
                        cursor: 'default',
                        opacity: hovering ? 0.7 : 1,
                    }}
                    onMouseEnter={() => updateHoveringCell({ layer: -1, seq: seqIdx })}
                    onMouseLeave={() => updateHoveringCell(null)}
                >
                    {tok}
                </div>;
            })}
        </div>;
    }

    const renderTokenRow = (layerIdx: number) => {
        return <div
            style={{
                display: 'flex',
                flexDirection: 'row',
            }}
        >
            <ReactTooltip
                id='logit-tooltip'
                place='top'
                multiline={true}
            />
            {example.tokens.map((tok, seqIdx) => {
                const topLogits = dataForExample.logits[layerIdx][seqIdx];
                const prevTopLogits = layerIdx > 0 ? dataForExample.logits[layerIdx - 1][seqIdx] : null;
                const changed = topLogits && prevTopLogits && prevTopLogits[0].tok !== topLogits[0].tok;
                const correct = topLogits && seqIdx - 1 < example.tokens.length && example.tokens[seqIdx + 1] === topLogits[0].tok;
                const hovering = hoveringCell && hoveringCell.layer === layerIdx && hoveringCell.seq === seqIdx;
                const tooltipText = topLogits ?
                    topLogits.map(l => `${l.tok.replace(' ', '␣')} ${Math.round(l.prob * 100)}%`).join('<br />')
                    : '';
                return <div
                    key={seqIdx}
                    style={{
                        flex: 1,
                        outline: changed ? '1px solid black' : '1px solid white',
                        zIndex: changed ? 1 : 0,
                        backgroundColor: correct ? 'lightgreen' : '#eee',
                        textAlign: 'center',
                        fontSize: '9px',
                        overflow: 'hidden',
                        cursor: 'default',
                        opacity: hovering ? 0.7 : 1,
                    }}
                    onMouseEnter={() => updateHoveringCell({ layer: layerIdx, seq: seqIdx })}
                    onMouseLeave={() => updateHoveringCell(null)}
                >
                    <div className='logit' data-for='logit-tooltip' data-tip={tooltipText}>
                        {topLogits ? topLogits[0].tok.replace(' ', '␣') : ''}
                    </div>
                </div>;
            })}
        </div>
    };

    return <div
        style={{
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
        }}
    >
        <div>
            <span
                style={{
                    padding: '4px',
                    marginLeft: '4px',
                }}
            >attention head:</span>
            {COLORS.map(([R, G, B], headIdxOfButton) => {
                return <button
                    key={headIdxOfButton}
                    className='head-index-button'
                    style={{
                        backgroundColor: rgba(R, G, B, headIdx === headIdxOfButton ? 1 : 0.1),
                    }}
                    onClick={() => {
                        updateHeadIdx(headIdxOfButton);
                    }}
                >{headIdxOfButton}</button>;
            })}
            <span
                style={{
                    marginLeft: '10px',
                }}
            >
                hide attentions w/ first token?
                <input
                    type='checkbox'
                    checked={hideFirstAttn}
                    onChange={(e) => {
                        updateHideFirstAttn(e.target.checked);
                    }}
                />
            </span>
            {renderInputTokenRow()}
        </div>
        <div style={{
            flex: 1,
            position: 'relative',
        }}>
            <div
                style={{
                    flex: 1,
                    width: '100%',
                    height: '100%',
                    overflowY: 'scroll',
                    overflowX: 'hidden',
                    position: 'absolute',
                }}
            >
                {range(0, 28).map(layerIdx => {
                    return <div
                        key={layerIdx}
                        style={{
                            // border: "1px solid #eee",
                            // margin: '6px',
                            paddingLeft: '13px',
                            paddingRight: '13px',
                        }}
                    >
                        <Attn
                            attentions={dataForExample.attentions}
                            seqLen={example.tokens.length}
                            layerIdx={layerIdx}
                            headIdx={headIdx}
                            hoveringCell={hoveringCell}
                            hideFirstAttn={hideFirstAttn}
                        />
                        {renderTokenRow(layerIdx)}
                    </div>;
                })}
            </div>
        </div>
    </div>;
}


function drawArcBetweenPoints(ctx: CanvasRenderingContext2D, a: number, b: number) {
    if (a === b) {
        ctx.moveTo(a, 0);
        ctx.beginPath();
        ctx.arc(a, 5, 8, 0, 2 * Math.PI);
        ctx.stroke();
    } else {
        ctx.beginPath();
        const radius = (b - a) * 0.75;
        const heightUpperBound = 50;
        ctx.moveTo(a, 0);
        ctx.arcTo((a + b) / 2, heightUpperBound, b, 0, radius);
        ctx.lineTo(b, 0);
        ctx.stroke();
    }
}


const redraw = (params: {
    canvas: HTMLCanvasElement,
    attentions: { [k: string]: number },
    layerIdx: number,
    headIdx: number,
    seqLen: number,
    hoveringCell: { layer: number, seq: number } | null,
    hideFirstAttn: boolean,
}) => {
    const ctx = params.canvas.getContext('2d');
    if (ctx) {
        const width = params.canvas.width;
        const height = params.canvas.height;
        const tokenWidth = width / params.seqLen;
        ctx.clearRect(0, 0, width, height);
        // if we're hovering over a cell
        if (params.hoveringCell && params.hoveringCell.layer === params.layerIdx - 1) {
            // draw all attentions that point to this token (from all heads, ignoring the given headIdx)
            for (const headIdx2 of range(0, 16)) {
                for (const tokIdxOther of range(params.hideFirstAttn ? 1 : 0, params.hoveringCell.seq + 1)) {
                    const key: string = `${params.layerIdx}:${headIdx2}:${params.hoveringCell.seq}:${tokIdxOther}`;
                    if (key in params.attentions) {
                        const value = params.attentions[key];
                        ctx.lineWidth = value * 5;
                        const [R, G, B] = COLORS[headIdx2];
                        ctx.strokeStyle = rgba(R, G, B, value);
                        const a = Math.min(params.hoveringCell.seq, tokIdxOther) * tokenWidth + tokenWidth / 2;
                        const b = Math.max(params.hoveringCell.seq, tokIdxOther) * tokenWidth + tokenWidth / 2;
                        drawArcBetweenPoints(ctx, a, b);
                    }
                }
            }
        } else {
            // draw all attentions between all tokens
            for (const tokIdx1 of range(0, params.seqLen)) {
                for (const tokIdx2 of range(params.hideFirstAttn ? 1 : 0, tokIdx1 + 1)) {
                    const key = `${params.layerIdx}:${params.headIdx}:${tokIdx1}:${tokIdx2}`;
                    if (key in params.attentions) {
                        const value = params.attentions[key];
                        ctx.lineWidth = value * 5;
                        const [R, G, B] = COLORS[params.headIdx];
                        ctx.strokeStyle = rgba(R, G, B, value);
                        const a = Math.min(tokIdx1, tokIdx2) * tokenWidth + tokenWidth / 2;
                        const b = Math.max(tokIdx1, tokIdx2) * tokenWidth + tokenWidth / 2;
                        drawArcBetweenPoints(ctx, a, b);
                    }
                }
            }
        }
    }
};


interface AttnProps {
    attentions: { [k: string]: number };
    layerIdx: number;
    headIdx: number;
    seqLen: number;
    hoveringCell: { layer: number, seq: number } | null;
    hideFirstAttn: boolean;
}

function Attn({
    attentions,
    layerIdx,
    headIdx,
    seqLen,
    hoveringCell,
    hideFirstAttn,
}: AttnProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        if (canvasRef.current) {
            redraw({
                canvas: canvasRef.current,
                attentions,
                layerIdx,
                headIdx,
                seqLen,
                hoveringCell,
                hideFirstAttn,
            });
        }
    }, [canvasRef, attentions, layerIdx, headIdx, seqLen, hoveringCell, hideFirstAttn]);

    return <div>
        <canvas
            ref={canvasRef}
            width={1000}
            height={50}
            style={{
                width: '100%',
            }}
        />
    </div>
}