import React, { useEffect, useRef } from "react";
import {
    Dataset,
    ExampleData,
    range,
} from "./utils";


interface LogitAttnViewProps {
    dataset: Dataset | null;
    exampleIdx: number;
    dataForExample: ExampleData | null;
    loading: boolean;
    headIdx: number;
    updateHeadIdx: (idx: number) => void;
    hoveringCell: { layer: number, seq: number } | null;
    updateHoveringCell: (cell: { layer: number, seq: number } | null) => void;
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
}: LogitAttnViewProps) {

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
            {example.tokens.map((tok, seqIdx) => {
                const topLogits = dataForExample.logits[layerIdx][seqIdx];
                const prevTopLogits = layerIdx > 0 ? dataForExample.logits[layerIdx - 1][seqIdx] : null;
                const changed = topLogits && prevTopLogits && prevTopLogits[0].tok !== topLogits[0].tok;
                const correct = topLogits && seqIdx - 1 < example.tokens.length && example.tokens[seqIdx + 1] === topLogits[0].tok;
                const hovering = hoveringCell && hoveringCell.layer === layerIdx && hoveringCell.seq === seqIdx;
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
                    {topLogits ? topLogits[0].tok : ''}
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
        <div
            style={{
            }}
        >
            <input
                type='range'
                min={0}
                max={15}
                value={headIdx}
                onChange={(e) => {
                    const newIdx = parseInt(e.target.value);
                    updateHeadIdx(newIdx);
                }}
                style={{
                    width: '200px',
                }}
            />
            <span style={{ fontSize: '0.9em', top: -2, position: 'relative', paddingLeft: 5 }}>
                head {headIdx}
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
}) => {
    const ctx = params.canvas.getContext('2d');
    if (ctx) {
        const width = params.canvas.width;
        const height = params.canvas.height;
        const tokenWidth = width / params.seqLen;
        ctx.clearRect(0, 0, width, height);
        for (const tok1Idx of range(0, params.seqLen)) {
            // don't draw attention for first token (noisy)
            for (const tok2Idx of range(1, params.seqLen)) {
                const draw = () => {
                    const key = `${params.layerIdx}:${params.headIdx}:${tok1Idx}:${tok2Idx}`;
                    if (key in params.attentions) {
                        const value = params.attentions[key];
                        ctx.lineWidth = value * 5;
                        ctx.strokeStyle = `rgba(52, 183, 235, ${value})`;
                        const a = Math.min(tok1Idx, tok2Idx) * tokenWidth + tokenWidth / 2;
                        const b = Math.max(tok1Idx, tok2Idx) * tokenWidth + tokenWidth / 2;
                        drawArcBetweenPoints(ctx, a, b);
                    }
                };
                if (params.hoveringCell && params.hoveringCell.layer === params.layerIdx - 1) {
                    // draw just the attention for the cell we're hovering over
                    const hovering = params.hoveringCell.seq === tok1Idx || params.hoveringCell.seq === tok2Idx;
                    if (hovering) {
                        draw();
                    }
                } else {
                    // draw everything
                    draw();
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
}

function Attn({
    attentions,
    layerIdx,
    headIdx,
    seqLen,
    hoveringCell,
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
            });
        }
    }, [canvasRef, attentions, layerIdx, headIdx, seqLen, hoveringCell]);

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
