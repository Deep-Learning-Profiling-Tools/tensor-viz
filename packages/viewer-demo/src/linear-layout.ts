import {
    createBundleManifest,
    product,
    unravelIndex,
    type LoadedBundleDocument,
    type TensorViewSnapshot,
    type ViewerSnapshot,
} from '@tensor-viz/viewer-core';

export type ComposeChannel = 'H' | 'S' | 'L';
export type ComposeMappingValue = string | 'none';

export type ComposeLayoutState = {
    specsText: string;
    operationText: string;
    inputName: string;
    visibleTensors: Record<string, boolean>;
    mapping: Record<ComposeChannel, ComposeMappingValue>;
    ranges: Record<ComposeChannel, [string, string]>;
};

export type NamedLayoutSpec = {
    name: string;
    inputs: string[];
    outputs: string[];
    bases: number[][][];
};

export type MatrixAxis = {
    label: string;
    axis: number;
};

export type MatrixBlock = {
    title: string;
    rows: MatrixAxis[];
    columns: MatrixAxis[];
    values: number[][];
};

export type ComposeTensorMeta = {
    id: string;
    title: string;
    exprText: string;
    kind: 'root' | 'step';
    axisLabels: string[];
    shape: number[];
    rootToTensor: number[][];
    visible: boolean;
};

export type ComposeLayoutMeta = {
    version: 1;
    specsText: string;
    operationText: string;
    inputName?: string;
    rootInputLabels: string[];
    rootInputBitCounts: number[];
    tensors: ComposeTensorMeta[];
};

export type ComposeRuntime = {
    specs: NamedLayoutSpec[];
    inputLabels: string[];
    inputBitCounts: number[];
    inputShape: number[];
    tensors: ComposeTensorMeta[];
    matrixBlocks: MatrixBlock[];
    pythonCode: string;
    meta: ComposeLayoutMeta;
};

type LayoutExpr =
    | { kind: 'name'; name: string }
    | { kind: 'inverse'; expr: LayoutExpr }
    | { kind: 'product'; left: LayoutExpr; right: LayoutExpr }
    | { kind: 'apply'; outer: LayoutExpr; inner: LayoutExpr };

type EvaluatedLayout = {
    kind: LayoutExpr['kind'];
    exprText: string;
    codeRef: string;
    inputs: string[];
    outputs: string[];
    inputBitCounts: number[];
    outputBitCounts: number[];
    matrix: number[][];
    spec?: NamedLayoutSpec;
};

type CodeLine = {
    title: string;
    layout: EvaluatedLayout;
    line: string;
};

type ExampleState = {
    title: string;
    state: ComposeLayoutState;
};

const DEFAULT_INPUT_NAME = 'Input Space';

const DEFAULT_COLOR_RANGES = {
    H: ['0', '0.8'],
    S: ['1', '1'],
    L: ['0', '1'],
} as const;

const DEFAULT_MAPPING = {
    H: 'T',
    S: 'W',
    L: 'R',
} as const;

const LEGACY_AXIS_ALIASES = {
    warp: 'W',
    thread: 'T',
    register: 'R',
} as const;

const BLOCKED_LAYOUT_TEXT = [
    'Blocked_Layout: [T,W,R] -> [A,B]',
    '[[4,0],[8,0],[0,1],[0,2],[0,4]]',
    '[[0,8],[0,16]]',
    '[[1,0],[2,0]]',
].join('\n');

const MMA_A_LAYOUT_TEXT = [
    'MMA_A_Layout_m16n8k16: [T,W,R] -> [A,B]',
    '[[0,2],[0,4],[1,0],[2,0],[4,0]]',
    '[]',
    '[[0,1],[8,0],[0,8]]',
].join('\n');

const MMA_B_LAYOUT_TEXT = [
    'MMA_B_Layout_m16n8k16: [T,W,R] -> [A,B]',
    '[[2,0],[4,0],[0,1],[0,2],[0,4]]',
    '[]',
    '[[1,0],[8,0]]',
].join('\n');

const MMA_C_LAYOUT_TEXT = [
    'MMA_C_Layout_m16n8k16: [T,W,R] -> [A,B]',
    '[[0,2],[0,4],[1,0],[2,0],[4,0]]',
    '[]',
    '[[0,1],[8,0]]',
].join('\n');

const SHARED_MEMORY_SWIZZLE_TEXT = [
    'Shared_Memory_128B_Swizzle: [Y,X] -> [S,B]',
    '[[1,1],[2,2],[4,4]]',
    '[[0,1],[0,2],[0,4]]',
].join('\n');

const DEFAULT_EMPTY_SPEC_TEXT = [
    'Layout_1: [T,W,R] -> [A,B]',
    '[[0,1],[0,2],[0,4],[0,8],[0,16]]',
    '[]',
    '[[1,0],[2,0]]',
].join('\n');

const BAKED_EXAMPLES: ExampleState[] = [
    bakedExample('Blocked Layout', BLOCKED_LAYOUT_TEXT, 'Blocked_Layout', 'Hardware Layout'),
    bakedExample('MMA A Layout (m16n8k16)', MMA_A_LAYOUT_TEXT, 'MMA_A_Layout_m16n8k16', 'Hardware Layout'),
    bakedExample('MMA B Layout (m16n8k16)', MMA_B_LAYOUT_TEXT, 'MMA_B_Layout_m16n8k16', 'Hardware Layout'),
    bakedExample('MMA C Layout (m16n8k16)', MMA_C_LAYOUT_TEXT, 'MMA_C_Layout_m16n8k16', 'Hardware Layout'),
    bakedExample('Shared Memory 128B Swizzle', SHARED_MEMORY_SWIZZLE_TEXT, 'Shared_Memory_128B_Swizzle', 'Logical Offsets'),
];

function bakedExample(
    title: string,
    specsText: string,
    operationText: string,
    inputName = DEFAULT_INPUT_NAME,
): ExampleState {
    return {
        title,
        state: {
            specsText,
            operationText,
            inputName,
            visibleTensors: {},
            mapping: { ...DEFAULT_MAPPING },
            ranges: {
                H: [...DEFAULT_COLOR_RANGES.H],
                S: [...DEFAULT_COLOR_RANGES.S],
                L: [...DEFAULT_COLOR_RANGES.L],
            },
        },
    };
}

export function defaultComposeLayoutState(): ComposeLayoutState {
    return cloneComposeLayoutState(BAKED_EXAMPLES[0]!.state);
}

export function emptyComposeLayoutState(): ComposeLayoutState {
    return {
        specsText: DEFAULT_EMPTY_SPEC_TEXT,
        operationText: 'Layout_1',
        inputName: DEFAULT_INPUT_NAME,
        visibleTensors: {},
        mapping: { ...DEFAULT_MAPPING },
        ranges: {
            H: [...DEFAULT_COLOR_RANGES.H],
            S: [...DEFAULT_COLOR_RANGES.S],
            L: [...DEFAULT_COLOR_RANGES.L],
        },
    };
}

export function bakedComposeLayoutExamples(): ExampleState[] {
    return BAKED_EXAMPLES.map(({ title, state }) => ({
        title,
        state: cloneComposeLayoutState(state),
    }));
}

export function cloneComposeLayoutState(state: ComposeLayoutState): ComposeLayoutState {
    return {
        specsText: state.specsText,
        operationText: state.operationText,
        inputName: state.inputName,
        visibleTensors: { ...state.visibleTensors },
        mapping: { ...state.mapping },
        ranges: {
            H: [...state.ranges.H],
            S: [...state.ranges.S],
            L: [...state.ranges.L],
        },
    };
}

export function isComposeLayoutState(value: unknown): value is ComposeLayoutState {
    if (!value || typeof value !== 'object') return false;
    const record = value as ComposeLayoutState;
    return typeof record.specsText === 'string'
        && typeof record.operationText === 'string'
        && typeof record.inputName === 'string'
        && !!record.visibleTensors
        && typeof record.visibleTensors === 'object'
        && ['H', 'S', 'L'].every((channel) => typeof record.mapping?.[channel as ComposeChannel] === 'string')
        && ['H', 'S', 'L'].every((channel) => Array.isArray(record.ranges?.[channel as ComposeChannel]));
}

export function isComposeLayoutMeta(value: unknown): value is ComposeLayoutMeta {
    if (!value || typeof value !== 'object') return false;
    const record = value as ComposeLayoutMeta;
    return record.version === 1
        && typeof record.specsText === 'string'
        && typeof record.operationText === 'string'
        && (record.inputName === undefined || typeof record.inputName === 'string')
        && Array.isArray(record.rootInputLabels)
        && Array.isArray(record.rootInputBitCounts)
        && Array.isArray(record.tensors);
}

export function composeLayoutStateFromLegacySpec(raw: unknown, fallbackTitle = 'Layout_1'): ComposeLayoutState {
    const legacy = parseLegacySpec(raw, fallbackTitle);
    const labelMap = new Map<string, string>();
    const inputEntries = legacy.inputs.map((name, axis) => {
        const label = canonicalLegacyLabel(name, axis);
        labelMap.set(name, label);
        return { label, bases: legacy.bases[axis] ?? [] };
    });
    inputEntries.sort((left, right) => legacyAxisOrder(left.label) - legacyAxisOrder(right.label));
    const outputs = legacy.outputs.map((name, axis) => canonicalLegacyOutputLabel(name, axis));
    return {
        specsText: formatSpecsText([{
            name: sanitizeIdentifier(legacy.name || fallbackTitle || 'Layout_1'),
            inputs: inputEntries.map((entry) => entry.label),
            outputs,
            bases: inputEntries.map((entry) => entry.bases),
        }]),
        operationText: sanitizeIdentifier(legacy.name || fallbackTitle || 'Layout_1'),
        inputName: DEFAULT_INPUT_NAME,
        visibleTensors: {},
        mapping: legacyMapping(labelMap, legacy.colorAxes),
        ranges: legacyRanges(legacy.colorRanges),
    };
}

export function buildComposeRuntime(state: Pick<ComposeLayoutState, 'specsText' | 'operationText' | 'inputName' | 'visibleTensors'>): ComposeRuntime {
    const specs = parseLayoutSpecs(state.specsText);
    if (specs.length === 0) {
        throw new Error('At least one layout specification is required.');
    }
    const operationText = state.operationText.trim();
    if (!operationText) throw new Error('Layout Operation is required.');
    const operation = parseOperation(operationText);
    const specLayouts = new Map(specs.map((spec) => [spec.name, namedLayout(spec)]));
    const codeLines: CodeLine[] = [];
    const tempRefs = new WeakMap<object, EvaluatedLayout>();
    let tempIndex = 0;
    const evaluate = (expr: LayoutExpr): EvaluatedLayout => {
        const cached = tempRefs.get(expr as object);
        if (cached) return cached;
        let layout: EvaluatedLayout;
        switch (expr.kind) {
            case 'name': {
                const named = specLayouts.get(expr.name);
                if (!named) throw new Error(`Unknown layout ${expr.name}.`);
                layout = named;
                break;
            }
            case 'inverse': {
                const inner = evaluate(expr.expr);
                const inverse = invertLayout(inner, exprToString(expr));
                const codeRef = `layout_tmp${tempIndex += 1}`;
                layout = { ...inverse, codeRef };
                codeLines.push({
                    title: `${codeRef} = ${layout.exprText}`,
                    layout,
                    line: `${codeRef} = ${inner.codeRef}.invert()`,
                });
                break;
            }
            case 'product': {
                const left = evaluate(expr.left);
                const right = evaluate(expr.right);
                const productResult = productLayout(left, right, exprToString(expr));
                const codeRef = `layout_tmp${tempIndex += 1}`;
                layout = { ...productResult, codeRef };
                codeLines.push({
                    title: `${codeRef} = ${layout.exprText}`,
                    layout,
                    line: `${codeRef} = ${left.codeRef} * ${right.codeRef}`,
                });
                break;
            }
            case 'apply': {
                const outer = evaluate(expr.outer);
                const inner = evaluate(expr.inner);
                const composed = composeLayouts(inner, outer, exprToString(expr));
                const codeRef = `layout_tmp${tempIndex += 1}`;
                layout = { ...composed, codeRef };
                codeLines.push({
                    title: `${codeRef} = ${layout.exprText}`,
                    layout,
                    line: `${codeRef} = ${inner.codeRef}.compose(${outer.codeRef})`,
                });
                break;
            }
        }
        tempRefs.set(expr as object, layout);
        return layout;
    };
    const renderChain = renderSteps(operation, evaluate);
    const finalLayout = renderChain.at(-1)?.layout;
    if (!finalLayout) throw new Error('Unable to evaluate Layout Operation.');
    const inputLabels = finalLayout.inputs.slice();
    const inputBitCounts = finalLayout.inputBitCounts.slice();
    const inputShape = shapeFromBitCounts(inputBitCounts);
    const rootCount = product(inputShape);
    const inputName = state.inputName.trim() || DEFAULT_INPUT_NAME;
    const visibleTensors = state.visibleTensors ?? {};
    const rootTensor: ComposeTensorMeta = {
        id: 'compose-root',
        title: inputName,
        exprText: inputName,
        kind: 'root',
        axisLabels: inputLabels.slice(),
        shape: inputShape.slice(),
        rootToTensor: Array.from({ length: rootCount }, (_entry, index) => unravelIndex(index, inputShape)),
        visible: visibleTensors['compose-root'] ?? true,
    };
    const tensors = [
        rootTensor,
        ...renderChain.map(({ layout, exprText }, index) => ({
            id: `compose-step-${index + 1}`,
            title: exprText,
            exprText,
            kind: 'step' as const,
            axisLabels: layout.outputs.slice(),
            shape: shapeFromBitCounts(layout.outputBitCounts),
            rootToTensor: Array.from({ length: rootCount }, (_entry, rootIndex) => (
                mapCoord(unravelIndex(rootIndex, inputShape), inputBitCounts, layout.matrix, layout.outputBitCounts)
            )),
            visible: visibleTensors[`compose-step-${index + 1}`] ?? true,
        })),
    ];
    const matrixBlocks = [
        ...specs.map((spec) => matrixBlock(spec.name, namedLayout(spec))),
        ...codeLines.map(({ title, layout }) => matrixBlock(title, layout)),
    ];
    const pythonCode = [
        ...specs.flatMap((spec) => pythonNamedLayout(spec)),
        ...(specs.length ? [''] : []),
        ...codeLines.map(({ line }) => line),
    ].filter((line, index, lines) => !(line === '' && lines[index - 1] === '')).join('\n').trim();
    const meta: ComposeLayoutMeta = {
        version: 1,
        specsText: state.specsText,
        operationText,
        inputName,
        rootInputLabels: inputLabels.slice(),
        rootInputBitCounts: inputBitCounts.slice(),
        tensors: tensors.map((tensor) => ({
            ...tensor,
            rootToTensor: tensor.rootToTensor.map((coord) => coord.slice()),
            axisLabels: tensor.axisLabels.slice(),
            shape: tensor.shape.slice(),
        })),
    };
    return {
        specs,
        inputLabels,
        inputBitCounts,
        inputShape,
        tensors,
        matrixBlocks,
        pythonCode,
        meta,
    };
}

export function createComposeLayoutDocument(
    state: ComposeLayoutState,
    viewer?: Partial<ViewerSnapshot>,
    title?: string,
    tensorViews?: Record<string, TensorViewSnapshot>,
): LoadedBundleDocument {
    const runtime = buildComposeRuntime(state);
    const visibleTensors = runtime.tensors.filter((tensor) => tensor.visible);
    if (visibleTensors.length === 0) {
        throw new Error('At least one tensor in the render chain must stay visible.');
    }
    const rootLabelToAxis = new Map(runtime.inputLabels.map((label, axis) => [label, axis]));
    const rootColors = Array.from({ length: product(runtime.inputShape) }, (_entry, rootIndex) => (
        rgbColorForRootCoord(
            unravelIndex(rootIndex, runtime.inputShape),
            runtime.inputShape,
            rootLabelToAxis,
            state.mapping,
            state.ranges,
        )
    ));
    const tensors = new Map<string, Float32Array>();
    const manifest = createBundleManifest({
        viewer: persistedViewerSettings(viewer),
        tensors: visibleTensors.map((tensor) => {
            const data = new Float32Array(product(tensor.shape)).fill(-1);
            const rgb = new Float32Array(product(tensor.shape) * 3);
            const rootToTensor = tensor.rootToTensor;
            rootToTensor.forEach((coord, rootIndex) => {
                const flat = flatIndex(coord, tensor.shape);
                data[flat] = rootIndex;
                rgb.set(rootColors[rootIndex]!, flat * 3);
            });
            const markerCoords = Array.from({ length: data.length }, (_entry, index) => index)
                .filter((index) => data[index] < 0)
                .map((index) => unravelIndex(index, tensor.shape));
            tensors.set(tensor.id, data);
            return {
                id: tensor.id,
                name: tensor.title,
                dtype: 'float32' as const,
                shape: tensor.shape,
                axisLabels: tensor.axisLabels,
                view: tensorViews?.[tensor.id]
                    ? {
                        view: tensorViews[tensor.id]!.view,
                        hiddenIndices: tensorViews[tensor.id]!.hiddenIndices.slice(),
                    }
                    : undefined,
                colorInstructions: [{ mode: 'rgb' as const, kind: 'dense' as const, values: Array.from(rgb) }],
                markerCoords,
            };
        }),
    });
    const viewerState = manifest.viewer as ViewerSnapshot & {
        composeLayoutState?: ComposeLayoutState;
        composeLayoutMeta?: ComposeLayoutMeta;
    };
    viewerState.composeLayoutState = cloneComposeLayoutState(state);
    viewerState.composeLayoutMeta = runtime.meta;
    return {
        id: 'compose-layout',
        title: title ?? visibleTensors.at(-1)?.title ?? runtime.meta.operationText,
        manifest,
        tensors,
    };
}

function parseLayoutSpecs(text: string): NamedLayoutSpec[] {
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    const specs: NamedLayoutSpec[] = [];
    let index = 0;
    while (index < lines.length) {
        while (index < lines.length && !lines[index]!.trim()) index += 1;
        if (index >= lines.length) break;
        const signatureLine = lines[index]!.trim();
        const signature = parseSignature(signatureLine);
        index += 1;
        const bases: number[][][] = [];
        for (let axis = 0; axis < signature.inputs.length; axis += 1) {
            const line = lines[index]?.trim();
            if (!line) {
                throw new Error(`Layout ${signature.name} is missing basis row ${axis + 1} for ${signature.inputs[axis]}.`);
            }
            bases.push(parseBasisRow(line, signature.outputs.length, signature.inputs[axis]!));
            index += 1;
        }
        const spec = {
            name: signature.name,
            inputs: signature.inputs,
            outputs: signature.outputs,
            bases,
        };
        assertInjective(spec);
        specs.push(spec);
        while (index < lines.length && !lines[index]!.trim()) index += 1;
    }
    const duplicate = duplicateValue(specs.map((spec) => spec.name));
    if (duplicate) throw new Error(`Layout names must be unique; received duplicate ${duplicate}.`);
    return specs;
}

function parseSignature(line: string): { name: string; inputs: string[]; outputs: string[] } {
    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[(.*)\]\s*->\s*\[(.*)\]\s*$/);
    if (!match) {
        throw new Error('Each specification must start with "<name>: [labels] -> [labels]".');
    }
    const name = match[1]!;
    const inputs = parseLabelList(match[2] ?? '', `${name} inputs`);
    const outputs = parseLabelList(match[3] ?? '', `${name} outputs`);
    const duplicateInput = duplicateValue(inputs);
    if (duplicateInput) throw new Error(`Layout ${name} has duplicate input label ${duplicateInput}.`);
    const duplicateOutput = duplicateValue(outputs);
    if (duplicateOutput) throw new Error(`Layout ${name} has duplicate output label ${duplicateOutput}.`);
    return { name, inputs, outputs };
}

function parseLabelList(source: string, label: string): string[] {
    const trimmed = source.trim();
    if (!trimmed) return [];
    return trimmed.split(',').map((entry) => {
        const value = entry.trim();
        if (!/^[A-Za-z][0-9]*$/.test(value)) {
            throw new Error(`${label} may only contain labels like T, A0, or B12 (received ${JSON.stringify(value)}).`);
        }
        return value;
    });
}

function parseBasisRow(line: string, outputCount: number, axisLabel: string): number[][] {
    let parsed: unknown;
    try {
        parsed = JSON.parse(line);
    } catch {
        throw new Error(`${axisLabel} bases must be valid JSON.`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${axisLabel} bases must be a JSON array.`);
    return parsed.map((basis, basisIndex) => {
        if (!Array.isArray(basis)) {
            throw new Error(`${axisLabel} basis ${basisIndex + 1} must be an array.`);
        }
        if (basis.length !== outputCount) {
            throw new Error(`${axisLabel} basis ${basisIndex + 1} must have length ${outputCount}.`);
        }
        return basis.map((value, outputAxis) => {
            if (!Number.isInteger(value) || Number(value) < 0) {
                throw new Error(`${axisLabel} basis ${basisIndex + 1}[${outputAxis + 1}] must be a non-negative integer.`);
            }
            return Number(value);
        });
    });
}

function parseOperation(source: string): LayoutExpr {
    const parser = new OperationParser(source);
    const expr = parser.parse();
    parser.finish();
    return expr;
}

class OperationParser {
    private index = 0;

    public constructor(private readonly source: string) {}

    public parse(): LayoutExpr {
        this.skipWhitespace();
        return this.parseProduct();
    }

    public finish(): void {
        this.skipWhitespace();
        if (this.index < this.source.length) {
            throw new Error(`Unexpected token ${JSON.stringify(this.source[this.index])} in Layout Operation.`);
        }
    }

    private parseProduct(): LayoutExpr {
        let expr = this.parseApplication();
        while (true) {
            this.skipWhitespace();
            if (!this.consume('*')) return expr;
            expr = {
                kind: 'product',
                left: expr,
                right: this.parseApplication(),
            };
        }
    }

    private parseApplication(): LayoutExpr {
        let expr = this.parseUnary();
        while (true) {
            this.skipWhitespace();
            if (!this.consume('(')) return expr;
            const inner = this.parseProduct();
            this.skipWhitespace();
            this.expect(')');
            expr = {
                kind: 'apply',
                outer: expr,
                inner,
            };
        }
    }

    private parseUnary(): LayoutExpr {
        this.skipWhitespace();
        if (this.peekWord('inv')) {
            this.index += 3;
            this.skipWhitespace();
            this.expect('(');
            const expr = this.parseProduct();
            this.skipWhitespace();
            this.expect(')');
            return { kind: 'inverse', expr };
        }
        return this.parsePrimary();
    }

    private parsePrimary(): LayoutExpr {
        this.skipWhitespace();
        if (this.consume('(')) {
            const expr = this.parseProduct();
            this.skipWhitespace();
            this.expect(')');
            return expr;
        }
        const name = this.parseName();
        if (!name) throw new Error('Expected a layout name, "inv(...)", or parenthesized expression.');
        return { kind: 'name', name };
    }

    private parseName(): string | null {
        this.skipWhitespace();
        const match = this.source.slice(this.index).match(/^[A-Za-z_][A-Za-z0-9_]*/);
        if (!match) return null;
        this.index += match[0].length;
        return match[0];
    }

    private peekWord(word: string): boolean {
        return this.source.slice(this.index, this.index + word.length) === word
            && !/[A-Za-z0-9_]/.test(this.source[this.index + word.length] ?? '');
    }

    private skipWhitespace(): void {
        while (/\s/.test(this.source[this.index] ?? '')) this.index += 1;
    }

    private consume(char: string): boolean {
        if (this.source[this.index] !== char) return false;
        this.index += 1;
        return true;
    }

    private expect(char: string): void {
        if (!this.consume(char)) {
            throw new Error(`Expected ${JSON.stringify(char)} in Layout Operation.`);
        }
    }
}

function namedLayout(spec: NamedLayoutSpec): EvaluatedLayout {
    const inputBitCounts = spec.bases.map((bases) => bases.length);
    const outputBitCounts = outputBitCountsFromBases(spec.bases, spec.outputs.length);
    return {
        kind: 'name',
        exprText: spec.name,
        codeRef: spec.name,
        inputs: spec.inputs.slice(),
        outputs: spec.outputs.slice(),
        inputBitCounts,
        outputBitCounts,
        matrix: matrixFromBases(spec.bases, inputBitCounts, outputBitCounts),
        spec,
    };
}

function composeLayouts(inner: EvaluatedLayout, outer: EvaluatedLayout, exprText: string): EvaluatedLayout {
    if (!sameLabels(inner.outputs, outer.inputs)) {
        throw new Error(`${outer.exprText} expects [${outer.inputs.join(',')}] but received [${inner.outputs.join(',')}].`);
    }
    const bridgeBitCounts = inner.outputBitCounts.map((bits, axis) => Math.max(bits, outer.inputBitCounts[axis] ?? 0));
    const matrix = multiplyMatrices(
        expandInputColumns(outer.matrix, outer.inputBitCounts, bridgeBitCounts),
        expandOutputRows(inner.matrix, inner.outputBitCounts, bridgeBitCounts),
    );
    const layout = {
        kind: 'apply',
        exprText,
        codeRef: exprText,
        inputs: inner.inputs.slice(),
        outputs: outer.outputs.slice(),
        inputBitCounts: inner.inputBitCounts.slice(),
        outputBitCounts: trimOutputBitCounts(matrix, outer.outputBitCounts),
        matrix,
    };
    assertEvaluatedInjective(layout);
    return layout;
}

function productLayout(left: EvaluatedLayout, right: EvaluatedLayout, exprText: string): EvaluatedLayout {
    const inputs = mergeAxisLabels(left.inputs, right.inputs);
    const inputBitCounts = inputs.map((label) => (
        (left.inputBitCounts[left.inputs.indexOf(label)] ?? 0)
        + (right.inputBitCounts[right.inputs.indexOf(label)] ?? 0)
    ));
    const outputs = mergeAxisLabels(left.outputs, right.outputs);
    const outputBitCounts = outputs.map((label) => (
        (left.outputBitCounts[left.outputs.indexOf(label)] ?? 0)
        + (right.outputBitCounts[right.outputs.indexOf(label)] ?? 0)
    ));
    const rightInputBitOffsets = inputs.map((label) => left.inputBitCounts[left.inputs.indexOf(label)] ?? 0);
    const rightOutputBitOffsets = outputs.map((label) => left.outputBitCounts[left.outputs.indexOf(label)] ?? 0);
    const matrix = mergeProductMatrices(
        embedProductMatrix(
            left,
            inputs,
            inputBitCounts,
            new Array(inputs.length).fill(0),
            outputs,
            outputBitCounts,
            new Array(outputs.length).fill(0),
        ),
        embedProductMatrix(
            right,
            inputs,
            inputBitCounts,
            rightInputBitOffsets,
            outputs,
            outputBitCounts,
            rightOutputBitOffsets,
        ),
    );
    const layout = {
        kind: 'product',
        exprText,
        codeRef: exprText,
        inputs,
        outputs,
        inputBitCounts,
        outputBitCounts: trimOutputBitCounts(matrix, outputBitCounts),
        matrix,
    };
    assertEvaluatedInjective(layout);
    return layout;
}

function invertLayout(layout: EvaluatedLayout, exprText: string): EvaluatedLayout {
    if (!isBijective(layout)) {
        throw new Error(`${layout.exprText} is not bijective, so inv(${layout.exprText}) is invalid.`);
    }
    const inverse = invertSquareMatrix(layout.matrix);
    if (!inverse) {
        throw new Error(`Unable to invert ${layout.exprText}.`);
    }
    return {
        kind: 'inverse',
        exprText,
        codeRef: exprText,
        inputs: layout.outputs.slice(),
        outputs: layout.inputs.slice(),
        inputBitCounts: layout.outputBitCounts.slice(),
        outputBitCounts: layout.inputBitCounts.slice(),
        matrix: inverse,
    };
}

function renderSteps(
    expr: LayoutExpr,
    evaluate: (expr: LayoutExpr) => EvaluatedLayout,
): Array<{ exprText: string; layout: EvaluatedLayout }> {
    if (expr.kind !== 'apply') {
        return [{ exprText: exprToString(expr), layout: evaluate(expr) }];
    }
    return [
        ...renderSteps(expr.inner, evaluate),
        { exprText: exprToString(expr), layout: evaluate(expr) },
    ];
}

function exprToString(expr: LayoutExpr, parentPrecedence = 0): string {
    const precedence = expr.kind === 'product' ? 1 : expr.kind === 'apply' ? 2 : 3;
    let text = '';
    switch (expr.kind) {
        case 'name':
            text = expr.name;
            break;
        case 'inverse':
            text = `inv(${exprToString(expr.expr, 0)})`;
            break;
        case 'product':
            text = `${exprToString(expr.left, precedence)} * ${exprToString(expr.right, precedence + 1)}`;
            break;
        case 'apply':
            text = `${exprToString(expr.outer, precedence)}(${exprToString(expr.inner, 0)})`;
            break;
    }
    return precedence < parentPrecedence ? `(${text})` : text;
}

function matrixBlock(title: string, layout: EvaluatedLayout): MatrixBlock {
    return {
        title,
        rows: bitLabels(layout.outputs, layout.outputBitCounts),
        columns: bitLabels(layout.inputs, layout.inputBitCounts),
        values: layout.matrix.map((row) => row.slice()),
    };
}

function bitLabels(labels: string[], bitCounts: number[]): MatrixAxis[] {
    return labels.flatMap((label, axis) => (
        Array.from({ length: bitCounts[axis] ?? 0 }, (_entry, bit) => ({ label: `${label}${bit}`, axis }))
    ));
}

function pythonNamedLayout(spec: NamedLayoutSpec): string[] {
    if (spec.inputs.length === 0) {
        return [
            `${spec.name} = LinearLayout.from_bases(`,
            '    [],',
            `    ${JSON.stringify(spec.outputs)},`,
            ')',
        ];
    }
    return [
        `${spec.name} = LinearLayout.from_bases(`,
        '    [',
        ...spec.inputs.map((label, axis) => `        (${JSON.stringify(label)}, ${JSON.stringify(spec.bases[axis] ?? [])}),`),
        '    ],',
        `    ${JSON.stringify(spec.outputs)},`,
        ')',
    ];
}

function matrixFromBases(
    bases: number[][][],
    inputBitCounts: number[],
    outputBitCounts: number[],
): number[][] {
    const rowCount = sum(outputBitCounts);
    const columnCount = sum(inputBitCounts);
    const matrix = Array.from({ length: rowCount }, () => new Array(columnCount).fill(0));
    const inputOffsets = offsets(inputBitCounts);
    const outputOffsets = offsets(outputBitCounts);
    bases.forEach((axisBases, inputAxis) => {
        axisBases.forEach((basis, bit) => {
            const column = inputOffsets[inputAxis]! + bit;
            basis.forEach((component, outputAxis) => {
                const outputBits = outputBitCounts[outputAxis] ?? 0;
                for (let outputBit = 0; outputBit < outputBits; outputBit += 1) {
                    if (((component >> outputBit) & 1) === 0) continue;
                    matrix[outputOffsets[outputAxis]! + outputBit]![column] = 1;
                }
            });
        });
    });
    return matrix;
}

function mapCoord(
    inputCoord: number[],
    inputBitCounts: number[],
    matrix: number[][],
    outputBitCounts: number[],
): number[] {
    const inputBits = bitsFromCoord(inputCoord, inputBitCounts);
    const outputBits = multiplyMatrixVector(matrix, inputBits);
    return coordFromBits(outputBits, outputBitCounts);
}

function bitsFromCoord(coord: number[], bitCounts: number[]): number[] {
    return bitCounts.flatMap((count, axis) => (
        Array.from({ length: count }, (_entry, bit) => ((coord[axis] ?? 0) >> bit) & 1)
    ));
}

function coordFromBits(bits: number[], bitCounts: number[]): number[] {
    const result = new Array(bitCounts.length).fill(0);
    const axisOffsets = offsets(bitCounts);
    bitCounts.forEach((count, axis) => {
        let value = 0;
        for (let bit = 0; bit < count; bit += 1) {
            value |= (bits[axisOffsets[axis]! + bit] ?? 0) << bit;
        }
        result[axis] = value;
    });
    return result;
}

function multiplyMatrixVector(matrix: number[][], vector: number[]): number[] {
    return matrix.map((row) => row.reduce((value, cell, column) => value ^ (cell & (vector[column] ?? 0)), 0));
}

function multiplyMatrices(left: number[][], right: number[][]): number[][] {
    const rows = left.length;
    const shared = left[0]?.length ?? 0;
    const columns = right[0]?.length ?? 0;
    if (shared !== right.length) {
        throw new Error(`Incompatible layout matrices: ${rows}x${shared} cannot multiply ${right.length}x${columns}.`);
    }
    return Array.from({ length: rows }, (_entry, row) => (
        Array.from({ length: columns }, (_cell, column) => {
            let value = 0;
            for (let index = 0; index < shared; index += 1) {
                value ^= (left[row]![index] ?? 0) & (right[index]![column] ?? 0);
            }
            return value;
        })
    ));
}

function mergeProductMatrices(left: number[][], right: number[][]): number[][] {
    return left.map((row, rowIndex) => row.map((value, columnIndex) => value | (right[rowIndex]?.[columnIndex] ?? 0)));
}

function embedProductMatrix(
    layout: EvaluatedLayout,
    targetInputs: string[],
    targetInputBitCounts: number[],
    sharedInputOffset: number[],
    targetOutputs: string[],
    targetOutputBitCounts: number[],
    sharedOutputOffset: number[],
): number[][] {
    const matrix = Array.from({ length: sum(targetOutputBitCounts) }, () => new Array(sum(targetInputBitCounts)).fill(0));
    const targetInputAxes = new Map(targetInputs.map((label, axis) => [label, axis]));
    const targetOutputAxes = new Map(targetOutputs.map((label, axis) => [label, axis]));
    const targetInputOffsets = offsets(targetInputBitCounts);
    const targetOutputOffsets = offsets(targetOutputBitCounts);
    const inputOffsets = offsets(layout.inputBitCounts);
    const outputOffsets = offsets(layout.outputBitCounts);
    layout.outputs.forEach((label, outputAxis) => {
        const targetOutputAxis = targetOutputAxes.get(label);
        if (targetOutputAxis === undefined) return;
        const outputBase = targetOutputOffsets[targetOutputAxis]! + (sharedOutputOffset[targetOutputAxis] ?? 0);
        for (let outputBit = 0; outputBit < (layout.outputBitCounts[outputAxis] ?? 0); outputBit += 1) {
            const row = matrix[outputBase + outputBit]!;
            layout.inputs.forEach((inputLabel, inputAxis) => {
                const targetInputAxis = targetInputAxes.get(inputLabel);
                if (targetInputAxis === undefined) return;
                const inputBase = targetInputOffsets[targetInputAxis]! + (sharedInputOffset[targetInputAxis] ?? 0);
                for (let inputBit = 0; inputBit < (layout.inputBitCounts[inputAxis] ?? 0); inputBit += 1) {
                    row[inputBase + inputBit] = layout.matrix[outputOffsets[outputAxis]! + outputBit]?.[inputOffsets[inputAxis]! + inputBit] ?? 0;
                }
            });
        }
    });
    return matrix;
}

function mergeAxisLabels(left: string[], right: string[]): string[] {
    return [...left, ...right.filter((label) => !left.includes(label))];
}

function invertSquareMatrix(matrix: number[][]): number[][] | null {
    const size = matrix.length;
    if (size !== (matrix[0]?.length ?? 0)) return null;
    const rows = matrix.map((row, index) => [
        ...row.slice(),
        ...Array.from({ length: size }, (_entry, column) => column === index ? 1 : 0),
    ]);
    let pivotRow = 0;
    for (let column = 0; column < size; column += 1) {
        const candidate = rows.findIndex((row, index) => index >= pivotRow && row[column] === 1);
        if (candidate === -1) return null;
        if (candidate !== pivotRow) {
            const swap = rows[pivotRow]!;
            rows[pivotRow] = rows[candidate]!;
            rows[candidate] = swap;
        }
        rows.forEach((row, index) => {
            if (index === pivotRow || row[column] === 0) return;
            for (let offset = column; offset < row.length; offset += 1) {
                row[offset] = row[offset]! ^ rows[pivotRow]![offset]!;
            }
        });
        pivotRow += 1;
    }
    return rows.map((row) => row.slice(size));
}

function gf2Rank(matrix: number[][]): number {
    if (matrix.length === 0) return 0;
    const rows = matrix.map((row) => row.slice());
    const columnCount = rows[0]?.length ?? 0;
    let rank = 0;
    for (let column = 0; column < columnCount; column += 1) {
        const pivot = rows.findIndex((row, index) => index >= rank && row[column] === 1);
        if (pivot === -1) continue;
        if (pivot !== rank) {
            const swap = rows[rank]!;
            rows[rank] = rows[pivot]!;
            rows[pivot] = swap;
        }
        rows.forEach((row, index) => {
            if (index === rank || row[column] === 0) return;
            for (let offset = column; offset < columnCount; offset += 1) {
                row[offset] = row[offset]! ^ rows[rank]![offset]!;
            }
        });
        rank += 1;
    }
    return rank;
}

function assertInjective(spec: NamedLayoutSpec): void {
    const layout = namedLayout(spec);
    if (!isInjectiveLayout(layout)) {
        throw new Error(`${spec.name} is not injective over its full input domain.`);
    }
}

function assertEvaluatedInjective(layout: EvaluatedLayout): void {
    if (isInjectiveLayout(layout)) return;
    throw new Error(`${layout.exprText} is not injective over its full input domain.`);
}

function isInjectiveLayout(layout: Pick<EvaluatedLayout, 'matrix' | 'inputBitCounts'>): boolean {
    return gf2Rank(layout.matrix) === sum(layout.inputBitCounts);
}

function isBijective(layout: EvaluatedLayout): boolean {
    const inputBits = sum(layout.inputBitCounts);
    const outputBits = sum(layout.outputBitCounts);
    return inputBits === outputBits && gf2Rank(layout.matrix) === inputBits;
}

function outputBitCountsFromBases(bases: number[][][], outputCount: number): number[] {
    return Array.from({ length: outputCount }, (_entry, outputAxis) => {
        let bits = 0;
        bases.forEach((axisBases) => {
            axisBases.forEach((basis) => {
                bits = Math.max(bits, bitLength(basis[outputAxis] ?? 0));
            });
        });
        return bits;
    });
}

function trimOutputBitCounts(matrix: number[][], outputBitCounts: number[]): number[] {
    let rowOffset = 0;
    return outputBitCounts.map((bitCount) => {
        let used = 0;
        for (let bit = 0; bit < bitCount; bit += 1) {
            const row = matrix[rowOffset + bit] ?? [];
            if (row.some((value) => value !== 0)) used = bit + 1;
        }
        rowOffset += bitCount;
        return used;
    });
}

function bitLength(value: number): number {
    return value <= 0 ? 0 : Math.floor(Math.log2(value)) + 1;
}

function offsets(values: number[]): number[] {
    let total = 0;
    return values.map((value) => {
        const offset = total;
        total += value;
        return offset;
    });
}

function shapeFromBitCounts(bitCounts: number[]): number[] {
    return bitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
}

function sum(values: number[]): number {
    return values.reduce((total, value) => total + value, 0);
}

function sameLabels(leftLabels: string[], rightLabels: string[]): boolean {
    return leftLabels.length === rightLabels.length
        && leftLabels.every((label, index) => label === rightLabels[index]);
}

function expandOutputRows(
    matrix: number[][],
    currentBitCounts: number[],
    targetBitCounts: number[],
): number[][] {
    const currentOffsets = offsets(currentBitCounts);
    return targetBitCounts.flatMap((targetBits, axis) => (
        Array.from({ length: targetBits }, (_entry, bit) => {
            const currentBits = currentBitCounts[axis] ?? 0;
            if (bit >= currentBits) return new Array(matrix[0]?.length ?? 0).fill(0);
            return matrix[currentOffsets[axis]! + bit]!.slice();
        })
    ));
}

function expandInputColumns(
    matrix: number[][],
    currentBitCounts: number[],
    targetBitCounts: number[],
): number[][] {
    const currentOffsets = offsets(currentBitCounts);
    return matrix.map((row) => targetBitCounts.flatMap((targetBits, axis) => (
        Array.from({ length: targetBits }, (_entry, bit) => {
            const currentBits = currentBitCounts[axis] ?? 0;
            if (bit >= currentBits) return 0;
            return row[currentOffsets[axis]! + bit] ?? 0;
        })
    )));
}

function duplicateValue(values: string[]): string | null {
    const seen = new Set<string>();
    for (const value of values) {
        if (seen.has(value)) return value;
        seen.add(value);
    }
    return null;
}

function rgbColorForRootCoord(
    coord: number[],
    shape: number[],
    labelToAxis: Map<string, number>,
    mapping: Record<ComposeChannel, ComposeMappingValue>,
    ranges: Record<ComposeChannel, [string, string]>,
): [number, number, number] {
    const hue = channelValue('H', coord, shape, labelToAxis.get(mapping.H) ?? null, ranges);
    const saturation = channelValue('S', coord, shape, labelToAxis.get(mapping.S) ?? null, ranges);
    const lightness = channelValue('L', coord, shape, labelToAxis.get(mapping.L) ?? null, ranges);
    return hsvToRgb(hue, saturation, lightness);
}

function channelValue(
    channel: ComposeChannel,
    coord: number[],
    shape: number[],
    axis: number | null,
    ranges: Record<ComposeChannel, [string, string]>,
): number {
    const [startText, endText] = ranges[channel];
    const start = Number(startText);
    const end = Number(endText);
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
        throw new Error(`${channel} range must contain numbers.`);
    }
    if (axis === null) return start;
    if ((shape[axis] ?? 1) <= 1) return start;
    const position = (coord[axis] ?? 0) / ((shape[axis] ?? 1) - 1);
    return start + ((end - start) * position);
}

function hsvToRgb(hue: number, saturation: number, value: number): [number, number, number] {
    const scaledHue = ((((hue % 1) + 1) % 1) * 6);
    const sector = Math.floor(scaledHue);
    const fraction = scaledHue - sector;
    const p = value * (1 - saturation);
    const q = value * (1 - (fraction * saturation));
    const t = value * (1 - ((1 - fraction) * saturation));
    switch (sector % 6) {
        case 0:
            return [value, t, p];
        case 1:
            return [q, value, p];
        case 2:
            return [p, value, t];
        case 3:
            return [p, q, value];
        case 4:
            return [t, p, value];
        default:
            return [value, p, q];
    }
}

function flatIndex(coord: number[], shape: number[]): number {
    return coord.reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}

function persistedViewerSettings(viewer: Partial<ViewerSnapshot> | undefined): Partial<ViewerSnapshot> {
    if (!viewer) return { dimensionMappingScheme: 'contiguous' };
    return {
        displayMode: viewer.displayMode,
        heatmap: viewer.heatmap,
        dimensionBlockGapMultiple: viewer.dimensionBlockGapMultiple,
        displayGaps: viewer.displayGaps,
        logScale: viewer.logScale,
        collapseHiddenAxes: viewer.collapseHiddenAxes,
        dimensionMappingScheme: viewer.dimensionMappingScheme ?? 'contiguous',
        showDimensionLines: viewer.showDimensionLines,
        showTensorNames: viewer.showTensorNames,
        showInspectorPanel: viewer.showInspectorPanel,
        showSelectionPanel: viewer.showSelectionPanel,
        showHoverDetailsPanel: viewer.showHoverDetailsPanel,
    };
}

function sanitizeIdentifier(value: string): string {
    const cleaned = value.replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '');
    return /^[A-Za-z_]/.test(cleaned) ? cleaned : `Layout_${cleaned || '1'}`;
}

function formatSpecsText(specs: NamedLayoutSpec[]): string {
    return specs.map((spec) => [
        `${spec.name}: [${spec.inputs.join(',')}] -> [${spec.outputs.join(',')}]`,
        ...spec.bases.map((row) => JSON.stringify(row)),
    ].join('\n')).join('\n\n');
}

function parseLegacySpec(raw: unknown, fallbackTitle: string): {
    name: string;
    inputs: string[];
    outputs: string[];
    bases: number[][][];
    colorAxes: Record<string, string>;
    colorRanges: Record<string, [number, number]>;
} {
    if (!raw || typeof raw !== 'object') {
        throw new Error('Unable to upgrade legacy layout state.');
    }
    const record = raw as Record<string, unknown>;
    const basesEntries = Array.isArray(record.input_dims)
        ? record.input_dims
        : Array.isArray(record.bases)
            ? record.bases
            : [];
    const inputs: string[] = [];
    const bases: number[][][] = [];
    basesEntries.forEach((entry, axis) => {
        const pair = Array.isArray(entry) ? entry : [entry?.name, entry?.bases];
        inputs.push(typeof pair[0] === 'string' ? pair[0] : `Axis${axis + 1}`);
        bases.push(Array.isArray(pair[1]) ? pair[1] as number[][] : []);
    });
    const outputsSource = Array.isArray(record.output_dims)
        ? record.output_dims
        : Array.isArray(record.out_dims)
            ? record.out_dims
            : [];
    const outputs = outputsSource.length
        ? outputsSource.map((entry, axis) => {
            if (typeof entry === 'string') return entry;
            if (Array.isArray(entry) && typeof entry[0] === 'string') return entry[0];
            if (entry && typeof entry === 'object' && typeof (entry as { name?: unknown }).name === 'string') {
                return (entry as { name: string }).name;
            }
            return `Axis${axis + 1}`;
        })
        : Array.from({ length: Math.max(1, ...bases.flatMap((row) => row.map((basis) => basis.length))) }, (_entry, axis) => String.fromCharCode(65 + axis));
    const colorAxes = record.color_axes && typeof record.color_axes === 'object'
        ? record.color_axes as Record<string, string>
        : {};
    const colorRanges = record.color_ranges && typeof record.color_ranges === 'object'
        ? Object.fromEntries(Object.entries(record.color_ranges).flatMap(([key, value]) => (
            Array.isArray(value) && value.length === 2 ? [[key, [Number(value[0]), Number(value[1])]]] : []
        ))) as Record<string, [number, number]>
        : {};
    return {
        name: typeof record.name === 'string' && record.name.trim() ? record.name : fallbackTitle,
        inputs,
        outputs,
        bases,
        colorAxes,
        colorRanges,
    };
}

function canonicalLegacyLabel(name: string, axis: number): string {
    const lowered = name.toLowerCase();
    if (lowered in LEGACY_AXIS_ALIASES) return LEGACY_AXIS_ALIASES[lowered as keyof typeof LEGACY_AXIS_ALIASES];
    const match = name.match(/[A-Za-z]/);
    const base = (match?.[0] ?? 'A').toUpperCase();
    return axis === 0 ? base : `${base}${axis}`;
}

function canonicalLegacyOutputLabel(name: string, axis: number): string {
    const match = name.match(/^([A-Za-z])([0-9]*)$/);
    if (match) return `${match[1]!.toUpperCase()}${match[2] ?? ''}`;
    const firstLetter = name.match(/[A-Za-z]/)?.[0];
    return (firstLetter ?? String.fromCharCode(65 + axis)).toUpperCase();
}

function legacyAxisOrder(label: string): number {
    if (label === 'T') return 0;
    if (label === 'W') return 1;
    if (label === 'R') return 2;
    return 3;
}

function legacyMapping(
    labelMap: Map<string, string>,
    colorAxes: Record<string, string>,
): Record<ComposeChannel, ComposeMappingValue> {
    const mapping: Record<ComposeChannel, ComposeMappingValue> = {
        H: labelMap.get('thread') ?? 'none',
        S: labelMap.get('warp') ?? 'none',
        L: labelMap.get('register') ?? 'none',
    };
    Object.entries(colorAxes).forEach(([axisName, channelName]) => {
        const channel = String(channelName).toUpperCase() as ComposeChannel;
        if (!['H', 'S', 'L'].includes(channel)) return;
        mapping[channel] = labelMap.get(axisName) ?? canonicalLegacyLabel(axisName, 0);
    });
    return mapping;
}

function legacyRanges(colorRanges: Record<string, [number, number]>): Record<ComposeChannel, [string, string]> {
    return {
        H: colorRanges.H ? [String(colorRanges.H[0]), String(colorRanges.H[1])] : [...DEFAULT_COLOR_RANGES.H],
        S: colorRanges.S ? [String(colorRanges.S[0]), String(colorRanges.S[1])] : [...DEFAULT_COLOR_RANGES.S],
        L: colorRanges.L ? [String(colorRanges.L[0]), String(colorRanges.L[1])] : [...DEFAULT_COLOR_RANGES.L],
    };
}
