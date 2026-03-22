import {
    createBundleManifest,
    product,
    unravelIndex,
    type LoadedBundleDocument,
    type ViewerSnapshot,
} from '@tensor-viz/viewer-core';

const DEFAULT_COLOR_RANGES = {
    H: [0, 0.85],
    S: [0.35, 0.95],
    L: [0.25, 0.95],
} as const;

const DEFAULT_LINEAR_LAYOUT_JSON = {
    name: 'Blocked Layout',
    bases: [
        ['warp', [[0, 8], [0, 16]]],
        ['thread', [[4, 0], [8, 0], [0, 1], [0, 2], [0, 4]]],
        ['register', [[1, 0], [2, 0]]],
    ],
    out_dims: ['x', 'y'],
    color_axes: {
        warp: 'H',
        thread: 'S',
        register: 'L',
    },
    color_ranges: {
        H: [0, 0.8],
        S: [0, 0],
        L: [0, 1],
    },
};

export const LINEAR_LAYOUT_TAB_ID = 'custom-linear-layout';

export type LinearLayoutColorAxes = Record<string, string>;
export type LinearLayoutColorRanges = Partial<Record<'H' | 'S' | 'L', [number, number]>>;

export type LinearLayoutInputDimSpec = {
    name: string;
    bases: number[][];
};

export type LinearLayoutOutputDimSpec = {
    name: string;
    size: number;
};

export type LinearLayoutSpec = {
    name?: string;
    input_dims: LinearLayoutInputDimSpec[];
    output_dims: LinearLayoutOutputDimSpec[];
    color_axes?: LinearLayoutColorAxes;
    color_ranges?: LinearLayoutColorRanges;
};

/** Return the default sidebar JSON shown by the static demo. */
export function defaultLinearLayoutSpecText(): string {
    return JSON.stringify(DEFAULT_LINEAR_LAYOUT_JSON, null, 2);
}

/** Build one loaded viewer document from a parsed linear-layout spec. */
export function createLinearLayoutDocument(
    spec: LinearLayoutSpec,
    viewer?: Partial<ViewerSnapshot>,
): LoadedBundleDocument {
    const inputShape = spec.input_dims.map(({ bases }) => 2 ** bases.length);
    const logicalOutputDims = logicalOutputDimsFor(spec.output_dims);
    const logicalShape = logicalOutputDims.map(({ size }) => size);
    const outputAxes = new Map(spec.output_dims.map((dim, axis) => [dim.name, axis]));
    const inputNames = spec.input_dims.map(({ name }) => name);
    const channelAxes = normalizeColorAxes(inputNames, inputShape, spec.color_axes);
    const hardwareTensor = new Float32Array(product(inputShape));
    const logicalTensor = new Float32Array(product(logicalShape)).fill(-1);
    const hardwareRgba = new Float32Array(product(inputShape) * 4);
    const logicalRgba = new Float32Array(product(logicalShape) * 4);

    for (let index = 0; index < hardwareTensor.length; index += 1) {
        const inputCoord = unravelIndex(index, inputShape);
        const hue = colorValue('H', inputCoord, inputShape, channelAxes.H, spec.color_ranges);
        const saturation = colorValue('S', inputCoord, inputShape, channelAxes.S, spec.color_ranges);
        const lightness = colorValue('L', inputCoord, inputShape, channelAxes.L, spec.color_ranges);
        const rgba = rgbaColor(hue, saturation, lightness);
        const outputCoord = mapLinearLayoutCoord(inputCoord, spec.input_dims, spec.output_dims.length);
        const logicalCoord = logicalOutputDims.map(({ name, size }) => {
            const axis = outputAxes.get(name);
            if (axis === undefined) throw new Error(`Unknown logical output axis ${name}.`);
            const value = outputCoord[axis] ?? 0;
            if (value < 0 || value >= size) {
                throw new Error(`Output coordinate ${name}=${value} exceeds declared size ${size}.`);
            }
            return value;
        });
        const logicalIndex = flatIndex(logicalCoord, logicalShape);
        hardwareTensor[index] = lightness;
        logicalTensor[logicalIndex] = lightness;
        hardwareRgba.set(rgba, index * 4);
        logicalRgba.set(rgba, logicalIndex * 4);
    }

    const manifest = createBundleManifest({
        viewer: persistedViewerSettings(viewer),
        tensors: [
            {
                id: 'tensor-1',
                name: 'Hardware tensor',
                dtype: 'float32',
                shape: inputShape,
                axisLabels: viewerAxisLabels(inputNames),
                colorInstructions: [{ mode: 'rgba', kind: 'dense', values: Array.from(hardwareRgba) }],
            },
            {
                id: 'tensor-2',
                name: 'Logical tensor',
                dtype: 'float32',
                shape: logicalShape,
                axisLabels: logicalAxisLabels(logicalOutputDims.map(({ name }) => name)),
                colorInstructions: [{ mode: 'rgba', kind: 'dense', values: Array.from(logicalRgba) }],
            },
        ],
    });

    return {
        id: LINEAR_LAYOUT_TAB_ID,
        title: spec.name?.trim() || 'Linear Layout',
        manifest,
        tensors: new Map([
            ['tensor-1', hardwareTensor],
            ['tensor-2', logicalTensor],
        ]),
    };
}

/** Parse the sidebar JSON into the canonical layout spec used by the viewer. */
export function parseLinearLayoutSpec(text: string): LinearLayoutSpec {
    const raw = expectRecord(JSON.parse(text), 'layout spec');
    const input_dims = parseInputDims(raw);
    const output_dims = parseOutputDims(raw, input_dims);
    return {
        name: optionalName(raw.name),
        input_dims,
        output_dims,
        color_axes: parseColorAxes(raw.color_axes),
        color_ranges: parseColorRanges(raw.color_ranges),
    };
}

/** Parse the supported input-dimension JSON shapes into canonical basis metadata. */
function parseInputDims(raw: Record<string, unknown>): LinearLayoutInputDimSpec[] {
    const entries = Array.isArray(raw.input_dims) ? raw.input_dims : Array.isArray(raw.bases) ? raw.bases : null;
    if (!entries?.length) throw new Error('input_dims or bases must be a non-empty array.');
    const dims = entries.map((entry, axis) => {
        if (Array.isArray(entry)) {
            const [name, bases] = entry;
            return {
                name: namedValue(name, `input dim ${axis + 1}`),
                bases: basisVectors(bases, `input dim ${axis + 1}`),
            };
        }
        const dim = expectRecord(entry, `input dim ${axis + 1}`);
        return {
            name: namedValue(dim.name, `input dim ${axis + 1}`),
            bases: basisVectors(dim.bases, `input dim ${axis + 1}`),
        };
    });
    const duplicate = duplicateName(dims.map(({ name }) => name));
    if (duplicate) throw new Error(`Input dims must be unique; received duplicate ${duplicate}.`);
    return dims;
}

/** Parse explicit or inferred output dimensions from the supported JSON schema. */
function parseOutputDims(
    raw: Record<string, unknown>,
    input_dims: LinearLayoutInputDimSpec[],
): LinearLayoutOutputDimSpec[] {
    const entries = Array.isArray(raw.output_dims) ? raw.output_dims : Array.isArray(raw.out_dims) ? raw.out_dims : null;
    if (!entries?.length) throw new Error('output_dims or out_dims must be a non-empty array.');
    const outputDims = entries.map((entry, axis) => {
        if (typeof entry === 'string') return { name: namedValue(entry, `output dim ${axis + 1}`), size: null };
        if (Array.isArray(entry)) {
            const [name, size] = entry;
            return {
                name: namedValue(name, `output dim ${axis + 1}`),
                size: size === undefined ? null : positiveInt(size, `output dim ${axis + 1} size`),
            };
        }
        const dim = expectRecord(entry, `output dim ${axis + 1}`);
        return {
            name: namedValue(dim.name, `output dim ${axis + 1}`),
            size: dim.size === undefined ? null : positiveInt(dim.size, `output dim ${axis + 1} size`),
        };
    });
    const duplicate = duplicateName(outputDims.map(({ name }) => name));
    if (duplicate) throw new Error(`Output dims must be unique; received duplicate ${duplicate}.`);
    if (outputDims.every(({ size }) => size !== null)) return outputDims as LinearLayoutOutputDimSpec[];
    if (outputDims.some(({ size }) => size !== null)) {
        throw new Error('Either specify sizes for every output dim or omit them for all output dims.');
    }
    const inputShape = input_dims.map(({ bases }) => 2 ** bases.length);
    const maxima = new Array(outputDims.length).fill(0);
    for (let index = 0; index < product(inputShape); index += 1) {
        const coord = mapLinearLayoutCoord(unravelIndex(index, inputShape), input_dims, outputDims.length);
        coord.forEach((value, axis) => {
            maxima[axis] = Math.max(maxima[axis], value);
        });
    }
    return outputDims.map(({ name }, axis) => ({ name, size: maxima[axis] + 1 }));
}

/** Normalize one optional layout name. */
function optionalName(value: unknown): string | undefined {
    if (value === undefined || value === null || value === '') return undefined;
    return namedValue(value, 'layout name');
}

/** Normalize the optional color-axis override map. */
function parseColorAxes(value: unknown): LinearLayoutColorAxes | undefined {
    if (value === undefined) return undefined;
    const raw = expectRecord(value, 'color_axes');
    const entries = Object.entries(raw).map(([axisName, channelName]) => [axisName, namedValue(channelName, `color_axes.${axisName}`)] as const);
    return entries.length ? Object.fromEntries(entries) : undefined;
}

/** Normalize the optional color-range override map. */
function parseColorRanges(value: unknown): LinearLayoutColorRanges | undefined {
    if (value === undefined) return undefined;
    const raw = expectRecord(value, 'color_ranges');
    const ranges = Object.entries(raw).map(([channel, range]) => {
        const values = numericTuple(range, `color_ranges.${channel}`);
        if (values.length !== 2) throw new Error(`${channel} must contain exactly two numeric endpoints.`);
        const normalized = channel.toUpperCase();
        if (!['H', 'S', 'L'].includes(normalized)) {
            throw new Error(`Unknown color channel ${channel}; expected H, S, or L.`);
        }
        return [normalized, [values[0], values[1]] as [number, number]] as const;
    });
    return ranges.length ? Object.fromEntries(ranges) : undefined;
}

/** Validate one object-like JSON value. */
function expectRecord(value: unknown, label: string): Record<string, unknown> {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
        throw new Error(`Expected ${label} to be an object.`);
    }
    return value as Record<string, unknown>;
}

/** Validate one required string-like name value. */
function namedValue(value: unknown, label: string): string {
    if (typeof value !== 'string' || !value.trim()) throw new Error(`Expected ${label} to be a non-empty string.`);
    return value.trim();
}

/** Validate one non-negative integer. */
function nonNegativeInt(value: unknown, label: string): number {
    if (!Number.isInteger(value) || Number(value) < 0) throw new Error(`Expected ${label} to be a non-negative integer.`);
    return Number(value);
}

/** Validate one strictly positive integer. */
function positiveInt(value: unknown, label: string): number {
    if (!Number.isInteger(value) || Number(value) <= 0) throw new Error(`Expected ${label} to be a positive integer.`);
    return Number(value);
}

/** Validate one numeric tuple-like array. */
function numericTuple(value: unknown, label: string): number[] {
    if (!Array.isArray(value) || value.some((entry) => typeof entry !== 'number' || Number.isNaN(entry))) {
        throw new Error(`Expected ${label} to be an array of numbers.`);
    }
    return value.map(Number);
}

/** Validate one list of basis vectors for an input dimension. */
function basisVectors(value: unknown, label: string): number[][] {
    if (!Array.isArray(value)) throw new Error(`Expected ${label} bases to be an array.`);
    return value.map((basis, bit) => {
        if (!Array.isArray(basis)) throw new Error(`Expected ${label} basis ${bit + 1} to be an array.`);
        return basis.map((entry, outAxis) => nonNegativeInt(entry, `${label} basis ${bit + 1}[${outAxis}]`));
    });
}

/** Return the first case-insensitive duplicate name, if any. */
function duplicateName(names: string[]): string | null {
    const seen = new Set<string>();
    for (const name of names) {
        const lowered = name.toLowerCase();
        if (seen.has(lowered)) return name;
        seen.add(lowered);
    }
    return null;
}

/** Map one input coordinate into output space using Triton-style xor basis composition. */
function mapLinearLayoutCoord(
    inputCoord: number[],
    input_dims: LinearLayoutInputDimSpec[],
    outputRank: number,
): number[] {
    const outputCoord = new Array(outputRank).fill(0);
    input_dims.forEach((dim, axis) => {
        const value = inputCoord[axis] ?? 0;
        dim.bases.forEach((basis, bit) => {
            if (((value >> bit) & 1) === 0) return;
            if (basis.length !== outputRank) {
                throw new Error(`Basis ${dim.name}[${bit}] has rank ${basis.length}, expected ${outputRank}.`);
            }
            basis.forEach((component, outAxis) => {
                outputCoord[outAxis] ^= component;
            });
        });
    });
    return outputCoord;
}

/** Return the viewer-visible output-dimension order used for the logical tensor. */
function logicalOutputDimsFor(output_dims: LinearLayoutOutputDimSpec[]): LinearLayoutOutputDimSpec[] {
    const names = output_dims.map(({ name }) => name.toLowerCase());
    return names.length === 2 && names[0] === 'x' && names[1] === 'y' ? [...output_dims].reverse() : output_dims;
}

/** Convert raw layout names into viewer-safe axis labels. */
function viewerAxisLabels(names: string[]): string[] {
    const counts = new Map<string, number>();
    return names.map((name) => {
        const base = [...name].find((char) => /[a-z]/i.test(char))?.toUpperCase() ?? 'A';
        const index = counts.get(base) ?? 0;
        counts.set(base, index + 1);
        return index === 0 ? base : `${base}${index}`;
    });
}

/** Return the logical output labels used by the tensor viewer. */
function logicalAxisLabels(outputNames: string[]): string[] {
    return outputNames.length === 2 && outputNames.map((name) => name.toLowerCase()).join(',') === 'y,x'
        ? ['Y', 'X']
        : viewerAxisLabels(outputNames);
}

/** Find the first axis whose name loosely matches the requested substring. */
function axisByName(names: string[], needle: string, fallback: number | null): number | null {
    const lowered = needle.toLowerCase();
    const axis = names.findIndex((name) => name.toLowerCase().includes(lowered));
    return axis === -1 ? fallback : axis;
}

/** Resolve one user-specified axis override against the input-dimension names. */
function resolveAxis(names: string[], axisName: string): number {
    const lowered = axisName.toLowerCase();
    const exact = names.findIndex((name) => name.toLowerCase() === lowered);
    if (exact !== -1) return exact;
    const partial = names
        .map((name, axis) => ({ name, axis }))
        .filter(({ name }) => name.toLowerCase().includes(lowered));
    if (partial.length === 1) return partial[0]!.axis;
    throw new Error(`Unknown layout axis ${axisName}; expected one of ${names.join(', ')}.`);
}

/** Resolve which input axes drive hue, saturation, and value in the sidebar preview. */
function normalizeColorAxes(
    inputNames: string[],
    inputShape: number[],
    color_axes: LinearLayoutColorAxes | undefined,
): Record<'H' | 'S' | 'L', number | null> {
    const defaults = {
        H: axisByName(inputNames, 'thread', inputShape.length >= 3 ? 1 : null),
        S: axisByName(inputNames, 'warp', inputShape.length >= 3 ? 0 : null),
        L: axisByName(inputNames, 'reg', inputShape.length ? inputShape.length - 1 : null),
    };
    if (!color_axes) return defaults;
    Object.entries(color_axes).forEach(([axisName, channelName]) => {
        const channel = channelName.toUpperCase();
        if (!['H', 'S', 'L'].includes(channel)) {
            throw new Error(`Unknown color channel ${channelName}; expected H, S, or L.`);
        }
        defaults[channel as 'H' | 'S' | 'L'] = resolveAxis(inputNames, axisName);
    });
    const usedAxes = Object.values(defaults).filter((axis): axis is number => axis !== null);
    if (usedAxes.length !== new Set(usedAxes).size) throw new Error('Each color axis must map to at most one channel.');
    return defaults;
}

/** Return one configured H, S, or V channel value for the current input coordinate. */
function colorValue(
    channel: 'H' | 'S' | 'L',
    coord: number[],
    shape: number[],
    axis: number | null,
    color_ranges: LinearLayoutColorRanges | undefined,
): number {
    if (axis === null) return { H: 0, S: 1, L: 0 }[channel];
    const ranges = { ...DEFAULT_COLOR_RANGES, ...color_ranges };
    const [start, stop] = ranges[channel];
    if (shape[axis] <= 1) return start;
    const position = coord[axis] / (shape[axis] - 1);
    return start + ((stop - start) * position);
}

/** Convert one HSV triple into the RGB tuple stored in custom viewer colors. */
function rgbaColor(hue: number, saturation: number, value: number): [number, number, number, number] {
    const [red, green, blue] = hsvToRgb(hue, Math.max(0, Math.min(1, saturation)), Math.max(0, Math.min(1, value)));
    return [red, green, blue, 1];
}

/** Convert one HSV triple into its RGB representation. */
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

/** Flatten one coordinate into row-major tensor storage order. */
function flatIndex(coord: number[], shape: number[]): number {
    return coord.reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}

/** Preserve interactive display settings while letting the new layout refit the camera. */
function persistedViewerSettings(viewer: Partial<ViewerSnapshot> | undefined): Partial<ViewerSnapshot> {
    if (!viewer) return {};
    return {
        displayMode: viewer.displayMode,
        heatmap: viewer.heatmap,
        dimensionBlockGapMultiple: viewer.dimensionBlockGapMultiple,
        displayGaps: viewer.displayGaps,
        logScale: viewer.logScale,
        collapseHiddenAxes: viewer.collapseHiddenAxes,
        dimensionMappingScheme: viewer.dimensionMappingScheme,
        showDimensionLines: viewer.showDimensionLines,
        showInspectorPanel: viewer.showInspectorPanel,
        showHoverDetailsPanel: viewer.showHoverDetailsPanel,
    };
}
