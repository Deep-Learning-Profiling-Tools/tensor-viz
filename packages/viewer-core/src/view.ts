import type { HiddenToken, TensorViewSpec, ViewParseResult, ViewToken } from './types.js';

function axisLabel(index: number): string {
    let value = index;
    let out = '';
    do {
        out = String.fromCharCode(65 + (value % 26)) + out;
        value = Math.floor(value / 26) - 1;
    } while (value >= 0);
    return out;
}

function parseAxisLabels(shape: number[], axisLabelsInput?: readonly string[]): {
    ok: true;
    axisLabels: string[];
} | {
    ok: false;
    errors: string[];
} {
    const defaults = shape.map((_dim, axis) => axisLabel(axis));
    if (!axisLabelsInput) return { ok: true, axisLabels: defaults };
    if (axisLabelsInput.length !== shape.length) {
        return { ok: false, errors: [`Expected ${shape.length} axis labels, got ${axisLabelsInput.length}.`] };
    }
    const axisLabels = axisLabelsInput.map((label) => String(label).trim().toUpperCase());
    if (axisLabels.some((label) => !/^[A-Z][^A-Z]*$/.test(label))) {
        return { ok: false, errors: ['Axis labels must start with a letter and may only use non-letters after it.'] };
    }
    if (new Set(axisLabels.map((label) => label.toLowerCase())).size !== axisLabels.length) {
        return { ok: false, errors: ['Axis labels must be unique.'] };
    }
    return { ok: true, axisLabels };
}

export function product(values: number[]): number {
    return values.reduce((acc, value) => acc * Math.max(1, value), 1);
}

export function normalizeShape(shape: number[]): number[] {
    return shape.map((dim) => {
        const value = Number(dim);
        return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1;
    });
}

export function defaultTensorView(shape: number[], axisLabelsInput?: readonly string[]): string {
    const normalizedShape = normalizeShape(shape);
    const axisLabels = parseAxisLabels(normalizedShape, axisLabelsInput);
    if (!axisLabels.ok) throw new Error(axisLabels.errors.join(' '));
    return axisLabels.axisLabels.join(' ');
}

function clampHiddenIndices(shape: number[], hiddenIndices?: number[]): number[] {
    return normalizeShape(shape).map((dim, axis) => {
        const value = Math.round(Number(hiddenIndices?.[axis] ?? 0));
        return Math.min(dim - 1, Math.max(0, Number.isFinite(value) ? value : 0));
    });
}

function flattenAxesIndex(axes: number[], values: number[], shape: number[]): number {
    let linear = 0;
    axes.forEach((axis) => {
        linear = (linear * shape[axis]) + values[axis];
    });
    return linear;
}

function unflattenAxesIndex(axes: number[], linearIndex: number, shape: number[]): number[] {
    const out = new Array(axes.length).fill(0);
    let remaining = linearIndex;
    for (let index = axes.length - 1; index >= 0; index -= 1) {
        const axis = axes[index];
        out[index] = remaining % shape[axis];
        remaining = Math.floor(remaining / shape[axis]);
    }
    return out;
}

export function expandGroupedIndex(axes: number[], linearIndex: number, shape: number[]): number[] {
    return unflattenAxesIndex(axes, linearIndex, normalizeShape(shape));
}

export function mapDisplayCoordToFullCoord(displayCoord: number[], spec: TensorViewSpec): number[] {
    const full = spec.hiddenIndices.slice();
    let displayAxis = 0;
    spec.tokens.forEach((token) => {
        if (!token.visible) return;
        if (token.kind === 'singleton') {
            displayAxis += 1;
            return;
        }
        const size = token.size;
        const value = Math.max(0, Math.min(size - 1, displayCoord[displayAxis] ?? 0));
        const expanded = unflattenAxesIndex(token.axes, value, spec.axisShape);
        token.axes.forEach((axis, axisIndex) => {
            full[axis] = expanded[axisIndex];
        });
        displayAxis += 1;
    });
    return full;
}

export function mapDisplayCoordToOutlineCoord(displayCoord: number[], spec: TensorViewSpec): number[] {
    const outline: number[] = [];
    let displayAxis = 0;
    spec.tokens.forEach((token) => {
        if (token.kind === 'singleton') {
            outline.push(0);
            if (token.visible) displayAxis += 1;
            return;
        }
        if (!token.visible) {
            const hidden = spec.hiddenTokens.find((entry) => entry.token === token.label.toLowerCase());
            outline.push(hidden?.value ?? 0);
            return;
        }
        outline.push(Math.max(0, Math.min(token.size - 1, displayCoord[displayAxis] ?? 0)));
        displayAxis += 1;
    });
    return outline;
}

export function viewedShape(spec: TensorViewSpec, showSlicesInSamePlace = false): number[] {
    const shape = showSlicesInSamePlace ? spec.displayShape : spec.outlineShape;
    return shape.length === 0 ? [1] : shape.slice();
}

export function mapDisplayCoordToViewedCoord(
    displayCoord: number[],
    spec: TensorViewSpec,
    showSlicesInSamePlace = false,
): number[] {
    if (!showSlicesInSamePlace) return mapDisplayCoordToOutlineCoord(displayCoord, spec);
    return spec.displayShape.length === 0 ? [] : displayCoord.slice();
}

export function mapOutlineCoordToDisplayCoord(outlineCoord: number[], spec: TensorViewSpec): number[] {
    const display: number[] = [];
    spec.tokens.forEach((token, outlineAxis) => {
        if (!token.visible) return;
        if (token.kind === 'singleton') {
            display.push(0);
            return;
        }
        display.push(Math.max(0, Math.min(token.size - 1, outlineCoord[outlineAxis] ?? 0)));
    });
    return display;
}

export function mapViewedCoordToDisplayCoord(
    viewedCoord: number[],
    spec: TensorViewSpec,
    showSlicesInSamePlace = false,
): number[] {
    if (!showSlicesInSamePlace) return mapOutlineCoordToDisplayCoord(viewedCoord, spec);
    return spec.displayShape.length === 0 ? [] : viewedCoord.slice();
}

/** returns whether an outline coordinate belongs to the currently visible hidden-token slice. */
export function outlineCoordIsVisible(outlineCoord: number[], spec: TensorViewSpec): boolean {
    let hiddenIndex = 0;
    for (let outlineAxis = 0; outlineAxis < spec.tokens.length; outlineAxis += 1) {
        const token = spec.tokens[outlineAxis];
        if (!token || token.kind !== 'axis_group' || token.visible) continue;
        const hidden = spec.hiddenTokens[hiddenIndex];
        if (!hidden || (outlineCoord[outlineAxis] ?? 0) !== hidden.value) return false;
        hiddenIndex += 1;
    }
    return true;
}

export function viewedCoordIsVisible(
    viewedCoord: number[],
    spec: TensorViewSpec,
    showSlicesInSamePlace = false,
): boolean {
    return showSlicesInSamePlace || outlineCoordIsVisible(viewedCoord, spec);
}

export function viewedTokenLabels(spec: TensorViewSpec, showSlicesInSamePlace = false): string[] {
    const tokens = showSlicesInSamePlace ? spec.tokens.filter((token) => token.visible) : spec.tokens;
    return tokens.map((token) => token.label.toUpperCase());
}

export function buildPreviewExpression(spec: TensorViewSpec): string {
    const visibleAxes = spec.tokens
        .filter((token) => token.kind === 'axis_group')
        .flatMap((token) => token.axes);
    let expr = 'tensor';
    const isIdentity = visibleAxes.length === spec.axisShape.length
        && visibleAxes.every((axis, index) => axis === index);
    if (!isIdentity) expr += `.permute(${visibleAxes.join(', ')})`;

    const reshapeTerms = spec.tokens.map((token) => {
        if (token.kind === 'singleton') return '1';
        if (token.axes.length === 1) return `${spec.axisShape[token.axes[0]]}`;
        return token.axes.map((axis) => `${spec.axisShape[axis]}`).join('*');
    });
    const defaultTerms = normalizeShape(spec.axisShape).map(String);
    if (reshapeTerms.join(',') !== defaultTerms.join(',')) {
        expr += `.reshape(${reshapeTerms.join(', ')})`;
    }

    const sliceTerms = spec.tokens.flatMap((token) => {
        if (token.visible) return [];
        const hidden = spec.hiddenTokens.find((entry) => entry.token === token.label.toLowerCase());
        return [String(hidden?.value ?? 0)];
    });
    if (sliceTerms.length > 0) expr += `[${sliceTerms.join(', ')}]`;
    return expr;
}

export function parseTensorView(
    shapeInput: number[],
    input: string,
    hiddenIndices?: number[],
    axisLabelsInput?: readonly string[],
): ViewParseResult {
    const shape = normalizeShape(shapeInput);
    const axisLabelsResult = parseAxisLabels(shape, axisLabelsInput);
    if (!axisLabelsResult.ok) return axisLabelsResult;
    const axisLabels = axisLabelsResult.axisLabels;
    const text = input.trim().replace(/\s+/g, ' ');
    const normalizedHiddenIndices = clampHiddenIndices(shape, hiddenIndices);
    if (text === '') {
        return parseTensorView(shape, defaultTensorView(shape, axisLabels), normalizedHiddenIndices, axisLabels);
    }

    const labelEntries = axisLabels
        .map((label, axis) => ({ axis, label, lower: label.toLowerCase() }))
        .sort((left, right) => right.label.length - left.label.length);
    const seenAxes = new Set<number>();
    const tokens: ViewToken[] = [];
    const errors: string[] = [];

    for (const rawToken of text.split(' ')) {
        if (rawToken === '1') {
            tokens.push({ kind: 'singleton', visible: true, label: '1', axes: [], size: 1 });
            continue;
        }

        const lower = rawToken.toLowerCase();
        const axes: number[] = [];
        let visible: boolean | null = null;
        let cursor = 0;
        while (cursor < lower.length) {
            const match = labelEntries.find((entry) => lower.startsWith(entry.lower, cursor));
            if (!match) {
                errors.push(`Token "${rawToken}" could not be parsed against tensor axes.`);
                break;
            }
            const segment = rawToken.slice(cursor, cursor + match.lower.length);
            const segmentVisible = segment === match.label;
            const segmentHidden = segment === match.lower;
            if (!segmentVisible && !segmentHidden) {
                errors.push(`Token "${rawToken}" could not be parsed against tensor axes.`);
                break;
            }
            if (visible === null) visible = segmentVisible;
            else if (visible !== segmentVisible) {
                errors.push(`Grouped token "${rawToken}" must be all upper or all lower case.`);
                break;
            }
            if (seenAxes.has(match.axis)) {
                errors.push(`Axis ${match.label} appears more than once.`);
                break;
            }
            seenAxes.add(match.axis);
            axes.push(match.axis);
            cursor += match.lower.length;
        }

        if (axes.length === 0) continue;
        if (axes.some((axis) => !Number.isInteger(axis))) continue;
        tokens.push({
            kind: 'axis_group',
            visible: visible ?? true,
            label: axes.map((axis) => axisLabels[axis]).join(''),
            axes,
            size: product(axes.map((axis) => shape[axis])),
        });
    }

    axisLabels.forEach((label, axis) => {
        if (!seenAxes.has(axis)) errors.push(`Axis ${label} is missing from the view string.`);
    });
    if (errors.length > 0) return { ok: false, errors };

    const hiddenTokens: HiddenToken[] = tokens
        .filter((token) => token.kind === 'axis_group' && !token.visible)
        .map((token) => ({
            token: token.label.toLowerCase(),
            axes: token.axes,
            size: token.size,
            value: flattenAxesIndex(token.axes, normalizedHiddenIndices, shape),
        }));
    const visibleAxisTokens = tokens.filter((token) => token.kind === 'axis_group' && token.visible);
    const displayShape = tokens
        .filter((token) => token.visible)
        .map((token) => token.size);

    return {
        ok: true,
        spec: {
            input,
            canonical: tokens.map((token) => {
                if (token.kind === 'singleton') return '1';
                return token.visible ? token.label : token.label.toLowerCase();
            }).join(' '),
            axisLabels,
            axisShape: shape,
            tokens,
            visibleAxes: visibleAxisTokens.flatMap((token) => token.axes),
            hiddenAxes: hiddenTokens.flatMap((token) => token.axes),
            hiddenIndices: normalizedHiddenIndices,
            hiddenTokens,
            displayShape,
            outlineShape: tokens.map((token) => token.size),
        },
    };
}
