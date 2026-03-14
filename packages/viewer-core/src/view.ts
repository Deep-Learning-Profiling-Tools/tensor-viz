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

export function product(values: number[]): number {
    return values.reduce((acc, value) => acc * Math.max(1, value), 1);
}

export function normalizeShape(shape: number[]): number[] {
    return shape.map((dim) => {
        const value = Number(dim);
        return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1;
    });
}

export function defaultTensorView(shape: number[]): string {
    return normalizeShape(shape).map((_dim, axis) => axisLabel(axis)).join(' ');
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

export function parseTensorView(shapeInput: number[], input: string, hiddenIndices?: number[]): ViewParseResult {
    const shape = normalizeShape(shapeInput);
    const axisLabels = shape.map((_dim, axis) => axisLabel(axis));
    const text = input.trim().replace(/\s+/g, ' ');
    const normalizedHiddenIndices = clampHiddenIndices(shape, hiddenIndices);
    if (text === '') {
        return parseTensorView(shape, defaultTensorView(shape), normalizedHiddenIndices);
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

        if (!/^[A-Za-z]+$/.test(rawToken)) {
            errors.push(`Invalid token "${rawToken}".`);
            continue;
        }

        const hasLower = /[a-z]/.test(rawToken);
        const hasUpper = /[A-Z]/.test(rawToken);
        if (hasLower && hasUpper) {
            errors.push(`Grouped token "${rawToken}" must be all upper or all lower case.`);
            continue;
        }

        const lower = rawToken.toLowerCase();
        const axes: number[] = [];
        let cursor = 0;
        while (cursor < lower.length) {
            const match = labelEntries.find((entry) => lower.startsWith(entry.lower, cursor));
            if (!match) {
                errors.push(`Token "${rawToken}" could not be parsed against tensor axes.`);
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
            visible: hasUpper,
            label: axes.map((axis) => axisLabels[axis]).join(hasUpper ? '' : '').toUpperCase(),
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
