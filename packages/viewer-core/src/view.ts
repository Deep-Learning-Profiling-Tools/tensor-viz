import type { SliceToken, TensorViewSpec, ViewParseResult, ViewToken } from './types.js';

function axisLabel(index: number): string {
    if (index < 26) return String.fromCharCode(65 + index);
    const suffix = Math.floor((index - 26) / 26);
    return `${String.fromCharCode(65 + ((index - 26) % 26))}${suffix}`;
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
        return {
            ok: false,
            errors: [`Expected ${shape.length} axis labels, got ${axisLabelsInput.length} (received ${JSON.stringify(axisLabelsInput)}).`],
        };
    }
    const axisLabels = axisLabelsInput.map((label) => String(label).trim().toUpperCase());
    const invalidLabel = axisLabels.find((label) => !/^[A-Z][^A-Z]*$/.test(label));
    if (invalidLabel) {
        return {
            ok: false,
            errors: [`Axis labels must start with a letter and may only use non-letters after it (received ${JSON.stringify(invalidLabel)}).`],
        };
    }
    const seen = new Set<string>();
    const duplicate = axisLabels.find((label) => {
        const lower = label.toLowerCase();
        if (seen.has(lower)) return true;
        seen.add(lower);
        return false;
    });
    if (duplicate) {
        return { ok: false, errors: [`Axis labels must be unique (received duplicate ${JSON.stringify(duplicate)}).`] };
    }
    return { ok: true, axisLabels };
}

/**
 * Compute the element count for one shape after clamping degenerate extents to `1`.
 *
 * Used anywhere the viewer needs a stable linear size, such as reshapes,
 * grouped-axis flattening, and instanced-mesh allocation.
 */
export function product(values: number[]): number {
    return values.reduce((acc, value) => acc * Math.max(1, value), 1);
}

/** Normalize one shape to positive integer extents before any view or layout math runs. */
export function normalizeShape(shape: number[]): number[] {
    return shape.map((dim) => {
        const value = Number(dim);
        return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1;
    });
}

/** Build the default canonical view string by exposing every tensor axis in order. */
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

/** Expand one grouped view index back into the original per-axis tensor indices. */
export function expandGroupedIndex(axes: number[], linearIndex: number, shape: number[]): number[] {
    return unflattenAxesIndex(axes, linearIndex, normalizeShape(shape));
}

/** Map one visible view coordinate into the original dense tensor coordinate. */
export function mapViewCoordToTensorCoord(viewCoord: number[], spec: TensorViewSpec): number[] {
    const tensorCoord = spec.hiddenIndices.slice();
    let viewAxis = 0;
    spec.tokens.forEach((token) => {
        if (!token.visible) return;
        if (token.kind === 'singleton') {
            viewAxis += 1;
            return;
        }
        const value = Math.max(0, Math.min(token.size - 1, viewCoord[viewAxis] ?? 0));
        const expanded = unflattenAxesIndex(token.axes, value, spec.tensorShape);
        token.axes.forEach((axis, axisIndex) => {
            tensorCoord[axis] = expanded[axisIndex];
        });
        viewAxis += 1;
    });
    return tensorCoord;
}

function mapViewCoordToFullLayoutCoord(viewCoord: number[], spec: TensorViewSpec): number[] {
    const layoutCoord: number[] = [];
    let viewAxis = 0;
    spec.tokens.forEach((token) => {
        if (token.kind === 'singleton') {
            layoutCoord.push(0);
            if (token.visible) viewAxis += 1;
            return;
        }
        if (!token.visible) {
            const sliceToken = spec.sliceTokens.find((entry) => entry.token === token.label.toLowerCase());
            layoutCoord.push(sliceToken?.value ?? 0);
            return;
        }
        layoutCoord.push(Math.max(0, Math.min(token.size - 1, viewCoord[viewAxis] ?? 0)));
        viewAxis += 1;
    });
    return layoutCoord;
}

/** Return the rendered layout shape, optionally collapsing hidden axes out of the scene. */
export function layoutShape(spec: TensorViewSpec, collapseHiddenAxes = false): number[] {
    const shape = collapseHiddenAxes ? spec.viewShape : spec.layoutShape;
    return shape.length === 0 ? [1] : shape.slice();
}

/** Map one visible view coordinate into the layout coordinate space drawn on screen. */
export function mapViewCoordToLayoutCoord(
    viewCoord: number[],
    spec: TensorViewSpec,
    collapseHiddenAxes = false,
): number[] {
    if (collapseHiddenAxes) return spec.viewShape.length === 0 ? [] : viewCoord.slice();
    return mapViewCoordToFullLayoutCoord(viewCoord, spec);
}

/** Recover the visible view coordinate from one layout coordinate picked from the scene. */
export function mapLayoutCoordToViewCoord(layoutCoord: number[], spec: TensorViewSpec, collapseHiddenAxes = false): number[] {
    if (collapseHiddenAxes) return spec.viewShape.length === 0 ? [] : layoutCoord.slice();
    const viewCoord: number[] = [];
    spec.tokens.forEach((token, layoutAxis) => {
        if (!token.visible) return;
        if (token.kind === 'singleton') {
            viewCoord.push(0);
            return;
        }
        viewCoord.push(Math.max(0, Math.min(token.size - 1, layoutCoord[layoutAxis] ?? 0)));
    });
    return viewCoord;
}

/** returns whether a full layout coordinate belongs to the active sliced tensor. */
export function layoutCoordMatchesSlice(layoutCoord: number[], spec: TensorViewSpec): boolean {
    let sliceIndex = 0;
    for (let layoutAxis = 0; layoutAxis < spec.tokens.length; layoutAxis += 1) {
        const token = spec.tokens[layoutAxis];
        if (!token || token.kind !== 'axis_group' || token.visible) continue;
        const sliceToken = spec.sliceTokens[sliceIndex];
        if (!sliceToken || (layoutCoord[layoutAxis] ?? 0) !== sliceToken.value) return false;
        sliceIndex += 1;
    }
    return true;
}

/** Test whether one layout coordinate belongs to the slice that is currently visible. */
export function layoutCoordIsVisible(layoutCoord: number[], spec: TensorViewSpec, collapseHiddenAxes = false): boolean {
    return collapseHiddenAxes || layoutCoordMatchesSlice(layoutCoord, spec);
}

/** Return axis labels in the same order as the active rendered layout. */
export function layoutAxisLabels(spec: TensorViewSpec, collapseHiddenAxes = false): string[] {
    const tokens = collapseHiddenAxes ? spec.tokens.filter((token) => token.visible) : spec.tokens;
    return tokens.map((token) => token.label.toUpperCase());
}

/** Return whether 2D contiguous selection can use the row-major fast path. */
export function supportsContiguousSelectionFastPath2D(spec: TensorViewSpec, collapseHiddenAxes = false): boolean {
    if (collapseHiddenAxes) return false;
    const flattenedAxes = spec.tokens
        .filter((token): token is ViewToken & { kind: 'axis_group' } => token.kind === 'axis_group')
        .flatMap((token) => token.axes);
    if (flattenedAxes.length !== spec.tensorShape.length || flattenedAxes.some((axis, index) => axis !== index)) return false;

    const split = Math.floor(spec.tokens.length / 2);
    const xVisibleAxes: number[] = [];
    let sawVisibleX = false;
    for (let layoutAxis = split; layoutAxis < spec.tokens.length; layoutAxis += 1) {
        const token = spec.tokens[layoutAxis];
        if (!token || token.kind !== 'axis_group') continue;
        if (token.visible) {
            sawVisibleX = true;
            xVisibleAxes.push(...token.axes);
            continue;
        }
        if (sawVisibleX) return false;
    }
    const firstXAxis = spec.tensorShape.length - xVisibleAxes.length;
    return xVisibleAxes.every((axis, index) => axis === firstXAxis + index);
}

/** Build a readable summary of the permute, reshape, and slice implied by one view string. */
export function buildPreviewExpression(spec: TensorViewSpec): string {
    const viewAxes = spec.tokens
        .filter((token) => token.kind === 'axis_group')
        .flatMap((token) => token.axes);
    let expr = 'tensor';
    const isIdentity = viewAxes.length === spec.tensorShape.length
        && viewAxes.every((axis, index) => axis === index);
    if (!isIdentity) expr += `.permute(${viewAxes.join(', ')})`;

    const reshapeTerms = spec.tokens.map((token) => {
        if (token.kind === 'singleton') return '1';
        if (token.axes.length === 1) return `${spec.tensorShape[token.axes[0]]}`;
        return token.axes.map((axis) => `${spec.tensorShape[axis]}`).join('*');
    });
    const defaultTerms = normalizeShape(spec.tensorShape).map(String);
    if (reshapeTerms.join(',') !== defaultTerms.join(',')) {
        expr += `.reshape(${reshapeTerms.join(', ')})`;
    }

    const sliceTokens = new Map(spec.sliceTokens.map((token) => [token.token, token.value]));
    const sliceTerms = spec.tokens.map((token) => {
        if (token.kind !== 'axis_group' || token.visible) return ':';
        return String(sliceTokens.get(token.label.toLowerCase()) ?? 0);
    });
    let hasHiddenAxis = false;
    spec.tokens.forEach((token) => {
        if (token.kind !== 'axis_group' || token.visible) return;
        hasHiddenAxis = true;
    });
    if (hasHiddenAxis) expr += `[${sliceTerms.join(', ')}]`;
    return expr;
}

/** Parse one tensor-view string into the grouped visible axes and hidden slice metadata the viewer renders. */
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

    const sliceTokens: SliceToken[] = tokens
        .filter((token) => token.kind === 'axis_group' && !token.visible)
        .map((token) => ({
            token: token.label.toLowerCase(),
            axes: token.axes,
            size: token.size,
            value: flattenAxesIndex(token.axes, normalizedHiddenIndices, shape),
        }));

    return {
        ok: true,
        spec: {
            input,
            canonical: tokens.map((token) => {
                if (token.kind === 'singleton') return '1';
                return token.visible ? token.label : token.label.toLowerCase();
            }).join(' '),
            axisLabels,
            tensorShape: shape,
            tokens,
            viewAxes: tokens
                .filter((token) => token.kind === 'axis_group' && token.visible)
                .flatMap((token) => token.axes),
            sliceAxes: sliceTokens.flatMap((token) => token.axes),
            hiddenIndices: normalizedHiddenIndices,
            sliceTokens,
            viewShape: tokens.filter((token) => token.visible).map((token) => token.size),
            layoutShape: tokens.map((token) => token.size),
        },
    };
}
