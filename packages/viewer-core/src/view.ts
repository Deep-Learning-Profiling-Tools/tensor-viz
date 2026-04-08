import type {
    SliceToken,
    TensorViewEditor,
    TensorViewEditorDim,
    TensorViewSpec,
    ViewParseResult,
    ViewToken,
} from './types.js';

const TENSOR_VIEW_EDITOR_PREFIX = 'tv2:';

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
    const axisLabels = axisLabelsInput.map((label) => String(label).trim());
    const invalidLabel = axisLabels.find((label) => label === '' || /[\s,\[\]=]/.test(label));
    if (invalidLabel) {
        return {
            ok: false,
            errors: [`Axis labels may not be empty or contain whitespace, commas, brackets, or "=" (received ${JSON.stringify(invalidLabel)}).`],
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

function unflattenLinearIndex(linearIndex: number, shape: number[]): number[] {
    return unflattenAxesIndex(Array.from({ length: shape.length }, (_entry, index) => index), linearIndex, shape);
}

function defaultEditorDims(shape: number[], axisLabels: string[]): TensorViewEditorDim[] {
    return shape.map((size, axis) => ({
        id: `axis-${axis}`,
        label: axisLabels[axis]!,
        size,
    }));
}

function formatViewTensorInput(baseDims: TensorViewEditorDim[]): string {
    return `[${baseDims.map((dim) => `${dim.label}=${dim.size}`).join(', ')}]`;
}

function formatFinalViewInputFromTokens(tokens: ViewToken[]): string {
    return `[${tokens.map((token) => token.kind === 'singleton' ? '1' : `${token.label}=${token.size}`).join(', ')}]`;
}

function parseViewLabelToken(part: string): { ok: true; label: string; sizeText?: string } | { ok: false } {
    const anonymous = part.match(/^((?:\*A|\*|_)\d+)(?:\s*=\s*(-?\d+))?$/);
    if (anonymous) return { ok: true, label: anonymous[1]!, sizeText: anonymous[2] };
    const explicit = part.match(/^([^=,\[\]]+?)(?:\s*=\s*(-?\d+))?$/);
    if (!explicit) return { ok: false };
    return { ok: true, label: explicit[1]!.trim(), sizeText: explicit[2] };
}

function parseExplicitViewInput(
    input: string,
    totalElements: number,
): { ok: true; dims: Array<{ label: string; size: number }> } | { ok: false; errors: string[] } {
    const text = input.trim();
    const inner = text.replace(/^\[/, '').replace(/\]$/, '').trim();
    if (inner === '') return { ok: true, dims: [] };
    const parts = inner.split(',').map((part) => part.trim()).filter(Boolean);
    const errors: string[] = [];
    const seen = new Set<string>();
    let inferredIndex = -1;
    let anonymousIndex = parts.reduce((maxIndex, part) => {
        const match = part.match(/^(?:\*A|\*|_)(\d+)(?:\s*=.*)?$/);
        return match ? Math.max(maxIndex, Number(match[1]) + 1) : maxIndex;
    }, 0);
    const dims = parts.map((part, index) => {
        if (/^-?\d+$/.test(part)) {
            const size = Number(part);
            if (size === -1) {
                if (inferredIndex >= 0) {
                    errors.push('Tensor View may contain at most one inferred -1 dimension.');
                    return null;
                }
                inferredIndex = index;
            }
            return { label: `*A${anonymousIndex++}`, size };
        }
        const match = parseViewLabelToken(part);
        if (!match.ok) {
            errors.push(`Tensor View token "${part}" is invalid.`);
            return null;
        }
        const label = match.label;
        if (seen.has(label)) {
            errors.push(`Tensor View label "${label}" appears more than once.`);
            return null;
        }
        seen.add(label);
        const explicitSize = match.sizeText;
        const size = explicitSize ? Number(explicitSize) : -1;
        if (size === -1) {
            if (inferredIndex >= 0) {
                errors.push('Tensor View may contain at most one inferred -1 dimension.');
                return null;
            }
            inferredIndex = index;
        }
        return { label, size };
    }).filter((dim): dim is { label: string; size: number } => dim !== null);
    if (inferredIndex >= 0) {
        const knownProduct = product(dims.filter((_dim, index) => index !== inferredIndex).map((dim) => dim.size));
        if (knownProduct === 0 || totalElements % knownProduct !== 0) {
            errors.push('Tensor View could not infer a valid size for -1.');
        } else {
            dims[inferredIndex]!.size = totalElements / knownProduct;
        }
    }
    if (product(dims.map((dim) => dim.size)) !== totalElements) {
        errors.push(`Tensor View shape product must equal ${totalElements}.`);
    }
    if (errors.length > 0) return { ok: false, errors };
    return { ok: true, dims };
}

function editorTokenKey(dimIds: string[]): string {
    return `group:${dimIds.join('+')}`;
}

function editorSingletonKey(singletonId: string): string {
    return `singleton:${singletonId}`;
}

function sameBaseAsTensor(baseDims: TensorViewEditorDim[], shape: number[], axisLabels: string[]): boolean {
    return baseDims.length === shape.length && baseDims.every((dim, axis) => (
        dim.size === shape[axis]
        && dim.label === axisLabels[axis]
        && dim.id === `axis-${axis}`
    ));
}

function buildEditorSpec(
    tensorShape: number[],
    axisLabels: string[],
    editor: TensorViewEditor,
): TensorViewSpec {
    const baseDims = editor.baseDims.slice();
    const baseShape = baseDims.map((dim) => dim.size);
    const baseIndexById = new Map(baseDims.map((dim, index) => [dim.id, index]));
    const permutedBaseIndices = editor.permutedDimIds
        .map((dimId) => baseIndexById.get(dimId) ?? -1)
        .filter((axis) => axis >= 0);
    const permutedBaseShape = permutedBaseIndices.map((axis) => baseShape[axis]!);
    const explicitFinalView = editor.finalViewInput?.trim();
    let tokens: ViewToken[] = [];
    if (explicitFinalView) {
        const parsedFinal = parseExplicitViewInput(explicitFinalView, product(permutedBaseShape));
        if (!parsedFinal.ok) throw new Error(parsedFinal.errors.join(' '));
        tokens = parsedFinal.dims.map((dim, index) => ({
            kind: 'axis_group',
            key: `view:${dim.label}`,
            visible: true,
            label: dim.label,
            axes: [],
            size: dim.size,
        }));
    } else {
        const groups: ViewToken[] = [];
        let currentGroup: TensorViewEditorDim[] = [];
        editor.permutedDimIds.forEach((dimId, index) => {
            const dim = baseDims[baseIndexById.get(dimId) ?? -1];
            if (!dim) return;
            currentGroup.push(dim);
            const split = editor.flattenSeparators[index] ?? true;
            if (!split) return;
            const axes = currentGroup.map((groupDim) => baseIndexById.get(groupDim.id) ?? -1).filter((axis) => axis >= 0);
            groups.push({
                kind: 'axis_group',
                key: editorTokenKey(currentGroup.map((groupDim) => groupDim.id)),
                visible: true,
                label: currentGroup.map((groupDim) => groupDim.label).join(''),
                axes,
                size: product(currentGroup.map((groupDim) => groupDim.size)),
            });
            currentGroup = [];
        });
        if (currentGroup.length > 0) {
            const axes = currentGroup.map((groupDim) => baseIndexById.get(groupDim.id) ?? -1).filter((axis) => axis >= 0);
            groups.push({
                kind: 'axis_group',
                key: editorTokenKey(currentGroup.map((groupDim) => groupDim.id)),
                visible: true,
                label: currentGroup.map((groupDim) => groupDim.label).join(''),
                axes,
                size: product(currentGroup.map((groupDim) => groupDim.size)),
            });
        }
        tokens = groups.slice();
        editor.singletons
            .slice()
            .sort((left, right) => left.position - right.position || left.id.localeCompare(right.id))
            .forEach((singleton) => {
                tokens.splice(Math.max(0, Math.min(tokens.length, singleton.position)), 0, {
                    kind: 'singleton',
                    key: editorSingletonKey(singleton.id),
                    visible: true,
                    label: '1',
                    axes: [],
                    size: 1,
                });
            });
    }

    const hiddenIndices = new Array(baseDims.length).fill(0);
    const slicedSet = new Set(editor.slicedTokenKeys);
    const sliceTokens: SliceToken[] = tokens
        .filter((token) => slicedSet.has(token.key))
        .map((token) => {
            const value = Math.max(0, Math.min(token.size - 1, Math.floor(editor.sliceValues[token.key] ?? 0)));
            if (!explicitFinalView && token.kind === 'axis_group') {
                const expanded = expandGroupedIndex(token.axes, value, baseShape);
                token.axes.forEach((axis, axisIndex) => {
                    hiddenIndices[axis] = expanded[axisIndex] ?? 0;
                });
            }
            return {
                token: explicitFinalView ? token.label : token.kind === 'singleton' ? '1' : token.label.toLowerCase(),
                key: token.key,
                axes: token.axes,
                size: token.size,
                value,
            };
        });
    tokens.forEach((token) => {
        token.visible = !slicedSet.has(token.key);
    });

    return {
        input: editor.viewTensorInput,
        canonical: serializeTensorViewEditor(editor),
        axisLabels,
        tensorShape,
        baseDims,
        baseShape,
        permutedBaseShape,
        permutedBaseIndices,
        baseIsTensorAxes: sameBaseAsTensor(baseDims, tensorShape, axisLabels),
        tokens,
        viewAxes: tokens.filter((token) => token.kind === 'axis_group' && token.visible).flatMap((token) => token.axes),
        sliceAxes: sliceTokens.flatMap((token) => token.axes),
        hiddenIndices,
        sliceTokens,
        viewShape: tokens.filter((token) => token.visible).map((token) => token.size),
        layoutShape: tokens.map((token) => token.size),
        editor,
    };
}

function parseViewTensorInput(
    tensorShape: number[],
    axisLabels: string[],
    input: string,
): { ok: true; baseDims: TensorViewEditorDim[]; canonicalInput: string } | { ok: false; errors: string[] } {
    const totalElements = product(tensorShape);
    const text = input.trim();
    const inner = text.replace(/^\[/, '').replace(/\]$/, '').trim();
    if (inner === '') {
        const baseDims = defaultEditorDims(tensorShape, axisLabels);
        return { ok: true, baseDims, canonicalInput: formatViewTensorInput(baseDims) };
    }
    const originalByLabel = new Map(axisLabels.map((label, axis) => [label, tensorShape[axis]!]));
    const parts = inner.split(',').map((part) => part.trim()).filter(Boolean);
    const errors: string[] = [];
    const seen = new Set<string>();
    let inferredIndex = -1;
    let anonymousIndex = parts.reduce((maxIndex, part) => {
        const match = part.match(/^(?:\*A|\*|_)(\d+)(?:\s*=.*)?$/);
        return match ? Math.max(maxIndex, Number(match[1]) + 1) : maxIndex;
    }, 0);
    const baseDims = parts.map((part, index) => {
        if (/^-?\d+$/.test(part)) {
            const size = Number(part);
            if (size === -1) {
                if (inferredIndex >= 0) {
                    errors.push('View Tensor may contain at most one inferred -1 dimension.');
                    return null;
                }
                inferredIndex = index;
            }
            return { id: `anon-${index}`, label: `*A${anonymousIndex++}`, size };
        }
        const match = parseViewLabelToken(part);
        if (!match.ok) {
            errors.push(`View Tensor token "${part}" is invalid.`);
            return null;
        }
        const label = match.label;
        const explicitSize = match.sizeText;
        const size = explicitSize ? Number(explicitSize) : originalByLabel.get(label);
        if (!size) {
            errors.push(`View Tensor token "${part}" needs a numeric size or an existing axis label.`);
            return null;
        }
        if (size === -1) {
            if (inferredIndex >= 0) {
                errors.push('View Tensor may contain at most one inferred -1 dimension.');
                return null;
            }
            inferredIndex = index;
        }
        return { id: `dim-${index}`, label, size };
    }).filter((dim): dim is TensorViewEditorDim => dim !== null);
    if (inferredIndex >= 0) {
        const knownProduct = product(baseDims.filter((_dim, index) => index !== inferredIndex).map((dim) => dim.size));
        if (knownProduct === 0 || totalElements % knownProduct !== 0) {
            errors.push('View Tensor could not infer a valid size for -1.');
        } else {
            baseDims[inferredIndex]!.size = totalElements / knownProduct;
        }
    }
    baseDims.forEach((dim) => {
        if (seen.has(dim.label)) errors.push(`View Tensor label "${dim.label}" appears more than once.`);
        seen.add(dim.label);
    });
    if (product(baseDims.map((dim) => dim.size)) !== totalElements) {
        errors.push(`View Tensor shape product must equal ${totalElements}.`);
    }
    if (errors.length > 0) return { ok: false, errors };
    return { ok: true, baseDims, canonicalInput: formatViewTensorInput(baseDims) };
}

function normalizeEditor(
    tensorShape: number[],
    axisLabels: string[],
    editor: TensorViewEditor,
): { ok: true; editor: TensorViewEditor } | { ok: false; errors: string[] } {
    const parsed = parseViewTensorInput(tensorShape, axisLabels, editor.viewTensorInput);
    if (!parsed.ok) return parsed;
    const viewTensorChanged = editor.viewTensorInput.trim() !== formatViewTensorInput(editor.baseDims);
    const baseDims = parsed.baseDims.map((dim, index) => {
        const previous = editor.baseDims[index];
        // non-view-tensor edits still round-trip through normalizeEditor(), so
        // once the input has been canonicalized to [2, 3, 4] we must keep the
        // existing labels/ids for unchanged positional sizes; otherwise a later
        // permute/flatten/slice click would re-infer anon dims and drop state
        if (!previous || previous.size !== dim.size) return dim;
        return viewTensorChanged ? { ...dim, id: previous.id } : { ...previous };
    });
    const baseDimIds = new Set(baseDims.map((dim) => dim.id));
    const permutedDimIds = editor.permutedDimIds.filter((dimId) => baseDimIds.has(dimId));
    const droppedPermutedDimIds = editor.permutedDimIds.filter((dimId) => !baseDimIds.has(dimId));
    const baseDimIdChanged = baseDims.some((dim, index) => dim.id !== parsed.baseDims[index]?.id);
    baseDims.forEach((dim) => {
        if (!permutedDimIds.includes(dim.id)) permutedDimIds.push(dim.id);
    });
    const flattenSeparators = Array.from({ length: Math.max(0, permutedDimIds.length - 1) }, (_entry, index) => (
        editor.flattenSeparators[index] ?? true
    ));
    const maxPosition = baseDims.length + editor.singletons.length + 1;
    const singletons = editor.singletons
        .map((singleton, index) => ({
            id: singleton.id || `singleton-${index}`,
            position: Math.max(0, Math.min(maxPosition, Math.floor(singleton.position))),
        }))
        .sort((left, right) => left.position - right.position || left.id.localeCompare(right.id));
    return {
        ok: true,
        editor: {
            version: 2,
            viewTensorInput: parsed.canonicalInput,
            finalViewInput: editor.finalViewInput,
            baseDims,
            permutedDimIds,
            flattenSeparators,
            singletons,
            slicedTokenKeys: editor.slicedTokenKeys.slice(),
            sliceValues: { ...editor.sliceValues },
        },
    };
}

function defaultEditor(shape: number[], axisLabels: string[]): TensorViewEditor {
    const baseDims = defaultEditorDims(shape, axisLabels);
    return {
        version: 2,
        viewTensorInput: formatViewTensorInput(baseDims),
        baseDims,
        permutedDimIds: baseDims.map((dim) => dim.id),
        flattenSeparators: new Array(Math.max(0, baseDims.length - 1)).fill(true),
        singletons: [],
        slicedTokenKeys: [],
        sliceValues: {},
    };
}

/** Build the default structured tensor-view editor for one tensor shape. */
export function defaultTensorViewEditor(shape: number[], axisLabelsInput?: readonly string[]): TensorViewEditor {
    const normalizedShape = normalizeShape(shape);
    const axisLabels = parseAxisLabels(normalizedShape, axisLabelsInput);
    if (!axisLabels.ok) throw new Error(axisLabels.errors.join(' '));
    return defaultEditor(normalizedShape, axisLabels.axisLabels);
}

export function serializeTensorViewEditor(editor: TensorViewEditor): string {
    return `${TENSOR_VIEW_EDITOR_PREFIX}${encodeURIComponent(JSON.stringify(editor))}`;
}

/** Map one visible view coordinate into the original dense tensor coordinate. */
export function mapViewCoordToTensorCoord(viewCoord: number[], spec: TensorViewSpec): number[] {
    if (spec.editor.finalViewInput?.trim()) {
        const layoutCoord = mapViewCoordToLayoutCoord(viewCoord, spec);
        const linearIndex = flattenAxesIndex(
            Array.from({ length: spec.layoutShape.length }, (_entry, index) => index),
            layoutCoord,
            spec.layoutShape,
        );
        const permutedCoord = unflattenLinearIndex(linearIndex, spec.permutedBaseShape);
        const baseCoord = new Array(spec.baseShape.length).fill(0);
        spec.permutedBaseIndices.forEach((axis, index) => {
            baseCoord[axis] = permutedCoord[index] ?? 0;
        });
        const baseLinear = flattenAxesIndex(
            Array.from({ length: spec.baseShape.length }, (_entry, index) => index),
            baseCoord,
            spec.baseShape,
        );
        return unflattenLinearIndex(baseLinear, spec.tensorShape);
    }
    const tensorCoord = spec.hiddenIndices.slice();
    let viewAxis = 0;
    spec.tokens.forEach((token) => {
        if (!token.visible) return;
        if (token.kind === 'singleton') {
            viewAxis += 1;
            return;
        }
        const value = Math.max(0, Math.min(token.size - 1, viewCoord[viewAxis] ?? 0));
        const expanded = unflattenAxesIndex(token.axes, value, spec.baseShape);
        token.axes.forEach((axis, axisIndex) => {
            tensorCoord[axis] = expanded[axisIndex];
        });
        viewAxis += 1;
    });
    const linearIndex = flattenAxesIndex(
        Array.from({ length: spec.baseShape.length }, (_entry, index) => index),
        tensorCoord,
        spec.baseShape,
    );
    return unflattenLinearIndex(linearIndex, spec.tensorShape);
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
            const sliceToken = spec.sliceTokens.find((entry) => entry.key === token.key);
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
        if (!token || token.visible) continue;
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
    return tokens.map((token) => token.label);
}

/** Return whether 2D contiguous selection can use the row-major fast path. */
export function supportsContiguousSelectionFastPath2D(spec: TensorViewSpec, collapseHiddenAxes = false): boolean {
    if (collapseHiddenAxes) return false;
    if (!spec.baseIsTensorAxes) return false;
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

export function buildTensorViewExpression(spec: TensorViewSpec): string {
    const viewInput = spec.editor.viewTensorInput.trim().replace(/^\[/, '').replace(/\]$/, '');
    const finalViewInput = spec.editor.finalViewInput?.trim().replace(/^\[/, '').replace(/\]$/, '') ?? '';
    let expr = `tensor.view(${viewInput || spec.baseShape.join(', ')})`;
    const permuteAxes = spec.permutedBaseIndices.length === 0
        ? spec.viewAxes
        : spec.permutedBaseIndices;
    const isIdentity = permuteAxes.length === spec.baseShape.length
        && permuteAxes.every((axis, index) => axis === index);
    if (!isIdentity) expr += `.permute(${permuteAxes.join(', ')})`;
    if (finalViewInput !== '') expr += `.view(${finalViewInput})`;
    const sliceTerms = spec.tokens.map((token) => (token.visible ? ':' : String(spec.sliceTokens.find((entry) => entry.key === token.key)?.value ?? 0)));
    if (spec.tokens.some((token) => !token.visible)) expr += `[${sliceTerms.join(', ')}]`;
    return expr;
}

/** Parse one tensor-view string into the grouped visible axes and hidden slice metadata the viewer renders. */
export function parseTensorView(
    shapeInput: number[],
    input: string,
    _hiddenIndices?: number[],
    axisLabelsInput?: readonly string[],
): ViewParseResult {
    const shape = normalizeShape(shapeInput);
    const axisLabelsResult = parseAxisLabels(shape, axisLabelsInput);
    if (!axisLabelsResult.ok) return axisLabelsResult;
    const axisLabels = axisLabelsResult.axisLabels;
    if (input.trim() === '') {
        return { ok: true, spec: buildEditorSpec(shape, axisLabels, defaultEditor(shape, axisLabels)) };
    }
    if (input.startsWith(TENSOR_VIEW_EDITOR_PREFIX)) {
        try {
            const editor = JSON.parse(decodeURIComponent(input.slice(TENSOR_VIEW_EDITOR_PREFIX.length))) as TensorViewEditor;
            const normalized = normalizeEditor(shape, axisLabels, editor);
            if (!normalized.ok) return normalized;
            return { ok: true, spec: buildEditorSpec(shape, axisLabels, normalized.editor) };
        } catch {
            return { ok: false, errors: ['Tensor view editor state is invalid.'] };
        }
    }
    return { ok: false, errors: ['Tensor view state must be serialized editor data.'] };
}
