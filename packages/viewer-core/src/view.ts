import type {
    SliceToken,
    TensorViewEditor,
    TensorViewEditorDim,
    TensorViewSpec,
    ViewParseResult,
    ViewToken,
} from './types.js';
import { unravelIndex } from './layout.js';
import { normalizeTensorViewEditor, VIEWER_LIMITS } from './validation.js';

const TENSOR_VIEW_EDITOR_PREFIX = 'tv2:';

/**
 * Generates the deterministic fallback label used for tensor axes that do not provide explicit names.
 *
 * @param index - Zero-based tensor axis index from a tensor shape.
 * @returns The stable axis label for saved tensor-view expressions: `A` through `Z`, then `A0`, `B0`, and so on.
 * @noThrows The function reads no external state and only performs numeric arithmetic plus string construction for the supplied index.
 * @example
 * axisLabel(0);
 * // => 'A'
 * axisLabel(25);
 * // => 'Z'
 * axisLabel(26);
 * // => 'A0'
 */
function axisLabel(index: number): string {
    // labels must stay deterministic because saved tensor-view strings refer to
    // them when a session is restored without explicit axis names.
    if (index < 26) return String.fromCharCode(65 + index);
    const suffix = Math.floor((index - 26) / 26);
    return `${String.fromCharCode(65 + ((index - 26) % 26))}${suffix}`;
}

/**
 * Builds the label list for a tensor shape and validates caller-provided axis names.
 *
 * When no labels are supplied, each axis receives the viewer's default axis label.
 * Supplied labels must match the shape rank, trim to non-empty names, avoid view-expression
 * separators, and be unique case-insensitively.
 *
 * @param shape - Normalized tensor extents; the number of entries defines the required label count.
 * @param axisLabelsInput - Optional labels from tensor metadata or viewer setup for each tensor axis.
 * @returns A success result with labels ready for tensor-view parsing, or an error result containing the validation messages to show to the caller.
 * @noThrows Label validation failures are reported through the `{ ok: false, errors }` result instead of throwing exceptions.
 * @example
 * parseAxisLabels([2, 3]);
 * // => { ok: true, axisLabels: ['A', 'B'] }
 *
 * parseAxisLabels([2, 3], ['row', ' row ']);
 * // => { ok: false, errors: ['Axis labels must be unique (received duplicate "row").'] }
 */
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
 *
 * @param values - Shape extents or grouped-axis sizes whose zero or negative entries should count as one element.
 * @returns The multiplicative element count used for linear indexing, reshape validation, or mesh instance allocation.
 * @noThrows The calculation only reduces the provided numeric array with `Math.max`; invalid shape semantics are handled by callers before or after this helper.
 * @example
 * product([2, 0, 4]);
 * // => 8
 */
export function product(values: number[]): number {
    return values.reduce((acc, value) => acc * Math.max(1, value), 1);
}

/**
 * Normalize one shape to positive integer extents before any view or layout math runs.
 *
 * @param shape - Raw tensor extents from caller metadata, which may include fractional, zero, negative, or non-finite values.
 * @returns A shape with every finite positive extent floored to an integer and every invalid or degenerate extent replaced with `1`.
 * @noThrows Dimension coercion is local and tolerant: values that cannot become positive finite extents are normalized to `1` rather than rejected.
 * @example
 * normalizeShape([2.9, 0, -4, Number.NaN, 5]);
 * // => [2, 1, 1, 1, 5]
 */
export function normalizeShape(shape: number[]): number[] {
    return shape.map((dim) => {
        const value = Number(dim);
        return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1;
    });
}

/**
 * Converts coordinates on selected tensor axes into their row-major linear offset.
 *
 * The axis order controls the stride calculation, so grouped axes can be flattened
 * consistently before being mapped between layout, view, and tensor coordinates.
 *
 * @param axes - Axis indexes to flatten, in most-significant to least-significant stride order.
 * @param values - Full coordinate vector for the tensor or layout; entries for `axes` are read as coordinate positions.
 * @param shape - Normalized extents for the coordinate space that owns the selected axes.
 * @returns The zero-based row-major linear index represented by the selected axis coordinates.
 * @noThrows View parsing and layout code supply valid axis indexes and in-range coordinates; this helper performs only arithmetic on those validated arrays.
 * @example
 * flattenAxesIndex([0, 1, 2], [1, 2, 3], [2, 3, 4]);
 * // => 23
 *
 * flattenAxesIndex([1, 2], [0, 2, 3], [2, 3, 4]);
 * // => 11
 */
function flattenAxesIndex(axes: number[], values: number[], shape: number[]): number {
    let linear = 0;
    axes.forEach((axis) => {
        linear = (linear * shape[axis]) + values[axis];
    });
    return linear;
}

/**
 * Converts a row-major flat offset within a selected axis group into coordinates for those axes.
 *
 * @param axes - Tensor axis numbers included in the grouped dimension, in the same order used when the group was flattened.
 * @param linearIndex - Zero-based offset inside the Cartesian product of the selected axes.
 * @param shape - Full tensor shape; each axis listed in `axes` must have a positive extent at `shape[axis]`.
 * @returns Coordinates for the selected axes, ordered to match `axes`, for writing back into a full tensor coordinate.
 * @noThrows Performs only array allocation and arithmetic for caller-normalized, in-range axis and shape data; it does not validate bounds itself.
 * @example
 * const coords = unflattenAxesIndex([1, 2], 5, [2, 3, 4]);
 * // Axis 1 has size 3 and axis 2 has size 4, so offset 5 maps to [1, 1].
 * expect(coords).toEqual([1, 1]);
 */
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

/**
 * Expands a grouped view-axis coordinate into the original tensor coordinates for that group.
 *
 * @param axes - Original tensor axes represented by the grouped view dimension, in flattening order.
 * @param linearIndex - Zero-based coordinate selected on the grouped view axis.
 * @param shape - Tensor shape used to determine each grouped axis extent before expansion.
 * @returns Per-axis tensor coordinates ordered to match `axes`, suitable for copying into hidden or visible tensor-coordinate arrays.
 * @noThrows Delegates to deterministic shape normalization and index expansion; callers provide parser-validated axes and extents.
 * @example
 * const expanded = expandGroupedIndex([0, 2], 7, [2, 3, 4]);
 * // The grouped axis spans shape[0] * shape[2] = 8 positions; offset 7 is axis 0 = 1, axis 2 = 3.
 * expect(expanded).toEqual([1, 3]);
 */
export function expandGroupedIndex(axes: number[], linearIndex: number, shape: number[]): number[] {
    return unflattenAxesIndex(axes, linearIndex, normalizeShape(shape));
}

/**
 * Converts a row-major flat tensor offset into one coordinate per tensor axis.
 *
 * @param linearIndex - Zero-based offset into a tensor with the provided shape.
 * @param shape - Tensor dimension extents in axis order.
 * @returns Full tensor coordinate where each entry is the index for the corresponding axis in `shape`.
 * @noThrows Performs deterministic arithmetic over the supplied shape and does not contain validation branches that throw.
 * @example
 * const coord = unflattenLinearIndex(17, [2, 3, 4]);
 * // 17 / (3 * 4) = axis 0 coordinate 1, with remainder mapping to [1, 1].
 * expect(coord).toEqual([1, 1, 1]);
 */
function unflattenLinearIndex(linearIndex: number, shape: number[]): number[] {
    return unflattenAxesIndex(Array.from({ length: shape.length }, (_entry, index) => index), linearIndex, shape);
}

/**
 * Builds the default editable dimension list for a tensor view, one entry for each tensor axis.
 *
 * @param shape - Tensor dimension extents in axis order.
 * @param axisLabels - Display labels parallel to `shape`; `axisLabels[axis]` becomes the editor label for that axis.
 * @returns Editor dimension records with stable `axis-N` ids, the matching axis label, and the axis size.
 * @noThrows Maps the provided arrays without parsing or validation; callers are expected to provide a label for every axis in `shape`.
 * @example
 * const dims = defaultEditorDims([2, 4], ['batch', 'feature']);
 * expect(dims).toEqual([
 *   { id: 'axis-0', label: 'batch', size: 2 },
 *   { id: 'axis-1', label: 'feature', size: 4 },
 * ]);
 */
function defaultEditorDims(shape: number[], axisLabels: string[]): TensorViewEditorDim[] {
    return shape.map((size, axis) => ({
        id: `axis-${axis}`,
        label: axisLabels[axis]!,
        size,
    }));
}

/**
 * Builds the canonical View Tensor editor text from the editor's base dimensions.
 *
 * @param baseDims - Ordered tensor-view dimensions whose `label` and `size` fields should be serialized as `label=size` entries.
 * @returns Bracketed, comma-separated View Tensor input text that can be displayed in the editor or parsed again later.
 * @noThrows Only reads dimension fields and joins strings; it does not validate shapes or parse user input.
 * @example
 * const baseDims = [
 *     { id: 'axis-0', label: 'batch', size: 2 },
 *     { id: 'axis-1', label: 'channel', size: 3 },
 * ];
 *
 * formatViewTensorInput(baseDims);
 * // => '[batch=2, channel=3]'
 */
function formatViewTensorInput(baseDims: TensorViewEditorDim[]): string {
    return `[${baseDims.map((dim) => `${dim.label}=${dim.size}`).join(', ')}]`;
}

/**
 * Parses one comma-delimited View Tensor token into a dimension label and optional explicit size.
 *
 * @param part - A trimmed token such as `batch=4`, `channel`, `*A0=8`, `*1`, or `_2` from a tensor-view expression.
 * @returns `{ ok: true, label, sizeText }` for recognized label tokens, or `{ ok: false }` when the token contains invalid separators or brackets.
 * @noThrows Token grammar failures are represented by `{ ok: false }`; the parser only runs regular-expression matches and string trimming.
 * @example
 * parseViewLabelToken('batch=4');
 * // => { ok: true, label: 'batch', sizeText: '4' }
 *
 * parseViewLabelToken('[bad]');
 * // => { ok: false }
 */
function parseViewLabelToken(part: string): { ok: true; label: string; sizeText?: string } | { ok: false } {
    const anonymous = part.match(/^((?:\*A|\*|_)\d+)(?:\s*=\s*(-?\d+))?$/);
    if (anonymous) return { ok: true, label: anonymous[1]!, sizeText: anonymous[2] };
    const explicit = part.match(/^([^=,\[\]]+?)(?:\s*=\s*(-?\d+))?$/);
    if (!explicit) return { ok: false };
    return { ok: true, label: explicit[1]!.trim(), sizeText: explicit[2] };
}

/**
 * Parses an explicit final View expression and verifies that its dimensions cover the expected tensor element count.
 *
 * @param input - Bracketed or comma-separated View expression containing named dimensions, anonymous dimensions, numeric sizes, and at most one `-1` inferred size.
 * @param totalElements - Required product of the resolved dimension sizes, usually the element count of the permuted base tensor being reshaped.
 * @returns Successful parsed dimensions with inferred sizes filled in, or validation messages that the editor can show to the user.
 * @noThrows Invalid tokens, duplicate labels, multiple inferred dimensions, and shape-product mismatches are collected in the returned `errors` array.
 * @example
 * parseExplicitViewInput('[row=2, col=-1]', 8);
 * // => { ok: true, dims: [{ label: 'row', size: 2 }, { label: 'col', size: 4 }] }
 *
 * parseExplicitViewInput('[row=-1, col=-1]', 8);
 * // => { ok: false, errors: ['Tensor View may contain at most one inferred -1 dimension.'] }
 */
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
    // anonymous labels are real labels after parse; choosing fresh suffixes here
    // prevents `[2, *A0=3]` from generating two dimensions with the same name.
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

/**
 * Checks whether the View Tensor editor still matches the tensor's original axes without reshaping, relabeling, or reordering.
 *
 * @param baseDims - Editor base dimensions to compare, including each dimension's `axis-${index}` id, label, and size.
 * @param shape - Tensor shape indexed by source axis.
 * @param axisLabels - Source-axis labels aligned with `shape` by index.
 * @returns `true` when every editor dimension has the same axis id, label, and size as the corresponding tensor axis; otherwise `false`.
 * @noThrows Length, id, label, and size mismatches are normal comparison results, so the function returns a boolean instead of throwing.
 * @example
 * const baseDims = [
 *     { id: 'axis-0', label: 'batch', size: 2 },
 *     { id: 'axis-1', label: 'channel', size: 3 },
 * ];
 *
 * sameBaseAsTensor(baseDims, [2, 3], ['batch', 'channel']);
 * // => true
 *
 * sameBaseAsTensor(baseDims, [2, 3], ['batch', 'feature']);
 * // => false
 */
function sameBaseAsTensor(baseDims: TensorViewEditorDim[], shape: number[], axisLabels: string[]): boolean {
    return baseDims.length === shape.length && baseDims.every((dim, axis) => (
        dim.size === shape[axis]
        && dim.label === axisLabels[axis]
        && dim.id === `axis-${axis}`
    ));
}

/**
 * Materializes the renderer-facing tensor view from a normalized editor state, including base dimensions,
 * permutation groups, inserted singleton axes, sliced tokens, hidden indices, and optional explicit final-view
 * reshape tokens.
 *
 * @param tensorShape - Original tensor extents in storage-axis order; the product is used to validate explicit final-view reshapes.
 * @param axisLabels - Labels for the original tensor axes, in the same order as `tensorShape`.
 * @param editor - Normalized tensor-view editor containing base dimensions, permutation ids, flatten separators, singleton axes, slice selections, and optional `finalViewInput`.
 * @returns A `TensorViewSpec` that downstream coordinate and rendering code uses to map visible layout axes and sliced dimensions back to the original tensor axes.
 * @throws Error when `editor.finalViewInput` is present but cannot be parsed as a reshape of the permuted base tensor, such as a shape product mismatch or invalid inferred dimension.
 * @example
 * const editor = defaultEditor([2, 3], ['Y', 'X']);
 * editor.slicedTokenKeys = ['group:axis-0'];
 * editor.sliceValues = { 'group:axis-0': 1 };
 *
 * const spec = buildEditorSpec([2, 3], ['Y', 'X'], editor);
 *
 * expect(spec.viewShape).toEqual([3]);
 * expect(spec.sliceTokens).toEqual([
 *   { token: 'y', key: 'group:axis-0', axes: [0], size: 2, value: 1 },
 * ]);
 * expect(spec.hiddenIndices).toEqual([1, 0]);
 * @example
 * const editor = defaultEditor([2, 3], ['Y', 'X']);
 * editor.finalViewInput = '[4, 4]';
 *
 * expect(() => buildEditorSpec([2, 3], ['Y', 'X'], editor)).toThrow(Error);
 */
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
        // final-view input is a reshape of the permuted base tensor, so tokens
        // deliberately have no source axes and map through linear indices later.
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
                key: `group:${currentGroup.map((groupDim) => groupDim.id).join('+')}`,
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
                key: `group:${currentGroup.map((groupDim) => groupDim.id).join('+')}`,
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
                    key: `singleton:${singleton.id}`,
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
                // sliced grouped axes need to write every original hidden axis;
                // otherwise value lookup would keep using stale zero indices.
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

/**
 * Parses the View Tensor text box into base editor dimensions, resolving existing axis labels, explicit numeric sizes,
 * anonymous dimensions, and a single inferred `-1` dimension while validating that the resulting product matches the tensor.
 *
 * @param tensorShape - Original tensor extents whose total element count the parsed view-tensor dimensions must preserve.
 * @param axisLabels - Existing labels for `tensorShape`; label tokens without an explicit size inherit their size from this list.
 * @param input - View Tensor expression such as `"[Y, X]"`, `"[Batch=2, X=3]"`, `"[2, -1]"`, or an empty string for the default axis view.
 * @returns `ok: true` with parsed base dimensions and canonical bracketed input, or `ok: false` with user-facing validation messages for invalid tokens, duplicate labels, product mismatches, or failed `-1` inference.
 * @noThrows Invalid View Tensor syntax and shape mismatches are accumulated in the returned `errors` array instead of being thrown.
 * @example
 * const parsed = parseViewTensorInput([2, 3], ['Y', 'X'], '[Rows=2, X]');
 *
 * expect(parsed).toMatchObject({
 *   ok: true,
 *   canonicalInput: '[Rows=2, X]',
 *   baseDims: [
 *     { label: 'Rows', size: 2 },
 *     { label: 'X', size: 3 },
 *   ],
 * });
 * @example
 * const parsed = parseViewTensorInput([2, 3], ['Y', 'X'], '[Y, Y]');
 *
 * expect(parsed).toEqual({
 *   ok: false,
 *   errors: ['View Tensor label "Y" appears more than once.'],
 * });
 */
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

/**
 * Validates and canonicalizes a tensor-view editor after a UI or serialized-state edit, preserving compatible dimension ids
 * and permutation order, dropping permutation ids that no longer exist, rebuilding flatten separators, and clamping singleton positions.
 *
 * @param tensorShape - Original tensor extents used to validate `editor.viewTensorInput` and infer any `-1` dimension.
 * @param axisLabels - Labels for the original tensor axes that view-tensor tokens may reference without explicit sizes.
 * @param editor - Editor state from the UI or decoded serialized view, including view tensor text, base dimensions, permutation ids, separators, singleton axes, slices, and optional final view text.
 * @returns `ok: true` with a version-2 normalized editor ready for `buildEditorSpec`, or `ok: false` with the View Tensor validation errors that prevented normalization.
 * @noThrows Parse failures in `editor.viewTensorInput` are returned as `{ ok: false, errors }`; normalization otherwise copies, filters, sorts, and clamps editor fields without throwing.
 * @example
 * const editor = defaultEditor([2, 3], ['Y', 'X']);
 * editor.viewTensorInput = '[Rows=2, X]';
 * editor.singletons = [{ id: 's', position: 99 }];
 *
 * const normalized = normalizeEditor([2, 3], ['Y', 'X'], editor);
 *
 * expect(normalized).toMatchObject({
 *   ok: true,
 *   editor: {
 *     version: 2,
 *     viewTensorInput: '[Rows=2, X]',
 *     baseDims: [
 *       { label: 'Rows', size: 2 },
 *       { label: 'X', size: 3 },
 *     ],
 *     singletons: [{ id: 's', position: 3 }],
 *   },
 * });
 * @example
 * const editor = defaultEditor([2, 3], ['Y', 'X']);
 * editor.viewTensorInput = '[2, 2]';
 *
 * expect(normalizeEditor([2, 3], ['Y', 'X'], editor)).toEqual({
 *   ok: false,
 *   errors: ['View Tensor shape product must equal 6.'],
 * });
 */
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
    // keep every valid prior permutation entry, then append new dims. this makes
    // view-tensor edits preserve user ordering whenever the old ids still exist.
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

/**
 * Creates the initial version-2 tensor-view editor for an unsliced tensor, with one base dimension per tensor axis,
 * identity permutation order, separators between every adjacent axis, and no singleton or final-view edits.
 *
 * @param shape - Original tensor extents used as the default base dimension sizes.
 * @param axisLabels - Per-axis labels, in `shape` order, used for the default base dimension labels and view tensor text.
 * @returns A `TensorViewEditor` that can be serialized, shown in the view editor UI, normalized, or passed to `buildEditorSpec` for the tensor's default view.
 * @noThrows The function only maps the supplied shape and label arrays into editor fields and formats those dimensions; it performs no validation or parsing.
 * @example
 * const editor = defaultEditor([2, 3], ['Y', 'X']);
 *
 * expect(editor).toMatchObject({
 *   version: 2,
 *   viewTensorInput: '[Y, X]',
 *   baseDims: [
 *     { id: 'axis-0', label: 'Y', size: 2 },
 *     { id: 'axis-1', label: 'X', size: 3 },
 *   ],
 *   permutedDimIds: ['axis-0', 'axis-1'],
 *   flattenSeparators: [true],
 *   singletons: [],
 *   slicedTokenKeys: [],
 *   sliceValues: {},
 * });
 */
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

/**
 * Convert an unknown persisted structured-editor payload into a complete version-2 tensor-view editor.
 *
 * Missing or wrongly typed fields are filled from the editor that would be created for the supplied tensor shape and axis labels, while slice-only collections default to empty values.
 *
 * @param shape - Normalized tensor extents used to build fallback dimension ids, labels, and view text.
 * @param axisLabels - One label per tensor axis, already validated for the same order and length as `shape`.
 * @param value - Decoded persisted editor payload from a serialized tensor-view string; non-object values are treated as absent state.
 * @returns A complete `TensorViewEditor` that can be passed to `serializeTensorViewEditor` or stored on a tensor view snapshot.
 * @noThrows The function only performs type guards, array checks, and fallback object construction; malformed persisted values are ignored instead of parsed or rejected.
 * @example
 * const editor = normalizeSerializedEditor([2, 3], ['row', 'col'], {
 *     viewTensorInput: '[row=2, col=3]',
 *     permutedDimIds: ['axis-1', 'axis-0'],
 * });
 *
 * expect(editor).toMatchObject({
 *     version: 2,
 *     viewTensorInput: '[row=2, col=3]',
 *     permutedDimIds: ['axis-1', 'axis-0'],
 *     singletons: [],
 *     slicedTokenKeys: [],
 *     sliceValues: {},
 * });
 *
 * expect(normalizeSerializedEditor([2, 3], ['row', 'col'], null).viewTensorInput).toBe('[row=2, col=3]');
 */
function normalizeSerializedEditor(
    shape: number[],
    axisLabels: string[],
    value: unknown,
): TensorViewEditor {
    const fallback = defaultEditor(shape, axisLabels);
    if (!value || typeof value !== 'object') return fallback;
    const editor = value as Partial<TensorViewEditor>;
    return {
        version: 2,
        viewTensorInput: typeof editor.viewTensorInput === 'string' ? editor.viewTensorInput : fallback.viewTensorInput,
        finalViewInput: typeof editor.finalViewInput === 'string' ? editor.finalViewInput : undefined,
        baseDims: Array.isArray(editor.baseDims) ? editor.baseDims : fallback.baseDims,
        permutedDimIds: Array.isArray(editor.permutedDimIds) ? editor.permutedDimIds : fallback.permutedDimIds,
        flattenSeparators: Array.isArray(editor.flattenSeparators) ? editor.flattenSeparators : fallback.flattenSeparators,
        singletons: Array.isArray(editor.singletons) ? editor.singletons : [],
        slicedTokenKeys: Array.isArray(editor.slicedTokenKeys) ? editor.slicedTokenKeys : [],
        sliceValues: editor.sliceValues && typeof editor.sliceValues === 'object' ? editor.sliceValues : {},
    };
}

/**
 * Create the initial structured tensor-view editor for a tensor before the user has changed grouping, permutation, or slice controls.
 *
 * @param shape - Tensor extents in storage order; each entry becomes one base editor dimension.
 * @param axisLabelsInput - Optional axis label list supplied with the tensor metadata; when present it must provide one valid label for each axis in `shape`.
 * @returns A version-2 editor whose view input describes the tensor axes, whose permutation matches storage order, and whose slice/singleton state is empty.
 * @throws Error When `axisLabelsInput` cannot be parsed for `shape`, such as when the label count does not match the number of tensor axes.
 * @example
 * const editor = defaultTensorViewEditor([2, 3], ['row', 'col']);
 *
 * expect(editor).toMatchObject({
 *     version: 2,
 *     viewTensorInput: '[row=2, col=3]',
 *     permutedDimIds: ['axis-0', 'axis-1'],
 *     slicedTokenKeys: [],
 *     sliceValues: {},
 * });
 *
 * expect(() => defaultTensorViewEditor([2, 3], ['row'])).toThrow(Error);
 */
export function defaultTensorViewEditor(shape: number[], axisLabelsInput?: readonly string[]): TensorViewEditor {
    const normalizedShape = normalizeShape(shape);
    const axisLabels = parseAxisLabels(normalizedShape, axisLabelsInput);
    if (!axisLabels.ok) throw new Error(axisLabels.errors.join(' '));
    return defaultEditor(normalizedShape, axisLabels.axisLabels);
}

/**
 * Drop the active slice selections from a structured editor without changing its staged view expression, final-view expression, or dimension permutation.
 *
 * @param editor - Structured tensor-view editor whose `slicedTokenKeys` and `sliceValues` represent currently selected slice controls.
 * @returns The original editor when no slice state is present; otherwise a shallow copy with `slicedTokenKeys` set to `[]` and `sliceValues` set to `{}`.
 * @noThrows The function only reads the editor's slice collections and returns either the same object or an object spread with empty slice collections.
 * @example
 * const editor: TensorViewEditor = {
 *     version: 2,
 *     viewTensorInput: '[A=2, B=3, C=4]',
 *     finalViewInput: undefined,
 *     baseDims: [],
 *     permutedDimIds: ['axis-0', 'axis-1', 'axis-2'],
 *     flattenSeparators: [],
 *     singletons: [],
 *     slicedTokenKeys: ['B'],
 *     sliceValues: { B: 1 },
 * };
 *
 * expect(clearTensorViewSlices(editor)).toMatchObject({
 *     viewTensorInput: '[A=2, B=3, C=4]',
 *     permutedDimIds: ['axis-0', 'axis-1', 'axis-2'],
 *     slicedTokenKeys: [],
 *     sliceValues: {},
 * });
 */
export function clearTensorViewSlices(editor: TensorViewEditor): TensorViewEditor {
    if (editor.slicedTokenKeys.length === 0 && Object.keys(editor.sliceValues).length === 0) return editor;
    return {
        ...editor,
        slicedTokenKeys: [],
        sliceValues: {},
    };
}

/**
 * Encode a structured tensor-view editor as the prefixed string form accepted by the tensor-view parsing and assignment APIs.
 *
 * @param editor - Version-2 structured editor snapshot containing the staged view text, final-view text, permutation, singleton, and slice state to persist.
 * @returns A string with the structured-editor prefix followed by URI-encoded normalized JSON, suitable for `parseTensorView`, `setTensorView`, and saved viewer snapshots.
 * @noThrows Normalizes and JSON-encodes a structured editor object without parsing untrusted text or mutating viewer state.
 * @example
 * const editor = defaultTensorViewEditor([2, 3], ['row', 'col']);
 * const serialized = serializeTensorViewEditor(editor);
 * const parsed = parseTensorView([2, 3], serialized, [], ['row', 'col']);
 *
 * expect(serialized).toContain(encodeURIComponent('"version":2'));
 * expect(parsed.ok).toBe(true);
 * if (parsed.ok) {
 *     expect(parsed.spec.editor.viewTensorInput).toBe('[row=2, col=3]');
 * }
 */
export function serializeTensorViewEditor(editor: TensorViewEditor): string {
    // the prefix distinguishes structured editor snapshots from legacy ad-hoc
    // view strings while still fitting into the existing string API.
    return `${TENSOR_VIEW_EDITOR_PREFIX}${encodeURIComponent(JSON.stringify(normalizeTensorViewEditor(editor)))}`;
}

/**
 * Convert a coordinate in the rendered visible view back to the corresponding coordinate in the source tensor.
 *
 * The mapping honors hidden slice tokens, grouped axes, permutations, singleton axes, and an explicit final
 * `.view(...)` reshape when the parsed TensorViewSpec includes one.
 *
 * @param viewCoord - Zero-based coordinate over `spec.viewShape`; missing components default to `0` and visible-axis values are clamped to the token extent.
 * @param spec - Parsed tensor-view specification that describes the source tensor shape, visible tokens, hidden slices, permutation, and optional final reshape.
 * @returns Zero-based coordinate in `spec.tensorShape`, suitable for looking up the underlying tensor value for the rendered cell.
 * @noThrows For a `TensorViewSpec` produced by the view parser, this only reads spec arrays, clamps/defaults view-coordinate components, and performs deterministic index flattening/unflattening.
 * @example
 * const result = parseTensorView([2, 3, 4], 'tensor.view(6, 4)[:, 3]');
 * if (!result.ok) throw new Error(result.message);
 *
 * mapViewCoordToTensorCoord([2], result.spec);
 * // => [0, 2, 3]
 */
export function mapViewCoordToTensorCoord(viewCoord: number[], spec: TensorViewSpec): number[] {
    if (spec.editor.finalViewInput?.trim()) {
        // explicit final views reshape after permutation, so coordinate mapping has
        // to go view -> layout -> linear -> permuted base -> original tensor.
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

/**
 * List the source tensor coordinate for each rendered cell in the active view.
 *
 * Coordinates are emitted in row-major order over `spec.viewShape`; a scalar view produces one coordinate for the single visible cell.
 *
 * @param spec - Parsed tensor-view specification whose visible shape and hidden slices determine which source cells appear in the viewer.
 * @returns Source tensor coordinates, ordered the same way the visible view cells are enumerated for mesh instances and hover lookup.
 * @noThrows For a valid parsed `TensorViewSpec`, enumeration is bounded by `spec.viewShape` and delegates each generated coordinate to the non-throwing view-to-tensor mapper.
 * @example
 * const result = parseTensorView([2, 3, 4], 'tensor[:, 1, 0:2]');
 * if (!result.ok) throw new Error(result.message);
 *
 * visibleTensorCoords(result.spec);
 * // => [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]]
 */
export function visibleTensorCoords(spec: TensorViewSpec): number[][] {
    const viewShape = spec.viewShape.length === 0 ? [1] : spec.viewShape;
    const total = product(viewShape);
    const coords: number[][] = [];
    for (let index = 0; index < total; index += 1) {
        const viewCoord = total === 1 && spec.viewShape.length === 0 ? [] : unravelIndex(index, viewShape);
        coords.push(mapViewCoordToTensorCoord(viewCoord, spec));
    }
    return coords;
}

/**
 * Expand a visible-view coordinate into the full rendered layout coordinate, including axes hidden by slices.
 *
 * Singleton layout axes are filled with `0`; hidden non-singleton axes use their slice-token value; visible axes consume
 * components from `viewCoord` in token order.
 *
 * @param viewCoord - Zero-based coordinate over the visible axes in `spec.viewShape`; missing visible components default to `0` and supplied values are clamped to the visible token size.
 * @param spec - Parsed tensor-view specification containing the token list and slice tokens needed to reinsert hidden layout axes.
 * @returns Coordinate over `spec.layoutShape`, suitable for display positioning and layout hit-test comparisons before hidden axes are collapsed.
 * @noThrows For a parsed `TensorViewSpec`, the function only iterates token metadata, uses optional lookup for slice tokens, and defaults absent slice or coordinate components to `0`.
 * @example
 * const result = parseTensorView([2, 3, 4], 'tensor[:, 2, :]');
 * if (!result.ok) throw new Error(result.message);
 *
 * mapViewCoordToFullLayoutCoord([1, 3], result.spec);
 * // => [1, 2, 3]
 */
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

/**
 * Return the layout dimensions that rendering and hit testing should use for a parsed tensor view.
 *
 * When hidden axes are collapsed, the shape contains only visible view axes; otherwise it preserves every layout token,
 * including singleton and sliced axes. Scalar layouts are represented as `[1]` so callers still have one drawable cell.
 *
 * @param spec - Parsed tensor-view specification containing both the full token layout shape and the visible-only view shape.
 * @param collapseHiddenAxes - Whether to omit hidden sliced axes from the returned display shape.
 * @returns A copied shape array for the active layout mode; callers may pass it to extent, mesh-instance, or hit-test calculations without mutating the spec.
 * @noThrows The function only chooses between precomputed shape arrays on the parsed spec, copies the selected array, and substitutes `[1]` for scalar views.
 * @example
 * const result = parseTensorView([2, 3, 4], 'tensor[:, 1, :]');
 * if (!result.ok) throw new Error(result.message);
 *
 * layoutShape(result.spec, false);
 * // => [2, 1, 4]
 * layoutShape(result.spec, true);
 * // => [2, 4]
 */
export function layoutShape(spec: TensorViewSpec, collapseHiddenAxes = false): number[] {
    const shape = collapseHiddenAxes ? spec.viewShape : spec.layoutShape;
    return shape.length === 0 ? [1] : shape.slice();
}

/**
 * Converts a coordinate in the visible tensor view into the layout coordinate used to place the cell on screen.
 *
 * Hidden sliced axes are reinserted from the `TensorViewSpec` when hidden axes are expanded in the layout; when
 * `collapseHiddenAxes` is enabled, the visible coordinate is already the rendered layout coordinate.
 *
 * @param viewCoord - Zero-based coordinate with one entry for each visible axis in `spec.viewShape`.
 * @param spec - Parsed tensor-view specification that describes visible axes, hidden slice tokens, grouping, and layout-axis order.
 * @param collapseHiddenAxes - Whether the viewer is rendering only visible axes instead of keeping hidden sliced axes in layout space.
 * @returns Layout-space coordinate for display positioning, hit metadata, and selection bounds.
 * @noThrows Reads coordinate/spec arrays and returns a copied or mapped coordinate; malformed lengths are tolerated by the mapping helpers rather than throwing here.
 * @example
 * ```ts
 * const result = parseTensorView('tensor[:, 1, :]', [2, 3, 4]);
 * if (!result.ok) throw new Error(result.message);
 *
 * mapViewCoordToLayoutCoord([1, 2], result.spec);
 * // => [1, 1, 2]
 *
 * mapViewCoordToLayoutCoord([1, 2], result.spec, true);
 * // => [1, 2]
 * ```
 */
export function mapViewCoordToLayoutCoord(
    viewCoord: number[],
    spec: TensorViewSpec,
    collapseHiddenAxes = false,
): number[] {
    if (collapseHiddenAxes) return spec.viewShape.length === 0 ? [] : viewCoord.slice();
    return mapViewCoordToFullLayoutCoord(viewCoord, spec);
}

/**
 * Projects a rendered layout coordinate back to the visible tensor-view coordinate used by value lookup and selection state.
 *
 * Hidden axes are skipped, singleton visible axes become `0`, and non-singleton visible axes are clamped to the axis size so
 * out-of-range hit-test coordinates still resolve to the nearest visible cell.
 *
 * @param layoutCoord - Zero-based coordinate from layout space, typically produced by display hit testing or mesh instance metadata.
 * @param spec - Parsed tensor-view specification whose tokens identify which layout axes are visible and their sizes.
 * @param collapseHiddenAxes - Whether `layoutCoord` already omits hidden sliced axes because the viewer collapsed them for display.
 * @returns Visible view coordinate that can be passed to tensor-coordinate mapping, value lookup, or selection helpers.
 * @noThrows Missing layout entries default to `0` and visible entries are clamped, so ordinary hit-test coordinates do not create a throw path.
 * @example
 * ```ts
 * const result = parseTensorView('tensor[:, 1, :]', [2, 3, 4]);
 * if (!result.ok) throw new Error(result.message);
 *
 * mapLayoutCoordToViewCoord([1, 1, 2], result.spec);
 * // => [1, 2]
 *
 * mapLayoutCoordToViewCoord([1, 2], result.spec, true);
 * // => [1, 2]
 * ```
 */
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

/**
 * Checks whether a full layout-space coordinate lies on the hidden-axis slice selected by the tensor view.
 *
 * Visible layout axes are ignored. Each hidden layout axis must equal the corresponding slice token value for the
 * coordinate to belong to the active sliced tensor.
 *
 * @param layoutCoord - Full layout coordinate, including entries for hidden sliced axes.
 * @param spec - Parsed tensor-view specification containing hidden layout tokens and their active slice values.
 * @returns `true` when every hidden-axis coordinate equals its active slice value; otherwise `false`.
 * @noThrows Missing coordinate entries are treated as `0` during comparison, and absent slice metadata simply makes the coordinate fail to match.
 * @example
 * ```ts
 * const result = parseTensorView('tensor[:, 1, :]', [2, 3, 4]);
 * if (!result.ok) throw new Error(result.message);
 *
 * layoutCoordMatchesSlice([1, 1, 2], result.spec);
 * // => true
 *
 * layoutCoordMatchesSlice([1, 0, 2], result.spec);
 * // => false
 * ```
 */
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

/**
 * Reports whether a layout-space coordinate should be rendered or accepted by picking for the current tensor view.
 *
 * In expanded-layout mode, the coordinate must match the active hidden-axis slice. In collapsed-hidden-axes mode,
 * all provided layout coordinates are already in the visible layout and are therefore accepted.
 *
 * @param layoutCoord - Coordinate from layout generation or hit testing to test against the active view slice.
 * @param spec - Parsed tensor-view specification containing the hidden-axis slice constraints.
 * @param collapseHiddenAxes - Whether hidden sliced axes were removed from the rendered layout coordinate space.
 * @returns `true` when the coordinate is visible in the current view; otherwise `false` so callers can skip rendering or hit results.
 * @noThrows Delegates to slice comparison or returns the collapsed-layout shortcut, neither of which validates by throwing.
 * @example
 * ```ts
 * const result = parseTensorView('tensor[:, 1, :]', [2, 3, 4]);
 * if (!result.ok) throw new Error(result.message);
 *
 * layoutCoordIsVisible([1, 1, 2], result.spec);
 * // => true
 *
 * layoutCoordIsVisible([1, 0, 2], result.spec);
 * // => false
 *
 * layoutCoordIsVisible([1, 2], result.spec, true);
 * // => true
 * ```
 */
export function layoutCoordIsVisible(layoutCoord: number[], spec: TensorViewSpec, collapseHiddenAxes = false): boolean {
    return collapseHiddenAxes || layoutCoordMatchesSlice(layoutCoord, spec);
}

/**
 * Lists the labels attached to the layout axes in the order the viewer renders them.
 *
 * @param spec - Normalized tensor view spec whose layout tokens carry axis labels and visibility flags.
 * @param collapseHiddenAxes - When true, omit tokens hidden by slice state so the labels match the collapsed visible layout.
 * @returns Axis-label strings in rendered layout order, suitable for row/column headers and layout metadata.
 * @noThrows Reads the already-normalized token array and maps labels without parsing, validation, or allocation-sensitive lookups that can fail.
 * @example
 * const result = parseTensorView([2, 3, 4, 5], serializeTensorViewEditor({
 *     version: 2,
 *     viewTensorInput: '[A=2, B=3, C=4, D=5]',
 *     permutedDimIds: ['axis-0', 'axis-1', 'axis-2', 'axis-3'],
 *     finalViewInput: '',
 *     hiddenDimIds: ['axis-1', 'axis-3'],
 *     sliceValues: { 'axis-1': 1, 'axis-3': 2 },
 * }));
 * if (!result.ok) throw new Error(result.errors.join('\n'));
 *
 * layoutAxisLabels(result.spec, true);
 * // => ['A', 'C']
 */
export function layoutAxisLabels(spec: TensorViewSpec, collapseHiddenAxes = false): string[] {
    const tokens = collapseHiddenAxes ? spec.tokens.filter((token) => token.visible) : spec.tokens;
    return tokens.map((token) => token.label);
}

/**
 * Reports whether the current 2D layout can translate a rectangular selection as one contiguous row-major tensor span.
 *
 * @param spec - Normalized tensor view spec to inspect for original-axis ordering, base tensor axes, visible X-axis grouping, and hidden-token placement.
 * @param collapseHiddenAxes - Collapsed hidden-axis mode requested by the caller; this mode always disables the contiguous selection fast path.
 * @returns True when visible X axes are the trailing original tensor axes with no hidden token after them; false when selection needs the general coordinate mapper.
 * @noThrows Performs only array filtering and numeric comparisons against the normalized spec, so unsupported layouts are returned as false instead of throwing.
 * @example
 * const result = parseTensorView([2, 3, 4], '');
 * if (!result.ok) throw new Error(result.errors.join('\n'));
 *
 * supportsContiguousSelectionFastPath2D(result.spec);
 * // => true
 * supportsContiguousSelectionFastPath2D(result.spec, true);
 * // => false
 */
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

/**
 * Formats the normalized editor state as the tensor expression shown to users for view, permute, final-view, and slice steps.
 *
 * @param spec - Normalized tensor view spec containing the editor inputs, base shape, permutation axes, token visibility, and slice values to render.
 * @returns Display string such as `tensor.view(...).permute(...)[...]` that callers can show as the canonical view expression.
 * @noThrows Formats fields from a normalized spec with string trimming, joins, and slice lookup; invalid editor data is expected to be rejected before a spec is built.
 * @example
 * const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
 *     version: 2,
 *     viewTensorInput: '[A=2, B=3, C=4]',
 *     permutedDimIds: ['axis-2', 'axis-0', 'axis-1'],
 *     finalViewInput: '',
 *     hiddenDimIds: ['axis-2'],
 *     sliceValues: { 'axis-2': 5 },
 * }));
 * if (!result.ok) throw new Error(result.errors.join('\n'));
 *
 * buildTensorViewExpression(result.spec);
 * // => 'tensor.view(A=2, B=3, C=4).permute(2, 0, 1)[5, :, :]'
 */
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

/**
 * Converts serialized tensor-view editor state into the grouped axes, slice metadata, labels, and coordinate mapping spec used by the viewer.
 *
 * @param shapeInput - Base tensor shape dimensions, one positive extent per original tensor axis.
 * @param input - Empty string to reset to the default editor, or a `tv2:` URL-encoded JSON editor payload produced by `serializeTensorViewEditor`.
 * @param _hiddenIndices - Legacy hidden-axis argument kept for older callers; structured editor state in `input` supplies the active slice state.
 * @param axisLabelsInput - Optional labels for the original tensor axes; when omitted, the parser generates default axis labels.
 * @returns `ok: true` with a normalized `TensorViewSpec` for rendering, or `ok: false` with user-facing parse/validation errors to display.
 * @noThrows Malformed serialized editor state, oversized payloads, and unsupported legacy strings are converted to `ok: false` results instead of escaping as parse errors.
 * @example
 * const parsed = parseTensorView([2, 3, 4], '');
 * if (!parsed.ok) throw new Error(parsed.errors.join('\n'));
 *
 * parsed.spec.viewShape;
 * // => [2, 3, 4]
 *
 * const invalid = parseTensorView([2, 3, 4], 'A B C');
 * invalid;
 * // => { ok: false, errors: ['Tensor view state must be serialized editor data.'] }
 */
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
        if (input.length > VIEWER_LIMITS.maxEditorInputLength) {
            return { ok: false, errors: ['Tensor view editor state is too large.'] };
        }
        try {
            const decoded = decodeURIComponent(input.slice(TENSOR_VIEW_EDITOR_PREFIX.length));
            if (decoded.length > VIEWER_LIMITS.maxEditorInputLength) {
                return { ok: false, errors: ['Tensor view editor state is too large.'] };
            }
            const editor = normalizeSerializedEditor(
                shape,
                axisLabels,
                JSON.parse(decoded),
            );
            const normalized = normalizeEditor(shape, axisLabels, normalizeTensorViewEditor(editor));
            if (!normalized.ok) return normalized;
            return { ok: true, spec: buildEditorSpec(shape, axisLabels, normalized.editor) };
        } catch {
            return { ok: false, errors: ['Tensor view editor state is invalid.'] };
        }
    }
    return { ok: false, errors: ['Tensor view state must be serialized editor data.'] };
}
