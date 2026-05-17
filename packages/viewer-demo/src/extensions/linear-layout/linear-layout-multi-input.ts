import {
    coordFromKey,
    coordKey,
    parseTensorView,
    serializeTensorViewEditor,
    visibleTensorCoords,
    type LoadedBundleDocument,
} from '@tensor-viz/viewer-core';
import { rootColorsForLayoutState } from './linear-layout.js';
import { composeLayoutMetaForTab, type LinearLayoutSelectionMap, type LinearLayoutUiContext } from './linear-layout-state.js';

/**
 * shape of linear layout display model data used by the viewer.
 *
 * @example
 * const value: LinearLayoutDisplayModel = {} as LinearLayoutDisplayModel;
 */
export type LinearLayoutDisplayModel = {
    rootIndexes: Set<number>;
    sliceRootIndexes: Set<number> | null;
    displayedRootIndexByTensor: Map<string, Array<number | null>>;
    visibleCoordsByTensor: Map<string, number[][]>;
    ghostRootIndexesByTensor: Map<string, Array<{ coord: number[]; rootIndex: number; layer: number }>>;
};

/**
 * shape of linear layout multi input model data used by the viewer.
 *
 * @example
 * const value: LinearLayoutMultiInputModel = {} as LinearLayoutMultiInputModel;
 */
export type LinearLayoutMultiInputModel = {
    focusedTensorId: string;
    value: number;
    size: number;
} | null;

/**
 * return linear layout selection map for meta for the current viewer state.
 *
 * @param tab - Session tab used by this operation.
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * linearLayoutSelectionMapForMeta(tab);
 */
export function linearLayoutSelectionMapForMeta(
    tab: LoadedBundleDocument,
): LinearLayoutSelectionMap | null {
    const meta = composeLayoutMetaForTab(tab);
    if (!meta || meta.tensors.length === 0) return null;
    const loadedTensorIds = new Set(tab.manifest.tensors.map((tensor) => tensor.id));
    const finalOutputShape = meta.finalOutputBitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
    const rootInputShape = meta.rootInputBitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
    const rootKeys = meta.tensors[0]!.rootToTensor.map((coord) => coordKey(coord));
    const rootToFinalKeys = meta.tensors[0]!.tensorToFinal.map((coord) => coord ? coordKey(coord) : '');
    const tensors = new Map<string, LinearLayoutSelectionMap['tensors'] extends Map<string, infer T> ? T : never>();
    meta.tensors.forEach((tensorMeta) => {
        if (!loadedTensorIds.has(tensorMeta.id)) return;
        const rootToTensorKeys = tensorMeta.rootToTensor.map((coord) => coordKey(coord));
        const coordKeyToFlatIndex = new Map<string, number>();
        const cellRootIndexes = Array.from({ length: tensorMeta.shape.reduce((total, value) => total * value, 1) }, () => [] as number[]);
        // non-injective tensors can map many root inputs into one cell.  Keep
        // all roots by flat cell so hover, selection, and ghost layers agree.
        rootToTensorKeys.forEach((tensorKey, rootIndex) => {
            const flat = coordFromKey(tensorKey).reduce((index, value, axis) => (index * tensorMeta.shape[axis]!) + value, 0);
            coordKeyToFlatIndex.set(tensorKey, flat);
            cellRootIndexes[flat]!.push(rootIndex);
        });
        tensors.set(tensorMeta.id, { meta: tensorMeta, rootToTensorKeys, coordKeyToFlatIndex, cellRootIndexes });
    });
    return {
        injective: meta.injective,
        rootInputLabels: meta.rootInputLabels.slice(),
        rootInputShape,
        rootKeys: rootKeys.slice(),
        rootKeyToIndex: new Map(rootKeys.map((key, index) => [key, index])),
        finalOutputLabels: meta.finalOutputLabels.slice(),
        finalOutputShape,
        rootToFinalKeys,
        tensors,
        orderedTensorIds: meta.tensors.map((tensor) => tensor.id).filter((id) => tensors.has(id)),
    };
}

/**
 * return linear layout multi input model for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param mapping - Mapping data used by this operation.
 * @returns Computed LinearLayoutMultiInputModel value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * linearLayoutMultiInputModel(ctx, mapping);
 */
export function linearLayoutMultiInputModel(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap | null,
): LinearLayoutMultiInputModel {
    const focusedTensorId = ctx.viewer.getState().activeTensorId;
    if (!mapping || !focusedTensorId) return null;
    if (ctx.state.linearLayoutState?.propagateOutputs) return null;
    const tensor = mapping.tensors.get(focusedTensorId);
    if (!tensor) return null;
    const size = Math.max(0, ...tensor.cellRootIndexes.map((roots) => roots.length));
    // the slider exists only for many-to-one cells; injective or currently
    // one-to-one views should not expose an extra control.
    if (size <= 1) return null;
    const storedValue = ctx.state.linearLayoutMultiInputState[focusedTensorId] ?? -1;
    const value = storedValue < 0 ? -1 : Math.min(size - 1, storedValue);
    return { focusedTensorId, value, size };
}

/**
 * apply linear layout display for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * applyLinearLayoutDisplay(ctx);
 */
export function applyLinearLayoutDisplay(ctx: LinearLayoutUiContext): void {
    const tab = ctx.getActiveTab();
    if (!tab) return;
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping) return;
    const display = linearLayoutDisplayModel(ctx, mapping);
    const [colorLabels, colorShape] = ctx.state.linearLayoutState.propagateOutputs
        ? [mapping.finalOutputLabels, mapping.finalOutputShape]
        : [mapping.rootInputLabels, mapping.rootInputShape];
    const colors = rootColorsForLayoutState(
        colorLabels,
        colorShape,
        ctx.state.linearLayoutState,
    );
    mapping.orderedTensorIds.forEach((tensorId) => {
        const tensor = mapping.tensors.get(tensorId)!;
        const displayed = display.displayedRootIndexByTensor.get(tensorId) ?? [];
        const data = new Float32Array(tensor.meta.shape.reduce((total, value) => total * value, 1)).fill(-1);
        const rgb = new Float32Array(data.length * 3);
        displayed.forEach((rootIndex, flat) => {
            if (rootIndex === null) return;
            data[flat] = rootIndex;
            rgb.set(colors[propagatedIndexForRoot(mapping, rootIndex, ctx.state.linearLayoutState.propagateOutputs)]!, flat * 3);
        });
        // data, colors, visible coords, and ghost layers are updated together so
        // rendering cannot show stale hidden roots after slicing or slider edits.
        ctx.viewer.setTensorData(tensorId, data, 'float32');
        ctx.viewer.colorTensor(tensorId, rgb);
        ctx.viewer.setTensorVisibleCoords(tensorId, display.visibleCoordsByTensor.get(tensorId) ?? []);
        ctx.viewer.setTensorGhostLayers(tensorId, ctx.state.linearLayoutState.propagateOutputs ? null : display.ghostRootIndexesByTensor.get(tensorId)?.map((entry) => ({
            coord: entry.coord,
            color: colors[propagatedIndexForRoot(mapping, entry.rootIndex, ctx.state.linearLayoutState.propagateOutputs)]!
                .map((value) => Math.round(value * 255)) as [number, number, number],
            bias: [entry.layer * 0.18, -(entry.layer * 0.18)] as const,
            layer: entry.layer,
            text: linearLayoutGhostText(
                propagatedCoordForRoot(mapping, entry.rootIndex, ctx.state.linearLayoutState.propagateOutputs),
                ctx.state.linearLayoutState.propagateOutputs ? mapping.finalOutputLabels : mapping.rootInputLabels,
                ctx.state.linearLayoutCellTextState,
            ),
        })) ?? null);
    });
}

/**
 * return linear layout display model for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param mapping - Mapping data used by this operation.
 * @returns Computed LinearLayoutDisplayModel value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * linearLayoutDisplayModel(ctx, mapping);
 */
export function linearLayoutDisplayModel(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap,
): LinearLayoutDisplayModel {
    const sliceVisibleRootIndexes = sliceVisibleRootIndexesByTensor(ctx, mapping);
    const slicedRoots = intersectRootIndexes(sliceVisibleRootIndexes.values(), mapping.rootKeys.length);
    const multiInput = linearLayoutMultiInputModel(ctx, mapping);
    // visibility is the intersection of active tensor-view slices, then
    // optionally narrowed to one many-to-one member by the focused tensor slider.
    const focusedRoots = multiInput
        ? focusedRootIndexes(mapping, multiInput.focusedTensorId, multiInput.value, sliceVisibleRootIndexes)
        : null;
    const rootIndexes = focusedRoots ?? slicedRoots ?? new Set(Array.from({ length: mapping.rootKeys.length }, (_entry, index) => index));
    const displayedRootIndexByTensor = new Map<string, Array<number | null>>();
    const visibleCoordsByTensor = new Map<string, number[][]>();
    const ghostRootIndexesByTensor = new Map<string, Array<{ coord: number[]; rootIndex: number; layer: number }>>();
    mapping.orderedTensorIds.forEach((tensorId) => {
        const tensor = mapping.tensors.get(tensorId)!;
        const visibleRoots = tensor.cellRootIndexes.map((roots) => roots.filter((rootIndex) => rootIndexes.has(rootIndex)));
        const displayed = visibleRoots.map((roots) => roots[0] ?? null);
        displayedRootIndexByTensor.set(tensorId, displayed);
        visibleCoordsByTensor.set(tensorId, displayed.flatMap((rootIndex, flat) => (
            rootIndex === null ? [] : [unravelIndex(flat, tensor.meta.shape)]
        )));
        ghostRootIndexesByTensor.set(tensorId, visibleRoots.flatMap((roots, flat) => (
            // root zero is rendered as the main cell; additional roots become
            // offset ghost layers so non-injective cells remain inspectable.
            roots.slice(1).map((rootIndex, layer) => ({
                coord: unravelIndex(flat, tensor.meta.shape),
                rootIndex,
                layer: layer + 1,
            }))
        )));
    });
    return { rootIndexes, sliceRootIndexes: slicedRoots, displayedRootIndexByTensor, visibleCoordsByTensor, ghostRootIndexesByTensor };
}

/**
 * return root indexes for coords for the current viewer state.
 *
 * @param mapping - Mapping data used by this operation.
 * @param tensorId - Stable identifier used by this operation.
 * @param coords - Coordinates used by this operation.
 * @returns Collection computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * rootIndexesForCoords(mapping, tensorId, coords);
 */
export function rootIndexesForCoords(
    mapping: LinearLayoutSelectionMap,
    tensorId: string,
    coords: number[][],
): Set<number> {
    const tensor = mapping.tensors.get(tensorId);
    if (!tensor) return new Set();
    return new Set(coords.flatMap((coord) => {
        const flat = tensor.coordKeyToFlatIndex.get(coordKey(coord));
        return flat === undefined ? [] : tensor.cellRootIndexes[flat] ?? [];
    }));
}

/**
 * return coords for root indexes for the current viewer state.
 *
 * @param mapping - Mapping data used by this operation.
 * @param tensorId - Stable identifier used by this operation.
 * @param selectedRootIndexes - selected root indexes input used by this operation (Set<number>).
 * @param visibleRootIndexes - visible root indexes input used by this operation (Set<number> | null).
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * coordsForRootIndexes(mapping, tensorId, selectedRootIndexes, visibleRootIndexes);
 */
export function coordsForRootIndexes(
    mapping: LinearLayoutSelectionMap,
    tensorId: string,
    selectedRootIndexes: Set<number>,
    visibleRootIndexes: Set<number> | null = null,
): number[][] {
    const tensor = mapping.tensors.get(tensorId);
    if (!tensor || selectedRootIndexes.size === 0) return [];
    return tensor.cellRootIndexes.flatMap((roots, flat) => {
        const matchesSelection = roots.some((rootIndex) => selectedRootIndexes.has(rootIndex));
        const matchesVisible = visibleRootIndexes === null || roots.some((rootIndex) => visibleRootIndexes.has(rootIndex));
        return matchesSelection && matchesVisible ? [unravelIndex(flat, tensor.meta.shape)] : [];
    });
}

/**
 * return displayed root index for coord for the current viewer state.
 *
 * @param display - display input used by this operation (LinearLayoutDisplayModel).
 * @param mapping - Mapping data used by this operation.
 * @param tensorId - Stable identifier used by this operation.
 * @param coord - Coordinate used by this operation.
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * displayedRootIndexForCoord(display, mapping, tensorId, coord);
 */
export function displayedRootIndexForCoord(
    display: LinearLayoutDisplayModel,
    mapping: LinearLayoutSelectionMap,
    tensorId: string,
    coord: number[],
): number | null {
    const tensor = mapping.tensors.get(tensorId);
    if (!tensor) return null;
    const flat = tensor.coordKeyToFlatIndex.get(coordKey(coord));
    if (flat === undefined) return null;
    return display.displayedRootIndexByTensor.get(tensorId)?.[flat] ?? null;
}

/**
 * return focused root indexes for the current viewer state.
 *
 * @param mapping - Mapping data used by this operation.
 * @param focusedTensorId - Stable identifier used by this operation.
 * @param index - Index used by this operation.
 * @param sliceVisibleRootIndexes - slice visible root indexes input used by this operation (Map<string, Set<number>>).
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * focusedRootIndexes(mapping, focusedTensorId, index, sliceVisibleRootIndexes);
 */
function focusedRootIndexes(
    mapping: LinearLayoutSelectionMap,
    focusedTensorId: string,
    index: number,
    sliceVisibleRootIndexes: Map<string, Set<number>>,
): Set<number> | null {
    if (index < 0) return null;
    const tensor = mapping.tensors.get(focusedTensorId);
    if (!tensor) return null;
    const visibleRoots = sliceVisibleRootIndexes.get(focusedTensorId) ?? null;
    return new Set(tensor.cellRootIndexes.flatMap((roots) => {
        const filteredRoots = visibleRoots ? roots.filter((rootIndex) => visibleRoots.has(rootIndex)) : roots;
        const rootIndex = filteredRoots[index];
        return rootIndex === undefined ? [] : [rootIndex];
    }));
}

/**
 * return slice visible root indexes by tensor for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param mapping - Mapping data used by this operation.
 * @returns Collection computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * sliceVisibleRootIndexesByTensor(ctx, mapping);
 */
function sliceVisibleRootIndexesByTensor(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap,
): Map<string, Set<number>> {
    return new Map(mapping.orderedTensorIds.map((tensorId) => {
        const coords = slicedTensorCoords(ctx, tensorId);
        return [tensorId, coords ? rootIndexesForCoords(mapping, tensorId, coords) : new Set<number>()] as const;
    }).filter(([_tensorId, roots]) => roots.size > 0));
}

/**
 * return intersect root indexes for the current viewer state.
 *
 * @param sets - sets input used by this operation (Iterable<Set<number>>).
 * @param rootCount - root count input used by this operation (number).
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * intersectRootIndexes(sets, rootCount);
 */
function intersectRootIndexes(sets: Iterable<Set<number>>, rootCount: number): Set<number> | null {
    let intersection: Set<number> | null = null;
    for (const set of sets) {
        intersection = intersection
            ? new Set(Array.from<number>(intersection).filter((rootIndex) => set.has(rootIndex)))
            : new Set(set);
    }
    if (intersection) return intersection;
    return null;
}

/**
 * return sliced tensor coords for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param tensorId - Stable identifier used by this operation.
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * slicedTensorCoords(ctx, tensorId);
 */
function slicedTensorCoords(ctx: LinearLayoutUiContext, tensorId: string): number[][] | null {
    const status = ctx.viewer.getTensorStatus(tensorId);
    const snapshot = ctx.viewer.getTensorView(tensorId);
    const parsed = parseTensorView(
        status.shape.slice(),
        serializeTensorViewEditor(snapshot.editor),
        snapshot.hiddenIndices,
        status.axisLabels,
    );
    return !parsed.ok ? null : visibleTensorCoords(parsed.spec);
}

/**
 * return unravel index for the current viewer state.
 *
 * @param index - Index used by this operation.
 * @param shape - Tensor shape used by this operation.
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * unravelIndex(index, shape);
 */
function unravelIndex(index: number, shape: number[]): number[] {
    if (shape.length === 0) return [];
    const coord = new Array(shape.length).fill(0);
    let remainder = index;
    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
        const size = shape[axis] ?? 1;
        coord[axis] = remainder % size;
        remainder = Math.floor(remainder / size);
    }
    return coord;
}

/**
 * return linear layout ghost text for the current viewer state.
 *
 * @param coord - Coordinate used by this operation.
 * @param labels - labels input used by this operation (string[]).
 * @param state - State object read or updated by this operation.
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * linearLayoutGhostText(coord, labels, state);
 */
function linearLayoutGhostText(coord: number[], labels: string[], state: Record<string, boolean>): string | null {
    const text = labels
        .flatMap((label, axis) => (state[label] && axis < coord.length ? [`${label}:${coord[axis] ?? 0}`] : []))
        .join('\n');
    return text || null;
}

/**
 * return propagated coord for root for the current viewer state.
 *
 * @param mapping - Mapping data used by this operation.
 * @param rootIndex - Index used by this operation.
 * @param propagateOutputs - propagate outputs input used by this operation (boolean).
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * propagatedCoordForRoot(mapping, rootIndex, propagateOutputs);
 */
function propagatedCoordForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number[] {
    const key = propagateOutputs ? mapping.rootToFinalKeys[rootIndex] : mapping.rootKeys[rootIndex];
    return coordFromKey(key ?? '');
}

/**
 * return propagated index for root for the current viewer state.
 *
 * @param mapping - Mapping data used by this operation.
 * @param rootIndex - Index used by this operation.
 * @param propagateOutputs - propagate outputs input used by this operation (boolean).
 * @returns Numeric result computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * propagatedIndexForRoot(mapping, rootIndex, propagateOutputs);
 */
function propagatedIndexForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number {
    const shape = propagateOutputs ? mapping.finalOutputShape : mapping.rootInputShape;
    return propagatedCoordForRoot(mapping, rootIndex, propagateOutputs)
        .reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}

/**
 * return linear layout selection map for tab for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param tab - Session tab used by this operation.
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * linearLayoutSelectionMapForTab(ctx, tab);
 */
function linearLayoutSelectionMapForTab(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): LinearLayoutSelectionMap | null {
    const cached = ctx.state.linearLayoutSelectionMaps.get(tab.id);
    if (cached) return cached;
    const mapping = linearLayoutSelectionMapForMeta(tab);
    if (!mapping) return null;
    ctx.state.linearLayoutSelectionMaps.set(tab.id, mapping);
    return mapping;
}
