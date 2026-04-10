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

export type LinearLayoutDisplayModel = {
    rootIndexes: Set<number>;
    sliceRootIndexes: Set<number> | null;
    displayedRootIndexByTensor: Map<string, Array<number | null>>;
    visibleCoordsByTensor: Map<string, number[][]>;
    ghostRootIndexesByTensor: Map<string, Array<{ coord: number[]; rootIndex: number; layer: number }>>;
};

export type LinearLayoutMultiInputModel = {
    focusedTensorId: string;
    value: number;
    size: number;
} | null;

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
        const cellRootIndexes = Array.from({ length: product(tensorMeta.shape) }, () => [] as number[]);
        rootToTensorKeys.forEach((tensorKey, rootIndex) => {
            const flat = flatIndex(coordFromKey(tensorKey), tensorMeta.shape);
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
    if (size <= 1) return null;
    const storedValue = ctx.state.linearLayoutMultiInputState[focusedTensorId] ?? -1;
    const value = storedValue < 0 ? -1 : Math.min(size - 1, storedValue);
    return { focusedTensorId, value, size };
}

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

export function linearLayoutDisplayModel(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap,
): LinearLayoutDisplayModel {
    const sliceVisibleRootIndexes = sliceVisibleRootIndexesByTensor(ctx, mapping);
    const slicedRoots = intersectRootIndexes(sliceVisibleRootIndexes.values(), mapping.rootKeys.length);
    const multiInput = linearLayoutMultiInputModel(ctx, mapping);
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
            roots.slice(1).map((rootIndex, layer) => ({
                coord: unravelIndex(flat, tensor.meta.shape),
                rootIndex,
                layer: layer + 1,
            }))
        )));
    });
    return { rootIndexes, sliceRootIndexes: slicedRoots, displayedRootIndexByTensor, visibleCoordsByTensor, ghostRootIndexesByTensor };
}

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

function sliceVisibleRootIndexesByTensor(
    ctx: LinearLayoutUiContext,
    mapping: LinearLayoutSelectionMap,
): Map<string, Set<number>> {
    return new Map(mapping.orderedTensorIds.map((tensorId) => {
        const coords = slicedTensorCoords(ctx, tensorId);
        return [tensorId, coords ? rootIndexesForCoords(mapping, tensorId, coords) : new Set<number>()] as const;
    }).filter(([_tensorId, roots]) => roots.size > 0));
}

function intersectRootIndexes(sets: Iterable<Set<number>>, rootCount: number): Set<number> | null {
    let intersection: Set<number> | null = null;
    for (const set of sets) {
        intersection = intersection
            ? new Set(Array.from(intersection).filter((rootIndex) => set.has(rootIndex)))
            : new Set(set);
    }
    if (intersection) return intersection;
    return null;
}

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

function flatIndex(coord: number[], shape: number[]): number {
    return coord.reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}

function product(values: number[]): number {
    return values.reduce((total, value) => total * value, 1);
}

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

function linearLayoutGhostText(coord: number[], labels: string[], state: Record<string, boolean>): string | null {
    const text = labels
        .flatMap((label, axis) => (state[label] && axis < coord.length ? [`${label}:${coord[axis] ?? 0}`] : []))
        .join('\n');
    return text || null;
}

function propagatedCoordForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number[] {
    const key = propagateOutputs ? mapping.rootToFinalKeys[rootIndex] : mapping.rootKeys[rootIndex];
    return coordFromKey(key ?? '');
}

function propagatedIndexForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number {
    return flatIndex(
        propagatedCoordForRoot(mapping, rootIndex, propagateOutputs),
        propagateOutputs ? mapping.finalOutputShape : mapping.rootInputShape,
    );
}

function linearLayoutSelectionMapForTab(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): LinearLayoutSelectionMap | null {
    const cached = ctx.state.linearLayoutSelectionMaps.get(tab.id);
    if (cached) return cached;
    const mapping = linearLayoutSelectionMapForMeta(tab);
    if (!mapping) return null;
    ctx.state.linearLayoutSelectionMaps.set(tab.id, mapping);
    return mapping;
}
