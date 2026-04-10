import {
    coordFromKey,
    coordKey,
    type LoadedBundleDocument,
    type SelectionCoords,
    type TensorViewer,
} from '@tensor-viz/viewer-core';
import {
    applyLinearLayoutDisplay,
    coordsForRootIndexes,
    displayedRootIndexForCoord,
    linearLayoutDisplayModel,
    linearLayoutSelectionMapForMeta,
    rootIndexesForCoords,
} from './linear-layout-multi-input.js';
import {
    isLinearLayoutTab,
    snapshotTensorViews,
    type InspectorCoordEntry,
    type LinearLayoutCellTextState,
    type LinearLayoutSelectionMap,
    type LinearLayoutTensorViewsState,
    type LinearLayoutUiContext,
} from './linear-layout-state.js';
import { rootColorsForLayoutState } from './linear-layout.js';

export type LinearLayoutHoverPopupEntry = {
    color: string;
    text: string;
};

export function preservedLinearLayoutTensorViews(
    ctx: LinearLayoutUiContext,
    tabId: string | null = ctx.getActiveTabId(),
): LinearLayoutTensorViewsState {
    const stored = tabId ? ctx.state.linearLayoutTensorViewsStates.get(tabId) ?? {} : {};
    if (!tabId || ctx.getActiveTabId() !== tabId) return { ...stored };
    return { ...stored, ...snapshotTensorViews(ctx.viewer.getSnapshot()) };
}

export function linearLayoutSelectionMapForTab(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): LinearLayoutSelectionMap | null {
    const cached = ctx.state.linearLayoutSelectionMaps.get(tab.id);
    if (cached) return cached;
    const map = linearLayoutSelectionMapForMeta(tab);
    if (!map) return null;
    ctx.state.linearLayoutSelectionMaps.set(tab.id, map);
    return map;
}

export function inspectorCoordEntries(
    _ctx: LinearLayoutUiContext,
    hover: ReturnType<TensorViewer['getHover']>,
    hoveredStatus: ReturnType<TensorViewer['getTensorStatus']>,
    linearLayout: LinearLayoutSelectionMap | null,
): InspectorCoordEntry[] {
    if (!hover) return [];
    if (!linearLayout) {
        return [{
            title: hover.tensorName,
            labels: hoveredStatus?.axisLabels ?? [],
            shape: hoveredStatus?.shape ?? [],
            coord: hover.tensorCoord,
            hovered: true,
        }];
    }
    const display = linearLayoutDisplayModel(_ctx, linearLayout);
    const rootIndex = displayedRootIndexForCoord(display, linearLayout, hover.tensorId, hover.tensorCoord);
    if (rootIndex === null) return [];
    return linearLayout.orderedTensorIds.map((tensorId) => {
        const entry = linearLayout.tensors.get(tensorId)!;
        return {
            title: entry.meta.title,
            labels: entry.meta.axisLabels,
            shape: entry.meta.shape,
            coord: coordFromKey(entry.rootToTensorKeys[rootIndex] ?? ''),
            hovered: tensorId === hover.tensorId,
        };
    });
}

export function applyLinearLayoutCellText(ctx: LinearLayoutUiContext): void {
    const tab = activeLinearLayoutTab(ctx);
    if (!tab) {
        ctx.viewer.getInspectorModel().tensors.forEach((tensor) => ctx.viewer.setTensorCellLabels(tensor.id, null));
        return;
    }
    const labels = linearLayoutCellLabelsForTab(ctx, tab, ctx.state.linearLayoutCellTextState);
    if (!labels) {
        const mapping = linearLayoutSelectionMapForTab(ctx, tab);
        mapping?.orderedTensorIds.forEach((tensorId) => ctx.viewer.setTensorCellLabels(tensorId, null));
        return;
    }
    labels.forEach(({ tensorId, labels: tensorLabels }) => {
        ctx.viewer.setTensorCellLabels(tensorId, tensorLabels);
    });
}

export function linearLayoutHoverPopupEntries(
    ctx: LinearLayoutUiContext,
    hover: ReturnType<TensorViewer['getHover']>,
    linearLayout: LinearLayoutSelectionMap | null,
): LinearLayoutHoverPopupEntry[] {
    if (!hover || !linearLayout) return [];
    if (linearLayout.injective || ctx.state.linearLayoutState.propagateOutputs) return [];
    const tensor = linearLayout.tensors.get(hover.tensorId);
    if (!tensor) return [];
    const flat = tensor.coordKeyToFlatIndex.get(coordKey(hover.tensorCoord));
    if (flat === undefined) return [];
    const rootColors = rootColorsForLayoutState(
        ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputLabels : linearLayout.rootInputLabels,
        ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputShape : linearLayout.rootInputShape,
        ctx.state.linearLayoutState,
    );
    return (tensor.cellRootIndexes[flat] ?? []).map((rootIndex) => {
        const coord = propagatedCoordForRoot(linearLayout, rootIndex, ctx.state.linearLayoutState.propagateOutputs);
        return {
            color: cssColor(rootColors[propagatedIndexForRoot(linearLayout, rootIndex, ctx.state.linearLayoutState.propagateOutputs)] ?? [0, 0, 0]),
            text: linearLayoutCellTextForCoord(
                coord,
                ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputLabels : linearLayout.rootInputLabels,
                ctx.state.linearLayoutCellTextState,
            ) || (
                ctx.state.linearLayoutState.propagateOutputs ? linearLayout.finalOutputLabels : linearLayout.rootInputLabels
            ).map((label, axis) => indexedAxisLabel(label, coord[axis] ?? 0)).join('\n'),
        };
    });
}

export function syncLinearLayoutViewFilters(ctx: LinearLayoutUiContext): void {
    const tab = ctx.getActiveTab();
    if (!tab || !isLinearLayoutTab(tab)) return;
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping) return;
    applyLinearLayoutDisplay(ctx);
    applyLinearLayoutCellText(ctx);
}

export function syncLinearLayoutSelection(ctx: LinearLayoutUiContext, selection: SelectionCoords): void {
    if (ctx.state.syncingLinearLayoutSelection) return;
    const tab = ctx.getActiveTab();
    if (!tab || !isLinearLayoutTab(tab)) return;
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping) return;
    const nextSelection = mappedSelectionFromSource(ctx, selection, mapping);
    if (nextSelection.size === 0 || selectionsMatch(selection, nextSelection)) return;
    ctx.state.syncingLinearLayoutSelection = true;
    ctx.viewer.setSelectedCoords(nextSelection);
    ctx.state.syncingLinearLayoutSelection = false;
}

export function syncLinearLayoutSelectionPreview(ctx: LinearLayoutUiContext, selection: SelectionCoords): void {
    const tab = ctx.getActiveTab();
    if (!tab || !isLinearLayoutTab(tab)) {
        ctx.viewer.setPreviewSelectedCoords(selection);
        return;
    }
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    ctx.viewer.setPreviewSelectedCoords(mapping ? mappedSelectionFromSource(ctx, selection, mapping) : selection);
}

function activeLinearLayoutTab(ctx: LinearLayoutUiContext): LoadedBundleDocument | null {
    const tab = ctx.getActiveTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

function selectionKeys(coords: number[][]): Set<string> {
    return new Set(coords.map((coord) => coordKey(coord)));
}

function selectionsMatch(left: SelectionCoords, right: Map<string, number[][]>): boolean {
    if (left.size !== right.size) return false;
    for (const [tensorId, coords] of right) {
        const leftCoords = left.get(tensorId);
        if (!leftCoords) return false;
        const leftKeys = selectionKeys(leftCoords);
        const rightKeys = selectionKeys(coords);
        if (leftKeys.size !== rightKeys.size) return false;
        for (const key of rightKeys) {
            if (!leftKeys.has(key)) return false;
        }
    }
    return true;
}

function selectionSourceTensorId(ctx: LinearLayoutUiContext, selection: SelectionCoords, mapping: LinearLayoutSelectionMap): string | null {
    const nonEmpty = mapping.orderedTensorIds.filter((tensorId) => (selection.get(tensorId)?.length ?? 0) > 0);
    if (nonEmpty.length === 0) return null;
    if (nonEmpty.length === 1) return nonEmpty[0]!;
    const activeId = ctx.viewer.getState().activeTensorId;
    return activeId && nonEmpty.includes(activeId) ? activeId : nonEmpty[0]!;
}

function mappedSelectionFromSource(
    ctx: LinearLayoutUiContext,
    selection: SelectionCoords,
    mapping: LinearLayoutSelectionMap,
): SelectionCoords {
    const sourceTensorId = selectionSourceTensorId(ctx, selection, mapping);
    if (!sourceTensorId) return new Map();
    const sourceCoords = selection.get(sourceTensorId) ?? [];
    if (sourceCoords.length === 0) return new Map();
    const display = linearLayoutDisplayModel(ctx, mapping);
    const rootIndexes = rootIndexesForCoords(mapping, sourceTensorId, sourceCoords);
    const nextSelection = new Map<string, number[][]>();
    mapping.orderedTensorIds.forEach((tensorId) => {
        const coords = coordsForRootIndexes(mapping, tensorId, rootIndexes, display.sliceRootIndexes);
        if (coords.length) nextSelection.set(tensorId, coords);
    });
    return nextSelection;
}

function linearLayoutCellTextForCoord(coord: number[], labels: string[], state: LinearLayoutCellTextState): string {
    return labels
        .flatMap((label, axis) => (state[label] && axis < coord.length ? [indexedAxisLabel(label, coord[axis] ?? 0)] : []))
        .join('\n');
}

function indexedAxisLabel(label: string, index: number): string {
    return `${label}:${index}`;
}

function cssColor(color: [number, number, number]): string {
    return `rgb(${color.map((value) => Math.round(value * 255)).join(' ')})`;
}

function linearLayoutCellLabelsForTab(
    ctx: LinearLayoutUiContext,
    tab: LoadedBundleDocument,
    state: LinearLayoutCellTextState,
): Array<{ tensorId: string; labels: Array<{ coord: number[]; text: string }> }> | null {
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    const labels = ctx.state.linearLayoutState.propagateOutputs ? mapping?.finalOutputLabels : mapping?.rootInputLabels;
    if (!mapping || !labels?.some((label) => state[label])) return null;
    const display = linearLayoutDisplayModel(ctx, mapping);
    return mapping.orderedTensorIds.map((tensorId) => {
        const tensor = mapping.tensors.get(tensorId)!;
        const labels = tensor.cellRootIndexes.map((roots, flat) => {
            const rootIndex = display.displayedRootIndexByTensor.get(tensorId)?.[flat] ?? roots[0] ?? null;
            if (rootIndex === null) return null;
            return {
                coord: coordFromKey(tensor.rootToTensorKeys[rootIndex] ?? ''),
                text: linearLayoutCellTextForCoord(
                    propagatedCoordForRoot(mapping, rootIndex, ctx.state.linearLayoutState.propagateOutputs),
                    ctx.state.linearLayoutState.propagateOutputs ? mapping.finalOutputLabels : mapping.rootInputLabels,
                    state,
                ),
            };
        }).filter((entry): entry is { coord: number[]; text: string } => entry !== null);
        return { tensorId, labels };
    });
}

function propagatedCoordForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number[] {
    const key = propagateOutputs ? mapping.rootToFinalKeys[rootIndex] : mapping.rootKeys[rootIndex];
    return coordFromKey(key ?? '');
}

function propagatedIndexForRoot(mapping: LinearLayoutSelectionMap, rootIndex: number, propagateOutputs: boolean): number {
    const coord = propagatedCoordForRoot(mapping, rootIndex, propagateOutputs);
    const shape = propagateOutputs ? mapping.finalOutputShape : mapping.rootInputShape;
    return coord.reduce((index, value, axis) => (index * shape[axis]!) + value, 0);
}
