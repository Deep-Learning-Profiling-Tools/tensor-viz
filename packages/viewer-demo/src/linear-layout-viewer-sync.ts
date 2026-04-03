import {
    coordFromKey,
    coordKey,
    parseTensorView,
    product,
    unravelIndex,
    type LoadedBundleDocument,
    type SelectionCoords,
    type TensorViewer,
} from '@tensor-viz/viewer-core';
import {
    composeLayoutMetaForTab,
    isLinearLayoutTab,
    snapshotTensorViews,
    type InspectorCoordEntry,
    type LinearLayoutCellTextState,
    type LinearLayoutSelectionMap,
    type LinearLayoutTensorViewsState,
    type LinearLayoutUiContext,
} from './linear-layout-state.js';

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
    const meta = composeLayoutMetaForTab(tab);
    if (!meta || meta.tensors.length === 0) return null;
    const rootInputShape = meta.rootInputBitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
    const rootKeys = meta.tensors[0]!.rootToTensor.map((coord) => coordKey(coord));
    const loadedTensorIds = new Set(tab.manifest.tensors.map((tensor) => tensor.id));
    const tensors = new Map<string, LinearLayoutSelectionMap['tensors'] extends Map<string, infer T> ? T : never>();
    meta.tensors.forEach((tensorMeta) => {
        if (!loadedTensorIds.has(tensorMeta.id)) return;
        const rootToTensorKeys = tensorMeta.rootToTensor.map((coord) => coordKey(coord));
        const tensorToRoot = new Map<string, string>();
        rootToTensorKeys.forEach((tensorKey, rootIndex) => {
            tensorToRoot.set(tensorKey, rootKeys[rootIndex]!);
        });
        tensors.set(tensorMeta.id, { meta: tensorMeta, rootToTensorKeys, tensorToRoot });
    });
    const map = {
        rootInputLabels: meta.rootInputLabels.slice(),
        rootInputShape,
        rootKeyToIndex: new Map(rootKeys.map((key, index) => [key, index])),
        tensors,
        orderedTensorIds: meta.tensors.map((tensor) => tensor.id).filter((id) => tensors.has(id)),
    };
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
    const rootKey = linearLayout.tensors.get(hover.tensorId)?.tensorToRoot.get(coordKey(hover.tensorCoord));
    if (!rootKey) return [];
    const rootIndex = linearLayout.rootKeyToIndex.get(rootKey);
    if (rootIndex === undefined) return [];
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

export function syncLinearLayoutViewFilters(ctx: LinearLayoutUiContext): void {
    const tab = ctx.getActiveTab();
    if (!tab || !isLinearLayoutTab(tab)) return;
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping) return;
    let allowedRootKeys: Set<string> | null = null;
    mapping.orderedTensorIds.forEach((tensorId) => {
        const coords = slicedTensorCoords(ctx, tensorId);
        if (!coords) return;
        const tensor = mapping.tensors.get(tensorId)!;
        const rootKeys = rootKeysForCoords(coords, tensor);
        allowedRootKeys = allowedRootKeys
            ? new Set(Array.from(allowedRootKeys).filter((key) => rootKeys.has(key)))
            : rootKeys;
    });
    mapping.orderedTensorIds.forEach((tensorId) => {
        ctx.viewer.setTensorVisibleCoords(
            tensorId,
            allowedRootKeys ? coordsForRootKeys(allowedRootKeys, mapping, tensorId) : null,
        );
    });
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

function rootKeysForCoords(coords: number[][], tensor: { tensorToRoot: Map<string, string> }): Set<string> {
    return new Set(coords.flatMap((coord) => {
        const rootKey = tensor.tensorToRoot.get(coordKey(coord));
        return rootKey ? [rootKey] : [];
    }));
}

function coordsForRootKeys(rootKeys: Set<string>, mapping: LinearLayoutSelectionMap, tensorId: string): number[][] {
    const tensor = mapping.tensors.get(tensorId);
    if (!tensor || rootKeys.size === 0) return [];
    return Array.from(rootKeys, (rootKey) => {
        const rootIndex = mapping.rootKeyToIndex.get(rootKey);
        return rootIndex === undefined ? null : coordFromKey(tensor.rootToTensorKeys[rootIndex]!);
    }).filter((coord): coord is number[] => coord !== null);
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
    const sourceTensor = mapping.tensors.get(sourceTensorId);
    const sourceCoords = selection.get(sourceTensorId) ?? [];
    if (!sourceTensor || sourceCoords.length === 0) return new Map();
    const rootKeys = rootKeysForCoords(sourceCoords, sourceTensor);
    const nextSelection = new Map<string, number[][]>();
    mapping.orderedTensorIds.forEach((tensorId) => {
        const coords = coordsForRootKeys(rootKeys, mapping, tensorId);
        if (coords.length) nextSelection.set(tensorId, coords);
    });
    return nextSelection;
}

function linearLayoutCellTextForCoord(coord: number[], labels: string[], state: LinearLayoutCellTextState): string {
    return labels
        .flatMap((label, axis) => (state[label] && axis < coord.length ? [`${label}${coord[axis] ?? 0}`] : []))
        .join('\n');
}

function linearLayoutCellLabelsForTab(
    ctx: LinearLayoutUiContext,
    tab: LoadedBundleDocument,
    state: LinearLayoutCellTextState,
): Array<{ tensorId: string; labels: Array<{ coord: number[]; text: string }> }> | null {
    const mapping = linearLayoutSelectionMapForTab(ctx, tab);
    if (!mapping || !mapping.rootInputLabels.some((label) => state[label])) return null;
    return mapping.orderedTensorIds.map((tensorId) => {
        const tensor = mapping.tensors.get(tensorId)!;
        const labels = tensor.rootToTensorKeys.map((tensorKey) => {
            const rootKey = tensor.tensorToRoot.get(tensorKey);
            if (!rootKey) return null;
            return {
                coord: coordFromKey(tensorKey),
                text: linearLayoutCellTextForCoord(coordFromKey(rootKey), mapping.rootInputLabels, state),
            };
        }).filter((entry): entry is { coord: number[]; text: string } => entry !== null);
        return { tensorId, labels };
    });
}

function slicedTensorCoords(ctx: LinearLayoutUiContext, tensorId: string): number[][] | null {
    const status = ctx.viewer.getTensorStatus(tensorId);
    const snapshot = ctx.viewer.getTensorView(tensorId);
    const parsed = parseTensorView(status.shape.slice(), snapshot.view, snapshot.hiddenIndices, status.axisLabels);
    if (!parsed.ok || parsed.spec.sliceTokens.length === 0) return null;
    const hiddenAxes = parsed.spec.tokens
        .filter((token) => token.kind === 'axis_group' && !token.visible)
        .flatMap((token) => token.axes);
    if (hiddenAxes.length === 0) return null;
    const coords: number[][] = [];
    const total = product(status.shape);
    for (let index = 0; index < total; index += 1) {
        const coord = unravelIndex(index, status.shape);
        if (hiddenAxes.every((axis) => coord[axis] === parsed.spec.hiddenIndices[axis])) coords.push(coord);
    }
    return coords;
}
