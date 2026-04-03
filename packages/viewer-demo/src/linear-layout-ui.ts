import {
    coordFromKey,
    coordKey,
    parseTensorView,
    product,
    unravelIndex,
    type LoadedBundleDocument,
    type SelectionCoords,
    type TensorViewSnapshot,
    type ViewerSnapshot,
    type TensorViewer,
} from '@tensor-viz/viewer-core';
import { escapeInfo, labelWithInfo, titleWithInfo } from './app-format.js';
import {
    bakedComposeLayoutExamples,
    buildComposeRuntime,
    cloneComposeLayoutState,
    composeLayoutStateFromLegacySpec,
    createComposeLayoutDocument,
    defaultComposeLayoutState,
    emptyComposeLayoutState,
    isComposeLayoutMeta,
    isComposeLayoutState,
    type ComposeLayoutMeta,
    type ComposeLayoutState,
    type ComposeTensorMeta,
    type MatrixBlock,
} from './linear-layout.js';

export type InspectorCoordEntry = {
    title: string;
    labels: string[];
    shape: number[];
    coord: number[] | null;
    hovered: boolean;
};

export type LinearLayoutNotice = {
    tone: 'error' | 'success';
    text: string;
};

export type LinearLayoutCellTextState = Record<string, boolean>;
export type LinearLayoutTensorViewsState = Record<string, TensorViewSnapshot>;
export type LinearLayoutFormState = ComposeLayoutState;
export type LinearLayoutChannel = 'H' | 'S' | 'L';

export type LinearLayoutSelectionMap = {
    rootInputLabels: string[];
    rootInputShape: number[];
    rootKeyToIndex: Map<string, number>;
    tensors: Map<string, {
        meta: ComposeTensorMeta;
        rootToTensorKeys: string[];
        tensorToRoot: Map<string, string>;
    }>;
    orderedTensorIds: string[];
};

export type LinearLayoutUiState = {
    linearLayoutState: LinearLayoutFormState;
    linearLayoutStates: Map<string, LinearLayoutFormState>;
    linearLayoutCellTextState: LinearLayoutCellTextState;
    linearLayoutCellTextStates: Map<string, LinearLayoutCellTextState>;
    linearLayoutTensorViewsStates: Map<string, LinearLayoutTensorViewsState>;
    linearLayoutSelectionMaps: Map<string, LinearLayoutSelectionMap>;
    linearLayoutNotice: LinearLayoutNotice | null;
    linearLayoutMatrixPreview: string;
    showLinearLayoutMatrix: boolean;
    syncingLinearLayoutSelection: boolean;
};

export type LinearLayoutUiContext = {
    viewer: TensorViewer;
    viewport: HTMLElement;
    linearLayoutWidget: HTMLElement;
    cellTextWidget: HTMLElement;
    linearLayoutColorWidget: HTMLElement;
    state: LinearLayoutUiState;
    getActiveTab: () => LoadedBundleDocument | undefined;
    getActiveTabId: () => string | null;
    getSessionTabs: () => LoadedBundleDocument[];
    setSessionTabs: (tabs: LoadedBundleDocument[]) => void;
    loadTab: (id: string) => Promise<void>;
};

const LINEAR_LAYOUT_CHANNELS: LinearLayoutChannel[] = ['H', 'S', 'L'];
const LINEAR_LAYOUT_STORAGE_KEY = 'tensor-viz-linear-layout-spec';

export function loadLinearLayoutState(): LinearLayoutFormState {
    const fallback = defaultLinearLayoutState();
    try {
        const stored = window.localStorage.getItem(LINEAR_LAYOUT_STORAGE_KEY);
        if (!stored) return fallback;
        const parsed = JSON.parse(stored);
        if (isLinearLayoutState(parsed)) return cloneLinearLayoutState(parsed);
        if (parsed && typeof parsed === 'object' && ('specsText' in parsed || 'operationText' in parsed)) {
            return {
                ...fallback,
                ...(parsed as Partial<LinearLayoutFormState>),
            };
        }
        if (parsed && typeof parsed === 'object' && (parsed.basesText || parsed.bases)) {
            return legacyEditorState(parsed as Record<string, unknown>, fallback);
        }
        if (parsed && typeof parsed === 'object' && (parsed.input_dims || parsed.bases)) {
            return composeLayoutStateFromLegacySpec(parsed, 'Layout_1');
        }
    } catch {
        return fallback;
    }
    return fallback;
}

export function storeLinearLayoutState(state: LinearLayoutFormState): void {
    try {
        window.localStorage.setItem(LINEAR_LAYOUT_STORAGE_KEY, JSON.stringify(state));
    } catch {
        // ignore storage failures in restricted browsers
    }
}

export function isLinearLayoutState(value: unknown): value is LinearLayoutFormState {
    return isComposeLayoutState(value);
}

export function defaultLinearLayoutState(): LinearLayoutFormState {
    return defaultComposeLayoutState();
}

export function emptyLinearLayoutState(): LinearLayoutFormState {
    return emptyComposeLayoutState();
}

export function cloneLinearLayoutState(state: LinearLayoutFormState): LinearLayoutFormState {
    return cloneComposeLayoutState(state);
}

export function defaultLinearLayoutCellTextState(): LinearLayoutCellTextState {
    return {};
}

export function cloneLinearLayoutCellTextState(state: LinearLayoutCellTextState): LinearLayoutCellTextState {
    return { ...state };
}

export function cloneLinearLayoutTensorViewsState(state: LinearLayoutTensorViewsState): LinearLayoutTensorViewsState {
    return Object.fromEntries(Object.entries(state).map(([tensorId, view]) => [
        tensorId,
        {
            view: view.view,
            hiddenIndices: view.hiddenIndices.slice(),
        },
    ]));
}

export function isLinearLayoutCellTextState(value: unknown): value is LinearLayoutCellTextState {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => typeof entry === 'boolean');
}

export function snapshotTensorViews(snapshot: ViewerSnapshot): LinearLayoutTensorViewsState {
    return Object.fromEntries(snapshot.tensors.map((tensor) => [
        tensor.id,
        {
            view: tensor.view.view,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        },
    ]));
}

export function preservedLinearLayoutTensorViews(
    ctx: LinearLayoutUiContext,
    tabId: string | null = ctx.getActiveTabId(),
): LinearLayoutTensorViewsState {
    const stored = tabId ? cloneLinearLayoutTensorViewsState(ctx.state.linearLayoutTensorViewsStates.get(tabId) ?? {}) : {};
    if (!tabId || ctx.getActiveTabId() !== tabId) return stored;
    return {
        ...stored,
        ...snapshotTensorViews(ctx.viewer.getSnapshot()),
    };
}

function legacyEditorState(raw: Record<string, unknown>, fallback: LinearLayoutFormState): LinearLayoutFormState {
    const textByLabel = new Map<string, string>([['T', '[]'], ['W', '[]'], ['R', '[]']]);
    if (typeof raw.basesText === 'string') {
        raw.basesText.split('\n').forEach((line) => {
            const match = line.trim().match(/^([TWR])\s*:\s*(.+)$/i);
            if (!match) return;
            textByLabel.set(match[1]!.toUpperCase(), match[2]!.trim());
        });
    }
    if (raw.bases && typeof raw.bases === 'object') {
        const bases = raw.bases as Record<string, unknown>;
        if (typeof bases.thread === 'string') textByLabel.set('T', bases.thread);
        if (typeof bases.warp === 'string') textByLabel.set('W', bases.warp);
        if (typeof bases.register === 'string') textByLabel.set('R', bases.register);
    }
    const rows = ['T', 'W', 'R'].map((label) => parseBasesField(label, textByLabel.get(label) ?? '[]'));
    const outputRank = Math.max(1, ...rows.flatMap((entry) => entry.map((basis) => basis.length)));
    const outputs = Array.from({ length: outputRank }, (_entry, axis) => String.fromCharCode(65 + axis));
    const mapping = { ...fallback.mapping };
    if (raw.mapping && typeof raw.mapping === 'object') {
        Object.entries(raw.mapping as Record<string, unknown>).forEach(([channel, axisName]) => {
            const normalized = String(channel).toUpperCase() as LinearLayoutChannel;
            if (!LINEAR_LAYOUT_CHANNELS.includes(normalized) || typeof axisName !== 'string') return;
            mapping[normalized] = axisName === 'thread' ? 'T' : axisName === 'warp' ? 'W' : axisName === 'register' ? 'R' : 'none';
        });
    }
    const ranges = {
        H: [...fallback.ranges.H],
        S: [...fallback.ranges.S],
        L: [...fallback.ranges.L],
    } as Record<LinearLayoutChannel, [string, string]>;
    if (raw.ranges && typeof raw.ranges === 'object') {
        Object.entries(raw.ranges as Record<string, unknown>).forEach(([channel, range]) => {
            const normalized = String(channel).toUpperCase() as LinearLayoutChannel;
            if (!LINEAR_LAYOUT_CHANNELS.includes(normalized) || !Array.isArray(range) || range.length !== 2) return;
            ranges[normalized] = [String(range[0]), String(range[1])];
        });
    }
    return {
        specsText: [
            `Layout_1: [T,W,R] -> [${outputs.join(',')}]`,
            ...rows.map((row) => JSON.stringify(row)),
        ].join('\n'),
        operationText: 'Layout_1',
        inputName: fallback.inputName,
        visibleTensors: {},
        mapping,
        ranges,
    };
}

export function composeLayoutMetaForTab(tab: LoadedBundleDocument): ComposeLayoutMeta | null {
    const candidate = (tab.manifest.viewer as { composeLayoutMeta?: unknown }).composeLayoutMeta;
    if (!isComposeLayoutMeta(candidate)) return null;
    return candidate;
}

export function isLinearLayoutTab(tab: LoadedBundleDocument): boolean {
    return composeLayoutMetaForTab(tab) !== null;
}

export function syncLinearLayoutState(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) return;
    const stored = ctx.state.linearLayoutStates.get(tab.id);
    if (stored) {
        ctx.state.linearLayoutState = cloneLinearLayoutState(stored);
        refreshLinearLayoutMatrixPreview(ctx);
        renderLinearLayoutWidget(ctx);
        return;
    }
    const candidate = (tab.manifest.viewer as { composeLayoutState?: unknown }).composeLayoutState;
    if (isComposeLayoutState(candidate)) {
        ctx.state.linearLayoutState = cloneLinearLayoutState(candidate);
        ctx.state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(candidate));
        refreshLinearLayoutMatrixPreview(ctx);
        renderLinearLayoutWidget(ctx);
        return;
    }
    const meta = composeLayoutMetaForTab(tab);
    ctx.state.linearLayoutState = meta
        ? {
            ...defaultLinearLayoutState(),
            specsText: meta.specsText,
            operationText: meta.operationText,
            inputName: meta.inputName ?? defaultLinearLayoutState().inputName,
            visibleTensors: Object.fromEntries(meta.tensors.map((tensor) => [tensor.id, tensor.visible])),
        }
        : defaultLinearLayoutState();
    ctx.state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(ctx.state.linearLayoutState));
    refreshLinearLayoutMatrixPreview(ctx);
    renderLinearLayoutWidget(ctx);
}

export function syncLinearLayoutCellTextState(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) {
        ctx.state.linearLayoutCellTextState = defaultLinearLayoutCellTextState();
        return;
    }
    const stored = ctx.state.linearLayoutCellTextStates.get(tab.id);
    if (stored) {
        ctx.state.linearLayoutCellTextState = cloneLinearLayoutCellTextState(stored);
        return;
    }
    const candidate = (tab.manifest.viewer as { linearLayoutCellTextState?: unknown }).linearLayoutCellTextState;
    if (isLinearLayoutCellTextState(candidate)) {
        ctx.state.linearLayoutCellTextState = cloneLinearLayoutCellTextState(candidate);
        ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(candidate));
        return;
    }
    ctx.state.linearLayoutCellTextState = defaultLinearLayoutCellTextState();
    ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
}

function normalizeCellTextState(state: LinearLayoutCellTextState, labels: string[]): LinearLayoutCellTextState {
    return Object.fromEntries(labels.map((label) => [label, state[label] ?? false]));
}

export function linearLayoutSelectionMapForTab(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): LinearLayoutSelectionMap | null {
    const cached = ctx.state.linearLayoutSelectionMaps.get(tab.id);
    if (cached) return cached;
    const meta = composeLayoutMetaForTab(tab);
    if (!meta || meta.tensors.length === 0) return null;
    const rootInputShape = meta.rootInputBitCounts.map((bits) => bits === 0 ? 1 : 2 ** bits);
    const rootKeys = meta.tensors[0]!.rootToTensor.map((coord) => coordKey(coord));
    const loadedTensorIds = new Set(tab.manifest.tensors.map((tensor) => tensor.id));
    const tensors = new Map<string, {
        meta: ComposeTensorMeta;
        rootToTensorKeys: string[];
        tensorToRoot: Map<string, string>;
    }>();
    meta.tensors.forEach((tensorMeta) => {
        if (!loadedTensorIds.has(tensorMeta.id)) return;
        const rootToTensorKeys = tensorMeta.rootToTensor.map((coord) => coordKey(coord));
        const tensorToRoot = new Map<string, string>();
        rootToTensorKeys.forEach((tensorKey, rootIndex) => {
            tensorToRoot.set(tensorKey, rootKeys[rootIndex]!);
        });
        tensors.set(tensorMeta.id, {
            meta: tensorMeta,
            rootToTensorKeys,
            tensorToRoot,
        });
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

export function inspectorCoordEntries(
    ctx: LinearLayoutUiContext,
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

function activeLinearLayoutTab(ctx: LinearLayoutUiContext): LoadedBundleDocument | null {
    const tab = ctx.getActiveTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

function clearCellTextLabels(ctx: LinearLayoutUiContext): void {
    ctx.viewer.getInspectorModel().tensors.forEach((tensor) => {
        ctx.viewer.setTensorCellLabels(tensor.id, null);
    });
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

export function applyLinearLayoutCellText(ctx: LinearLayoutUiContext): void {
    const tab = activeLinearLayoutTab(ctx);
    if (!tab) {
        clearCellTextLabels(ctx);
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

function slicedTensorCoords(ctx: LinearLayoutUiContext, tensorId: string): number[][] | null {
    const status = ctx.viewer.getTensorStatus(tensorId);
    const snapshot = ctx.viewer.getTensorView(tensorId);
    const parsed = parseTensorView(
        status.shape.slice(),
        snapshot.view,
        snapshot.hiddenIndices,
        status.axisLabels,
    );
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
    if (nextSelection.size === 0) return;
    if (selectionsMatch(selection, nextSelection)) return;
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
    if (!mapping) {
        ctx.viewer.setPreviewSelectedCoords(selection);
        return;
    }
    ctx.viewer.setPreviewSelectedCoords(mappedSelectionFromSource(ctx, selection, mapping));
}

function parseBasesField(label: string, value: string): number[][] {
    if (!value.trim()) return [];
    let parsed: unknown;
    try {
        parsed = JSON.parse(value);
    } catch (error) {
        throw new Error(`${label} bases must be valid JSON.`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${label} bases must be a JSON array.`);
    return parsed.map((basis, index) => {
        if (!Array.isArray(basis)) throw new Error(`${label} basis ${index + 1} must be an array.`);
        return basis.map((entry, axis) => {
            if (typeof entry !== 'number' || Number.isNaN(entry)) {
                throw new Error(`${label} basis ${index + 1}[${axis + 1}] must be a number.`);
            }
            return entry;
        });
    });
}

function matrixAxisColorClass(axis: number): string {
    return `matrix-axis-${axis % 3}`;
}

function matrixPreviewFromBlocks(blocks: MatrixBlock[]): string {
    return blocks.map((block) => {
        const labelWidth = Math.max(1, ...block.rows.map((row) => row.label.length));
        const columnWidths = block.columns.map((column) => Math.max(1, column.label.length));
        const header = block.columns.length === 0
            ? '<span class="matrix-zero">0</span>'
            : `${' '.repeat(labelWidth)} | ${block.columns.map((column, index) => (
                `<span class="matrix-label ${matrixAxisColorClass(column.axis)}">${escapeInfo(column.label.padStart(columnWidths[index] ?? 1))}</span>`
            )).join(' ')}`;
        const rows = block.rows.length === 0
            ? []
            : block.rows.map((row, rowIndex) => (
                `<span class="matrix-label ${matrixAxisColorClass(row.axis)}">${escapeInfo(row.label.padStart(labelWidth))}</span> | ${block.columns.map((_column, columnIndex) => {
                    const value = block.values[rowIndex]?.[columnIndex] === 1 ? '1' : '0';
                    const klass = value === '1' ? 'matrix-one' : 'matrix-zero';
                    return `<span class="${klass}">${value.padStart(columnWidths[columnIndex] ?? 1)}</span>`;
                }).join(' ')}`
            ));
        return [
            '<div class="matrix-block">',
            `<div class="matrix-block-title">${escapeInfo(block.title)}</div>`,
            '<div class="matrix-block-body">',
            header,
            ...rows,
            '</div>',
            '</div>',
        ].join('\n');
    }).join('\n');
}

async function copyText(text: string): Promise<void> {
    if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return;
    }
    const input = document.createElement('textarea');
    input.value = text;
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.select();
    document.execCommand('copy');
    document.body.removeChild(input);
}

export function refreshLinearLayoutMatrixPreview(ctx: LinearLayoutUiContext, state = ctx.state.linearLayoutState): void {
    try {
        ctx.state.linearLayoutMatrixPreview = matrixPreviewFromBlocks(buildComposeRuntime(state).matrixBlocks);
    } catch {
        ctx.state.linearLayoutMatrixPreview = '';
    }
}

function linearLayoutRootLabels(ctx: LinearLayoutUiContext): string[] {
    const tab = activeLinearLayoutTab(ctx);
    const meta = tab ? composeLayoutMetaForTab(tab) : null;
    if (meta) return meta.rootInputLabels.slice();
    try {
        return buildComposeRuntime(ctx.state.linearLayoutState).inputLabels;
    } catch {
        return [];
    }
}

function renderLinearLayoutColorWidget(ctx: LinearLayoutUiContext): void {
    const activeElement = document.activeElement;
    const focusedInput = activeElement instanceof HTMLInputElement && ctx.linearLayoutColorWidget.contains(activeElement)
        ? {
            id: activeElement.id,
            start: activeElement.selectionStart,
            end: activeElement.selectionEnd,
        }
        : null;
    const channelLabels: Record<LinearLayoutChannel, string> = { H: 'Hue', S: 'Sat', L: 'Light' };
    const rootLabels = linearLayoutRootLabels(ctx);
    const assignedLabels = new Set(
        LINEAR_LAYOUT_CHANNELS
            .map((channel) => ctx.state.linearLayoutState.mapping[channel])
            .filter((label): label is string => label !== 'none' && rootLabels.includes(label)),
    );
    const availableLabels = rootLabels.filter((label) => !assignedLabels.has(label));
    ctx.linearLayoutColorWidget.innerHTML = `
      ${titleWithInfo('Color Mapping', 'Select which root-input axis drives each color channel and set channel ranges.')}
      <div class="widget-body">
        <div class="field">
          ${labelWithInfo('Available Axes', 'Drag one root-input axis onto H, S, or L. Drag a colored axis back here to clear that channel.')}
          <div class="mapping-pool mapping-drop-zone" data-pool="true">
            ${availableLabels.map((label) => `
              <button class="mapping-chip" type="button" draggable="true" data-axis="${label}">${label}</button>
            `).join('')}
            ${availableLabels.length === 0 ? '<span class="mapping-empty">all axes assigned</span>' : ''}
          </div>
        </div>
        ${LINEAR_LAYOUT_CHANNELS.map((channel) => `
          <div class="inline-row mapping-row">
            <span class="range-label">${channelLabels[channel]}</span>
            <div class="mapping-drop-zone" data-channel="${channel}">
              ${ctx.state.linearLayoutState.mapping[channel] !== 'none' && rootLabels.includes(ctx.state.linearLayoutState.mapping[channel] as string)
        ? `<button class="mapping-chip mapping-chip-assigned" type="button" draggable="true" data-channel="${channel}" data-axis="${ctx.state.linearLayoutState.mapping[channel]}">
                ${ctx.state.linearLayoutState.mapping[channel]}
              </button>`
        : '<span class="mapping-empty">none</span>'}
            </div>
            <input id="linear-layout-${channel.toLowerCase()}-min" type="number" step="0.01" value="${escapeInfo(ctx.state.linearLayoutState.ranges[channel][0])}" />
            <span class="range-separator">to</span>
            <input id="linear-layout-${channel.toLowerCase()}-max" type="number" step="0.01" value="${escapeInfo(ctx.state.linearLayoutState.ranges[channel][1])}" />
          </div>
        `).join('')}
        <div class="button-row">
          <button class="primary-button" id="linear-layout-recolor" type="button">Recolor Layout</button>
        </div>
      </div>
    `;

    const writeDragPayload = (event: DragEvent, payload: Record<string, string>): void => {
        event.dataTransfer?.setData('application/x-linear-layout-mapping', JSON.stringify(payload));
        event.dataTransfer?.setData('text/plain', JSON.stringify(payload));
        if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
    };
    const readDragPayload = (event: DragEvent): Record<string, string> | null => {
        const raw = event.dataTransfer?.getData('application/x-linear-layout-mapping')
            || event.dataTransfer?.getData('text/plain');
        if (!raw) return null;
        try {
            return JSON.parse(raw) as Record<string, string>;
        } catch {
            return null;
        }
    };
    ctx.linearLayoutColorWidget.querySelectorAll<HTMLElement>('[draggable="true"]').forEach((element) => {
        element.addEventListener('dragstart', (event) => {
            const channel = element.dataset.channel;
            const axis = element.dataset.axis;
            if (!axis) return;
            writeDragPayload(event, channel ? { kind: 'channel', channel, axis } : { kind: 'axis', axis });
        });
    });
    ctx.linearLayoutColorWidget.querySelectorAll<HTMLElement>('.mapping-drop-zone').forEach((element) => {
        element.addEventListener('dragover', (event) => {
            event.preventDefault();
            element.classList.add('drag-over');
        });
        element.addEventListener('dragleave', () => {
            element.classList.remove('drag-over');
        });
        element.addEventListener('drop', (event) => {
            event.preventDefault();
            element.classList.remove('drag-over');
            const payload = readDragPayload(event);
            const targetChannel = element.dataset.channel as LinearLayoutChannel | undefined;
            if (!payload) return;
            if (element.dataset.pool === 'true') {
                if (payload.kind !== 'channel') return;
                const sourceChannel = payload.channel as LinearLayoutChannel;
                if (!sourceChannel) return;
                ctx.state.linearLayoutState.mapping[sourceChannel] = 'none';
                renderLinearLayoutColorWidget(ctx);
                return;
            }
            if (!targetChannel) return;
            if (payload.kind === 'channel') {
                const sourceChannel = payload.channel as LinearLayoutChannel;
                if (!sourceChannel || sourceChannel === targetChannel) return;
                const sourceAxis = ctx.state.linearLayoutState.mapping[sourceChannel];
                ctx.state.linearLayoutState.mapping[sourceChannel] = ctx.state.linearLayoutState.mapping[targetChannel];
                ctx.state.linearLayoutState.mapping[targetChannel] = sourceAxis;
            } else {
                const axis = payload.axis;
                if (!axis) return;
                LINEAR_LAYOUT_CHANNELS.forEach((channel) => {
                    if (ctx.state.linearLayoutState.mapping[channel] === axis) ctx.state.linearLayoutState.mapping[channel] = 'none';
                });
                ctx.state.linearLayoutState.mapping[targetChannel] = axis;
            }
            renderLinearLayoutColorWidget(ctx);
        });
    });
    const hueMinInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-h-min');
    const hueMaxInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-h-max');
    const saturationMinInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-s-min');
    const saturationMaxInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-s-max');
    const lightnessMinInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-l-min');
    const lightnessMaxInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-l-max');
    const recolorButton = ctx.linearLayoutColorWidget.querySelector<HTMLButtonElement>('#linear-layout-recolor');
    hueMinInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.ranges.H[0] = hueMinInput.value;
    });
    hueMaxInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.ranges.H[1] = hueMaxInput.value;
    });
    saturationMinInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.ranges.S[0] = saturationMinInput.value;
    });
    saturationMaxInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.ranges.S[1] = saturationMaxInput.value;
    });
    lightnessMinInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.ranges.L[0] = lightnessMinInput.value;
    });
    lightnessMaxInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.ranges.L[1] = lightnessMaxInput.value;
    });
    recolorButton?.addEventListener('click', async () => {
        await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
    });
    if (focusedInput) {
        const nextInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#${focusedInput.id}`);
        nextInput?.focus();
        if (nextInput && focusedInput.start !== null && focusedInput.end !== null) {
            nextInput.setSelectionRange(focusedInput.start, focusedInput.end);
        }
    }
}

export function renderCellTextWidget(ctx: LinearLayoutUiContext): void {
    const labels = linearLayoutRootLabels(ctx);
    if (labels.length === 0) {
        ctx.cellTextWidget.innerHTML = '';
        return;
    }
    ctx.cellTextWidget.innerHTML = `
      ${titleWithInfo('Cell Text', 'Overlay selected root-input label values directly on every visible tensor cell in 2D. Labels only appear when cells are large enough to read.')}
      <div class="widget-body">
        <p class="widget-copy">Choose which root-input label values to draw on each cell. Every visible tensor shows the values of the root coordinate that maps to that cell.</p>
        <div class="checklist-field">
          ${labels.map((label) => `
            <label class="checklist-row" for="cell-text-${label}">
              <span>${label}</span>
              <input id="cell-text-${label}" type="checkbox" ${ctx.state.linearLayoutCellTextState[label] ? 'checked' : ''} />
            </label>
          `).join('')}
        </div>
      </div>
    `;
    const sync = (): void => {
        ctx.state.linearLayoutCellTextState = Object.fromEntries(labels.map((label) => [
            label,
            ctx.cellTextWidget.querySelector<HTMLInputElement>(`#cell-text-${CSS.escape(label)}`)?.checked ?? false,
        ]));
        const tab = activeLinearLayoutTab(ctx);
        if (tab) ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
        applyLinearLayoutCellText(ctx);
    };
    labels.forEach((label) => {
        ctx.cellTextWidget.querySelector<HTMLInputElement>(`#cell-text-${CSS.escape(label)}`)?.addEventListener('change', sync);
    });
}

function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

export function renderLinearLayoutWidget(ctx: LinearLayoutUiContext): void {
    const statusClass = ctx.state.linearLayoutNotice?.tone === 'success' ? 'success-box' : 'error-box';
    const status = ctx.state.linearLayoutNotice ? `<div class="${statusClass}">${escapeInfo(ctx.state.linearLayoutNotice.text)}</div>` : '';
    const tab = activeLinearLayoutTab(ctx);
    const meta = tab ? composeLayoutMetaForTab(tab) : null;
    const visibilityField = !meta ? '' : `
      <div class="field">
        ${labelWithInfo('Visible Tensors', 'Toggle which tensors in the render chain stay visible for the current tab.')}
        <div class="checklist-field">
          ${meta.tensors.map((tensor) => `
            <label class="checklist-row" for="linear-layout-visible-${tensor.id}">
              <span>${escapeInfo(tensor.title)}</span>
              <input id="linear-layout-visible-${tensor.id}" type="checkbox" ${ctx.state.linearLayoutState.visibleTensors[tensor.id] !== false ? 'checked' : ''} />
            </label>
          `).join('')}
        </div>
      </div>
    `;
    const matrixBlock = !ctx.state.showLinearLayoutMatrix
        ? ''
        : `<div class="mono-block linear-layout-matrix-preview">${ctx.state.linearLayoutMatrixPreview}</div>`;
    ctx.linearLayoutWidget.innerHTML = `
      ${titleWithInfo('Linear Layout Specifications', 'Define one or more named injective layouts, then use Layout Operation to build the rendered tensor chain.')}
      <div class="widget-body">
        <p class="widget-copy">Each specification starts with <span class="inline-code">name: [inputs] -> [outputs]</span>, followed by exactly one JSON basis row per input label. Separate specifications with blank lines. Layout Operation supports names, <span class="inline-code">inv(...)</span>, <span class="inline-code">*</span>, and parentheses.</p>
        <div class="field">
          ${labelWithInfo('Linear Layout Specifications', 'Enter one or more specification blocks. Each block has one signature line plus one basis row per input label.', 'linear-layout-specs')}
          <textarea id="linear-layout-specs" class="compact-textarea" rows="8" spellcheck="false">${escapeInfo(ctx.state.linearLayoutState.specsText)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Layout Operation', 'Enter the layout expression to visualize, such as A(B), inv(A), A * B, or parenthesized combinations.', 'linear-layout-operation')}
          <textarea id="linear-layout-operation" class="compact-textarea" rows="2" spellcheck="false">${escapeInfo(ctx.state.linearLayoutState.operationText)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Input Tensor Name', 'Sets the display name of the root tensor at the start of the rendered chain.', 'linear-layout-input-name')}
          <input id="linear-layout-input-name" type="text" value="${escapeInfo(ctx.state.linearLayoutState.inputName)}" />
        </div>
        ${visibilityField}
        <div class="button-row">
          <button class="primary-button" id="linear-layout-apply" type="button">Render Layout</button>
          <button class="secondary-button" id="linear-layout-copy" type="button">Copy Init Code</button>
          <button class="secondary-button" id="linear-layout-matrix" type="button">${ctx.state.showLinearLayoutMatrix ? 'Hide Matrix' : 'Show Matrix'}</button>
        </div>
        ${matrixBlock}
        ${status}
      </div>
    `;

    const specsInput = ctx.linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-specs');
    const operationInput = ctx.linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-operation');
    const inputNameInput = ctx.linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-input-name');
    const apply = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-apply');
    const copy = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-copy');
    const matrix = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-matrix');
    if (specsInput) autosizeTextarea(specsInput);
    if (operationInput) autosizeTextarea(operationInput);
    specsInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.specsText = specsInput.value;
        autosizeTextarea(specsInput);
    });
    operationInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.operationText = operationInput.value;
        autosizeTextarea(operationInput);
    });
    inputNameInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.inputName = inputNameInput.value;
    });
    meta?.tensors.forEach((tensor) => {
        ctx.linearLayoutWidget.querySelector<HTMLInputElement>(`#linear-layout-visible-${CSS.escape(tensor.id)}`)?.addEventListener('change', async (event) => {
            const target = event.currentTarget as HTMLInputElement;
            ctx.state.linearLayoutState.visibleTensors[tensor.id] = target.checked;
            await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
        });
    });
    apply?.addEventListener('click', async () => {
        await applyLinearLayoutSpec(ctx);
    });
    copy?.addEventListener('click', async () => {
        try {
            const code = buildComposeRuntime(ctx.state.linearLayoutState).pythonCode;
            await copyText(code);
            ctx.state.linearLayoutNotice = { tone: 'success', text: 'Copied Python init code.' };
        } catch (error) {
            ctx.state.linearLayoutNotice = { tone: 'error', text: error instanceof Error ? error.message : String(error) };
        }
        renderLinearLayoutWidget(ctx);
    });
    matrix?.addEventListener('click', () => {
        ctx.state.showLinearLayoutMatrix = !ctx.state.showLinearLayoutMatrix;
        renderLinearLayoutWidget(ctx);
    });
    renderLinearLayoutColorWidget(ctx);
}

export async function upsertLinearLayoutTab(
    ctx: LinearLayoutUiContext,
    document: LoadedBundleDocument,
    replaceTabs = false,
): Promise<void> {
    const activeTab = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId());
    const targetId = activeTab?.id ?? document.id;
    const targetTitle = activeTab?.title ?? document.title;
    const nextDocument = {
        ...document,
        id: targetId,
        title: targetTitle,
    };
    ctx.state.linearLayoutStates.set(targetId, cloneLinearLayoutState(ctx.state.linearLayoutState));
    ctx.state.linearLayoutCellTextStates.set(targetId, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
    ctx.state.linearLayoutSelectionMaps.delete(targetId);
    if (replaceTabs || ctx.getSessionTabs().length === 0) {
        ctx.setSessionTabs([nextDocument]);
    } else {
        const index = ctx.getSessionTabs().findIndex((tab) => tab.id === targetId);
        ctx.setSessionTabs(index === -1
            ? [...ctx.getSessionTabs(), nextDocument]
            : ctx.getSessionTabs().map((tab, tabIndex) => tabIndex === index ? nextDocument : tab));
    }
    await ctx.loadTab(targetId);
}

export async function applyLinearLayoutSpec(
    ctx: LinearLayoutUiContext,
    options: { replaceTabs?: boolean; silent?: boolean; preserveTensorViews?: boolean } = {},
): Promise<boolean> {
    try {
        refreshLinearLayoutMatrixPreview(ctx);
        const document = createComposeLayoutDocument(
            ctx.state.linearLayoutState,
            ctx.viewer.getSnapshot(),
            undefined,
            options.preserveTensorViews ? preservedLinearLayoutTensorViews(ctx) : undefined,
        );
        const runtime = buildComposeRuntime(ctx.state.linearLayoutState);
        ctx.state.linearLayoutCellTextState = normalizeCellTextState(ctx.state.linearLayoutCellTextState, runtime.inputLabels);
        storeLinearLayoutState(ctx.state.linearLayoutState);
        await upsertLinearLayoutTab(ctx, document, options.replaceTabs);
        const activeTitle = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId())?.title ?? document.title;
        ctx.state.linearLayoutNotice = options.silent ? null : { tone: 'success', text: `Rendered ${activeTitle}.` };
        renderLinearLayoutWidget(ctx);
        return true;
    } catch (error) {
        ctx.state.linearLayoutNotice = {
            tone: 'error',
            text: error instanceof Error ? error.message : String(error),
        };
        renderLinearLayoutWidget(ctx);
        return false;
    }
}

export async function loadBakedLinearLayoutTabs(ctx: LinearLayoutUiContext): Promise<boolean> {
    const examples = bakedComposeLayoutExamples();
    if (examples.length === 0) return false;
    const baseViewer = ctx.viewer.getSnapshot();
    ctx.state.linearLayoutStates.clear();
    ctx.state.linearLayoutCellTextStates.clear();
    ctx.state.linearLayoutTensorViewsStates.clear();
    ctx.setSessionTabs(examples.map(({ state, title }, index) => {
        const document = createComposeLayoutDocument(state, baseViewer, title);
        const id = `tab-${index + 1}`;
        ctx.state.linearLayoutStates.set(id, cloneLinearLayoutState(state));
        ctx.state.linearLayoutCellTextStates.set(id, defaultLinearLayoutCellTextState());
        ctx.state.linearLayoutTensorViewsStates.set(id, snapshotTensorViews(document.manifest.viewer));
        return {
            ...document,
            id,
            title,
        };
    }));
    ctx.state.linearLayoutSelectionMaps.clear();
    const initialTabId = ctx.getSessionTabs()[0]?.id ?? null;
    if (!initialTabId) return false;
    await ctx.loadTab(initialTabId);
    const settle = async (): Promise<void> => {
        if ('fonts' in document) {
            try {
                await (document as Document & { fonts: { ready: Promise<unknown> } }).fonts.ready;
            } catch {
                // ignore font-settlement errors
            }
        }
        let stableFrames = 0;
        let previousSize = '';
        while (stableFrames < 2) {
            await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
            const nextSize = `${ctx.viewport.clientWidth}x${ctx.viewport.clientHeight}`;
            stableFrames = nextSize === previousSize ? stableFrames + 1 : 0;
            previousSize = nextSize;
        }
        ctx.viewer.resize();
        ctx.viewer.refitView();
    };
    await settle();
    return true;
}
