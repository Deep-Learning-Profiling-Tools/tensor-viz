import type { LoadedBundleDocument, TensorViewSnapshot, ViewerSnapshot, TensorViewer } from '@tensor-viz/viewer-core';
import { escapeInfo } from '../../app-format.js';
import {
    autoColorLayoutState,
    buildComposeRuntime,
    cloneComposeLayoutState,
    composeLayoutStateFromLegacySpec,
    defaultComposeLayoutState,
    emptyComposeLayoutState,
    isComposeLayoutMeta,
    isComposeLayoutState,
    matchedComposeLayoutPresetSelection,
    type ComposeLayoutMeta,
    type ComposeLayoutState,
    type ComposeTensorMeta,
    type MatrixBlock,
} from './linear-layout.js';

/**
 * shape of inspector coord entry data used by the viewer.
 *
 * @example
 * const value: InspectorCoordEntry = {} as InspectorCoordEntry;
 */
export type InspectorCoordEntry = {
    title: string;
    labels: string[];
    shape: number[];
    coord: number[] | null;
    hovered: boolean;
};

/**
 * shape of linear layout notice data used by the viewer.
 *
 * @example
 * const value: LinearLayoutNotice = {} as LinearLayoutNotice;
 */
export type LinearLayoutNotice = {
    tone: 'error' | 'success';
    text: string;
};

/**
 * shape of linear layout cell text state data used by the viewer.
 *
 * @example
 * const value: LinearLayoutCellTextState = {} as LinearLayoutCellTextState;
 */
export type LinearLayoutCellTextState = Record<string, boolean>;
/**
 * shape of linear layout multi input state data used by the viewer.
 *
 * @example
 * const value: LinearLayoutMultiInputState = {} as LinearLayoutMultiInputState;
 */
export type LinearLayoutMultiInputState = Record<string, number>;
/**
 * shape of linear layout tensor views state data used by the viewer.
 *
 * @example
 * const value: LinearLayoutTensorViewsState = {} as LinearLayoutTensorViewsState;
 */
export type LinearLayoutTensorViewsState = Record<string, TensorViewSnapshot>;
/**
 * shape of linear layout form state data used by the viewer.
 *
 * @example
 * const value: LinearLayoutFormState = {} as LinearLayoutFormState;
 */
export type LinearLayoutFormState = ComposeLayoutState;
/**
 * shape of linear layout channel data used by the viewer.
 *
 * @example
 * const value: LinearLayoutChannel = {} as LinearLayoutChannel;
 */
export type LinearLayoutChannel = 'H' | 'S' | 'L';

/**
 * shape of linear layout selection map data used by the viewer.
 *
 * @example
 * const value: LinearLayoutSelectionMap = {} as LinearLayoutSelectionMap;
 */
export type LinearLayoutSelectionMap = {
    injective: boolean;
    rootInputLabels: string[];
    rootInputShape: number[];
    rootKeys: string[];
    rootKeyToIndex: Map<string, number>;
    finalOutputLabels: string[];
    finalOutputShape: number[];
    rootToFinalKeys: string[];
    tensors: Map<string, {
        meta: ComposeTensorMeta;
        rootToTensorKeys: string[];
        coordKeyToFlatIndex: Map<string, number>;
        cellRootIndexes: number[][];
    }>;
    orderedTensorIds: string[];
};

/**
 * shape of linear layout ui state data used by the viewer.
 *
 * @example
 * const value: LinearLayoutUiState = {} as LinearLayoutUiState;
 */
export type LinearLayoutUiState = {
    linearLayoutState: LinearLayoutFormState;
    linearLayoutStates: Map<string, LinearLayoutFormState>;
    linearLayoutCellTextState: LinearLayoutCellTextState;
    linearLayoutCellTextStates: Map<string, LinearLayoutCellTextState>;
    linearLayoutMultiInputState: LinearLayoutMultiInputState;
    linearLayoutMultiInputStates: Map<string, LinearLayoutMultiInputState>;
    linearLayoutTensorViewsStates: Map<string, LinearLayoutTensorViewsState>;
    linearLayoutSelectionMaps: Map<string, LinearLayoutSelectionMap>;
    linearLayoutNotice: LinearLayoutNotice | null;
    linearLayoutMatrixPreview: string;
    showLinearLayoutMatrix: boolean;
    syncingLinearLayoutSelection: boolean;
};

/**
 * shape of linear layout ui context data used by the viewer.
 *
 * @example
 * const value: LinearLayoutUiContext = {} as LinearLayoutUiContext;
 */
export type LinearLayoutUiContext = {
    viewer: TensorViewer;
    viewport: HTMLElement;
    linearLayoutPresetWidget: HTMLElement;
    linearLayoutWidget: HTMLElement;
    linearLayoutVisibleTensorsWidget: HTMLElement;
    cellTextWidget: HTMLElement;
    linearLayoutColorWidget: HTMLElement;
    state: LinearLayoutUiState;
    widgetTitle: (widgetId: string, info: string) => string;
    getActiveTab: () => LoadedBundleDocument | undefined;
    getActiveTabId: () => string | null;
    getSessionTabs: () => LoadedBundleDocument[];
    setSessionTabs: (tabs: LoadedBundleDocument[]) => void;
    loadTab: (id: string) => Promise<void>;
    renderLinearLayoutEditorWidgets: () => void;
};

const LINEAR_LAYOUT_CHANNELS: LinearLayoutChannel[] = ['H', 'S', 'L'];
const LINEAR_LAYOUT_STORAGE_KEY = 'tensor-viz-linear-layout-spec';

/**
 * load linear layout state for the current viewer state.
 *
 * @returns Computed LinearLayoutFormState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * loadLinearLayoutState();
 */
export function loadLinearLayoutState(): LinearLayoutFormState {
    const fallback = defaultLinearLayoutState();
    try {
        const stored = window.localStorage.getItem(LINEAR_LAYOUT_STORAGE_KEY);
        if (!stored) return fallback;
        const parsed = JSON.parse(stored);
        if (isLinearLayoutState(parsed)) return cloneLinearLayoutState(parsed);
        if (parsed && typeof parsed === 'object' && ('specsText' in parsed || 'operationText' in parsed)) {
            return { ...fallback, ...(parsed as Partial<LinearLayoutFormState>) };
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

/**
 * return store linear layout state for the current viewer state.
 *
 * @param state - State object read or updated by this operation.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * storeLinearLayoutState(state);
 */
export function storeLinearLayoutState(state: LinearLayoutFormState): void {
    try {
        window.localStorage.setItem(LINEAR_LAYOUT_STORAGE_KEY, JSON.stringify(state));
    } catch {
        // ignore storage failures in restricted browsers
    }
}

/**
 * return whether linear layout state for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @returns Computed value is LinearLayoutFormState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * isLinearLayoutState(value);
 */
export function isLinearLayoutState(value: unknown): value is LinearLayoutFormState {
    return isComposeLayoutState(value);
}

/**
 * return default linear layout state for the current viewer state.
 *
 * @returns Computed LinearLayoutFormState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * defaultLinearLayoutState();
 */
export function defaultLinearLayoutState(): LinearLayoutFormState {
    return defaultComposeLayoutState();
}

/**
 * return empty linear layout state for the current viewer state.
 *
 * @returns Computed LinearLayoutFormState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * emptyLinearLayoutState();
 */
export function emptyLinearLayoutState(): LinearLayoutFormState {
    return emptyComposeLayoutState();
}

/**
 * clone linear layout state for the current viewer state.
 *
 * @param state - State object read or updated by this operation.
 * @returns Computed LinearLayoutFormState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * cloneLinearLayoutState(state);
 */
export function cloneLinearLayoutState(state: LinearLayoutFormState): LinearLayoutFormState {
    return cloneComposeLayoutState(state);
}

/**
 * return default linear layout cell text state for the current viewer state.
 *
 * @param labels - labels input used by this operation (string[]).
 * @returns Computed LinearLayoutCellTextState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * defaultLinearLayoutCellTextState(labels);
 */
export function defaultLinearLayoutCellTextState(labels: string[] = []): LinearLayoutCellTextState {
    return Object.fromEntries(labels.map((label) => [label, true]));
}

/**
 * clone linear layout cell text state for the current viewer state.
 *
 * @param state - State object read or updated by this operation.
 * @returns Computed LinearLayoutCellTextState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * cloneLinearLayoutCellTextState(state);
 */
export function cloneLinearLayoutCellTextState(state: LinearLayoutCellTextState): LinearLayoutCellTextState {
    return { ...state };
}

/**
 * return default linear layout multi input state for the current viewer state.
 *
 * @returns Computed LinearLayoutMultiInputState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * defaultLinearLayoutMultiInputState();
 */
export function defaultLinearLayoutMultiInputState(): LinearLayoutMultiInputState {
    return {};
}

/**
 * clone linear layout multi input state for the current viewer state.
 *
 * @param state - State object read or updated by this operation.
 * @returns Computed LinearLayoutMultiInputState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * cloneLinearLayoutMultiInputState(state);
 */
export function cloneLinearLayoutMultiInputState(state: LinearLayoutMultiInputState): LinearLayoutMultiInputState {
    return { ...state };
}

/**
 * return whether linear layout multi input state for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @returns Computed value is LinearLayoutMultiInputState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * isLinearLayoutMultiInputState(value);
 */
export function isLinearLayoutMultiInputState(value: unknown): value is LinearLayoutMultiInputState {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => Number.isInteger(entry) && Number(entry) >= -1);
}

/**
 * clone linear layout tensor views state for the current viewer state.
 *
 * @param state - State object read or updated by this operation.
 * @returns Computed LinearLayoutTensorViewsState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * cloneLinearLayoutTensorViewsState(state);
 */
export function cloneLinearLayoutTensorViewsState(state: LinearLayoutTensorViewsState): LinearLayoutTensorViewsState {
    return Object.fromEntries(Object.entries(state).map(([tensorId, view]) => [
        tensorId,
        { editor: view.editor, hiddenIndices: view.hiddenIndices.slice() },
    ]));
}

/**
 * return whether linear layout cell text state for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @returns Computed value is LinearLayoutCellTextState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * isLinearLayoutCellTextState(value);
 */
export function isLinearLayoutCellTextState(value: unknown): value is LinearLayoutCellTextState {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => typeof entry === 'boolean');
}

/**
 * return snapshot tensor views for the current viewer state.
 *
 * @param snapshot - Viewer snapshot used by this operation.
 * @returns Computed LinearLayoutTensorViewsState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * snapshotTensorViews(snapshot);
 */
export function snapshotTensorViews(snapshot: ViewerSnapshot): LinearLayoutTensorViewsState {
    return Object.fromEntries(snapshot.tensors.map((tensor) => [
        tensor.id,
        { editor: tensor.view.editor, hiddenIndices: tensor.view.hiddenIndices.slice() },
    ]));
}

/**
 * compose layout meta for tab for the current viewer state.
 *
 * @param tab - Session tab used by this operation.
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * composeLayoutMetaForTab(tab);
 */
export function composeLayoutMetaForTab(tab: LoadedBundleDocument): ComposeLayoutMeta | null {
    const candidate = (tab.manifest.viewer as { composeLayoutMeta?: unknown }).composeLayoutMeta;
    return isComposeLayoutMeta(candidate) ? candidate : null;
}

/**
 * return whether linear layout tab for the current viewer state.
 *
 * @param tab - Session tab used by this operation.
 * @returns Whether the requested condition holds.
 * @noThrows This function has no direct throw path.
 * @example
 * isLinearLayoutTab(tab);
 */
export function isLinearLayoutTab(tab: LoadedBundleDocument): boolean {
    return composeLayoutMetaForTab(tab) !== null;
}

/**
 * sync linear layout state for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param tab - Session tab used by this operation.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * syncLinearLayoutState(ctx, tab);
 */
export function syncLinearLayoutState(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) return;
    const stored = ctx.state.linearLayoutStates.get(tab.id);
    if (stored) {
        ctx.state.linearLayoutState = cloneLinearLayoutState(stored);
        refreshLinearLayoutMatrixPreview(ctx);
        return;
    }
    const candidate = (tab.manifest.viewer as { composeLayoutState?: unknown }).composeLayoutState;
    if (isComposeLayoutState(candidate)) {
        ctx.state.linearLayoutState = cloneLinearLayoutState(candidate);
        ctx.state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(candidate));
        refreshLinearLayoutMatrixPreview(ctx);
        return;
    }
    const meta = composeLayoutMetaForTab(tab);
    const autoColor = meta ? autoColorLayoutState(meta.specsText, meta.operationText) : null;
    ctx.state.linearLayoutState = meta
        ? {
            ...defaultLinearLayoutState(),
            specsText: meta.specsText,
            operationText: meta.operationText,
            inputName: meta.inputName ?? defaultLinearLayoutState().inputName,
            presetSelection: matchedComposeLayoutPresetSelection({
                specsText: meta.specsText,
                operationText: meta.operationText,
                inputName: meta.inputName ?? defaultLinearLayoutState().inputName,
            }),
            visibleTensors: Object.fromEntries(meta.tensors.map((tensor) => [tensor.id, tensor.visible])),
            mapping: autoColor?.mapping ?? defaultLinearLayoutState().mapping,
            ranges: autoColor?.ranges ?? defaultLinearLayoutState().ranges,
        }
        : defaultLinearLayoutState();
    ctx.state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(ctx.state.linearLayoutState));
    refreshLinearLayoutMatrixPreview(ctx);
}

/**
 * sync linear layout cell text state for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param tab - Session tab used by this operation.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * syncLinearLayoutCellTextState(ctx, tab);
 */
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
    const meta = composeLayoutMetaForTab(tab);
    const labels = ctx.state.linearLayoutState.propagateOutputs
        ? meta?.finalOutputLabels ?? []
        : meta?.rootInputLabels ?? [];
    ctx.state.linearLayoutCellTextState = defaultLinearLayoutCellTextState(labels);
    ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
}

/**
 * sync linear layout multi input state for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param tab - Session tab used by this operation.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * syncLinearLayoutMultiInputState(ctx, tab);
 */
export function syncLinearLayoutMultiInputState(ctx: LinearLayoutUiContext, tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) {
        ctx.state.linearLayoutMultiInputState = defaultLinearLayoutMultiInputState();
        return;
    }
    const stored = ctx.state.linearLayoutMultiInputStates.get(tab.id);
    if (stored) {
        ctx.state.linearLayoutMultiInputState = cloneLinearLayoutMultiInputState(stored);
        return;
    }
    const candidate = (tab.manifest.viewer as { linearLayoutMultiInputState?: unknown }).linearLayoutMultiInputState;
    if (isLinearLayoutMultiInputState(candidate)) {
        ctx.state.linearLayoutMultiInputState = cloneLinearLayoutMultiInputState(candidate);
        ctx.state.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(candidate));
        return;
    }
    ctx.state.linearLayoutMultiInputState = defaultLinearLayoutMultiInputState();
    ctx.state.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(ctx.state.linearLayoutMultiInputState));
}

/**
 * refresh linear layout matrix preview for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @param state - State object read or updated by this operation.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * refreshLinearLayoutMatrixPreview(ctx, state);
 */
export function refreshLinearLayoutMatrixPreview(ctx: LinearLayoutUiContext, state = ctx.state.linearLayoutState): void {
    try {
        ctx.state.linearLayoutMatrixPreview = matrixPreviewFromBlocks(buildComposeRuntime(state).matrixBlocks);
    } catch {
        ctx.state.linearLayoutMatrixPreview = '';
    }
}

/**
 * return legacy editor state for the current viewer state.
 *
 * @param raw - raw input used by this operation (Record<string, unknown>).
 * @param fallback - fallback input used by this operation (LinearLayoutFormState).
 * @returns Computed LinearLayoutFormState value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * legacyEditorState(raw, fallback);
 */
function legacyEditorState(raw: Record<string, unknown>, fallback: LinearLayoutFormState): LinearLayoutFormState {
    const textByLabel = new Map<string, string>([['T', '[]'], ['W', '[]'], ['R', '[]']]);
    if (typeof raw.basesText === 'string') {
        raw.basesText.split('\n').forEach((line) => {
            const match = line.trim().match(/^([TWR])\s*:\s*(.+)$/i);
            if (match) textByLabel.set(match[1]!.toUpperCase(), match[2]!.trim());
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
    const specsText = [
        `Layout_1: [T,W,R] -> [${outputs.join(',')}]`,
        ...['T', 'W', 'R'].map((label, axis) => `${label}: ${JSON.stringify(rows[axis])}`),
    ].join('\n');
    const operationText = 'Layout_1';
    return {
        specsText,
        operationText,
        inputName: fallback.inputName,
        presetSelection: matchedComposeLayoutPresetSelection({
            specsText,
            operationText,
            inputName: fallback.inputName,
        }),
        visibleTensors: {},
        propagateOutputs: false,
        mapping,
        ranges,
    };
}

/**
 * parse bases field for the current viewer state.
 *
 * @param label - label input used by this operation (string).
 * @param value - Value supplied by the caller.
 * @returns Array of computed entries for the caller.
 * @throws Error when the requested input or state is invalid.
 * @example
 * parseBasesField(label, value);
 */
function parseBasesField(label: string, value: string): number[][] {
    if (!value.trim()) return [];
    let parsed: unknown;
    try {
        parsed = JSON.parse(value);
    } catch {
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

/**
 * return matrix preview from blocks for the current viewer state.
 *
 * @param blocks - blocks input used by this operation (MatrixBlock[]).
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * matrixPreviewFromBlocks(blocks);
 */
function matrixPreviewFromBlocks(blocks: MatrixBlock[]): string {
    return blocks.map((block) => {
        const labelWidth = Math.max(1, ...block.rows.map((row) => row.label.length));
        const columnWidths = block.columns.map((column) => Math.max(1, column.label.length));
        const header = block.columns.length === 0
            ? '<span class="matrix-zero">0</span>'
            : `${' '.repeat(labelWidth)} | ${block.columns.map((column, index) => (
                `<span class="matrix-label matrix-axis-${column.axis % 3}">${escapeInfo(column.label.padStart(columnWidths[index] ?? 1))}</span>`
            )).join(' ')}`;
        const rows = block.rows.length === 0
            ? []
            : block.rows.map((row, rowIndex) => (
                `<span class="matrix-label matrix-axis-${row.axis % 3}">${escapeInfo(row.label.padStart(labelWidth))}</span> | ${block.columns.map((_column, columnIndex) => {
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
