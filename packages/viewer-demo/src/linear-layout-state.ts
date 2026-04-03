import type { LoadedBundleDocument, TensorViewSnapshot, ViewerSnapshot, TensorViewer } from '@tensor-viz/viewer-core';
import { escapeInfo } from './app-format.js';
import {
    buildComposeRuntime,
    cloneComposeLayoutState,
    composeLayoutStateFromLegacySpec,
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
        { view: view.view, hiddenIndices: view.hiddenIndices.slice() },
    ]));
}

export function isLinearLayoutCellTextState(value: unknown): value is LinearLayoutCellTextState {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => typeof entry === 'boolean');
}

export function snapshotTensorViews(snapshot: ViewerSnapshot): LinearLayoutTensorViewsState {
    return Object.fromEntries(snapshot.tensors.map((tensor) => [
        tensor.id,
        { view: tensor.view.view, hiddenIndices: tensor.view.hiddenIndices.slice() },
    ]));
}

export function composeLayoutMetaForTab(tab: LoadedBundleDocument): ComposeLayoutMeta | null {
    const candidate = (tab.manifest.viewer as { composeLayoutMeta?: unknown }).composeLayoutMeta;
    return isComposeLayoutMeta(candidate) ? candidate : null;
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

export function refreshLinearLayoutMatrixPreview(ctx: LinearLayoutUiContext, state = ctx.state.linearLayoutState): void {
    try {
        ctx.state.linearLayoutMatrixPreview = matrixPreviewFromBlocks(buildComposeRuntime(state).matrixBlocks);
    } catch {
        ctx.state.linearLayoutMatrixPreview = '';
    }
}

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
    return {
        specsText: [
            `Layout_1: [T,W,R] -> [${outputs.join(',')}]`,
            ...['T', 'W', 'R'].map((label, axis) => `${label}: ${JSON.stringify(rows[axis])}`),
        ].join('\n'),
        operationText: 'Layout_1',
        inputName: fallback.inputName,
        visibleTensors: {},
        mapping,
        ranges,
    };
}

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
