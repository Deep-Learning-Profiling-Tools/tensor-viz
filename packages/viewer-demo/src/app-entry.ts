import {
    createTypedArray,
    coordFromKey,
    coordKey,
    parseTensorView,
    TensorViewer,
    product,
    unravelIndex,
    type BundleManifest,
    type DimensionMappingScheme,
    type LoadedBundleDocument,
    type NumericArray,
    type SessionBundleManifest,
    type SelectionCoords,
    type ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import {
    escapeInfo,
    formatAxisTokens,
    formatAxisValues,
    formatNamedAxisValues,
    formatRangeValue,
    infoButton,
    labelWithInfo,
    selectionEnabled,
    titleWithInfo,
} from './app-format.js';
import {
    bakedLinearLayoutSpecTexts,
    createLinearLayoutDocument,
    defaultLinearLayoutSpecText,
    linearLayoutAxisOrder,
    logicalOutputDimsFor,
    mapLinearLayoutCoord,
    parseLinearLayoutSpec,
    type LinearLayoutSpec,
} from './linear-layout.js';
import { getAppRoot, mountAppShell, renderWebglUnavailable, supportsWebGL } from './app-shell.js';
import './styles.css';

const app = getAppRoot();

if (!supportsWebGL()) {
    renderWebglUnavailable(app);
} else {
const {
    viewport,
    tabStrip,
    sidebarSplitter,
    linearLayoutWidget,
    cellTextWidget,
    linearLayoutColorWidget,
    tensorViewWidget,
    inspectorWidget,
    selectionWidget,
    advancedSettingsWidget,
    colorbarWidget,
    commandPalette,
    commandPaletteBackdrop,
    commandPaletteInput,
    commandPaletteList,
    fileInput,
} = mountAppShell(app);

const viewer = new TensorViewer(viewport);
viewer.toggleSelectionPanel(false);
const viewErrors = new Map<string, string>();
let suspendTensorViewRender = false;
let showTensorViewWidget = true;
let showAdvancedSettingsWidget = false;
let showColorbarWidget = false;
let inspectorReady = false;
let sessionTabs: LoadedBundleDocument[] = [];
let activeTabId: string | null = null;
let switchingTab = false;
let resizingSidebar = false;
let commandPaletteOpen = false;
let commandPaletteIndex = 0;
let commandPaletteMode: 'actions' | 'tabs' = 'actions';
let appliedStartupWidgetDefaults = false;
let showLinearLayoutMatrix = false;
let linearLayoutMatrixPreview = '';

const MIN_VIEWPORT_WIDTH = 280;
const MAX_SIDEBAR_WIDTH = 720;

type InspectorRefs = {
    hoveredTensor: HTMLDivElement;
    hardwareCoord: HTMLDivElement;
    logicalCoord: HTMLDivElement;
    hardwareCoordBinary: HTMLDivElement;
    logicalCoordBinary: HTMLDivElement;
    hoveredTensorValue: HTMLSpanElement;
    hardwareCoordValue: HTMLSpanElement;
    logicalCoordValue: HTMLSpanElement;
    hardwareCoordBinaryValue: HTMLSpanElement;
    logicalCoordBinaryValue: HTMLSpanElement;
    tensorShapeValue: HTMLSpanElement;
    rankValue: HTMLSpanElement;
};

let inspectorRefs: InspectorRefs | null = null;
type LinearLayoutNotice = {
    tone: 'error' | 'success';
    text: string;
};

type LinearLayoutCellTextState = {
    thread: boolean;
    warp: boolean;
    register: boolean;
};

const LINEAR_LAYOUT_AXES = ['thread', 'warp', 'register'] as const;
const LINEAR_LAYOUT_CHANNELS = ['H', 'S', 'L'] as const;
const HARDWARE_LAYOUT_NAMES = new Set(['hardware tensor', 'hardware layout']);
const LOGICAL_LAYOUT_NAMES = new Set(['logical tensor', 'logical layout']);
const OUTPUT_AXIS_NAMES = [
    'x',
    'y',
    'z',
    'w',
    'v',
    'u',
    't',
    's',
    'r',
    'q',
    'p',
    'o',
    'n',
    'm',
    'l',
    'k',
    'j',
    'i',
    'h',
    'g',
    'f',
    'e',
    'd',
    'c',
    'b',
    'a',
] as const;

type LinearLayoutAxis = typeof LINEAR_LAYOUT_AXES[number];
type LinearLayoutChannel = typeof LINEAR_LAYOUT_CHANNELS[number];
type LinearLayoutMappingValue = LinearLayoutAxis | 'none';

type LinearLayoutFormState = {
    bases: Record<LinearLayoutAxis, string>;
    basesText: string;
    mapping: Record<LinearLayoutChannel, LinearLayoutMappingValue>;
    ranges: Record<LinearLayoutChannel, [string, string]>;
};

const LINEAR_LAYOUT_STORAGE_KEY = 'tensor-viz-linear-layout-spec';
let linearLayoutState = loadLinearLayoutState();
const linearLayoutStates = new Map<string, LinearLayoutFormState>();
let linearLayoutCellTextState = defaultLinearLayoutCellTextState();
const linearLayoutCellTextStates = new Map<string, LinearLayoutCellTextState>();
let linearLayoutNotice: LinearLayoutNotice | null = null;
type LinearLayoutSelectionMap = {
    hardwareTensorId: string;
    logicalTensorId: string;
    hardwareToLogical: Map<string, string>;
    logicalToHardware: Map<string, string[]>;
};

const linearLayoutSelectionMaps = new Map<string, LinearLayoutSelectionMap>();
let syncingLinearLayoutSelection = false;

type CommandAction = {
    action: string;
    label: string;
    shortcut: string;
    keywords: string;
};

function logUi(event: string, details?: unknown): void {
    if (details === undefined) console.log('[tensor-viz-ui]', event);
    else console.log('[tensor-viz-ui]', event, details);
}

function selectionCountValue(summary: ReturnType<TensorViewer['getSelectionSummary']>, enabled: boolean): string {
    if (!enabled) return 'Unavailable';
    if (summary.count === 0) return '0';
    return summary.availableCount === summary.count ? String(summary.count) : `${summary.count} (${summary.availableCount} with values)`;
}

function selectionStatValue(summary: ReturnType<TensorViewer['getSelectionSummary']>, enabled: boolean, key: keyof NonNullable<ReturnType<TensorViewer['getSelectionSummary']>['stats']>): string {
    if (!enabled || !summary.stats) return '—';
    return formatRangeValue(summary.stats[key]);
}

/** Load the persisted linear-layout JSON used by the static demo sidebar. */
function loadLinearLayoutState(): LinearLayoutFormState {
    const fallback = defaultLinearLayoutState();
    try {
        const stored = window.localStorage.getItem(LINEAR_LAYOUT_STORAGE_KEY);
        if (!stored) return fallback;
        const parsed = JSON.parse(stored);
        if (isLinearLayoutState(parsed)) return cloneLinearLayoutState(parsed);
        if (parsed && typeof parsed === 'object' && (parsed.bases || parsed.input_dims)) {
            return stateFromLayoutSpec(parsed, fallback);
        }
    } catch {
        return fallback;
    }
    return fallback;
}

/** Persist the current linear-layout JSON for the next browser session. */
function storeLinearLayoutState(): void {
    try {
        window.localStorage.setItem(LINEAR_LAYOUT_STORAGE_KEY, JSON.stringify(linearLayoutState));
    } catch {
        // ignore storage failures in restricted browsers
    }
}

function isLinearLayoutState(value: unknown): value is LinearLayoutFormState {
    if (!value || typeof value !== 'object') return false;
    const record = value as LinearLayoutFormState;
    return LINEAR_LAYOUT_AXES.every((axis) => typeof record.bases?.[axis] === 'string')
        && (record.basesText === undefined || typeof record.basesText === 'string')
        && LINEAR_LAYOUT_CHANNELS.every((channel) => typeof record.mapping?.[channel] === 'string')
        && LINEAR_LAYOUT_CHANNELS.every((channel) => Array.isArray(record.ranges?.[channel]));
}

function defaultLinearLayoutState(): LinearLayoutFormState {
    return stateFromLayoutSpec(JSON.parse(defaultLinearLayoutSpecText()));
}

function cloneLinearLayoutState(state: LinearLayoutFormState): LinearLayoutFormState {
    return {
        bases: { ...state.bases },
        basesText: state.basesText ?? formatLabeledBasesText(state.bases),
        mapping: { ...state.mapping },
        ranges: {
            H: [...state.ranges.H],
            S: [...state.ranges.S],
            L: [...state.ranges.L],
        } as Record<LinearLayoutChannel, [string, string]>,
    };
}

function defaultLinearLayoutCellTextState(): LinearLayoutCellTextState {
    return {
        thread: false,
        warp: false,
        register: false,
    };
}

function cloneLinearLayoutCellTextState(state: LinearLayoutCellTextState): LinearLayoutCellTextState {
    return { ...state };
}

function isLinearLayoutCellTextState(value: unknown): value is LinearLayoutCellTextState {
    if (!value || typeof value !== 'object') return false;
    const record = value as LinearLayoutCellTextState;
    return typeof record.thread === 'boolean'
        && typeof record.warp === 'boolean'
        && typeof record.register === 'boolean';
}

function stateFromLayoutSpec(raw: any, fallback?: LinearLayoutFormState): LinearLayoutFormState {
    const axes = {
        thread: formatBases([]),
        warp: formatBases([]),
        register: formatBases([]),
    } satisfies Record<LinearLayoutAxis, string>;
    const entries = Array.isArray(raw?.input_dims)
        ? raw.input_dims
        : Array.isArray(raw?.bases)
            ? raw.bases
            : [];
    entries.forEach((entry: any) => {
        const pair = Array.isArray(entry) ? entry : [entry?.name, entry?.bases];
        const [name, bases] = pair;
        if (typeof name !== 'string' || !(name in axes)) return;
        axes[name as LinearLayoutAxis] = formatBases(bases ?? []);
    });
    const mappingFallback = fallback?.mapping ?? {
        H: 'warp',
        S: 'thread',
        L: 'register',
    };
    const rangesFallback = fallback?.ranges ?? {
        H: ['0', '0.85'],
        S: ['0.35', '0.95'],
        L: ['0.25', '0.95'],
    };
    const mapping: Record<LinearLayoutChannel, LinearLayoutMappingValue> = { ...mappingFallback };
    if (raw?.color_axes && typeof raw.color_axes === 'object') {
        Object.entries(raw.color_axes).forEach(([axisName, channelName]) => {
            const channel = String(channelName).toUpperCase();
            if (!LINEAR_LAYOUT_CHANNELS.includes(channel as LinearLayoutChannel)) return;
            if (!LINEAR_LAYOUT_AXES.includes(axisName as LinearLayoutAxis)) return;
            mapping[channel as LinearLayoutChannel] = axisName as LinearLayoutAxis;
        });
    }
    const ranges: Record<LinearLayoutChannel, [string, string]> = { ...rangesFallback };
    if (raw?.color_ranges && typeof raw.color_ranges === 'object') {
        Object.entries(raw.color_ranges).forEach(([channelName, range]) => {
            if (!LINEAR_LAYOUT_CHANNELS.includes(channelName.toUpperCase() as LinearLayoutChannel)) return;
            if (!Array.isArray(range) || range.length !== 2) return;
            ranges[channelName.toUpperCase() as LinearLayoutChannel] = [String(range[0]), String(range[1])];
        });
    }
    return {
        bases: axes,
        basesText: formatLabeledBasesText(axes),
        mapping,
        ranges,
    };
}

function formatBases(value: unknown): string {
    return JSON.stringify(value ?? []);
}

function formatLabeledBasesText(bases: Record<LinearLayoutAxis, string>): string {
    return [
        `T: ${bases.thread}`,
        `W: ${bases.warp}`,
        `R: ${bases.register}`,
    ].join('\n');
}

function parseLabeledBasesText(value: string): Record<LinearLayoutAxis, number[][]> {
    const grouped: Record<LinearLayoutAxis, number[][]> = {
        thread: [],
        warp: [],
        register: [],
    };
    value.split('\n').forEach((line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        const match = trimmed.match(/^([TWR])\s*:\s*(.+)$/i);
        if (!match) {
            throw new Error('Each bases line must look like T: [...], W: [...], or R: [...].');
        }
        const label = match[1]?.toUpperCase() ?? '';
        const basisText = match[2] ?? '[]';
        const axis = label.startsWith('T')
            ? 'thread'
            : label.startsWith('W')
                ? 'warp'
                : 'register';
        grouped[axis].push(...parseBasesField(label, basisText));
    });
    return grouped;
}

function applyLabeledBasesText(value: string): void {
    linearLayoutState.basesText = value;
    const grouped = parseLabeledBasesText(value);
    linearLayoutState.bases.thread = JSON.stringify(grouped.thread);
    linearLayoutState.bases.warp = JSON.stringify(grouped.warp);
    linearLayoutState.bases.register = JSON.stringify(grouped.register);
}

function isLinearLayoutTab(tab: LoadedBundleDocument): boolean {
    const names = tab.manifest.tensors.map((tensor) => tensor.name.toLowerCase());
    return names.some((name) => HARDWARE_LAYOUT_NAMES.has(name)) && names.some((name) => LOGICAL_LAYOUT_NAMES.has(name));
}

function hardwareLayoutTensor(tab: LoadedBundleDocument): BundleManifest['tensors'][number] | undefined {
    return tab.manifest.tensors.find((tensor) => HARDWARE_LAYOUT_NAMES.has(tensor.name.toLowerCase()));
}

function logicalLayoutTensor(tab: LoadedBundleDocument): BundleManifest['tensors'][number] | undefined {
    return tab.manifest.tensors.find((tensor) => LOGICAL_LAYOUT_NAMES.has(tensor.name.toLowerCase()));
}

function syncLinearLayoutState(tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) return;
    const stored = linearLayoutStates.get(tab.id);
    if (stored) {
        linearLayoutState = cloneLinearLayoutState(stored);
        refreshLinearLayoutMatrixPreview(linearLayoutState);
        renderLinearLayoutWidget();
        return;
    }
    const candidate = (tab.manifest.viewer as { linearLayoutState?: unknown }).linearLayoutState;
    if (isLinearLayoutState(candidate)) {
        linearLayoutState = cloneLinearLayoutState(candidate);
        linearLayoutStates.set(tab.id, cloneLinearLayoutState(candidate));
        refreshLinearLayoutMatrixPreview(linearLayoutState);
        renderLinearLayoutWidget();
        return;
    }
    linearLayoutState = defaultLinearLayoutState();
    linearLayoutStates.set(tab.id, cloneLinearLayoutState(linearLayoutState));
    refreshLinearLayoutMatrixPreview(linearLayoutState);
    renderLinearLayoutWidget();
}

function syncLinearLayoutCellTextState(tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) {
        linearLayoutCellTextState = defaultLinearLayoutCellTextState();
        return;
    }
    const stored = linearLayoutCellTextStates.get(tab.id);
    if (stored) {
        linearLayoutCellTextState = cloneLinearLayoutCellTextState(stored);
        return;
    }
    const candidate = (tab.manifest.viewer as { linearLayoutCellTextState?: unknown }).linearLayoutCellTextState;
    if (isLinearLayoutCellTextState(candidate)) {
        linearLayoutCellTextState = cloneLinearLayoutCellTextState(candidate);
        linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(candidate));
        return;
    }
    linearLayoutCellTextState = defaultLinearLayoutCellTextState();
    linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(linearLayoutCellTextState));
}

function linearLayoutSpecForTab(tab: LoadedBundleDocument): LinearLayoutSpec | null {
    const candidate = (tab.manifest.viewer as { linearLayoutSpec?: unknown }).linearLayoutSpec;
    if (!candidate) return null;
    try {
        return parseLinearLayoutSpec(JSON.stringify(candidate));
    } catch (error) {
        logUi('linear-layout:spec-error', error);
        return null;
    }
}

function linearLayoutSelectionMapForTab(tab: LoadedBundleDocument): LinearLayoutSelectionMap | null {
    const cached = linearLayoutSelectionMaps.get(tab.id);
    if (cached) return cached;
    const spec = linearLayoutSpecForTab(tab);
    if (!spec) return null;
    const hardwareTensor = hardwareLayoutTensor(tab);
    const logicalTensor = logicalLayoutTensor(tab);
    if (!hardwareTensor || !logicalTensor) return null;
    const inputShape = spec.input_dims.map(({ bases }) => 2 ** bases.length);
    const outputAxisByName = new Map(spec.output_dims.map((dim, axis) => [dim.name, axis]));
    const logicalOutputDims = logicalOutputDimsFor(spec.output_dims);
    const hardwareAxisOrder = linearLayoutAxisOrder(spec.input_dims);
    const hardwareToLogical = new Map<string, string>();
    const logicalToHardware = new Map<string, string[]>();
    const total = product(inputShape);
    for (let index = 0; index < total; index += 1) {
        const inputCoord = unravelIndex(index, inputShape);
        const outputCoord = mapLinearLayoutCoord(inputCoord, spec.input_dims, spec.output_dims.length);
        const logicalCoord = logicalOutputDims.map(({ name }) => {
            const axis = outputAxisByName.get(name);
            return axis === undefined ? 0 : (outputCoord[axis] ?? 0);
        });
        const hardwareCoord = hardwareAxisOrder.map((axis) => inputCoord[axis] ?? 0);
        const hardwareKey = coordKey(hardwareCoord);
        const logicalKey = coordKey(logicalCoord);
        hardwareToLogical.set(hardwareKey, logicalKey);
        const bucket = logicalToHardware.get(logicalKey) ?? [];
        bucket.push(hardwareKey);
        logicalToHardware.set(logicalKey, bucket);
    }
    const map = {
        hardwareTensorId: hardwareTensor.id,
        logicalTensorId: logicalTensor.id,
        hardwareToLogical,
        logicalToHardware,
    };
    linearLayoutSelectionMaps.set(tab.id, map);
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

function mapHardwareSelection(coords: number[][], mapping: LinearLayoutSelectionMap): number[][] {
    const logicalKeys = new Set<string>();
    coords.forEach((coord) => {
        const key = coordKey(coord);
        const mapped = mapping.hardwareToLogical.get(key);
        if (mapped) logicalKeys.add(mapped);
    });
    return Array.from(logicalKeys, (key) => coordFromKey(key));
}

function mapLogicalSelection(coords: number[][], mapping: LinearLayoutSelectionMap): number[][] {
    const hardwareKeys = new Set<string>();
    coords.forEach((coord) => {
        const key = coordKey(coord);
        const mapped = mapping.logicalToHardware.get(key);
        if (!mapped) return;
        mapped.forEach((hardwareKey) => hardwareKeys.add(hardwareKey));
    });
    return Array.from(hardwareKeys, (key) => coordFromKey(key));
}

function activeLinearLayoutTab(): LoadedBundleDocument | null {
    const tab = activeTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

function clearCellTextLabels(): void {
    viewer.getInspectorModel().tensors.forEach((tensor) => {
        viewer.setTensorCellLabels(tensor.id, null);
    });
}

function linearLayoutCellTextForCoord(coord: number[], spec: LinearLayoutSpec, state: LinearLayoutCellTextState): string {
    const axisOrder = linearLayoutAxisOrder(spec.input_dims);
    const valuesByName = new Map<string, number>();
    axisOrder.forEach((inputAxis, tensorAxis) => {
        valuesByName.set(spec.input_dims[inputAxis]?.name ?? '', coord[tensorAxis] ?? 0);
    });
    return [
        state.warp ? `W${valuesByName.get('warp') ?? 0}` : null,
        state.thread ? `T${valuesByName.get('thread') ?? 0}` : null,
        state.register ? `R${valuesByName.get('register') ?? 0}` : null,
    ].filter((value): value is string => value !== null).join('\n');
}

function linearLayoutCellLabelsForTab(
    tab: LoadedBundleDocument,
    state: LinearLayoutCellTextState,
): {
    hardwareTensorId: string;
    logicalTensorId: string;
    hardwareLabels: Array<{ coord: number[]; text: string }>;
    logicalLabels: Array<{ coord: number[]; text: string }>;
} | null {
    if (!state.warp && !state.thread && !state.register) return null;
    const spec = linearLayoutSpecForTab(tab);
    const mapping = linearLayoutSelectionMapForTab(tab);
    const hardwareTensor = hardwareLayoutTensor(tab);
    const logicalTensor = logicalLayoutTensor(tab);
    if (!spec || !mapping || !hardwareTensor || !logicalTensor) return null;
    const hardwareLabels = Array.from({ length: product(hardwareTensor.shape) }, (_entry, index) => {
        const coord = unravelIndex(index, hardwareTensor.shape);
        return { coord, text: linearLayoutCellTextForCoord(coord, spec, state) };
    });
    const logicalLabels = Array.from({ length: product(logicalTensor.shape) }, (_entry, index) => {
        const coord = unravelIndex(index, logicalTensor.shape);
        const hardwareKey = mapping.logicalToHardware.get(coordKey(coord))?.[0];
        if (!hardwareKey) return null;
        return {
            coord,
            text: linearLayoutCellTextForCoord(coordFromKey(hardwareKey), spec, state),
        };
    }).filter((entry): entry is { coord: number[]; text: string } => entry !== null);
    return {
        hardwareTensorId: mapping.hardwareTensorId,
        logicalTensorId: mapping.logicalTensorId,
        hardwareLabels,
        logicalLabels,
    };
}

function applyLinearLayoutCellText(): void {
    const tab = activeLinearLayoutTab();
    if (!tab) {
        clearCellTextLabels();
        return;
    }
    const hardwareTensor = hardwareLayoutTensor(tab);
    const logicalTensor = logicalLayoutTensor(tab);
    const labels = linearLayoutCellLabelsForTab(tab, linearLayoutCellTextState);
    if (!labels) {
        if (hardwareTensor) viewer.setTensorCellLabels(hardwareTensor.id, null);
        if (logicalTensor) viewer.setTensorCellLabels(logicalTensor.id, null);
        return;
    }
    viewer.setTensorCellLabels(labels.hardwareTensorId, labels.hardwareLabels);
    viewer.setTensorCellLabels(labels.logicalTensorId, labels.logicalLabels);
}

function slicedTensorCoords(tensorId: string): number[][] | null {
    const status = viewer.getTensorStatus(tensorId);
    const snapshot = viewer.getTensorView(tensorId);
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

function syncLinearLayoutViewFilters(): void {
    const tab = activeTab();
    if (!tab || !isLinearLayoutTab(tab)) return;
    const mapping = linearLayoutSelectionMapForTab(tab);
    if (!mapping) return;
    const hardwareSlice = slicedTensorCoords(mapping.hardwareTensorId);
    const logicalSlice = slicedTensorCoords(mapping.logicalTensorId);
    const hardwareMask = logicalSlice ? mapLogicalSelection(logicalSlice, mapping) : null;
    const logicalMask = hardwareSlice ? mapHardwareSelection(hardwareSlice, mapping) : null;
    viewer.setTensorVisibleCoords(mapping.hardwareTensorId, hardwareMask);
    viewer.setTensorVisibleCoords(mapping.logicalTensorId, logicalMask);
}

function syncLinearLayoutSelection(selection: SelectionCoords): void {
    if (syncingLinearLayoutSelection) return;
    const tab = activeTab();
    if (!tab || !isLinearLayoutTab(tab)) return;
    const mapping = linearLayoutSelectionMapForTab(tab);
    if (!mapping) return;
    const hardwareCoords = selection.get(mapping.hardwareTensorId) ?? [];
    const logicalCoords = selection.get(mapping.logicalTensorId) ?? [];
    if (hardwareCoords.length === 0 && logicalCoords.length === 0) return;
    const activeId = viewer.getState().activeTensorId;
    const source = selection.size === 1
        ? (hardwareCoords.length ? 'hardware' : 'logical')
        : activeId === mapping.hardwareTensorId
            ? 'hardware'
            : activeId === mapping.logicalTensorId
                ? 'logical'
                : hardwareCoords.length && !logicalCoords.length
                    ? 'hardware'
                    : logicalCoords.length
                        ? 'logical'
                        : 'hardware';
    const nextSelection = new Map<string, number[][]>();
    if (source === 'hardware') {
        if (hardwareCoords.length) nextSelection.set(mapping.hardwareTensorId, hardwareCoords);
        const mapped = mapHardwareSelection(hardwareCoords, mapping);
        if (mapped.length) nextSelection.set(mapping.logicalTensorId, mapped);
    } else {
        if (logicalCoords.length) nextSelection.set(mapping.logicalTensorId, logicalCoords);
        const mapped = mapLogicalSelection(logicalCoords, mapping);
        if (mapped.length) nextSelection.set(mapping.hardwareTensorId, mapped);
    }
    if (selectionsMatch(selection, nextSelection)) return;
    syncingLinearLayoutSelection = true;
    viewer.setSelectedCoords(nextSelection);
    syncingLinearLayoutSelection = false;
}

function syncLinearLayoutSelectionPreview(selection: SelectionCoords): void {
    const tab = activeTab();
    if (!tab || !isLinearLayoutTab(tab)) {
        viewer.setPreviewSelectedCoords(selection);
        return;
    }
    const mapping = linearLayoutSelectionMapForTab(tab);
    if (!mapping) {
        viewer.setPreviewSelectedCoords(selection);
        return;
    }
    const hardwareCoords = selection.get(mapping.hardwareTensorId) ?? [];
    const logicalCoords = selection.get(mapping.logicalTensorId) ?? [];
    if (hardwareCoords.length === 0 && logicalCoords.length === 0) {
        viewer.setPreviewSelectedCoords(new Map());
        return;
    }
    const activeId = viewer.getState().activeTensorId;
    const source = selection.size === 1
        ? (hardwareCoords.length ? 'hardware' : 'logical')
        : activeId === mapping.hardwareTensorId
            ? 'hardware'
            : activeId === mapping.logicalTensorId
                ? 'logical'
                : hardwareCoords.length && !logicalCoords.length
                    ? 'hardware'
                    : logicalCoords.length
                        ? 'logical'
                        : 'hardware';
    const nextSelection = new Map<string, number[][]>();
    if (source === 'hardware') {
        if (hardwareCoords.length) nextSelection.set(mapping.hardwareTensorId, hardwareCoords);
        const mapped = mapHardwareSelection(hardwareCoords, mapping);
        if (mapped.length) nextSelection.set(mapping.logicalTensorId, mapped);
    } else {
        if (logicalCoords.length) nextSelection.set(mapping.logicalTensorId, logicalCoords);
        const mapped = mapLogicalSelection(logicalCoords, mapping);
        if (mapped.length) nextSelection.set(mapping.hardwareTensorId, mapped);
    }
    viewer.setPreviewSelectedCoords(nextSelection);
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

function currentLinearLayoutBases(): Record<LinearLayoutAxis, number[][]> {
    return parseLabeledBasesText(linearLayoutState.basesText);
}

function outputDimsFromBases(bases: Record<LinearLayoutAxis, number[][]>): string[] {
    let outputRank = 0;
    LINEAR_LAYOUT_AXES.forEach((axis) => {
        bases[axis].forEach((basis) => {
            outputRank = Math.max(outputRank, basis.length);
        });
    });
    const rank = outputRank || 1;
    if (rank > OUTPUT_AXIS_NAMES.length) {
        throw new Error(`Output rank ${rank} exceeds supported axes (${OUTPUT_AXIS_NAMES.length}).`);
    }
    return OUTPUT_AXIS_NAMES.slice(0, rank).reverse();
}

function pythonOutputDimsFromBases(bases: Record<LinearLayoutAxis, number[][]>): string[] {
    let outputRank = 0;
    LINEAR_LAYOUT_AXES.forEach((axis) => {
        bases[axis].forEach((basis) => {
            outputRank = Math.max(outputRank, basis.length);
        });
    });
    const rank = outputRank || 1;
    return Array.from({ length: rank }, (_entry, axis) => String.fromCharCode(65 + axis));
}

function pythonInitCodeFromBases(bases: Record<LinearLayoutAxis, number[][]>): string {
    const outDims = pythonOutputDimsFromBases(bases);
    return [
        'LinearLayout.from_bases(',
        '    [',
        `        ("warp", ${JSON.stringify(bases.warp)}),`,
        `        ("thread", ${JSON.stringify(bases.thread)}),`,
        `        ("register", ${JSON.stringify(bases.register)}),`,
        '    ],',
        `    ${JSON.stringify(outDims)},`,
        ')',
    ].join('\n');
}

function matrixPreviewFromBases(bases: Record<LinearLayoutAxis, number[][]>): string {
    const logicalAxisColorClass = (axis: number): string => {
        const worldAxis = axis % 3 === 0 ? 1 : axis % 3 === 1 ? 0 : 2;
        return `matrix-axis-${worldAxis}`;
    };
    const hardwareAxisColorClass = (label: string): string => (
        label.startsWith('W') ? 'matrix-axis-0' : label.startsWith('T') ? 'matrix-axis-1' : 'matrix-axis-2'
    );
    const outputRank = Math.max(
        1,
        ...LINEAR_LAYOUT_AXES.flatMap((axis) => bases[axis].map((basis) => basis.length)),
    );
    const outputAxisNames = pythonOutputDimsFromBases(bases);
    const rowEntries = outputAxisNames.flatMap((axisName, axis) => {
        const bitCount = Math.max(
            1,
            ...LINEAR_LAYOUT_AXES.flatMap((layoutAxis) => bases[layoutAxis].map((basis) => (basis[axis] ?? 0).toString(2).length)),
        );
        return Array.from({ length: bitCount }, (_entry, bit) => ({ label: `${axisName}${bit}`, axis, bit }));
    });
    const columns = [
        ...bases.warp.map((_basis, index) => ({ label: `W${index}`, basis: bases.warp[index] ?? [] })),
        ...bases.thread.map((_basis, index) => ({ label: `T${index}`, basis: bases.thread[index] ?? [] })),
        ...bases.register.map((_basis, index) => ({ label: `R${index}`, basis: bases.register[index] ?? [] })),
    ];
    if (columns.length === 0) {
        const rowLabel = rowEntries[0]?.label ?? 'A0';
        const rowClass = logicalAxisColorClass(rowEntries[0]?.axis ?? 0);
        return `<span class="matrix-label ${rowClass}">${escapeInfo(rowLabel)}</span> | <span class="matrix-zero">0</span>`;
    }
    const labelWidth = Math.max(...rowEntries.map((entry) => entry.label.length), 1);
    const columnWidths = columns.map((column) => Math.max(column.label.length, 1));
    const header = `${' '.repeat(labelWidth)} | ${columns.map((column, index) => (
        `<span class="matrix-label ${hardwareAxisColorClass(column.label)}">${escapeInfo(column.label.padStart(columnWidths[index] ?? column.label.length))}</span>`
    )).join(' ')}`;
    const rows = rowEntries.map(({ label, axis, bit }) => (
        `<span class="matrix-label ${logicalAxisColorClass(axis)}">${escapeInfo(label.padStart(labelWidth))}</span> | ${columns.map((column, index) => {
            const value = (((column.basis[axis] ?? 0) >> bit) & 1) === 0 ? '0' : '1';
            const klass = value === '1' ? 'matrix-one' : 'matrix-zero';
            return `<span class="${klass}">${value.padStart(columnWidths[index] ?? 1)}</span>`;
        }).join(' ')}`
    ));
    return [header, ...rows].join('\n');
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

function refreshLinearLayoutMatrixPreview(state = linearLayoutState): void {
    try {
        linearLayoutMatrixPreview = matrixPreviewFromBases(parseLabeledBasesText(state.basesText));
    } catch {
        linearLayoutMatrixPreview = '';
    }
}

function colorAxesFromMapping(mapping: Record<LinearLayoutChannel, LinearLayoutMappingValue>): Record<string, string> | undefined {
    const entries: Array<[string, string]> = [];
    const used = new Set<string>();
    LINEAR_LAYOUT_CHANNELS.forEach((channel) => {
        const axis = mapping[channel];
        if (axis === 'none') return;
        if (used.has(axis)) {
            throw new Error(`Each axis can map to only one color channel (duplicate ${axis}).`);
        }
        used.add(axis);
        entries.push([axis, channel]);
    });
    return entries.length ? Object.fromEntries(entries) : undefined;
}

function colorRangesFromState(ranges: Record<LinearLayoutChannel, [string, string]>): Record<string, [number, number]> {
    const output: Record<string, [number, number]> = {};
    LINEAR_LAYOUT_CHANNELS.forEach((channel) => {
        const [start, end] = ranges[channel];
        const startValue = Number(start);
        const endValue = Number(end);
        if (!Number.isFinite(startValue) || !Number.isFinite(endValue)) {
            throw new Error(`${channel} range must contain two numbers.`);
        }
        output[channel] = [startValue, endValue];
    });
    return output;
}

function renderLinearLayoutColorWidget(): void {
    const activeElement = document.activeElement;
    const focusedInput = activeElement instanceof HTMLInputElement && linearLayoutColorWidget.contains(activeElement)
        ? {
            id: activeElement.id,
            start: activeElement.selectionStart,
            end: activeElement.selectionEnd,
        }
        : null;
    const channelLabels: Record<LinearLayoutChannel, string> = { H: 'Hue', S: 'Sat', L: 'Light' };
    linearLayoutColorWidget.innerHTML = `
      ${titleWithInfo('HSL Mapping', 'Select which axis drives each color channel and set channel ranges.')}
      <div class="widget-body">
        ${LINEAR_LAYOUT_CHANNELS.map((channel) => `
          <div class="inline-row mapping-row">
            <span class="range-label">${channelLabels[channel]}</span>
            <div class="mapping-drop-zone" data-channel="${channel}">
              <button class="mapping-chip mapping-chip-assigned" type="button" draggable="true" data-channel="${channel}" data-axis="${linearLayoutState.mapping[channel]}">
                ${linearLayoutState.mapping[channel]}
              </button>
            </div>
            <input id="linear-layout-${channel.toLowerCase()}-min" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges[channel][0])}" />
            <span class="range-separator">to</span>
            <input id="linear-layout-${channel.toLowerCase()}-max" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges[channel][1])}" />
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
    linearLayoutColorWidget.querySelectorAll<HTMLElement>('[draggable="true"]').forEach((element) => {
        element.addEventListener('dragstart', (event) => {
            const channel = element.dataset.channel;
            const axis = element.dataset.axis;
            if (!axis) return;
            writeDragPayload(event, channel ? { kind: 'channel', channel, axis } : { kind: 'axis', axis });
        });
    });
    linearLayoutColorWidget.querySelectorAll<HTMLElement>('.mapping-drop-zone').forEach((element) => {
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
            if (!payload || !targetChannel) return;
            const sourceChannel = payload.channel as LinearLayoutChannel;
            if (payload.kind !== 'channel' || sourceChannel === targetChannel) return;
            const sourceAxis = linearLayoutState.mapping[sourceChannel];
            linearLayoutState.mapping[sourceChannel] = linearLayoutState.mapping[targetChannel];
            linearLayoutState.mapping[targetChannel] = sourceAxis;
            renderLinearLayoutColorWidget();
        });
    });
    const hueMinInput = linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-h-min');
    const hueMaxInput = linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-h-max');
    const saturationMinInput = linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-s-min');
    const saturationMaxInput = linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-s-max');
    const lightnessMinInput = linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-l-min');
    const lightnessMaxInput = linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-l-max');
    const recolorButton = linearLayoutColorWidget.querySelector<HTMLButtonElement>('#linear-layout-recolor');
    hueMinInput?.addEventListener('input', () => {
        linearLayoutState.ranges.H[0] = hueMinInput.value;
    });
    hueMaxInput?.addEventListener('input', () => {
        linearLayoutState.ranges.H[1] = hueMaxInput.value;
    });
    saturationMinInput?.addEventListener('input', () => {
        linearLayoutState.ranges.S[0] = saturationMinInput.value;
    });
    saturationMaxInput?.addEventListener('input', () => {
        linearLayoutState.ranges.S[1] = saturationMaxInput.value;
    });
    lightnessMinInput?.addEventListener('input', () => {
        linearLayoutState.ranges.L[0] = lightnessMinInput.value;
    });
    lightnessMaxInput?.addEventListener('input', () => {
        linearLayoutState.ranges.L[1] = lightnessMaxInput.value;
    });
    recolorButton?.addEventListener('click', async () => {
        await applyLinearLayoutSpec({ silent: true });
    });
    if (focusedInput) {
        const nextInput = linearLayoutColorWidget.querySelector<HTMLInputElement>(`#${focusedInput.id}`);
        nextInput?.focus();
        if (nextInput && focusedInput.start !== null && focusedInput.end !== null) {
            nextInput.setSelectionRange(focusedInput.start, focusedInput.end);
        }
    }
}

function renderCellTextWidget(): void {
    const tab = activeLinearLayoutTab();
    if (!tab) {
        cellTextWidget.innerHTML = '';
        return;
    }
    cellTextWidget.innerHTML = `
      ${titleWithInfo('Cell Text', 'Overlay W, T, and R ids directly on hardware and logical tensor cells in 2D. Labels follow the current linear-layout tab and only appear when cells are large enough to read.')}
      <div class="widget-body">
        <p class="widget-copy">Choose which hardware ids to draw on each cell. Logical cells show the mapped hardware ids for the corresponding layout position.</p>
        <div class="checklist-field">
          <label class="checklist-row" for="cell-text-warp">
            <span>Warp Id</span>
            <input id="cell-text-warp" type="checkbox" ${linearLayoutCellTextState.warp ? 'checked' : ''} />
          </label>
          <label class="checklist-row" for="cell-text-thread">
            <span>Thread Id</span>
            <input id="cell-text-thread" type="checkbox" ${linearLayoutCellTextState.thread ? 'checked' : ''} />
          </label>
          <label class="checklist-row" for="cell-text-register">
            <span>Register Id</span>
            <input id="cell-text-register" type="checkbox" ${linearLayoutCellTextState.register ? 'checked' : ''} />
          </label>
        </div>
      </div>
    `;
    const warpInput = cellTextWidget.querySelector<HTMLInputElement>('#cell-text-warp');
    const threadInput = cellTextWidget.querySelector<HTMLInputElement>('#cell-text-thread');
    const registerInput = cellTextWidget.querySelector<HTMLInputElement>('#cell-text-register');
    const sync = (): void => {
        linearLayoutCellTextState = {
            warp: warpInput?.checked ?? false,
            thread: threadInput?.checked ?? false,
            register: registerInput?.checked ?? false,
        };
        if (tab) linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(linearLayoutCellTextState));
        applyLinearLayoutCellText();
    };
    warpInput?.addEventListener('change', sync);
    threadInput?.addEventListener('change', sync);
    registerInput?.addEventListener('change', sync);
}

function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

/** Render the browser-side linear-layout editor that powers the Pages build. */
function renderLinearLayoutWidget(): void {
    const statusClass = linearLayoutNotice?.tone === 'success' ? 'success-box' : 'error-box';
    const status = linearLayoutNotice ? `<div class="${statusClass}">${escapeInfo(linearLayoutNotice.text)}</div>` : '';
    const matrixBlock = !showLinearLayoutMatrix
        ? ''
        : `<div class="mono-block linear-layout-matrix-preview">${linearLayoutMatrixPreview}</div>`;
    linearLayoutWidget.innerHTML = `
      ${titleWithInfo('Linear Layout', 'Build hardware and logical tensors directly in the browser. Output dims are inferred from the basis vector length.')}
      <div class="widget-body">
        <p class="widget-copy">Enter one labeled line for each axis: <span class="inline-code">T:</span> for thread, <span class="inline-code">W:</span> for warp, and <span class="inline-code">R:</span> for register. Each line value should be the same JSON list you would pass to <span class="inline-code">LinearLayout.from_bases</span>.</p>
        <div class="field">
          ${labelWithInfo('Bases', 'Use exactly three labeled lines: T: for thread, W: for warp, and R: for register. Each line must be a JSON array of basis vectors.', 'linear-layout-bases')}
          <textarea id="linear-layout-bases" class="compact-textarea" rows="3" spellcheck="false">${escapeInfo(linearLayoutState.basesText)}</textarea>
        </div>
        <div class="button-row">
          <button class="primary-button" id="linear-layout-apply" type="button">Render Layout</button>
          <button class="secondary-button" id="linear-layout-copy" type="button">Copy Init Code</button>
          <button class="secondary-button" id="linear-layout-matrix" type="button">${showLinearLayoutMatrix ? 'Hide Matrix' : 'Show Matrix'}</button>
          <button class="secondary-button" id="linear-layout-reset" type="button">Reset Example</button>
        </div>
        ${matrixBlock}
        ${status}
      </div>
    `;

    const basesInput = linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-bases');
    const apply = linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-apply');
    const copy = linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-copy');
    const matrix = linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-matrix');
    const reset = linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-reset');
    if (basesInput) autosizeTextarea(basesInput);
    basesInput?.addEventListener('input', () => {
        try {
            applyLabeledBasesText(basesInput.value);
        } catch {
            linearLayoutState.basesText = basesInput.value;
        }
        autosizeTextarea(basesInput);
    });
    apply?.addEventListener('click', async () => {
        await applyLinearLayoutSpec();
    });
    copy?.addEventListener('click', async () => {
        try {
            const code = pythonInitCodeFromBases(currentLinearLayoutBases());
            await copyText(code);
            linearLayoutNotice = { tone: 'success', text: 'Copied Python init code.' };
        } catch (error) {
            linearLayoutNotice = { tone: 'error', text: error instanceof Error ? error.message : String(error) };
        }
        renderLinearLayoutWidget();
    });
    matrix?.addEventListener('click', () => {
        showLinearLayoutMatrix = !showLinearLayoutMatrix;
        renderLinearLayoutWidget();
    });
    reset?.addEventListener('click', () => {
        linearLayoutState = defaultLinearLayoutState();
        linearLayoutNotice = null;
        renderLinearLayoutWidget();
    });
    renderLinearLayoutColorWidget();
}

/** Add or replace the linear-layout tab and switch the viewer to it. */
async function upsertLinearLayoutTab(document: LoadedBundleDocument, replaceTabs = false): Promise<void> {
    const activeTab = sessionTabs.find((tab) => tab.id === activeTabId);
    const targetId = activeTab?.id ?? document.id;
    const targetTitle = activeTab?.title ?? document.title;
    const nextDocument = {
        ...document,
        id: targetId,
        title: targetTitle,
    };
    linearLayoutStates.set(targetId, cloneLinearLayoutState(linearLayoutState));
    linearLayoutCellTextStates.set(targetId, cloneLinearLayoutCellTextState(linearLayoutCellTextState));
    linearLayoutSelectionMaps.delete(targetId);
    if (replaceTabs || sessionTabs.length === 0) {
        sessionTabs = [nextDocument];
    } else {
        const index = sessionTabs.findIndex((tab) => tab.id === targetId);
        sessionTabs = index === -1
            ? [...sessionTabs, nextDocument]
            : sessionTabs.map((tab, tabIndex) => tabIndex === index ? nextDocument : tab);
    }
    await loadTab(targetId);
}

/** Parse the sidebar JSON, rebuild the layout tensors, and load them into the viewer. */
async function applyLinearLayoutSpec(
    options: { replaceTabs?: boolean; silent?: boolean } = {},
): Promise<boolean> {
    try {
        const bases = currentLinearLayoutBases();
        const spec = {
            bases: [
                ['warp', bases.warp],
                ['thread', bases.thread],
                ['register', bases.register],
            ],
            out_dims: outputDimsFromBases(bases),
            color_axes: colorAxesFromMapping(linearLayoutState.mapping),
            color_ranges: colorRangesFromState(linearLayoutState.ranges),
        };
        refreshLinearLayoutMatrixPreview();
        const document = createLinearLayoutDocument(
            parseLinearLayoutSpec(JSON.stringify(spec)),
            viewer.getSnapshot(),
        );
        storeLinearLayoutState();
        await upsertLinearLayoutTab(document, options.replaceTabs);
        const activeTitle = sessionTabs.find((tab) => tab.id === activeTabId)?.title ?? document.title;
        linearLayoutNotice = options.silent ? null : { tone: 'success', text: `Rendered ${activeTitle}.` };
        renderLinearLayoutWidget();
        return true;
    } catch (error) {
        linearLayoutNotice = {
            tone: 'error',
            text: error instanceof Error ? error.message : String(error),
        };
        renderLinearLayoutWidget();
        return false;
    }
}

function commandActions(): CommandAction[] {
    return [
        { action: 'command-palette', label: 'Command Palette', shortcut: '?', keywords: 'command palette search actions' },
        { action: 'save-svg', label: 'Save as SVG', shortcut: 'Ctrl+S', keywords: 'file save export svg vector image 2d' },
        { action: '2d', label: 'Display as 2D', shortcut: 'Ctrl+2', keywords: 'display 2d orthographic' },
        { action: '3d', label: 'Display as 3D', shortcut: 'Ctrl+3', keywords: 'display 3d perspective' },
        { action: 'mapping-contiguous', label: 'Set Contiguous Axis Family Mapping', shortcut: '', keywords: 'display axis family mapping contiguous layout' },
        { action: 'mapping-z-order', label: 'Set Z-Order Axis Family Mapping', shortcut: '', keywords: 'display axis family mapping z-order z order layout' },
        { action: 'display-gaps', label: 'Toggle Block Gaps', shortcut: '', keywords: 'display advanced block gaps spacing' },
        { action: 'collapse-hidden-axes', label: 'Toggle Collapse Hidden Axes', shortcut: '', keywords: 'display advanced collapse hidden axes slices same place' },
        { action: 'log-scale', label: 'Toggle Log Scale', shortcut: '', keywords: 'display advanced log scale heatmap colorbar' },
        { action: 'heatmap', label: 'Toggle Heatmap', shortcut: 'Ctrl+H', keywords: 'display heatmap colors' },
        { action: 'add-tab', label: 'Add New Tab', shortcut: '', keywords: 'tabs add new create layout' },
        { action: 'close-tab', label: 'Close Tab', shortcut: '', keywords: 'tabs close remove current layout' },
        { action: 'dims', label: 'Toggle Dimension Lines', shortcut: 'Ctrl+D', keywords: 'display dimensions guides labels' },
        { action: 'tensor-names', label: 'Toggle Tensor Names', shortcut: '', keywords: 'display tensor names labels title' },
        { action: 'tensor-view', label: 'Toggle Tensor View', shortcut: 'Ctrl+V', keywords: 'widgets tensor view panel' },
        { action: 'inspector', label: 'Toggle Inspector', shortcut: '', keywords: 'widgets inspector panel' },
        { action: 'selection', label: 'Toggle Selection', shortcut: '', keywords: 'widgets selection panel stats highlighted cells' },
        { action: 'colorbar', label: 'Toggle Colorbar', shortcut: '', keywords: 'widgets colorbar panel heatmap range' },
        { action: 'advanced-settings', label: 'Toggle Advanced Settings', shortcut: '', keywords: 'widgets advanced settings layout gap' },
        { action: 'view', label: 'Focus Tensor View Input', shortcut: '', keywords: 'focus tensor view input field' },
    ];
}

function tabActions(): CommandAction[] {
    return sessionTabs.map((tab) => ({
        action: `tab:${tab.id}`,
        label: tab.title,
        shortcut: tab.id === activeTabId ? 'Current' : '',
        keywords: `tab ${tab.title}`,
    }));
}

function paletteActions(): CommandAction[] {
    return commandPaletteMode === 'tabs' ? tabActions() : commandActions();
}

function fuzzyScore(candidate: string, query: string): number | null {
    let score = 0;
    let queryIndex = 0;
    let previousMatch = -1;
    for (let candidateIndex = 0; candidateIndex < candidate.length && queryIndex < query.length; candidateIndex += 1) {
        if (candidate[candidateIndex] !== query[queryIndex]) continue;
        score += 1;
        if (candidateIndex === 0 || ' -_/'.includes(candidate[candidateIndex - 1] ?? '')) score += 8;
        if (previousMatch >= 0) {
            const gap = candidateIndex - previousMatch - 1;
            if (gap === 0) score += 12;
            else score -= Math.min(4, gap);
        }
        previousMatch = candidateIndex;
        queryIndex += 1;
    }
    if (queryIndex !== query.length) return null;
    return score - Math.max(0, candidate.length - query.length) * 0.01;
}

function filteredCommandActions(): CommandAction[] {
    const query = commandPaletteInput.value.trim().toLowerCase().replace(/\s+/g, ' ');
    const actions = paletteActions();
    if (!query) return actions;
    return actions
        .map((entry) => ({
            entry,
            score: fuzzyScore(`${entry.label} ${entry.keywords}`.toLowerCase(), query),
        }))
        .filter((entry): entry is { entry: CommandAction; score: number } => entry.score !== null)
        .sort((left, right) => right.score - left.score || left.entry.label.localeCompare(right.entry.label))
        .map(({ entry }) => entry);
}

function renderCommandPalette(): void {
    if (!commandPaletteOpen) return;
    const actions = filteredCommandActions();
    if (actions.length === 0) {
        commandPaletteIndex = 0;
        commandPaletteList.innerHTML = '<div class="command-palette-empty">No matching commands.</div>';
        return;
    }
    commandPaletteIndex = Math.max(0, Math.min(commandPaletteIndex, actions.length - 1));
    commandPaletteList.replaceChildren(...actions.map((entry, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'command-palette-item';
        if (index === commandPaletteIndex) button.classList.add('active');
        button.innerHTML = `<span>${entry.label}</span><span>${entry.shortcut}</span>`;
        button.addEventListener('click', async () => {
            await runAction(entry.action);
        });
        return button;
    }));
}

function openCommandPalette(): void {
    commandPaletteMode = 'actions';
    commandPaletteOpen = true;
    commandPaletteIndex = 0;
    commandPalette.classList.remove('hidden');
    commandPaletteInput.value = '';
    commandPaletteInput.placeholder = 'Type a command';
    renderCommandPalette();
    commandPaletteInput.focus();
    commandPaletteInput.select();
}

function openTabPalette(): void {
    commandPaletteMode = 'tabs';
    commandPaletteOpen = true;
    commandPaletteIndex = 0;
    commandPalette.classList.remove('hidden');
    commandPaletteInput.value = '';
    commandPaletteInput.placeholder = 'Select a tab';
    renderCommandPalette();
    commandPaletteInput.focus();
    commandPaletteInput.select();
}

function closeCommandPalette(): void {
    if (!commandPaletteOpen) return;
    commandPaletteOpen = false;
    commandPaletteMode = 'actions';
    commandPalette.classList.add('hidden');
    commandPaletteInput.value = '';
    commandPaletteList.replaceChildren();
}

function setSidebarWidth(width: number): void {
    const maxWidth = Math.max(0, Math.min(MAX_SIDEBAR_WIDTH, app.clientWidth - MIN_VIEWPORT_WIDTH));
    const clamped = Math.max(0, Math.min(maxWidth, width));
    app.style.setProperty('--sidebar-width', `${clamped}px`);
    viewer.resize();
}

setSidebarWidth(MAX_SIDEBAR_WIDTH);

function activeTab(): LoadedBundleDocument | undefined {
    return sessionTabs.find((tab) => tab.id === activeTabId);
}

function nextTabTitle(): string {
    const used = new Set(sessionTabs.map((tab) => tab.title));
    let index = 1;
    while (used.has(`Layout ${index}`)) index += 1;
    return `Layout ${index}`;
}

function cloneTabDocument(tab: LoadedBundleDocument, id: string, title: string): LoadedBundleDocument {
    return {
        id,
        title,
        manifest: structuredClone(tab.manifest),
        tensors: new Map(Array.from(tab.tensors.entries(), ([tensorId, data]) => [tensorId, data.slice()])),
    };
}

function captureActiveTabSnapshot(): void {
    const tab = activeTab();
    if (!tab) return;
    const snapshot = viewer.getSnapshot() as ViewerSnapshot & {
        linearLayoutSpec?: LinearLayoutSpec;
        linearLayoutState?: LinearLayoutFormState;
        linearLayoutCellTextState?: LinearLayoutCellTextState;
    };
    if (isLinearLayoutTab(tab)) {
        const cloned = cloneLinearLayoutState(linearLayoutState);
        const clonedCellText = cloneLinearLayoutCellTextState(linearLayoutCellTextState);
        linearLayoutStates.set(tab.id, cloned);
        linearLayoutCellTextStates.set(tab.id, clonedCellText);
        snapshot.linearLayoutState = cloned;
        snapshot.linearLayoutCellTextState = clonedCellText;
        const linearLayoutSpec = (tab.manifest.viewer as { linearLayoutSpec?: LinearLayoutSpec }).linearLayoutSpec;
        if (linearLayoutSpec) snapshot.linearLayoutSpec = linearLayoutSpec;
    }
    tab.manifest.viewer = snapshot;
}

function renderTabStrip(): void {
    tabStrip.classList.remove('hidden');
    const tabs = sessionTabs.map((tab) => {
        const wrapper = document.createElement('div');
        wrapper.className = `tab-item${tab.id === activeTabId ? ' active' : ''}`;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'tab-button';
        button.textContent = tab.title;
        button.addEventListener('click', async () => {
            if (tab.id === activeTabId) return;
            await loadTab(tab.id);
        });
        wrapper.append(button);
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'tab-close-button';
        closeButton.textContent = 'x';
        closeButton.setAttribute('aria-label', `Close ${tab.title}`);
        closeButton.addEventListener('click', async (event) => {
            event.stopPropagation();
            if (activeTabId !== tab.id) await loadTab(tab.id);
            await closeCurrentTab();
        });
        wrapper.append(closeButton);
        return wrapper;
    });
    const addButton = document.createElement('button');
    addButton.type = 'button';
    addButton.className = 'tab-strip-action';
    addButton.textContent = 'Add New Tab';
    addButton.addEventListener('click', async () => {
        await runAction('add-tab');
    });
    const actions = document.createElement('div');
    actions.className = 'tab-strip-actions';
    actions.append(addButton);
    tabStrip.replaceChildren(...tabs, actions);
}

async function addNewTab(): Promise<void> {
    const currentTab = activeTab();
    const id = `tab-${Date.now()}`;
    const title = nextTabTitle();
    if (!currentTab) {
        const document = createLinearLayoutDocument(
            parseLinearLayoutSpec(defaultLinearLayoutSpecText()),
            viewer.getSnapshot(),
        );
        linearLayoutState = defaultLinearLayoutState();
        linearLayoutCellTextState = defaultLinearLayoutCellTextState();
        linearLayoutStates.set(id, cloneLinearLayoutState(linearLayoutState));
        linearLayoutCellTextStates.set(id, cloneLinearLayoutCellTextState(linearLayoutCellTextState));
        sessionTabs = [{ ...document, id, title }];
        await loadTab(id);
        return;
    }
    captureActiveTabSnapshot();
    const nextTab = cloneTabDocument(currentTab, id, title);
    const currentLinearLayoutState = linearLayoutStates.get(currentTab.id);
    if (currentLinearLayoutState) {
        linearLayoutStates.set(id, cloneLinearLayoutState(currentLinearLayoutState));
    }
    const currentCellTextState = linearLayoutCellTextStates.get(currentTab.id);
    if (currentCellTextState) {
        linearLayoutCellTextStates.set(id, cloneLinearLayoutCellTextState(currentCellTextState));
    }
    linearLayoutSelectionMaps.delete(id);
    sessionTabs = [...sessionTabs, nextTab];
    await loadTab(id);
}

async function closeCurrentTab(): Promise<void> {
    if (!activeTabId) return;
    const activeIndex = sessionTabs.findIndex((tab) => tab.id === activeTabId);
    if (activeIndex === -1) return;
    const nextActiveId = sessionTabs[Math.max(0, activeIndex - 1)]?.id === activeTabId
        ? sessionTabs[activeIndex + 1]?.id ?? null
        : sessionTabs[Math.max(0, activeIndex - 1)]?.id ?? null;
    linearLayoutStates.delete(activeTabId);
    linearLayoutCellTextStates.delete(activeTabId);
    linearLayoutSelectionMaps.delete(activeTabId);
    sessionTabs = sessionTabs.filter((tab) => tab.id !== activeTabId);
    activeTabId = null;
    renderTabStrip();
    if (nextActiveId) {
        await loadTab(nextActiveId);
        return;
    }
    const emptySnapshot = {
        ...viewer.getSnapshot(),
        activeTensorId: null,
        tensors: [],
    };
    viewer.loadBundleData({ version: 1, viewer: emptySnapshot, tensors: [] }, new Map());
    render(viewer.getSnapshot());
}

async function loadTab(tabId: string): Promise<void> {
    const tab = sessionTabs.find((entry) => entry.id === tabId);
    if (!tab) return;
    if (!switchingTab && activeTabId && activeTabId !== tabId) captureActiveTabSnapshot();
    switchingTab = true;
    activeTabId = tabId;
    viewer.loadBundleData(tab.manifest, tab.tensors);
    if (tab.manifest.viewer.dimensionMappingScheme) {
        viewer.setDimensionMappingScheme(tab.manifest.viewer.dimensionMappingScheme);
    }
    if (!appliedStartupWidgetDefaults) {
        viewer.toggleSelectionPanel(false);
        appliedStartupWidgetDefaults = true;
    }
    switchingTab = false;
    renderTabStrip();
    syncLinearLayoutState(tab);
    syncLinearLayoutCellTextState(tab);
    renderCellTextWidget();
    syncLinearLayoutViewFilters();
    applyLinearLayoutCellText();
    syncLinearLayoutSelectionPreview(new Map());
}

window.addEventListener('pointerup', () => {
    resizingSidebar = false;
    if (!suspendTensorViewRender) return;
    suspendTensorViewRender = false;
    render(viewer.getSnapshot());
});

window.addEventListener('pointermove', (event) => {
    if (!resizingSidebar) return;
    const bounds = app.getBoundingClientRect();
    setSidebarWidth(bounds.right - event.clientX);
});

sidebarSplitter.addEventListener('pointerdown', (event) => {
    resizingSidebar = true;
    sidebarSplitter.setPointerCapture(event.pointerId);
});

sidebarSplitter.addEventListener('dblclick', () => {
    setSidebarWidth(MAX_SIDEBAR_WIDTH);
});

commandPaletteBackdrop.addEventListener('click', () => {
    closeCommandPalette();
});

commandPaletteInput.addEventListener('input', () => {
    renderCommandPalette();
});

commandPaletteInput.addEventListener('keydown', async (event) => {
    if (event.key === 'Escape') {
        event.preventDefault();
        closeCommandPalette();
        return;
    }
    if (event.key === 'ArrowDown') {
        event.preventDefault();
        const actions = filteredCommandActions();
        if (actions.length === 0) return;
        commandPaletteIndex = Math.min(actions.length - 1, commandPaletteIndex + 1);
        renderCommandPalette();
        return;
    }
    if (event.key === 'ArrowUp') {
        event.preventDefault();
        const actions = filteredCommandActions();
        if (actions.length === 0) return;
        commandPaletteIndex = Math.max(0, commandPaletteIndex - 1);
        renderCommandPalette();
        return;
    }
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const action = filteredCommandActions()[commandPaletteIndex];
    if (!action) return;
    await runAction(action.action);
});

function updateSidebar(snapshot: ViewerSnapshot): void {
    cellTextWidget.classList.toggle('hidden', !activeLinearLayoutTab());
    tensorViewWidget.classList.toggle('hidden', !showTensorViewWidget);
    inspectorWidget.classList.toggle('hidden', !snapshot.showInspectorPanel);
    selectionWidget.classList.toggle('hidden', !snapshot.showSelectionPanel);
    advancedSettingsWidget.classList.toggle('hidden', !showAdvancedSettingsWidget);
    const model = viewer.getInspectorModel();
    colorbarWidget.classList.toggle('hidden', !showColorbarWidget || !snapshot.heatmap || model.colorRanges.length === 0);
}

function renderTensorViewWidget(snapshot: ViewerSnapshot): void {
    if (suspendTensorViewRender) return;
    const model = viewer.getInspectorModel();
    if (!model.handle) {
        tensorViewWidget.innerHTML = `${titleWithInfo('Tensor View', 'Edit the active tensor view string, inspect the preview expression, and control slice tokens.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }

    const error = viewErrors.get(model.handle.id);
    const parsedView = parseTensorView(model.handle.shape.slice(), model.viewInput, undefined, model.handle.axisLabels);
    const visibleTokens = parsedView.ok ? parsedView.spec.tokens.filter((token) => token.visible).map((token) => token.label) : [];
    const dimensionMappingScheme = snapshot.dimensionMappingScheme ?? 'z-order';
    const tensorOptions = model.tensors.map((tensor) => `
      <option value="${tensor.id}" ${tensor.id === model.handle!.id ? 'selected' : ''}>${tensor.id} (${tensor.name})</option>
    `).join('');
    tensorViewWidget.innerHTML = `
      ${titleWithInfo('Tensor View', 'Edit the active tensor view string, inspect the preview expression, and control slice tokens.')}
      <div class="widget-body">
        <div class="field">
          ${labelWithInfo('Tensor', 'Choose which loaded tensor the Tensor View editor controls.', 'tensor-select')}
          <select id="tensor-select">${tensorOptions}</select>
        </div>
        <div class="field">
          ${labelWithInfo('View String', 'Use the Tensor View grammar. Uppercase tokens stay visible, lowercase tokens become slices, and 1 inserts a singleton dimension.', 'view-input')}
          <input id="view-input" type="text" value="${model.viewInput}" placeholder="empty resets to default" />
          <div class="axis-value">${formatAxisTokens(visibleTokens, snapshot.displayMode, dimensionMappingScheme)}</div>
        </div>
        ${error ? `<div class="error-box">${error}</div>` : ''}
        <div class="field">
          ${labelWithInfo('Preview', 'Shows the implied permute, reshape, and slice operations for the current Tensor View string.')}
          <div class="mono-block">${model.preview}</div>
        </div>
        <div class="slider-list" id="slice-token-controls"></div>
      </div>
    `;

    const input = tensorViewWidget.querySelector<HTMLInputElement>('#view-input');
    const select = tensorViewWidget.querySelector<HTMLSelectElement>('#tensor-select');
    const sliceHost = tensorViewWidget.querySelector<HTMLElement>('#slice-token-controls');
    select?.addEventListener('change', () => {
        logUi('tensor-select', select.value);
        viewer.setActiveTensor(select.value);
    });
    if (input) {
        input.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            input.blur();
        });
        input.addEventListener('change', () => {
            try {
                logUi('tensor-view:change', { tensorId: model.handle!.id, value: input.value });
                viewer.setTensorView(model.handle!.id, input.value);
                syncLinearLayoutViewFilters();
                viewErrors.delete(model.handle!.id);
            } catch (error) {
                viewErrors.set(model.handle!.id, error instanceof Error ? error.message : String(error));
            }
            render(snapshot);
        });
    }

    sliceHost?.replaceChildren(...model.sliceTokens.map((token) => {
        const row = document.createElement('div');
        row.className = 'slider-row';
        row.innerHTML = `
          <label for="slice-${token.token}">${token.token}</label>
          <input id="slice-${token.token}" type="range" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
          <input id="slice-${token.token}-number" type="number" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
        `;
        const slider = row.querySelector<HTMLInputElement>(`#slice-${token.token}`);
        const number = row.querySelector<HTMLInputElement>(`#slice-${token.token}-number`);
        const applyValue = (nextValue: number): void => {
            logUi('slice-token:update', { tensorId: model.handle!.id, token: token.token, value: nextValue });
            viewer.setSliceTokenValue(model.handle!.id, token.token, nextValue);
            syncLinearLayoutViewFilters();
        };
        slider?.addEventListener('pointerdown', () => {
            suspendTensorViewRender = true;
        });
        slider?.addEventListener('input', () => {
            if (!number || !slider) return;
            number.value = slider.value;
            applyValue(Number(slider.value));
        });
        slider?.addEventListener('change', () => {
            suspendTensorViewRender = false;
            render(snapshot);
        });
        number?.addEventListener('change', () => {
            const clamped = Math.max(0, Math.min(token.size - 1, Number(number.value)));
            number.value = String(clamped);
            if (slider) slider.value = String(clamped);
            applyValue(clamped);
            suspendTensorViewRender = false;
            render(snapshot);
        });
        return row;
    }));
}

function renderInspectorWidget(snapshot: ViewerSnapshot): void {
    const model = viewer.getInspectorModel();
    const dimensionMappingScheme = snapshot.dimensionMappingScheme ?? 'z-order';
    const tab = activeTab();
    const linearLayoutTab = tab && isLinearLayoutTab(tab) ? tab : null;
    if (!model.handle) {
        inspectorReady = false;
        inspectorRefs = null;
        inspectorWidget.innerHTML = `${titleWithInfo('Inspector', 'Shows metadata for the active tensor and hover data for the current cell.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }
    if (!inspectorReady) {
        inspectorWidget.innerHTML = `
          ${titleWithInfo('Inspector', 'Shows metadata for the active tensor and hover data for the current cell.')}
          <div class="widget-body meta-grid">
            <div id="inspector-hovered-tensor"><div class="label-row"><span class="meta-label">Hovered Tensor</span>${infoButton('The loaded tensor currently under the cursor.')}</div><span class="meta-value" id="inspector-hovered-tensor-value"></span></div>
            <div id="inspector-hardware-coord"><div class="label-row"><span class="meta-label">Hardware Coord</span>${infoButton('Coordinate in hardware tensor order. For linear-layout tabs this is the T/W/R coordinate for the hovered logical or hardware cell.')}</div><span class="meta-value" id="inspector-hardware-coord-value"></span></div>
            <div id="inspector-logical-coord"><div class="label-row"><span class="meta-label">Logical Coord</span>${infoButton('Coordinate in logical tensor order. For linear-layout tabs this is the matching logical coordinate for the hovered hardware or logical cell.')}</div><span class="meta-value" id="inspector-logical-coord-value"></span></div>
            <div id="inspector-hardware-coord-binary"><div class="label-row"><span class="meta-label">Hardware Coord (Binary)</span>${infoButton('Hardware coordinate encoded in binary, padded per axis width.')}</div><span class="meta-value mono-value" id="inspector-hardware-coord-binary-value"></span></div>
            <div id="inspector-logical-coord-binary"><div class="label-row"><span class="meta-label">Logical Coord (Binary)</span>${infoButton('Logical coordinate encoded in binary, padded per axis width.')}</div><span class="meta-value mono-value" id="inspector-logical-coord-binary-value"></span></div>
            <div><div class="label-row"><span class="meta-label">Tensor Shape</span>${infoButton('Original tensor shape before Tensor View transformations.')}</div><span class="meta-value" id="inspector-tensor-shape"></span></div>
            <div><div class="label-row"><span class="meta-label">Rank</span>${infoButton('Number of dimensions in the original tensor.')}</div><span class="meta-value" id="inspector-rank"></span></div>
          </div>
        `;
        inspectorRefs = {
            hoveredTensor: inspectorWidget.querySelector<HTMLDivElement>('#inspector-hovered-tensor')!,
            hardwareCoord: inspectorWidget.querySelector<HTMLDivElement>('#inspector-hardware-coord')!,
            logicalCoord: inspectorWidget.querySelector<HTMLDivElement>('#inspector-logical-coord')!,
            hardwareCoordBinary: inspectorWidget.querySelector<HTMLDivElement>('#inspector-hardware-coord-binary')!,
            logicalCoordBinary: inspectorWidget.querySelector<HTMLDivElement>('#inspector-logical-coord-binary')!,
            hoveredTensorValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-hovered-tensor-value')!,
            hardwareCoordValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-hardware-coord-value')!,
            logicalCoordValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-logical-coord-value')!,
            hardwareCoordBinaryValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-hardware-coord-binary-value')!,
            logicalCoordBinaryValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-logical-coord-binary-value')!,
            tensorShapeValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-tensor-shape')!,
            rankValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-rank')!,
        };
        inspectorReady = true;
    }
    if (!inspectorRefs) return;
    const hover = viewer.getHover();
    const hoveredStatus = hover ? viewer.getTensorStatus(hover.tensorId) : null;
    const linearLayout = linearLayoutTab ? linearLayoutSelectionMapForTab(linearLayoutTab) : null;
    const hardwareStatus = linearLayout ? viewer.getTensorStatus(linearLayout.hardwareTensorId) : null;
    const logicalStatus = linearLayout ? viewer.getTensorStatus(linearLayout.logicalTensorId) : null;
    const hardwareCoord = !hover || !linearLayout
        ? hover?.layoutCoord ?? null
        : hover.tensorId === linearLayout.hardwareTensorId
            ? hover.tensorCoord
            : coordFromKey(linearLayout.logicalToHardware.get(coordKey(hover.tensorCoord))?.[0] ?? '');
    const logicalCoord = !hover || !linearLayout
        ? hover?.tensorCoord ?? null
        : hover.tensorId === linearLayout.logicalTensorId
            ? hover.tensorCoord
            : coordFromKey(linearLayout.hardwareToLogical.get(coordKey(hover.tensorCoord)) ?? '');
    const hardwareLabels = hardwareStatus?.axisLabels ?? [];
    const logicalLabels = logicalStatus?.axisLabels ?? [];
    const binaryCoord = (coord: number[] | null, shape: readonly number[] | undefined): string => {
        if (!coord || !shape) return '';
        return formatAxisTokens(
            coord.map((value, axis) => {
                const width = Math.max(1, Math.ceil(Math.log2(Math.max(2, shape[axis] ?? 1))));
                return value.toString(2).padStart(width, '0');
            }),
            snapshot.displayMode,
            dimensionMappingScheme,
        );
    };
    inspectorRefs.hoveredTensor.classList.toggle('hidden', !hover);
    inspectorRefs.hardwareCoord.classList.toggle('hidden', !hover);
    inspectorRefs.logicalCoord.classList.toggle('hidden', !hover);
    inspectorRefs.hardwareCoordBinary.classList.toggle('hidden', !hover);
    inspectorRefs.logicalCoordBinary.classList.toggle('hidden', !hover);
    inspectorRefs.hoveredTensorValue.textContent = hover?.tensorName ?? '';
    inspectorRefs.hardwareCoordValue.innerHTML = !hardwareCoord
        ? ''
        : formatNamedAxisValues(
            hardwareLabels,
            hardwareCoord,
            snapshot.displayMode,
            dimensionMappingScheme,
        );
    inspectorRefs.logicalCoordValue.innerHTML = !logicalCoord
        ? ''
        : formatNamedAxisValues(
            logicalLabels,
            logicalCoord,
            snapshot.displayMode,
            dimensionMappingScheme,
        );
    inspectorRefs.hardwareCoordBinaryValue.innerHTML = binaryCoord(hardwareCoord, hardwareStatus?.shape);
    inspectorRefs.logicalCoordBinaryValue.innerHTML = binaryCoord(logicalCoord, logicalStatus?.shape);
    inspectorRefs.tensorShapeValue.innerHTML = formatAxisValues(
        hoveredStatus?.shape ?? model.handle.shape,
        snapshot.displayMode,
        dimensionMappingScheme,
    );
    inspectorRefs.rankValue.textContent = String(hoveredStatus?.rank ?? model.handle.rank);
}

function renderSelectionWidget(snapshot: ViewerSnapshot): void {
    const model = viewer.getInspectorModel();
    if (!model.handle) {
        selectionWidget.innerHTML = `${titleWithInfo('Selection', 'Shows how many cells are highlighted and summary statistics across their loaded numeric values. Selection is only available in 2D contiguous mapping.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }
    const summary = viewer.getSelectionSummary();
    const enabled = selectionEnabled(snapshot);
    const note = enabled
        ? 'Left-click and drag to draw a selection box, then release to apply it. Hold Shift to add cells, or hold Ctrl to remove cells.'
        : 'Selection is only available in 2D contiguous mapping.';
    selectionWidget.innerHTML = `
      ${titleWithInfo('Selection', 'Shows how many cells are highlighted and summary statistics across their loaded numeric values. Selection is only available in 2D contiguous mapping.')}
      <div class="widget-body meta-grid">
        <div><div class="label-row"><span class="meta-label">Highlighted Cells</span>${infoButton(note)}</div><span class="meta-value">${selectionCountValue(summary, enabled)}</span></div>
        <div><div class="label-row"><span class="meta-label">Min</span>${infoButton('Minimum across the selected cells with loaded values.')}</div><span class="meta-value">${selectionStatValue(summary, enabled, 'min')}</span></div>
        <div><div class="label-row"><span class="meta-label">25th</span>${infoButton('25th percentile across the selected cells with loaded values.')}</div><span class="meta-value">${selectionStatValue(summary, enabled, 'p25')}</span></div>
        <div><div class="label-row"><span class="meta-label">50th</span>${infoButton('Median across the selected cells with loaded values.')}</div><span class="meta-value">${selectionStatValue(summary, enabled, 'p50')}</span></div>
        <div><div class="label-row"><span class="meta-label">75th</span>${infoButton('75th percentile across the selected cells with loaded values.')}</div><span class="meta-value">${selectionStatValue(summary, enabled, 'p75')}</span></div>
        <div><div class="label-row"><span class="meta-label">Max</span>${infoButton('Maximum across the selected cells with loaded values.')}</div><span class="meta-value">${selectionStatValue(summary, enabled, 'max')}</span></div>
        <div><div class="label-row"><span class="meta-label">Mean</span>${infoButton('Mean across the selected cells with loaded values.')}</div><span class="meta-value">${selectionStatValue(summary, enabled, 'mean')}</span></div>
        <div><div class="label-row"><span class="meta-label">Std</span>${infoButton('Population standard deviation across the selected cells with loaded values.')}</div><span class="meta-value">${selectionStatValue(summary, enabled, 'std')}</span></div>
      </div>
    `;
}

function renderColorbarWidget(snapshot: ViewerSnapshot): void {
    const model = viewer.getInspectorModel();
    if (!snapshot.heatmap || model.colorRanges.length === 0) {
        colorbarWidget.innerHTML = '';
        return;
    }
    const scaleMode = snapshot.logScale ? '<div class="colorbar-mode">Log Scale</div>' : '<div class="colorbar-mode">Linear Scale</div>';
    const sections = model.colorRanges.map((range) => `
      <div class="colorbar-section">
        <div class="colorbar-title">${range.id} (${range.name})</div>
        <div class="colorbar"></div>
        <div class="colorbar-labels">
          <span>${formatRangeValue(range.min)}</span>
          <span>${formatRangeValue(range.max)}</span>
        </div>
      </div>
    `).join('');
    colorbarWidget.innerHTML = `
      ${titleWithInfo('Colorbar', 'Shows the heatmap range for each loaded tensor using its current minimum and maximum values. The gradient can be linear or signed-log depending on Advanced Settings.')}
      <div class="widget-body">
        ${scaleMode}
        ${sections}
      </div>
    `;
}

function renderAdvancedSettingsWidget(snapshot: ViewerSnapshot): void {
    const currentValue = snapshot.dimensionBlockGapMultiple ?? 3;
    const displayGaps = snapshot.displayGaps ?? true;
    const logScale = snapshot.logScale ?? false;
    const collapseHiddenAxes = snapshot.collapseHiddenAxes ?? snapshot.showSlicesInSamePlace ?? false;
    const dimensionMappingScheme = snapshot.dimensionMappingScheme ?? 'z-order';
    advancedSettingsWidget.innerHTML = `
      ${titleWithInfo('Advanced Settings', 'Adjust lower-level layout tuning that changes how tensor dimension blocks are spaced and assigned to x, y, and z families.')}
      <div class="widget-body">
        <div class="field">
          ${labelWithInfo('Block Gap Scale', 'Sets the factor used to grow the gap between higher-level dimension blocks in both 2D and 3D layouts.', 'dimension-block-gap-multiple')}
          <input id="dimension-block-gap-multiple" type="number" min="1" step="0.25" value="${currentValue}" />
        </div>
        <div class="field">
          ${labelWithInfo('Axis Family Mapping', 'Controls how tensor dimensions are assigned to the x, y, and z layout families. Z-Order alternates from the last axis. Contiguous keeps nearby axes in the same family.', 'dimension-mapping-scheme')}
          <select id="dimension-mapping-scheme">
            <option value="z-order" ${dimensionMappingScheme === 'z-order' ? 'selected' : ''}>Z-Order</option>
            <option value="contiguous" ${dimensionMappingScheme === 'contiguous' ? 'selected' : ''}>Contiguous</option>
          </select>
        </div>
        <label class="toggle-field" for="display-gaps">
          <span>Show Block Gaps</span>
          <input id="display-gaps" type="checkbox" ${displayGaps ? 'checked' : ''} />
        </label>
        <label class="toggle-field" for="collapse-hidden-axes">
          <span class="label-row">
            <span>Collapse Hidden Axes</span>
            ${infoButton('When enabled, sliced views are rendered using only their visible dimensions, so different slices occupy the same position and the outline is based on visible axes only.')}
          </span>
          <input id="collapse-hidden-axes" type="checkbox" ${collapseHiddenAxes ? 'checked' : ''} />
        </label>
        <label class="toggle-field" for="log-scale">
          <span class="label-row">
            <span>Log</span>
            ${infoButton('Uses signed-log scaling for the heatmap and colorbar so large magnitudes are compressed while preserving negative versus positive values.')}
          </span>
          <input id="log-scale" type="checkbox" ${logScale ? 'checked' : ''} />
        </label>
      </div>
    `;
    const input = advancedSettingsWidget.querySelector<HTMLInputElement>('#dimension-block-gap-multiple');
    const dimensionMappingSchemeInput = advancedSettingsWidget.querySelector<HTMLSelectElement>('#dimension-mapping-scheme');
    const displayGapsInput = advancedSettingsWidget.querySelector<HTMLInputElement>('#display-gaps');
    const collapseHiddenAxesInput = advancedSettingsWidget.querySelector<HTMLInputElement>('#collapse-hidden-axes');
    const logScaleInput = advancedSettingsWidget.querySelector<HTMLInputElement>('#log-scale');
    if (!input) return;
    input.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        input.blur();
    });
    input.addEventListener('change', () => {
        const nextValue = viewer.setDimensionBlockGapMultiple(Number(input.value));
        input.value = String(nextValue);
        logUi('advanced-settings:dimension-block-gap-multiple', nextValue);
    });
    dimensionMappingSchemeInput?.addEventListener('change', () => {
        const nextValue = viewer.setDimensionMappingScheme(dimensionMappingSchemeInput.value as DimensionMappingScheme);
        dimensionMappingSchemeInput.value = nextValue;
        logUi('advanced-settings:dimension-mapping-scheme', nextValue);
    });
    displayGapsInput?.addEventListener('change', () => {
        const nextValue = viewer.toggleDisplayGaps(displayGapsInput.checked);
        displayGapsInput.checked = nextValue;
        logUi('advanced-settings:display-gaps', nextValue);
    });
    collapseHiddenAxesInput?.addEventListener('change', () => {
        const nextValue = viewer.toggleCollapseHiddenAxes(collapseHiddenAxesInput.checked);
        collapseHiddenAxesInput.checked = nextValue;
        logUi('advanced-settings:collapse-hidden-axes', nextValue);
    });
    logScaleInput?.addEventListener('change', () => {
        const nextValue = viewer.toggleLogScale(logScaleInput.checked);
        logScaleInput.checked = nextValue;
        logUi('advanced-settings:log-scale', nextValue);
    });
}

function render(snapshot: ViewerSnapshot): void {
    if (!switchingTab) captureActiveTabSnapshot();
    updateSidebar(snapshot);
    renderTabStrip();
    renderCellTextWidget();
    renderTensorViewWidget(snapshot);
    renderInspectorWidget(snapshot);
    renderSelectionWidget(snapshot);
    renderAdvancedSettingsWidget(snapshot);
    renderColorbarWidget(snapshot);
}

function sanitizeFilename(name: string): string {
    return name.replace(/[^a-z0-9._-]+/gi, '_').replace(/^_+|_+$/g, '') || 'tensor';
}

async function saveTensorToDisk(): Promise<void> {
    const blob = await viewer.saveFile();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${sanitizeFilename(viewer.getInspectorModel().handle?.name ?? 'tensor')}.npy`;
    anchor.click();
    URL.revokeObjectURL(url);
}

async function saveSvgToDisk(): Promise<void> {
    const blob = viewer.saveSvg();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${sanitizeFilename(activeTab()?.title ?? 'tensor-view')}.svg`;
    anchor.click();
    URL.revokeObjectURL(url);
}

/** loads one tab's raw tensor payloads from the local python session server. */
async function loadTabTensors(tensors: BundleManifest['tensors']): Promise<Map<string, NumericArray>> {
    const entries = await Promise.all(tensors
        .filter((tensor) => tensor.dataFile)
        .map(async (tensor) => {
            const dataFile = tensor.dataFile;
            if (!dataFile) throw new Error(`Session tensor ${tensor.id} has no data file.`);
            const response = await fetch(`/api/${dataFile}`, { cache: 'no-store' });
            if (!response.ok) throw new Error(`Missing tensor payload ${dataFile}.`);
            return [tensor.id, createTypedArray(tensor.dtype, await response.arrayBuffer())] as const;
        }));
    return new Map(entries);
}

/** loads one session tab from the raw manifest plus tensor-byte endpoints. */
async function loadSessionTab(tab: SessionBundleManifest['tabs'][number]): Promise<LoadedBundleDocument> {
    const tensorNames = tab.tensors.map((tensor) => tensor.name.toLowerCase());
    const isLinearLayout = tensorNames.some((name) => HARDWARE_LAYOUT_NAMES.has(name))
        && tensorNames.some((name) => LOGICAL_LAYOUT_NAMES.has(name));
    const viewerState = {
        ...tab.viewer,
        dimensionMappingScheme: isLinearLayout ? 'contiguous' : tab.viewer.dimensionMappingScheme,
        showSelectionPanel: false,
    };
    const storedLinearLayoutState = (viewerState as { linearLayoutState?: unknown }).linearLayoutState;
    if (isLinearLayout && isLinearLayoutState(storedLinearLayoutState)) {
        linearLayoutStates.set(tab.id, cloneLinearLayoutState(storedLinearLayoutState));
    }
    const storedCellTextState = (viewerState as { linearLayoutCellTextState?: unknown }).linearLayoutCellTextState;
    if (isLinearLayout && isLinearLayoutCellTextState(storedCellTextState)) {
        linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(storedCellTextState));
    }
    return {
        id: tab.id,
        title: tab.title,
        manifest: {
            version: 1,
            viewer: viewerState,
            tensors: tab.tensors,
        },
        tensors: await loadTabTensors(tab.tensors),
    };
}

async function tryLoadSession(): Promise<boolean> {
    const response = await fetch('/api/session.json', { cache: 'no-store' });
    if (!response.ok) return false;
    const manifest = await response.json() as SessionBundleManifest;
    if (manifest.version !== 1) throw new Error(`Unsupported session version ${manifest.version}.`);
    const initialMapping = manifest.tabs[0]?.viewer.dimensionMappingScheme;
    if (initialMapping) viewer.setDimensionMappingScheme(initialMapping);
    linearLayoutSelectionMaps.clear();
    sessionTabs = await Promise.all(manifest.tabs.map((tab) => loadSessionTab(tab)));
    activeTabId = null;
    const initialTabId = sessionTabs[0]?.id ?? null;
    if (initialTabId) await loadTab(initialTabId);
    return true;
}

function seedDemoTensor(): void {
    sessionTabs = [];
    activeTabId = null;
    linearLayoutSelectionMaps.clear();
    const shape = [4, 4, 4];
    const data = new Float64Array(product(shape));
    for (let index = 0; index < data.length; index += 1) {
        data[index] = Math.sin(index / 3) * 10;
    }
    viewer.addTensor(shape, data, 'Sample');
}

async function loadBakedLinearLayoutTabs(): Promise<boolean> {
    const specs = bakedLinearLayoutSpecTexts();
    if (specs.length === 0) return false;
    const baseViewer = viewer.getSnapshot();
    sessionTabs = specs.map((text, index) => {
        const parsed = parseLinearLayoutSpec(text);
        const linearLayoutState = stateFromLayoutSpec(JSON.parse(text));
        const bases = parseLabeledBasesText(linearLayoutState.basesText);
        const document = createLinearLayoutDocument(parseLinearLayoutSpec(JSON.stringify({
            name: parsed.name,
            bases: [
                ['warp', bases.warp],
                ['thread', bases.thread],
                ['register', bases.register],
            ],
            out_dims: outputDimsFromBases(bases),
            color_axes: colorAxesFromMapping(linearLayoutState.mapping),
            color_ranges: colorRangesFromState(linearLayoutState.ranges),
        })), baseViewer);
        (document.manifest.viewer as ViewerSnapshot & { linearLayoutState?: LinearLayoutFormState }).linearLayoutState = linearLayoutState;
        linearLayoutStates.set(`tab-${index + 1}`, cloneLinearLayoutState(linearLayoutState));
        return {
            ...document,
            id: `tab-${index + 1}`,
        };
    });
    activeTabId = null;
    linearLayoutSelectionMaps.clear();
    const initialTabId = sessionTabs[0]?.id ?? null;
    if (!initialTabId) return false;
    await loadTab(initialTabId);
    return true;
}

async function openLocalFile(file: File): Promise<void> {
    sessionTabs = [];
    activeTabId = null;
    linearLayoutSelectionMaps.clear();
    renderTabStrip();
    await viewer.openFile(file);
}

async function runAction(action: string): Promise<void> {
    logUi('action', action);
    closeCommandPalette();
    if (action.startsWith('tab:')) {
        await loadTab(action.slice(4));
        return;
    }
    switch (action) {
        case 'command-palette':
            openCommandPalette();
            return;
        case 'open':
            fileInput.click();
            return;
        case 'save':
            await saveTensorToDisk();
            return;
        case 'save-svg':
            try {
                await saveSvgToDisk();
            } catch (error) {
                window.alert(error instanceof Error ? error.message : String(error));
            }
            return;
        case '2d':
            viewer.setDisplayMode('2d');
            return;
        case '3d':
            viewer.setDisplayMode('3d');
            return;
        case 'mapping-contiguous':
            viewer.setDimensionMappingScheme('contiguous');
            return;
        case 'mapping-z-order':
            viewer.setDimensionMappingScheme('z-order');
            return;
        case 'display-gaps':
            viewer.toggleDisplayGaps();
            return;
        case 'collapse-hidden-axes':
            viewer.toggleCollapseHiddenAxes();
            return;
        case 'log-scale':
            viewer.toggleLogScale();
            return;
        case 'heatmap':
            viewer.toggleHeatmap();
            return;
        case 'add-tab':
            await addNewTab();
            return;
        case 'close-tab':
            await closeCurrentTab();
            return;
        case 'tensor-view':
            showTensorViewWidget = !showTensorViewWidget;
            render(viewer.getSnapshot());
            return;
        case 'selection':
            viewer.toggleSelectionPanel();
            return;
        case 'colorbar':
            showColorbarWidget = !showColorbarWidget;
            render(viewer.getSnapshot());
            return;
        case 'advanced-settings':
            showAdvancedSettingsWidget = !showAdvancedSettingsWidget;
            render(viewer.getSnapshot());
            return;
        case 'view': {
        const input = tensorViewWidget.querySelector<HTMLInputElement>('#view-input');
        input?.focus();
        input?.select();
        logUi('tensor-view:focus');
        return;
        }
        case 'dims':
            viewer.toggleDimensionLines();
            return;
        case 'tensor-names':
            viewer.toggleTensorNames();
            return;
        case 'inspector':
            viewer.toggleInspectorPanel();
            return;
    }
}

document.querySelectorAll<HTMLButtonElement>('.menu-list button').forEach((button) => {
    button.addEventListener('click', async () => {
        logUi('menu:click', button.dataset.action ?? '');
        await runAction(button.dataset.action ?? '');
        button.blur();
    });
});

fileInput.addEventListener('change', async () => {
    const [file] = Array.from(fileInput.files ?? []);
    if (!file) return;
    logUi('file:input', file.name);
    await openLocalFile(file);
    fileInput.value = '';
});

window.addEventListener('keydown', async (event) => {
    const target = event.target as HTMLElement | null;
    const isEditing = target && ['INPUT', 'TEXTAREA'].includes(target.tagName);
    const isPaletteInput = target === commandPaletteInput;
    if (isPaletteInput && event.key === 'Escape') {
        event.preventDefault();
        closeCommandPalette();
        return;
    }
    if (isEditing && !isPaletteInput && !(event.ctrlKey && event.key.toLowerCase() === 's')) return;

    if (event.ctrlKey && event.key.toLowerCase() === 'o') {
        event.preventDefault();
        await runAction('open');
    } else if (event.ctrlKey && event.key.toLowerCase() === 's') {
        event.preventDefault();
        await runAction('save-svg');
    } else if (event.ctrlKey && event.key === '2') {
        event.preventDefault();
        await runAction('2d');
    } else if (event.ctrlKey && event.key === '3') {
        event.preventDefault();
        await runAction('3d');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'h') {
        event.preventDefault();
        await runAction('heatmap');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'd') {
        event.preventDefault();
        await runAction('dims');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'v') {
        event.preventDefault();
        await runAction('tensor-view');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'p') {
        event.preventDefault();
        if (commandPaletteOpen && commandPaletteMode === 'tabs') closeCommandPalette();
        else openTabPalette();
    } else if (event.key === '?' || (event.shiftKey && event.key === '/')) {
        event.preventDefault();
        if (commandPaletteOpen) closeCommandPalette();
        else openCommandPalette();
    } else if (event.key === 'Escape') {
        closeCommandPalette();
    }
});

viewer.subscribe(render);
viewer.subscribeHover(() => renderInspectorWidget(viewer.getSnapshot()));
viewer.subscribeSelectionPreview((selection) => {
    syncLinearLayoutSelectionPreview(selection);
});
viewer.subscribeSelection((selection) => {
    renderSelectionWidget(viewer.getSnapshot());
    syncLinearLayoutSelection(selection);
});
renderLinearLayoutWidget();

tryLoadSession().then(async (loaded) => {
    if (loaded) return;
    if (await loadBakedLinearLayoutTabs()) return;
    seedDemoTensor();
}).catch(async () => {
    if (await loadBakedLinearLayoutTabs()) return;
    seedDemoTensor();
});
}
