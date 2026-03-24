import {
    createTypedArray,
    parseTensorView,
    TensorViewer,
    product,
    type BundleManifest,
    type DimensionMappingScheme,
    type LoadedBundleDocument,
    type NumericArray,
    type SessionBundleManifest,
    type ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import {
    escapeInfo,
    formatAxisTokens,
    formatAxisValues,
    formatRangeValue,
    infoButton,
    labelWithInfo,
    selectionEnabled,
    titleWithInfo,
} from './app-format.js';
import {
    createLinearLayoutDocument,
    defaultLinearLayoutSpecText,
    parseLinearLayoutSpec,
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
const viewErrors = new Map<string, string>();
let suspendTensorViewRender = false;
let showTensorViewWidget = true;
let showAdvancedSettingsWidget = false;
let inspectorReady = false;
let sessionTabs: LoadedBundleDocument[] = [];
let activeTabId: string | null = null;
let switchingTab = false;
let resizingSidebar = false;
let commandPaletteOpen = false;
let commandPaletteIndex = 0;
let commandPaletteMode: 'actions' | 'tabs' = 'actions';

const MIN_VIEWPORT_WIDTH = 280;
const MAX_SIDEBAR_WIDTH = 720;

type InspectorRefs = {
    hoveredTensor: HTMLDivElement;
    layoutCoord: HTMLDivElement;
    tensorCoord: HTMLDivElement;
    value: HTMLDivElement;
    hoveredTensorValue: HTMLSpanElement;
    layoutCoordValue: HTMLSpanElement;
    tensorCoordValue: HTMLSpanElement;
    valueField: HTMLSpanElement;
    dtypeValue: HTMLSpanElement;
    tensorShapeValue: HTMLSpanElement;
    rankValue: HTMLSpanElement;
};

let inspectorRefs: InspectorRefs | null = null;
type LinearLayoutNotice = {
    tone: 'error' | 'success';
    text: string;
};

const LINEAR_LAYOUT_AXES = ['thread', 'warp', 'register'] as const;
const LINEAR_LAYOUT_CHANNELS = ['H', 'S', 'L'] as const;
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
    mapping: Record<LinearLayoutChannel, LinearLayoutMappingValue>;
    ranges: Record<LinearLayoutChannel, [string, string]>;
};

const LINEAR_LAYOUT_STORAGE_KEY = 'tensor-viz-linear-layout-spec';
let linearLayoutState = loadLinearLayoutState();
const linearLayoutStates = new Map<string, LinearLayoutFormState>();
let linearLayoutNotice: LinearLayoutNotice | null = null;

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
        if (isLinearLayoutState(parsed)) return parsed;
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
        && LINEAR_LAYOUT_CHANNELS.every((channel) => typeof record.mapping?.[channel] === 'string')
        && LINEAR_LAYOUT_CHANNELS.every((channel) => Array.isArray(record.ranges?.[channel]));
}

function defaultLinearLayoutState(): LinearLayoutFormState {
    return stateFromLayoutSpec(JSON.parse(defaultLinearLayoutSpecText()));
}

function cloneLinearLayoutState(state: LinearLayoutFormState): LinearLayoutFormState {
    return {
        bases: { ...state.bases },
        mapping: { ...state.mapping },
        ranges: {
            H: [...state.ranges.H],
            S: [...state.ranges.S],
            L: [...state.ranges.L],
        } as Record<LinearLayoutChannel, [string, string]>,
    };
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
        mapping,
        ranges,
    };
}

function formatBases(value: unknown): string {
    return JSON.stringify(value ?? []);
}

function isLinearLayoutTab(tab: LoadedBundleDocument): boolean {
    const names = tab.manifest.tensors.map((tensor) => tensor.name.toLowerCase());
    return names.includes('hardware tensor') && names.includes('logical tensor');
}

function syncLinearLayoutState(tab: LoadedBundleDocument): void {
    if (!isLinearLayoutTab(tab)) return;
    const stored = linearLayoutStates.get(tab.id);
    if (stored) {
        linearLayoutState = cloneLinearLayoutState(stored);
        renderLinearLayoutWidget();
        return;
    }
    const candidate = (tab.manifest.viewer as { linearLayoutState?: unknown }).linearLayoutState;
    if (isLinearLayoutState(candidate)) {
        linearLayoutState = cloneLinearLayoutState(candidate);
        linearLayoutStates.set(tab.id, cloneLinearLayoutState(candidate));
        renderLinearLayoutWidget();
        return;
    }
    linearLayoutState = defaultLinearLayoutState();
    linearLayoutStates.set(tab.id, cloneLinearLayoutState(linearLayoutState));
    renderLinearLayoutWidget();
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

/** Render the browser-side linear-layout editor that powers the Pages build. */
function renderLinearLayoutWidget(): void {
    const statusClass = linearLayoutNotice?.tone === 'success' ? 'success-box' : 'error-box';
    const status = linearLayoutNotice ? `<div class="${statusClass}">${escapeInfo(linearLayoutNotice.text)}</div>` : '';
    const axisOptions = (value: LinearLayoutMappingValue): string => [
        `<option value="none" ${value === 'none' ? 'selected' : ''}>None</option>`,
        ...LINEAR_LAYOUT_AXES.map((axis) => `<option value="${axis}" ${value === axis ? 'selected' : ''}>${axis}</option>`),
    ].join('');
    linearLayoutWidget.innerHTML = `
      ${titleWithInfo('Linear Layout', 'Build hardware and logical tensors directly in the browser. Output dims are inferred from the basis vector length.')}
      <div class="widget-body">
        <p class="widget-copy">Enter basis vectors for each axis. Use the same JSON lists you would pass to <span class="inline-code">LinearLayout.from_bases</span>.</p>
        <div class="field">
          ${labelWithInfo('Thread Bases', 'Basis vectors for the thread axis.', 'linear-layout-thread')}
          <textarea id="linear-layout-thread" class="compact-textarea" rows="4" spellcheck="false">${escapeInfo(linearLayoutState.bases.thread)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Warp Bases', 'Basis vectors for the warp axis.', 'linear-layout-warp')}
          <textarea id="linear-layout-warp" class="compact-textarea" rows="4" spellcheck="false">${escapeInfo(linearLayoutState.bases.warp)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Register Bases', 'Basis vectors for the register axis.', 'linear-layout-register')}
          <textarea id="linear-layout-register" class="compact-textarea" rows="4" spellcheck="false">${escapeInfo(linearLayoutState.bases.register)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('HSL Mapping', 'Select which axis drives each color channel and set channel ranges.', 'linear-layout-color')}
          <div class="inline-row">
            <span class="range-label">Hue</span>
            <select id="linear-layout-hue-axis">${axisOptions(linearLayoutState.mapping.H)}</select>
            <input id="linear-layout-hue-min" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges.H[0])}" />
            <span class="range-separator">to</span>
            <input id="linear-layout-hue-max" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges.H[1])}" />
          </div>
          <div class="inline-row">
            <span class="range-label">Sat</span>
            <select id="linear-layout-saturation-axis">${axisOptions(linearLayoutState.mapping.S)}</select>
            <input id="linear-layout-saturation-min" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges.S[0])}" />
            <span class="range-separator">to</span>
            <input id="linear-layout-saturation-max" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges.S[1])}" />
          </div>
          <div class="inline-row">
            <span class="range-label">Light</span>
            <select id="linear-layout-lightness-axis">${axisOptions(linearLayoutState.mapping.L)}</select>
            <input id="linear-layout-lightness-min" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges.L[0])}" />
            <span class="range-separator">to</span>
            <input id="linear-layout-lightness-max" type="number" step="0.01" value="${escapeInfo(linearLayoutState.ranges.L[1])}" />
          </div>
        </div>
        <div class="button-row">
          <button class="primary-button" id="linear-layout-apply" type="button">Render Layout</button>
          <button class="secondary-button" id="linear-layout-reset" type="button">Reset Example</button>
        </div>
        ${status}
      </div>
    `;

    const threadInput = linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-thread');
    const warpInput = linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-warp');
    const registerInput = linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-register');
    const hueAxisInput = linearLayoutWidget.querySelector<HTMLSelectElement>('#linear-layout-hue-axis');
    const saturationAxisInput = linearLayoutWidget.querySelector<HTMLSelectElement>('#linear-layout-saturation-axis');
    const lightnessAxisInput = linearLayoutWidget.querySelector<HTMLSelectElement>('#linear-layout-lightness-axis');
    const hueMinInput = linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-hue-min');
    const hueMaxInput = linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-hue-max');
    const saturationMinInput = linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-saturation-min');
    const saturationMaxInput = linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-saturation-max');
    const lightnessMinInput = linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-lightness-min');
    const lightnessMaxInput = linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-lightness-max');
    const apply = linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-apply');
    const reset = linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-reset');
    threadInput?.addEventListener('input', () => {
        linearLayoutState.bases.thread = threadInput.value;
    });
    warpInput?.addEventListener('input', () => {
        linearLayoutState.bases.warp = warpInput.value;
    });
    registerInput?.addEventListener('input', () => {
        linearLayoutState.bases.register = registerInput.value;
    });
    hueAxisInput?.addEventListener('change', () => {
        linearLayoutState.mapping.H = (hueAxisInput.value as LinearLayoutMappingValue) ?? 'none';
    });
    saturationAxisInput?.addEventListener('change', () => {
        linearLayoutState.mapping.S = (saturationAxisInput.value as LinearLayoutMappingValue) ?? 'none';
    });
    lightnessAxisInput?.addEventListener('change', () => {
        linearLayoutState.mapping.L = (lightnessAxisInput.value as LinearLayoutMappingValue) ?? 'none';
    });
    hueMinInput?.addEventListener('change', () => {
        linearLayoutState.ranges.H[0] = hueMinInput.value;
    });
    hueMaxInput?.addEventListener('change', () => {
        linearLayoutState.ranges.H[1] = hueMaxInput.value;
    });
    saturationMinInput?.addEventListener('change', () => {
        linearLayoutState.ranges.S[0] = saturationMinInput.value;
    });
    saturationMaxInput?.addEventListener('change', () => {
        linearLayoutState.ranges.S[1] = saturationMaxInput.value;
    });
    lightnessMinInput?.addEventListener('change', () => {
        linearLayoutState.ranges.L[0] = lightnessMinInput.value;
    });
    lightnessMaxInput?.addEventListener('change', () => {
        linearLayoutState.ranges.L[1] = lightnessMaxInput.value;
    });
    apply?.addEventListener('click', async () => {
        await applyLinearLayoutSpec();
    });
    reset?.addEventListener('click', () => {
        linearLayoutState = defaultLinearLayoutState();
        linearLayoutNotice = null;
        renderLinearLayoutWidget();
    });
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
        const bases = {
            thread: parseBasesField('Thread', linearLayoutState.bases.thread),
            warp: parseBasesField('Warp', linearLayoutState.bases.warp),
            register: parseBasesField('Register', linearLayoutState.bases.register),
        };
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
        { action: 'open', label: 'Open Tensor', shortcut: 'Ctrl+O', keywords: 'file open load tensor npy' },
        { action: 'save', label: 'Save Tensor', shortcut: 'Ctrl+S', keywords: 'file save export tensor npy' },
        { action: '2d', label: 'Display as 2D', shortcut: 'Ctrl+2', keywords: 'display 2d orthographic' },
        { action: '3d', label: 'Display as 3D', shortcut: 'Ctrl+3', keywords: 'display 3d perspective' },
        { action: 'mapping-contiguous', label: 'Set Contiguous Axis Family Mapping', shortcut: '', keywords: 'display axis family mapping contiguous layout' },
        { action: 'mapping-z-order', label: 'Set Z-Order Axis Family Mapping', shortcut: '', keywords: 'display axis family mapping z-order z order layout' },
        { action: 'display-gaps', label: 'Toggle Block Gaps', shortcut: '', keywords: 'display advanced block gaps spacing' },
        { action: 'collapse-hidden-axes', label: 'Toggle Collapse Hidden Axes', shortcut: '', keywords: 'display advanced collapse hidden axes slices same place' },
        { action: 'log-scale', label: 'Toggle Log Scale', shortcut: '', keywords: 'display advanced log scale heatmap colorbar' },
        { action: 'heatmap', label: 'Toggle Heatmap', shortcut: 'Ctrl+H', keywords: 'display heatmap colors' },
        { action: 'dims', label: 'Toggle Dimension Lines', shortcut: 'Ctrl+D', keywords: 'display dimensions guides labels' },
        { action: 'tensor-names', label: 'Toggle Tensor Names', shortcut: '', keywords: 'display tensor names labels title' },
        { action: 'tensor-view', label: 'Toggle Tensor View', shortcut: 'Ctrl+V', keywords: 'widgets tensor view panel' },
        { action: 'inspector', label: 'Toggle Inspector', shortcut: '', keywords: 'widgets inspector panel' },
        { action: 'selection', label: 'Toggle Selection', shortcut: '', keywords: 'widgets selection panel stats highlighted cells' },
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

function activeTab(): LoadedBundleDocument | undefined {
    return sessionTabs.find((tab) => tab.id === activeTabId);
}

function captureActiveTabSnapshot(): void {
    const tab = activeTab();
    if (!tab) return;
    const snapshot = viewer.getSnapshot() as ViewerSnapshot & { linearLayoutState?: LinearLayoutFormState };
    if (isLinearLayoutTab(tab)) {
        const cloned = cloneLinearLayoutState(linearLayoutState);
        linearLayoutStates.set(tab.id, cloned);
        snapshot.linearLayoutState = cloned;
    }
    tab.manifest.viewer = snapshot;
}

function renderTabStrip(): void {
    tabStrip.classList.toggle('hidden', sessionTabs.length < 2);
    if (sessionTabs.length < 2) {
        tabStrip.replaceChildren();
        return;
    }
    tabStrip.replaceChildren(...sessionTabs.map((tab) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `tab-button${tab.id === activeTabId ? ' active' : ''}`;
        button.textContent = tab.title;
        button.addEventListener('click', async () => {
            if (tab.id === activeTabId) return;
            await loadTab(tab.id);
        });
        return button;
    }));
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
    switchingTab = false;
    renderTabStrip();
    syncLinearLayoutState(tab);
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
    setSidebarWidth(360);
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
    tensorViewWidget.classList.toggle('hidden', !showTensorViewWidget);
    inspectorWidget.classList.toggle('hidden', !snapshot.showInspectorPanel);
    selectionWidget.classList.toggle('hidden', !snapshot.showSelectionPanel);
    advancedSettingsWidget.classList.toggle('hidden', !showAdvancedSettingsWidget);
    const model = viewer.getInspectorModel();
    colorbarWidget.classList.toggle('hidden', !snapshot.heatmap || model.colorRanges.length === 0);
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
            <div id="inspector-layout-coord"><div class="label-row"><span class="meta-label">Layout Coord</span>${infoButton('Coordinate in the rendered layout after grouping and optional slice collapsing.')}</div><span class="meta-value" id="inspector-layout-coord-value"></span></div>
            <div id="inspector-tensor-coord"><div class="label-row"><span class="meta-label">Tensor Coord</span>${infoButton('Coordinate in the original dense tensor before view regrouping or slicing.')}</div><span class="meta-value" id="inspector-tensor-coord-value"></span></div>
            <div id="inspector-value"><div class="label-row"><span class="meta-label">Value</span>${infoButton('Numeric value at the hovered tensor element.')}</div><span class="meta-value" id="inspector-value-field"></span></div>
            <div><div class="label-row"><span class="meta-label">DType</span>${infoButton('Underlying numeric storage type for the active tensor.')}</div><span class="meta-value" id="inspector-dtype"></span></div>
            <div><div class="label-row"><span class="meta-label">Tensor Shape</span>${infoButton('Original tensor shape before Tensor View transformations.')}</div><span class="meta-value" id="inspector-tensor-shape"></span></div>
            <div><div class="label-row"><span class="meta-label">Rank</span>${infoButton('Number of dimensions in the original tensor.')}</div><span class="meta-value" id="inspector-rank"></span></div>
          </div>
        `;
        inspectorRefs = {
            hoveredTensor: inspectorWidget.querySelector<HTMLDivElement>('#inspector-hovered-tensor')!,
            layoutCoord: inspectorWidget.querySelector<HTMLDivElement>('#inspector-layout-coord')!,
            tensorCoord: inspectorWidget.querySelector<HTMLDivElement>('#inspector-tensor-coord')!,
            value: inspectorWidget.querySelector<HTMLDivElement>('#inspector-value')!,
            hoveredTensorValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-hovered-tensor-value')!,
            layoutCoordValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-layout-coord-value')!,
            tensorCoordValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-tensor-coord-value')!,
            valueField: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-value-field')!,
            dtypeValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-dtype')!,
            tensorShapeValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-tensor-shape')!,
            rankValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-rank')!,
        };
        inspectorReady = true;
    }
    if (!inspectorRefs) return;
    const hover = viewer.getHover();
    const hoveredStatus = hover ? viewer.getTensorStatus(hover.tensorId) : null;
    inspectorRefs.hoveredTensor.classList.toggle('hidden', !hover);
    inspectorRefs.layoutCoord.classList.toggle('hidden', !hover);
    inspectorRefs.tensorCoord.classList.toggle('hidden', !hover);
    inspectorRefs.value.classList.toggle('hidden', !hover);
    inspectorRefs.hoveredTensorValue.textContent = hover?.tensorName ?? '';
    inspectorRefs.layoutCoordValue.innerHTML = hover ? formatAxisValues(hover.layoutCoord, snapshot.displayMode, dimensionMappingScheme) : '';
    inspectorRefs.tensorCoordValue.innerHTML = hover ? formatAxisValues(hover.tensorCoord, snapshot.displayMode, dimensionMappingScheme) : '';
    inspectorRefs.valueField.textContent = !hover ? '' : hover.value === null ? 'Unavailable' : String(hover.value);
    inspectorRefs.dtypeValue.textContent = model.handle.dtype;
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
    const isLinearLayout = tensorNames.includes('hardware tensor') && tensorNames.includes('logical tensor');
    const viewerState = {
        ...tab.viewer,
        dimensionMappingScheme: isLinearLayout ? 'contiguous' : tab.viewer.dimensionMappingScheme,
    };
    const storedLinearLayoutState = (viewerState as { linearLayoutState?: unknown }).linearLayoutState;
    if (isLinearLayout && isLinearLayoutState(storedLinearLayoutState)) {
        linearLayoutStates.set(tab.id, cloneLinearLayoutState(storedLinearLayoutState));
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
    sessionTabs = await Promise.all(manifest.tabs.map((tab) => loadSessionTab(tab)));
    activeTabId = null;
    const initialTabId = sessionTabs[0]?.id ?? null;
    if (initialTabId) await loadTab(initialTabId);
    return true;
}

function seedDemoTensor(): void {
    sessionTabs = [];
    activeTabId = null;
    const shape = [4, 4, 4];
    const data = new Float64Array(product(shape));
    for (let index = 0; index < data.length; index += 1) {
        data[index] = Math.sin(index / 3) * 10;
    }
    viewer.addTensor(shape, data, 'Sample');
}

async function openLocalFile(file: File): Promise<void> {
    sessionTabs = [];
    activeTabId = null;
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
        case 'tensor-view':
            showTensorViewWidget = !showTensorViewWidget;
            render(viewer.getSnapshot());
            return;
        case 'selection':
            viewer.toggleSelectionPanel();
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
        await runAction('save');
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
viewer.subscribeSelection(() => renderSelectionWidget(viewer.getSnapshot()));
renderLinearLayoutWidget();

tryLoadSession().then(async (loaded) => {
    if (loaded) return;
    if (await applyLinearLayoutSpec({ replaceTabs: true, silent: true })) return;
    linearLayoutState = defaultLinearLayoutState();
    if (await applyLinearLayoutSpec({ replaceTabs: true, silent: true })) return;
    seedDemoTensor();
}).catch(async () => {
    if (await applyLinearLayoutSpec({ replaceTabs: true, silent: true })) return;
    linearLayoutState = defaultLinearLayoutState();
    if (await applyLinearLayoutSpec({ replaceTabs: true, silent: true })) return;
    seedDemoTensor();
});
}
