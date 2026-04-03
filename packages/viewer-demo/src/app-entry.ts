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
    formatNamedAxisValues,
    formatRangeValue,
    infoButton,
    labelWithInfo,
    selectionEnabled,
    titleWithInfo,
} from './app-format.js';
import {
    composeLayoutStateFromLegacySpec,
    createComposeLayoutDocument,
} from './linear-layout.js';
import {
    applyLinearLayoutCellText,
    applyLinearLayoutSpec,
    cloneLinearLayoutCellTextState,
    cloneLinearLayoutState,
    cloneLinearLayoutTensorViewsState,
    composeLayoutMetaForTab,
    defaultLinearLayoutCellTextState,
    emptyLinearLayoutState,
    inspectorCoordEntries,
    isLinearLayoutCellTextState,
    isLinearLayoutState,
    isLinearLayoutTab,
    linearLayoutSelectionMapForTab,
    loadBakedLinearLayoutTabs,
    loadLinearLayoutState,
    preservedLinearLayoutTensorViews,
    renderCellTextWidget,
    renderLinearLayoutWidget,
    snapshotTensorViews,
    syncLinearLayoutCellTextState,
    syncLinearLayoutSelection,
    syncLinearLayoutSelectionPreview,
    syncLinearLayoutState,
    syncLinearLayoutViewFilters,
    type InspectorCoordEntry,
    type LinearLayoutCellTextState,
    type LinearLayoutFormState,
    type LinearLayoutNotice,
    type LinearLayoutSelectionMap,
    type LinearLayoutTensorViewsState,
    type LinearLayoutUiContext,
    type LinearLayoutUiState,
} from './linear-layout-ui.js';
import { getAppRoot, mountAppShell, renderWebglUnavailable, supportsWebGL } from './app-shell.js';
import './styles.css';

const app = getAppRoot();

if (!supportsWebGL()) {
    renderWebglUnavailable(app);
} else {
const {
    viewport,
    tabStrip,
    controlDock,
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

const MIN_VIEWPORT_WIDTH = 280;
const MAX_SIDEBAR_WIDTH = 720;

type InspectorRefs = {
    hoveredTensor: HTMLDivElement;
    coordList: HTMLDivElement;
    hoveredTensorValue: HTMLSpanElement;
    tensorShapeValue: HTMLSpanElement;
    rankValue: HTMLSpanElement;
};

let inspectorRefs: InspectorRefs | null = null;
const linearLayoutUiState: LinearLayoutUiState = {
    linearLayoutState: loadLinearLayoutState(),
    linearLayoutStates: new Map<string, LinearLayoutFormState>(),
    linearLayoutCellTextState: defaultLinearLayoutCellTextState(),
    linearLayoutCellTextStates: new Map<string, LinearLayoutCellTextState>(),
    linearLayoutTensorViewsStates: new Map<string, LinearLayoutTensorViewsState>(),
    linearLayoutSelectionMaps: new Map<string, LinearLayoutSelectionMap>(),
    linearLayoutNotice: null,
    linearLayoutMatrixPreview: '',
    showLinearLayoutMatrix: false,
    syncingLinearLayoutSelection: false,
};

const linearLayoutUi: LinearLayoutUiContext = {
    viewer,
    viewport,
    linearLayoutWidget,
    cellTextWidget,
    linearLayoutColorWidget,
    state: linearLayoutUiState,
    getActiveTab: () => activeTab(),
    getActiveTabId: () => activeTabId,
    getSessionTabs: () => sessionTabs,
    setSessionTabs: (tabs) => {
        sessionTabs = tabs;
    },
    loadTab: async (tabId) => {
        await loadTab(tabId);
    },
};

type CommandAction = {
    action: string;
    label: string;
    shortcut: string;
    keywords: string;
};

type ControlSpec = {
    id: string;
    label: string;
    description: string;
    shortcut: string;
    active: boolean;
    disabled?: boolean;
    content: string;
    onClick: () => void | Promise<void>;
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
    const maxWidth = Math.max(0, app.clientWidth - MIN_VIEWPORT_WIDTH);
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
        composeLayoutMeta?: ComposeLayoutMeta;
        composeLayoutState?: LinearLayoutFormState;
        linearLayoutCellTextState?: LinearLayoutCellTextState;
        composeLayoutTensorViews?: LinearLayoutTensorViewsState;
    };
    if (isLinearLayoutTab(tab)) {
        const cloned = cloneLinearLayoutState(linearLayoutUiState.linearLayoutState);
        const clonedCellText = cloneLinearLayoutCellTextState(linearLayoutUiState.linearLayoutCellTextState);
        const tensorViews = preservedLinearLayoutTensorViews(linearLayoutUi, tab.id);
        linearLayoutUiState.linearLayoutStates.set(tab.id, cloned);
        linearLayoutUiState.linearLayoutCellTextStates.set(tab.id, clonedCellText);
        linearLayoutUiState.linearLayoutTensorViewsStates.set(tab.id, tensorViews);
        snapshot.composeLayoutState = cloned;
        snapshot.linearLayoutCellTextState = clonedCellText;
        snapshot.composeLayoutTensorViews = cloneLinearLayoutTensorViewsState(tensorViews);
        const composeLayoutMeta = composeLayoutMetaForTab(tab);
        if (composeLayoutMeta) snapshot.composeLayoutMeta = composeLayoutMeta;
    }
    tab.manifest.viewer = snapshot;
}

async function closeTab(tabId: string): Promise<void> {
    const index = sessionTabs.findIndex((tab) => tab.id === tabId);
    if (index < 0) return;
    const wasActive = activeTabId === tabId;
    if (wasActive) captureActiveTabSnapshot();
    linearLayoutUiState.linearLayoutStates.delete(tabId);
    linearLayoutUiState.linearLayoutCellTextStates.delete(tabId);
    linearLayoutUiState.linearLayoutTensorViewsStates.delete(tabId);
    linearLayoutUiState.linearLayoutSelectionMaps.delete(tabId);
    sessionTabs.splice(index, 1);
    if (sessionTabs.length === 0) {
        activeTabId = null;
        seedDemoTensor();
        renderTabStrip();
        render(viewer.getSnapshot());
        return;
    }
    if (!wasActive) {
        renderTabStrip();
        return;
    }
    await loadTab(sessionTabs[Math.max(0, index - 1)]!.id);
}

function renderTabStrip(): void {
    tabStrip.classList.toggle('hidden', sessionTabs.length === 0);
    if (sessionTabs.length === 0) {
        tabStrip.replaceChildren();
        return;
    }
    const tabs = sessionTabs.map((tab) => {
        const tabElement = document.createElement('div');
        tabElement.className = `tab-button${tab.id === activeTabId ? ' active' : ''}`;

        const label = document.createElement('button');
        label.type = 'button';
        label.className = 'tab-label';
        label.textContent = tab.title;
        label.addEventListener('click', async () => {
            if (tab.id === activeTabId) return;
            await loadTab(tab.id);
        });

        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'tab-close';
        closeButton.textContent = 'x';
        closeButton.setAttribute('aria-label', `Close ${tab.title}`);
        closeButton.addEventListener('click', async (event) => {
            event.preventDefault();
            event.stopPropagation();
            await closeTab(tab.id);
        });

        tabElement.addEventListener('auxclick', async (event) => {
            if (event.button !== 1) return;
            event.preventDefault();
            await closeTab(tab.id);
        });
        tabElement.append(label, closeButton);
        return tabElement;
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
        linearLayoutUiState.linearLayoutState = emptyLinearLayoutState();
        const document = createComposeLayoutDocument(linearLayoutUiState.linearLayoutState, viewer.getSnapshot(), title);
        linearLayoutUiState.linearLayoutCellTextState = defaultLinearLayoutCellTextState();
        linearLayoutUiState.linearLayoutStates.set(id, cloneLinearLayoutState(linearLayoutUiState.linearLayoutState));
        linearLayoutUiState.linearLayoutCellTextStates.set(id, cloneLinearLayoutCellTextState(linearLayoutUiState.linearLayoutCellTextState));
        linearLayoutUiState.linearLayoutTensorViewsStates.set(id, snapshotTensorViews(document.manifest.viewer));
        sessionTabs = [{ ...document, id, title }];
        await loadTab(id);
        return;
    }
    captureActiveTabSnapshot();
    const nextTab = cloneTabDocument(currentTab, id, title);
    const currentLinearLayoutState = linearLayoutUiState.linearLayoutStates.get(currentTab.id);
    if (currentLinearLayoutState) {
        linearLayoutUiState.linearLayoutStates.set(id, cloneLinearLayoutState(currentLinearLayoutState));
    }
    const currentCellTextState = linearLayoutUiState.linearLayoutCellTextStates.get(currentTab.id);
    if (currentCellTextState) {
        linearLayoutUiState.linearLayoutCellTextStates.set(id, cloneLinearLayoutCellTextState(currentCellTextState));
    }
    const currentTensorViewsState = linearLayoutUiState.linearLayoutTensorViewsStates.get(currentTab.id);
    if (currentTensorViewsState) {
        linearLayoutUiState.linearLayoutTensorViewsStates.set(id, cloneLinearLayoutTensorViewsState(currentTensorViewsState));
    }
    linearLayoutUiState.linearLayoutSelectionMaps.delete(id);
    sessionTabs = [...sessionTabs, nextTab];
    await loadTab(id);
}

async function closeCurrentTab(): Promise<void> {
    if (activeTabId) await closeTab(activeTabId);
}

function iconSelection(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 5h3v3H5zM16 5h3v3h-3zM5 16h3v3H5zM16 16h3v3h-3z" />
        <path d="M9.5 6.5h5M9.5 17.5h5M6.5 9.5v5M17.5 9.5v5" stroke-dasharray="2.2 2.2" />
      </svg>
    `;
}

function iconRotate(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 4.8A7.2 7.2 0 1 1 4.8 12" />
        <path d="M4.8 9.6l1.92 3.6H2.88z" fill="currentColor" stroke="none" />
      </svg>
    `;
}

function iconHeatmap(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="4" width="8" height="8" fill="#111111" stroke="#111111" stroke-width="0.8" />
        <rect x="12" y="4" width="8" height="8" fill="#555555" stroke="#111111" stroke-width="0.8" />
        <rect x="4" y="12" width="8" height="8" fill="#cccccc" stroke="#111111" stroke-width="0.8" />
        <rect x="12" y="12" width="8" height="8" fill="#ffffff" stroke="#111111" stroke-width="0.8" />
      </svg>
    `;
}

function iconDimensionLines(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <polyline points="7,7 7,4 17,4 17,7" stroke="#ef4444" />
        <polyline points="7,7 4,7 4,17 7,17" stroke="#22c55e" />
        <rect x="7" y="7" width="10" height="10" />
      </svg>
    `;
}

function iconTensorNames(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 6h14" />
        <path d="M12 6v12" />
        <path d="M8 18h8" />
      </svg>
    `;
}

function iconGaps(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="6" width="4" height="12" />
        <rect x="10" y="6" width="4" height="12" />
        <rect x="16" y="6" width="4" height="12" />
      </svg>
    `;
}

function iconContiguousMapping(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <polyline points="4,5.5 20,5.5 4,10.5 20,10.5 4,15.5 20,15.5 4,20.5 20,20.5" />
      </svg>
    `;
}

function iconZOrderMapping(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <polyline points="4,4 9,4 4,9 9,9 15,4 20,4 15,9 20,9 4,15 9,15 4,20 9,20 15,15 20,15 15,20 20,20" />
      </svg>
    `;
}

function iconPan(): string {
    return `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 13v-7.5a1.5 1.5 0 0 1 3 0v6.5M11 5.5v-2a1.5 1.5 0 1 1 3 0v8.5M14 5.5a1.5 1.5 0 0 1 3 0v6.5M17 7.5a1.5 1.5 0 0 1 3 0v8.5a6 6 0 0 1-6 6h-2h.208a6 6 0 0 1-5.012-2.7a69.74 69.74 0 0 1-.196-.3c-.312-.479-1.407-2.388-3.286-5.728a1.5 1.5 0 0 1 .536-2.022a1.867 1.867 0 0 1 2.28.28l1.47 1.47" />
      </svg>
    `;
}

function renderControlDock(snapshot: ViewerSnapshot): void {
    const canSelect = selectionEnabled(snapshot);
    const canRotate = snapshot.displayMode === '3d';
    const interactionMode = snapshot.interactionMode ?? viewer.getInteractionMode();
    const controls: ControlSpec[] = [
        {
            id: 'pan',
            label: 'Pan',
            description: 'Left click and drag to move the viewport without changing the tensor data.',
            shortcut: 'P',
            active: interactionMode === 'pan',
            content: iconPan(),
            onClick: () => viewer.setInteractionMode('pan'),
        },
        {
            id: 'select',
            label: 'Select',
            description: canSelect
                ? 'Left click and drag to draw a selection box.<br />Shift + left click + drag: add cells.<br />Ctrl + left click + drag: remove cells.'
                : 'Selection is available in 2D contiguous layouts.<br />Left click and drag: select cells.<br />Shift + left click + drag: add cells.<br />Ctrl + left click + drag: remove cells.',
            shortcut: 'S',
            active: interactionMode === 'select',
            disabled: !canSelect,
            content: iconSelection(),
            onClick: () => viewer.setInteractionMode('select'),
        },
        {
            id: 'rotate',
            label: 'Rotate',
            description: canRotate
                ? 'Left click and drag to orbit the 3D camera around the tensor layout.'
                : 'Rotate is available in 3D mode, where left click and drag orbits the camera.',
            shortcut: 'R',
            active: interactionMode === 'rotate',
            disabled: !canRotate,
            content: iconRotate(),
            onClick: () => viewer.setInteractionMode('rotate'),
        },
        {
            id: '2d',
            label: '2D',
            description: 'Switch to the flat 2D viewer for grid inspection, panning, zooming, and box selection.',
            shortcut: 'Ctrl+2',
            active: snapshot.displayMode === '2d',
            content: '<span class="control-button-text" aria-hidden="true">2D</span>',
            onClick: () => viewer.setDisplayMode('2d'),
        },
        {
            id: '3d',
            label: '3D',
            description: 'Switch to the 3D viewer to orbit the scene and inspect stacked tensor layouts in depth.',
            shortcut: 'Ctrl+3',
            active: snapshot.displayMode === '3d',
            content: '<span class="control-button-text" aria-hidden="true">3D</span>',
            onClick: () => viewer.setDisplayMode('3d'),
        },
        {
            id: 'heatmap',
            label: 'Heatmap',
            description: 'Toggle heatmap coloring to map numeric values onto a grayscale intensity view.',
            shortcut: 'Ctrl+H',
            active: snapshot.heatmap,
            content: iconHeatmap(),
            onClick: () => viewer.toggleHeatmap(),
        },
        {
            id: 'dim-lines',
            label: 'Dim Lines',
            description: 'Toggle dimension guide lines to show axis extents and family orientation in the current layout.',
            shortcut: 'Ctrl+D',
            active: snapshot.showDimensionLines,
            content: iconDimensionLines(),
            onClick: () => viewer.toggleDimensionLines(),
        },
        {
            id: 'tensor-names',
            label: 'Tensor Names',
            description: 'Toggle tensor name labels above each rendered tensor in the current layout.',
            shortcut: 'Ctrl+N',
            active: snapshot.showTensorNames ?? true,
            content: iconTensorNames(),
            onClick: () => viewer.toggleTensorNames(),
        },
        {
            id: 'gaps',
            label: 'Block Gaps',
            description: 'Toggle spacing inside higher-level tensor blocks so grouped dimensions appear either separated or packed.',
            shortcut: 'Ctrl+G',
            active: snapshot.displayGaps ?? true,
            content: iconGaps(),
            onClick: () => viewer.toggleDisplayGaps(),
        },
        {
            id: 'mapping-contiguous',
            label: 'Contiguous Mapping',
            description: 'Use contiguous axis-family mapping so neighboring cells follow one continuous zig-zag traversal.',
            shortcut: 'Ctrl+M',
            active: snapshot.dimensionMappingScheme === 'contiguous',
            content: iconContiguousMapping(),
            onClick: () => viewer.setDimensionMappingScheme('contiguous'),
        },
        {
            id: 'mapping-z-order',
            label: 'Z-Order Mapping',
            description: 'Use z-order axis-family mapping so traversal breaks into smaller interleaved zig-zag groups.',
            shortcut: 'Ctrl+M',
            active: snapshot.dimensionMappingScheme === 'z-order',
            content: iconZOrderMapping(),
            onClick: () => viewer.setDimensionMappingScheme('z-order'),
        },
    ];
    controlDock.replaceChildren(...controls.map((control, index) => {
        const buttonClass = `control-button${control.active ? ' active' : ''}${control.disabled ? ' disabled' : ''}`;
        const tooltip = `<span class="control-tooltip"><strong>${control.label}</strong><span>${control.description}</span><span class="control-tooltip-shortcut">Shortcut: ${control.shortcut}</span></span>`;
        if (index === 3 || index === 5 || index === 7 || index === 9) {
            const fragment = document.createDocumentFragment();
            const divider = document.createElement('div');
            divider.className = 'control-dock-divider';
            fragment.appendChild(divider);
            const button = document.createElement('button');
            button.type = 'button';
            button.className = buttonClass;
            button.innerHTML = `${control.content}${tooltip}`;
            button.disabled = Boolean(control.disabled);
            button.setAttribute('aria-label', control.label);
            button.addEventListener('click', async () => {
                if (control.disabled) return;
                await control.onClick();
            });
            fragment.appendChild(button);
            return fragment;
        }
        const button = document.createElement('button');
        button.type = 'button';
        button.className = buttonClass;
        button.innerHTML = `${control.content}${tooltip}`;
        button.disabled = Boolean(control.disabled);
        button.setAttribute('aria-label', control.label);
        button.addEventListener('click', async () => {
            if (control.disabled) return;
            await control.onClick();
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
    if (!appliedStartupWidgetDefaults) {
        viewer.toggleSelectionPanel(false);
        appliedStartupWidgetDefaults = true;
    }
    switchingTab = false;
    renderTabStrip();
    syncLinearLayoutState(linearLayoutUi, tab);
    syncLinearLayoutCellTextState(linearLayoutUi, tab);
    renderCellTextWidget(linearLayoutUi);
    syncLinearLayoutViewFilters(linearLayoutUi);
    applyLinearLayoutCellText(linearLayoutUi);
    syncLinearLayoutSelectionPreview(linearLayoutUi, new Map());
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
    cellTextWidget.classList.toggle('hidden', !(activeTab() && isLinearLayoutTab(activeTab()!)));
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
          <div class="mono-block" id="view-preview"></div>
        </div>
        <div class="slider-list" id="slice-token-controls"></div>
      </div>
    `;

    const input = tensorViewWidget.querySelector<HTMLInputElement>('#view-input');
    const preview = tensorViewWidget.querySelector<HTMLElement>('#view-preview');
    const select = tensorViewWidget.querySelector<HTMLSelectElement>('#tensor-select');
    const sliceHost = tensorViewWidget.querySelector<HTMLElement>('#slice-token-controls');
    if (preview) preview.textContent = model.preview;
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
                syncLinearLayoutViewFilters(linearLayoutUi);
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
            syncLinearLayoutViewFilters(linearLayoutUi);
            if (preview) preview.textContent = viewer.getInspectorModel().preview;
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
            <div id="inspector-coords"><div class="label-row"><span class="meta-label">Visible Tensor Coords</span>${infoButton('Coordinates for every visible tensor in the active layout chain. The hovered tensor title is highlighted.')}</div><div class="inspector-coord-list" id="inspector-coord-list"></div></div>
            <div><div class="label-row"><span class="meta-label">Tensor Shape</span>${infoButton('Original tensor shape before Tensor View transformations.')}</div><span class="meta-value" id="inspector-tensor-shape"></span></div>
            <div><div class="label-row"><span class="meta-label">Rank</span>${infoButton('Number of dimensions in the original tensor.')}</div><span class="meta-value" id="inspector-rank"></span></div>
          </div>
        `;
        inspectorRefs = {
            hoveredTensor: inspectorWidget.querySelector<HTMLDivElement>('#inspector-hovered-tensor')!,
            coordList: inspectorWidget.querySelector<HTMLDivElement>('#inspector-coords')!,
            hoveredTensorValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-hovered-tensor-value')!,
            tensorShapeValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-tensor-shape')!,
            rankValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-rank')!,
        };
        inspectorReady = true;
    }
    if (!inspectorRefs) return;
    const hover = viewer.getHover();
    const hoveredStatus = hover ? viewer.getTensorStatus(hover.tensorId) : null;
    const linearLayout = linearLayoutTab ? linearLayoutSelectionMapForTab(linearLayoutUi, linearLayoutTab) : null;
    const coordEntries = inspectorCoordEntries(linearLayoutUi, hover, hoveredStatus, linearLayout);
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
    inspectorRefs.coordList.classList.toggle('hidden', !hover);
    inspectorRefs.hoveredTensorValue.textContent = hover?.tensorName ?? '';
    const coordList = inspectorWidget.querySelector<HTMLDivElement>('#inspector-coord-list');
    if (coordList) {
        coordList.innerHTML = coordEntries.map((entry) => `
            <div class="inspector-coord-item">
              <span class="meta-label inspector-coord-title${entry.hovered ? ' is-hovered-tensor' : ''}">${escapeInfo(entry.title)}</span>
              <span class="meta-value">${entry.coord ? formatNamedAxisValues(entry.labels, entry.coord, snapshot.displayMode, dimensionMappingScheme) : ''}</span>
              <span class="meta-value mono-value">${binaryCoord(entry.coord, entry.shape)}</span>
            </div>
        `).join('');
    }
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
    renderCellTextWidget(linearLayoutUi);
    renderControlDock(snapshot);
    renderTensorViewWidget(snapshot);
    renderInspectorWidget(snapshot);
    renderSelectionWidget(snapshot);
    renderAdvancedSettingsWidget(snapshot);
    renderColorbarWidget(snapshot);
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
    const legacySpec = (tab.viewer as { linearLayoutSpec?: unknown }).linearLayoutSpec;
    const storedComposeState = (tab.viewer as { composeLayoutState?: unknown }).composeLayoutState;
    const storedTensorViews = (tab.viewer as { composeLayoutTensorViews?: unknown }).composeLayoutTensorViews;
    const composeMeta = (tab.viewer as { composeLayoutMeta?: unknown }).composeLayoutMeta;
    if (legacySpec) {
        const state = isLinearLayoutState(storedComposeState)
            ? cloneLinearLayoutState(storedComposeState)
            : composeLayoutStateFromLegacySpec(legacySpec, tab.title);
        const document = createComposeLayoutDocument(state, {
            ...tab.viewer,
            showSelectionPanel: false,
        }, tab.title);
        linearLayoutUiState.linearLayoutStates.set(tab.id, cloneLinearLayoutState(state));
        if (storedTensorViews && typeof storedTensorViews === 'object') {
            linearLayoutUiState.linearLayoutTensorViewsStates.set(tab.id, cloneLinearLayoutTensorViewsState(storedTensorViews as LinearLayoutTensorViewsState));
        } else {
            linearLayoutUiState.linearLayoutTensorViewsStates.set(tab.id, snapshotTensorViews(document.manifest.viewer));
        }
        return {
            ...document,
            id: tab.id,
            title: tab.title,
        };
    }
    const isLinearLayout = isComposeLayoutMeta(composeMeta);
    const viewerState = {
        ...tab.viewer,
        dimensionMappingScheme: isLinearLayout ? 'contiguous' : tab.viewer.dimensionMappingScheme,
        showSelectionPanel: false,
    };
    const storedLinearLayoutState = (viewerState as { composeLayoutState?: unknown }).composeLayoutState;
    if (isLinearLayout && isLinearLayoutState(storedLinearLayoutState)) {
        linearLayoutUiState.linearLayoutStates.set(tab.id, cloneLinearLayoutState(storedLinearLayoutState));
    }
    if (isLinearLayout && storedTensorViews && typeof storedTensorViews === 'object') {
        linearLayoutUiState.linearLayoutTensorViewsStates.set(tab.id, cloneLinearLayoutTensorViewsState(storedTensorViews as LinearLayoutTensorViewsState));
    } else if (isLinearLayout) {
        linearLayoutUiState.linearLayoutTensorViewsStates.set(tab.id, snapshotTensorViews(viewerState));
    }
    const storedCellTextState = (viewerState as { linearLayoutCellTextState?: unknown }).linearLayoutCellTextState;
    if (isLinearLayout && isLinearLayoutCellTextState(storedCellTextState)) {
        linearLayoutUiState.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(storedCellTextState));
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
    linearLayoutUiState.linearLayoutSelectionMaps.clear();
    sessionTabs = await Promise.all(manifest.tabs.map((tab) => loadSessionTab(tab)));
    activeTabId = null;
    const initialTabId = sessionTabs[0]?.id ?? null;
    if (initialTabId) await loadTab(initialTabId);
    return true;
}

function seedDemoTensor(): void {
    sessionTabs = [];
    activeTabId = null;
    linearLayoutUiState.linearLayoutStates.clear();
    linearLayoutUiState.linearLayoutCellTextStates.clear();
    linearLayoutUiState.linearLayoutTensorViewsStates.clear();
    linearLayoutUiState.linearLayoutSelectionMaps.clear();
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
    linearLayoutUiState.linearLayoutStates.clear();
    linearLayoutUiState.linearLayoutCellTextStates.clear();
    linearLayoutUiState.linearLayoutTensorViewsStates.clear();
    linearLayoutUiState.linearLayoutSelectionMaps.clear();
    renderTabStrip();
    await viewer.openFile(file);
}

function downloadSvg(filename: string, svg: string): void {
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

function svgFilename(): string {
    const title = activeTab()?.title ?? 'tensor-viz';
    const base = title.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    return `${base || 'tensor-viz'}.svg`;
}

async function currentSvgDocument(): Promise<string> {
    if (viewer.getSnapshot().displayMode !== '2d') return viewer.exportCurrentViewSvg();
    return viewer.saveSvg().text();
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
        case 'save-svg':
            downloadSvg(svgFilename(), await currentSvgDocument());
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
    const isEditing = Boolean(target && (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable));
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
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key.toLowerCase() === 'p') {
        event.preventDefault();
        viewer.setInteractionMode('pan');
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key.toLowerCase() === 's') {
        event.preventDefault();
        viewer.setInteractionMode('select');
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key.toLowerCase() === 'r') {
        event.preventDefault();
        viewer.setInteractionMode('rotate');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'h') {
        event.preventDefault();
        await runAction('heatmap');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'd') {
        event.preventDefault();
        await runAction('dims');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'n') {
        event.preventDefault();
        await runAction('tensor-names');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'g') {
        event.preventDefault();
        await runAction('display-gaps');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'm') {
        event.preventDefault();
        viewer.setDimensionMappingScheme(
            (viewer.getSnapshot().dimensionMappingScheme ?? 'z-order') === 'contiguous' ? 'z-order' : 'contiguous',
        );
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
    syncLinearLayoutSelectionPreview(linearLayoutUi, selection);
});
viewer.subscribeSelection((selection) => {
    renderSelectionWidget(viewer.getSnapshot());
    syncLinearLayoutSelection(linearLayoutUi, selection);
});
renderLinearLayoutWidget(linearLayoutUi);

tryLoadSession().then(async (loaded) => {
    if (loaded) return;
    if (await loadBakedLinearLayoutTabs(linearLayoutUi)) return;
    seedDemoTensor();
}).catch(async () => {
    if (await loadBakedLinearLayoutTabs(linearLayoutUi)) return;
    seedDemoTensor();
});
}
