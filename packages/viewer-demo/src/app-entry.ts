import {
    createTypedArray,
    serializeTensorViewEditor,
    TensorViewer,
    product,
    type BundleManifest,
    type DimensionMappingScheme,
    type LoadedBundleDocument,
    type NumericArray,
    type SessionBundleManifest,
    type TensorViewEditor,
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
} from './app-format.js';
import {
    composeLayoutStateFromLegacySpec,
    createComposeLayoutDocument,
} from './linear-layout.js';
import {
    applyLinearLayoutCellText,
    applyLinearLayoutSpec,
    cloneLinearLayoutCellTextState,
    cloneLinearLayoutMultiInputState,
    cloneLinearLayoutState,
    cloneLinearLayoutTensorViewsState,
    composeLayoutMetaForTab,
    defaultLinearLayoutCellTextState,
    defaultLinearLayoutMultiInputState,
    emptyLinearLayoutState,
    inspectorCoordEntries,
    isLinearLayoutCellTextState,
    isLinearLayoutMultiInputState,
    isLinearLayoutState,
    isLinearLayoutTab,
    linearLayoutHoverPopupEntries,
    linearLayoutMultiInputModel,
    linearLayoutSelectionMapForTab,
    loadBakedLinearLayoutTabs,
    loadLinearLayoutState,
    preservedLinearLayoutTensorViews,
    renderCellTextWidget,
    renderLinearLayoutWidget,
    snapshotTensorViews,
    syncLinearLayoutCellTextState,
    syncLinearLayoutMultiInputState,
    syncLinearLayoutSelection,
    syncLinearLayoutSelectionPreview,
    syncLinearLayoutState,
    syncLinearLayoutViewFilters,
    type InspectorCoordEntry,
    type LinearLayoutCellTextState,
    type LinearLayoutFormState,
    type LinearLayoutMultiInputState,
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
    linearLayoutVisibleTensorsWidget,
    cellTextWidget,
    linearLayoutColorWidget,
    sidebarHeader,
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
const hoverPopup = document.createElement('div');
hoverPopup.className = 'linear-layout-hover-popup hidden';
viewport.appendChild(hoverPopup);
const sidebar = tensorViewWidget.parentElement as HTMLElement;
const sidebarScrollPad = document.createElement('div');
sidebarScrollPad.className = 'sidebar-scroll-pad';
const viewErrors = new Map<string, string>();
let suspendTensorViewRender = false;
let tensorViewHelpOpen = false;
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
let lastLinearLayoutActiveTensorId: string | null = null;
let hoverPopupPointer = { x: 16, y: 16 };

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
    linearLayoutMultiInputState: defaultLinearLayoutMultiInputState(),
    linearLayoutMultiInputStates: new Map<string, LinearLayoutMultiInputState>(),
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
    linearLayoutVisibleTensorsWidget,
    cellTextWidget,
    linearLayoutColorWidget,
    state: linearLayoutUiState,
    widgetTitle: (widgetId, info) => widgetTitle(widgetId as SidebarWidgetId, info),
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

type SidebarWidgetId =
    | 'linear-layout'
    | 'linear-layout-visible-tensors'
    | 'linear-layout-color'
    | 'cell-text'
    | 'tensor-view'
    | 'inspector'
    | 'selection'
    | 'advanced-settings'
    | 'colorbar';

const sidebarWidgets: Record<SidebarWidgetId, HTMLElement> = {
    'linear-layout': linearLayoutWidget,
    'linear-layout-visible-tensors': linearLayoutVisibleTensorsWidget,
    'linear-layout-color': linearLayoutColorWidget,
    'cell-text': cellTextWidget,
    'tensor-view': tensorViewWidget,
    inspector: inspectorWidget,
    selection: selectionWidget,
    'advanced-settings': advancedSettingsWidget,
    colorbar: colorbarWidget,
};

const sidebarWidgetLabels: Record<SidebarWidgetId, string> = {
    'linear-layout': 'Linear Layout Specifications',
    'linear-layout-visible-tensors': 'Visible Tensors',
    'linear-layout-color': 'Color Mapping',
    'cell-text': 'Cell Text',
    'tensor-view': 'Permute/Slice',
    inspector: 'Hover Info',
    selection: 'Selection',
    'advanced-settings': 'Advanced Settings',
    colorbar: 'Heatmap',
};

let widgetOrder: SidebarWidgetId[] = [
    'linear-layout',
    'linear-layout-visible-tensors',
    'linear-layout-color',
    'cell-text',
    'tensor-view',
    'inspector',
    'selection',
    'advanced-settings',
    'colorbar',
];
sidebarHeader.classList.add('label-row');
sidebarHeader.innerHTML = `<span>Widgets</span>${infoButton('Extra settings to inspect/change the visible tensor(s). Click the arrows/widget header text on each widget to expand/collapse them. Change widget position by left-clicking + dragging on the grabber by the right of each widget.')}`;
let draggedWidgetId: SidebarWidgetId | null = null;
let draggedWidgetSlot: number | null = null;
let draggedWidgetPointerId: number | null = null;
const collapsedWidgets = new Set<SidebarWidgetId>(widgetOrder);
collapsedWidgets.delete('linear-layout');

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
        { action: 'tensor-view', label: 'Toggle Permute/Slice', shortcut: 'Ctrl+V', keywords: 'widgets permute slice tensor permutation slicing tensor view panel' },
        { action: 'inspector', label: 'Toggle Hover Info', shortcut: '', keywords: 'widgets hover info inspector panel' },
        { action: 'selection', label: 'Toggle Selection', shortcut: '', keywords: 'widgets selection panel stats highlighted cells' },
        { action: 'colorbar', label: 'Toggle Colorbar', shortcut: '', keywords: 'widgets colorbar panel heatmap range' },
        { action: 'advanced-settings', label: 'Toggle Advanced Settings', shortcut: '', keywords: 'widgets advanced settings layout gap' },
        { action: 'view', label: 'Focus Permute/Slice Input', shortcut: '', keywords: 'focus permute slice tensor permutation slicing tensor view input field' },
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

function visibleSidebarWidgets(snapshot: ViewerSnapshot): SidebarWidgetId[] {
    const model = viewer.getInspectorModel();
    const linearLayoutActive = Boolean(activeTab() && isLinearLayoutTab(activeTab()!));
    return widgetOrder.filter((widgetId) => (
        (widgetId === 'linear-layout' && linearLayoutActive)
        || (widgetId === 'linear-layout-visible-tensors' && linearLayoutActive)
        || (widgetId === 'linear-layout-color' && linearLayoutActive)
        || (widgetId === 'cell-text' && linearLayoutActive)
        || (widgetId === 'tensor-view' && showTensorViewWidget)
        || (widgetId === 'inspector' && snapshot.showInspectorPanel)
        || (widgetId === 'selection'
            && snapshot.showSelectionPanel
            && (snapshot.interactionMode ?? viewer.getInteractionMode()) === 'select')
        || (widgetId === 'advanced-settings' && showAdvancedSettingsWidget)
        || (widgetId === 'colorbar' && snapshot.heatmap && model.colorRanges.length !== 0)
    ));
}

function widgetTitle(widgetId: SidebarWidgetId, info: string): string {
    const title = sidebarWidgetLabels[widgetId];
    const collapsed = collapsedWidgets.has(widgetId);
    return `
      <div class="widget-header">
        <div class="title-row widget-title-row">
          <div class="widget-title-main" data-widget-collapse="${widgetId}" role="button" tabindex="0" aria-label="${collapsed ? 'Expand' : 'Collapse'} ${title}" aria-expanded="${String(!collapsed)}">
            <span class="widget-title-chevron" data-widget-chevron="${widgetId}" aria-hidden="true">${collapsed ? '▸' : '▾'}</span>
            <h2>${title}</h2>
          </div>
          <div class="widget-title-controls">
            <span class="widget-title-icon" aria-hidden="true">${widgetIcon(widgetId)}</span>
            ${infoButton(info)}
          </div>
        </div>
        <button class="widget-drag-button" data-widget-handle="${widgetId}" type="button" aria-label="Drag ${title}" title="Drag ${title}"></button>
      </div>
    `;
}

function applySidebarOrder(): void {
    sidebar.replaceChildren(sidebarHeader, ...widgetOrder.map((widgetId) => sidebarWidgets[widgetId]), sidebarScrollPad);
    syncSidebarDragState();
}

function syncWidgetHeaderState(widgetId: SidebarWidgetId, widget: HTMLElement): void {
    const collapsed = collapsedWidgets.has(widgetId);
    const button = widget.querySelector<HTMLElement>(`[data-widget-collapse="${widgetId}"]`);
    const chevron = widget.querySelector<HTMLElement>(`[data-widget-chevron="${widgetId}"]`);
    if (!button || !chevron) return;
    chevron.textContent = collapsed ? '▸' : '▾';
    button.setAttribute('aria-label', `${collapsed ? 'Expand' : 'Collapse'} ${sidebarWidgetLabels[widgetId]}`);
    button.setAttribute('aria-expanded', String(!collapsed));
}

function syncSidebarDragState(): void {
    const visible = visibleSidebarWidgets(viewer.getSnapshot());
    (Object.entries(sidebarWidgets) as [SidebarWidgetId, HTMLElement][]).forEach(([widgetId, widget]) => {
        widget.classList.toggle('widget-dragging', widgetId === draggedWidgetId);
        widget.classList.remove('widget-drop-before', 'widget-drop-after');
    });
    if (draggedWidgetSlot === null || visible.length === 0) return;
    const boundedSlot = Math.max(0, Math.min(visible.length, draggedWidgetSlot));
    const targetId = visible[Math.min(boundedSlot, visible.length - 1)];
    if (!targetId) return;
    sidebarWidgets[targetId].classList.add(boundedSlot >= visible.length ? 'widget-drop-after' : 'widget-drop-before');
}

function toggleWidgetCollapse(widgetId: SidebarWidgetId): void {
    const header = sidebarWidgets[widgetId].querySelector<HTMLElement>(`[data-widget-collapse="${widgetId}"]`);
    const headerOffset = header
        ? header.getBoundingClientRect().top - sidebar.getBoundingClientRect().top
        : null;
    if (collapsedWidgets.has(widgetId)) collapsedWidgets.delete(widgetId);
    else collapsedWidgets.add(widgetId);
    render(viewer.getSnapshot());
    if (headerOffset === null) return;
    const nextHeader = sidebarWidgets[widgetId].querySelector<HTMLElement>(`[data-widget-collapse="${widgetId}"]`);
    if (!nextHeader) return;
    const nextOffset = nextHeader.getBoundingClientRect().top - sidebar.getBoundingClientRect().top;
    const targetScrollTop = Math.max(0, sidebar.scrollTop + nextOffset - headerOffset);
    const currentPadHeight = sidebarScrollPad.getBoundingClientRect().height;
    const naturalMaxScrollTop = Math.max(0, sidebar.scrollHeight - sidebar.clientHeight - currentPadHeight);
    const requiredPadHeight = Math.max(0, targetScrollTop - naturalMaxScrollTop);
    sidebarScrollPad.style.height = `${requiredPadHeight}px`; // keep enough gray gutter to reach the anchored scroll target without an intermediate clamp
    sidebar.scrollTop = targetScrollTop;
}

function sidebarWidgetSlot(clientY: number): number | null {
    const visible = visibleSidebarWidgets(viewer.getSnapshot());
    if (visible.length === 0) return null;
    for (let index = 0; index < visible.length; index += 1) {
        const rect = sidebarWidgets[visible[index]!].getBoundingClientRect();
        if (clientY <= rect.top + rect.height / 2) return index;
    }
    return visible.length;
}

function moveSidebarWidgetToSlot(widgetId: SidebarWidgetId, slot: number): void {
    const visible = visibleSidebarWidgets(viewer.getSnapshot());
    const visibleIndex = visible.indexOf(widgetId);
    if (visibleIndex < 0) return;
    const boundedSlot = Math.max(0, Math.min(visible.length, slot));
    const nextVisible = visible.filter((visibleWidgetId) => visibleWidgetId !== widgetId);
    const nextSlot = boundedSlot - Number(visibleIndex < boundedSlot);
    const anchor = nextVisible[nextSlot];
    const nextOrder = widgetOrder.filter((orderedWidgetId) => orderedWidgetId !== widgetId);
    if (!anchor) nextOrder.push(widgetId);
    else nextOrder.splice(nextOrder.indexOf(anchor), 0, widgetId);
    widgetOrder = nextOrder;
    applySidebarOrder();
}

function clearSidebarDragState(): void {
    draggedWidgetId = null;
    draggedWidgetSlot = null;
    draggedWidgetPointerId = null;
    syncSidebarDragState();
}

function widgetIcon(widgetId: SidebarWidgetId): string {
    switch (widgetId) {
        case 'linear-layout':
            return `
              <svg viewBox="0 0 24 24">
                <rect x="4" y="4" width="16" height="16" style="fill: #ffffff; stroke: #111827; stroke-width: 1.25;" />
                <rect x="4" y="4" width="4" height="4" style="fill: #111827; stroke: none;" />
                <rect x="8" y="8" width="4" height="4" style="fill: #111827; stroke: none;" />
                <rect x="4" y="12" width="4" height="4" style="fill: #111827; stroke: none;" />
                <rect x="12" y="12" width="4" height="4" style="fill: #111827; stroke: none;" />
                <rect x="8" y="16" width="4" height="4" style="fill: #111827; stroke: none;" />
                <rect x="16" y="16" width="4" height="4" style="fill: #111827; stroke: none;" />
              </svg>
            `;
        case 'linear-layout-visible-tensors':
            return `
              <svg viewBox="0 0 24 24">
                <path d="M3 12s3.5-5 9-5 9 5 9 5-3.5 5-9 5-9-5-9-5z" />
                <circle cx="12" cy="12" r="2.5" />
              </svg>
            `;
        case 'linear-layout-color':
            return `
              <svg viewBox="0 0 24 24">
                <text x="5.2" y="6.3" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 5.8px 'IBM Plex Sans', sans-serif;">H</text>
                <text x="12" y="6.3" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 5.8px 'IBM Plex Sans', sans-serif;">S</text>
                <text x="18.8" y="6.3" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 5.8px 'IBM Plex Sans', sans-serif;">L</text>
                <path d="M5.2 8.7v6.1M3.7 12.8l1.5 2 1.5-2" style="stroke-width: 1.2;" />
                <path d="M12 8.7v6.1M10.5 12.8l1.5 2 1.5-2" style="stroke-width: 1.2;" />
                <path d="M18.8 8.7v6.1M17.3 12.8l1.5 2 1.5-2" style="stroke-width: 1.2;" />
                <text x="5.2" y="19.1" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 5.8px 'IBM Plex Sans', sans-serif;">W</text>
                <text x="12" y="19.1" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 5.8px 'IBM Plex Sans', sans-serif;">T</text>
                <text x="18.8" y="19.1" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 5.8px 'IBM Plex Sans', sans-serif;">R</text>
              </svg>
            `;
        case 'cell-text':
            return `
              <svg viewBox="0 0 24 24">
                <rect x="2.5" y="4" width="19" height="16" style="fill: #ffffff; stroke: #111827; stroke-width: 1.5;" />
                <text x="12" y="14.2" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 7px 'IBM Plex Mono', monospace;">T:0</text>
              </svg>
            `;
        case 'tensor-view':
            return '<span class="widget-title-text-icon widget-title-text-icon-wide">A<sup>T</sup>[i,:]</span>';
        case 'inspector':
            return `
              <svg viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="5.5" />
                <path d="M15 15l4 4" />
              </svg>
            `;
        case 'selection':
            return iconSelection();
        case 'advanced-settings':
            return `
              <svg viewBox="0 0 24 24">
                <path d="M5 6h14M8 12h11M5 18h14" />
                <circle cx="8" cy="6" r="1.7" fill="currentColor" stroke="none" />
                <circle cx="13" cy="12" r="1.7" fill="currentColor" stroke="none" />
                <circle cx="10" cy="18" r="1.7" fill="currentColor" stroke="none" />
              </svg>
            `;
        case 'colorbar':
            return iconHeatmap();
    }
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
        linearLayoutMultiInputState?: LinearLayoutMultiInputState;
        composeLayoutTensorViews?: LinearLayoutTensorViewsState;
    };
    if (isLinearLayoutTab(tab)) {
        const cloned = cloneLinearLayoutState(linearLayoutUiState.linearLayoutState);
        const clonedCellText = cloneLinearLayoutCellTextState(linearLayoutUiState.linearLayoutCellTextState);
        const clonedMultiInput = cloneLinearLayoutMultiInputState(linearLayoutUiState.linearLayoutMultiInputState);
        const tensorViews = preservedLinearLayoutTensorViews(linearLayoutUi, tab.id);
        linearLayoutUiState.linearLayoutStates.set(tab.id, cloned);
        linearLayoutUiState.linearLayoutCellTextStates.set(tab.id, clonedCellText);
        linearLayoutUiState.linearLayoutMultiInputStates.set(tab.id, clonedMultiInput);
        linearLayoutUiState.linearLayoutTensorViewsStates.set(tab.id, tensorViews);
        snapshot.composeLayoutState = cloned;
        snapshot.linearLayoutCellTextState = clonedCellText;
        snapshot.linearLayoutMultiInputState = clonedMultiInput;
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
    linearLayoutUiState.linearLayoutMultiInputStates.delete(tabId);
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
        linearLayoutUiState.linearLayoutMultiInputState = defaultLinearLayoutMultiInputState();
        linearLayoutUiState.linearLayoutStates.set(id, cloneLinearLayoutState(linearLayoutUiState.linearLayoutState));
        linearLayoutUiState.linearLayoutCellTextStates.set(id, cloneLinearLayoutCellTextState(linearLayoutUiState.linearLayoutCellTextState));
        linearLayoutUiState.linearLayoutMultiInputStates.set(id, cloneLinearLayoutMultiInputState(linearLayoutUiState.linearLayoutMultiInputState));
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
    const currentMultiInputState = linearLayoutUiState.linearLayoutMultiInputStates.get(currentTab.id);
    if (currentMultiInputState) {
        linearLayoutUiState.linearLayoutMultiInputStates.set(id, cloneLinearLayoutMultiInputState(currentMultiInputState));
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
            shortcut: 'W',
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
            active: snapshot.displayGaps ?? false,
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
    syncLinearLayoutMultiInputState(linearLayoutUi, tab);
    renderLinearLayoutWidget(linearLayoutUi);
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

sidebar.addEventListener('click', (event) => {
    const target = event.target as HTMLElement | null;
    const collapse = target?.closest<HTMLElement>('[data-widget-collapse]');
    if (collapse?.dataset.widgetCollapse) {
        toggleWidgetCollapse(collapse.dataset.widgetCollapse as SidebarWidgetId);
        return;
    }
});

sidebar.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const target = event.target as HTMLElement | null;
    const collapse = target?.closest<HTMLElement>('[data-widget-collapse]');
    if (!collapse?.dataset.widgetCollapse) return;
    event.preventDefault();
    toggleWidgetCollapse(collapse.dataset.widgetCollapse as SidebarWidgetId);
});

sidebar.addEventListener('pointerdown', (event) => {
    const target = event.target as HTMLElement | null;
    const handle = target?.closest<HTMLElement>('[data-widget-handle]');
    const widgetId = handle?.dataset.widgetHandle as SidebarWidgetId | undefined;
    if (!widgetId) return;
    draggedWidgetId = widgetId;
    draggedWidgetSlot = null;
    draggedWidgetPointerId = event.pointerId;
    sidebar.setPointerCapture(event.pointerId);
    syncSidebarDragState();
    event.preventDefault();
});

sidebar.addEventListener('pointermove', (event) => {
    if (!draggedWidgetId || draggedWidgetPointerId !== event.pointerId) return;
    draggedWidgetSlot = sidebarWidgetSlot(event.clientY);
    syncSidebarDragState();
});

sidebar.addEventListener('pointerup', (event) => {
    if (!draggedWidgetId || draggedWidgetPointerId !== event.pointerId) return;
    if (draggedWidgetSlot !== null) moveSidebarWidgetToSlot(draggedWidgetId, draggedWidgetSlot);
    if (sidebar.hasPointerCapture(event.pointerId)) sidebar.releasePointerCapture(event.pointerId);
    clearSidebarDragState();
});

sidebar.addEventListener('pointercancel', (event) => {
    if (draggedWidgetPointerId !== event.pointerId) return;
    if (sidebar.hasPointerCapture(event.pointerId)) sidebar.releasePointerCapture(event.pointerId);
    clearSidebarDragState();
});

function updateSidebar(snapshot: ViewerSnapshot): void {
    const visible = new Set(visibleSidebarWidgets(snapshot));
    applySidebarOrder();
    (Object.entries(sidebarWidgets) as [SidebarWidgetId, HTMLElement][]).forEach(([widgetId, widget]) => {
        widget.classList.toggle('hidden', !visible.has(widgetId));
        widget.classList.toggle('collapsed', collapsedWidgets.has(widgetId));
        syncWidgetHeaderState(widgetId, widget);
    });
    syncSidebarDragState();
}

function applyTensorViewEditor(tensorId: string, editor: TensorViewEditor): void {
    try {
        viewer.setTensorView(tensorId, serializeTensorViewEditor(editor));
        syncLinearLayoutViewFilters(linearLayoutUi);
        viewErrors.delete(tensorId);
    } catch (error) {
        viewErrors.set(tensorId, error instanceof Error ? error.message : String(error));
    }
    render(viewer.getSnapshot());
}

function tensorCallInputValue(value: string): string {
    return value.replace(/^\[/, '').replace(/\]$/, '');
}

function parseIntegerTerm(value: string): number {
    const term = value.trim();
    if (term === '') return Number.NaN;
    if (term === '-1') return -1;
    const parts = term.split('*').map((part) => Number(part.trim()));
    if (parts.some((part) => !Number.isFinite(part))) return Number.NaN;
    return parts.reduce((acc, part) => acc * part, 1);
}

function parseShapeSpec(
    value: string,
    totalElements: number,
): Array<{ label: string; size: number }> {
    const parts = value.split(',').map((part) => part.trim()).filter(Boolean);
    let inferredIndex = -1;
    let anonymousIndex = parts.reduce((maxIndex, part) => {
        const match = part.match(/^(?:\*A|\*|_)(\d+)(?:\s*=.*)?$/);
        return match ? Math.max(maxIndex, Number(match[1]) + 1) : maxIndex;
    }, 0);
    const dims = parts.map((part, index) => {
        if (/^-?\d+$/.test(part)) {
            const size = Number(part);
            if (size === -1) inferredIndex = index;
            return { label: `*A${anonymousIndex++}`, size };
        }
        const anonymous = part.match(/^((?:\*A|\*|_)\d+)(?:\s*=\s*(-?\d+))?$/);
        const explicit = part.match(/^([^=,\[\]]+?)(?:\s*=\s*(-?\d+))?$/);
        const match = anonymous ?? explicit;
        if (!match) throw new Error(`Invalid view term "${part}".`);
        const rawLabel = match[1]!.trim();
        const label = rawLabel;
        const size = match[2] ? Number(match[2]) : -1;
        if (size === -1) inferredIndex = index;
        return { label, size };
    });
    if (inferredIndex >= 0) {
        const known = product(dims.filter((_dim, index) => index !== inferredIndex).map((dim) => dim.size));
        if (known === 0 || totalElements % known !== 0) throw new Error('Could not infer a valid -1 dimension.');
        dims[inferredIndex]!.size = totalElements / known;
    }
    return dims;
}

function parseIntegerListInput(value: string): number[] {
    return value.split(',').map(parseIntegerTerm).filter((part) => Number.isFinite(part));
}

function buildStep4Editor(
    previous: TensorViewEditor,
    viewInput: string,
    permuteInput: string,
    finalViewInput: string | null,
    totalElements: number,
): TensorViewEditor {
    const viewChanged = tensorCallInputValue(previous.viewTensorInput).trim() !== viewInput.trim();
    const parsedView = parseShapeSpec(viewInput, totalElements);
    const baseDims = viewChanged
        ? parsedView.map((dim, index) => ({ id: `dim-${index}`, label: dim.label, size: dim.size }))
        : previous.baseDims.map((dim, index) => ({ ...dim, label: parsedView[index]?.label ?? dim.label, size: parsedView[index]?.size ?? dim.size }));
    const permuteIndices = parseIntegerListInput(permuteInput);
    const permutedDimIds = permuteIndices.map((index) => baseDims[index]?.id).filter((dimId): dimId is string => Boolean(dimId));
    const flattenSeparators = new Array(Math.max(0, permutedDimIds.length - 1)).fill(true);
    return {
        ...previous,
        viewTensorInput: `[${viewInput}]`,
        finalViewInput: finalViewInput?.trim() ? `[${finalViewInput}]` : undefined,
        baseDims,
        permutedDimIds,
        flattenSeparators,
        singletons: [],
    };
}

function tensorViewHelpHtml(shape: readonly number[], axisLabels: readonly string[]): string {
    const shapeText = shape.join(', ');
    const reversedRangeText = shape.map((_dim, index) => shape.length - index - 1).join(', ');
    const labeledShapeText = shape.map((size, index) => `${axisLabels[index] ?? `A${index}`}=${size}`).join(', ');
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>In "Tensor View", input the tensor view/permutation/slice.</span>
          </div>
          <div class="usage-guide-step">
            <span>The input is of format "<strong>tensor</strong>[<strong>.view(A)</strong>][<strong>.permute(B)</strong>][<strong>.view(C)</strong>][<strong>[D]</strong>]".</span>
          </div>
          <div class="usage-guide-step">
            <span>View/permutation/slice semantics are similar to torch, but None indexes aren't allowed (instead, just insert a 1 dimension via a view).</span>
          </div>
          <div class="usage-guide-column">
            <div class="usage-guide-subtitle">Examples</div>
            <div class="usage-guide-example"><code>tensor.view(${shapeText})</code></div>
            <div class="usage-guide-example"><code>tensor.view(-1)</code></div>
            <div class="usage-guide-example"><code>tensor.permute(${reversedRangeText})</code></div>
            <div class="usage-guide-example"><code>tensor[0]</code></div>
            <div class="usage-guide-example"><code>tensor.view(${labeledShapeText})</code></div>
            <div class="usage-guide-example"><code>tensor.view(${labeledShapeText}).permute(${reversedRangeText})</code></div>
            <div class="usage-guide-example"><code>tensor.view(${labeledShapeText}).permute(${reversedRangeText}).view(${labeledShapeText})</code></div>
            <div class="usage-guide-example"><code>tensor.view(${labeledShapeText}).permute(${reversedRangeText}).view(${labeledShapeText})[0]</code></div>
          </div>
        </div>
      </details>
    `;
}

function parseTensorViewExpressionInput(
    value: string,
    previous: TensorViewEditor,
    shape: readonly number[],
): TensorViewEditor {
    const text = value.trim();
    if (!text.startsWith('tensor')) throw new Error('Tensor View must start with "tensor".');
    let rest = text.slice('tensor'.length);
    const consumeCall = (name: 'view' | 'permute'): string | null => {
        if (!rest.startsWith(`.${name}(`)) return null;
        const start = name.length + 2;
        let depth = 1;
        let index = start;
        while (index < rest.length && depth > 0) {
            const char = rest[index]!;
            if (char === '(') depth += 1;
            else if (char === ')') depth -= 1;
            index += 1;
        }
        if (depth !== 0) throw new Error(`Unclosed ${name}(...) in Tensor View.`);
        const content = rest.slice(start, index - 1).trim();
        rest = rest.slice(index);
        return content;
    };
    const defaultView = [...shape].join(', ');
    const firstView = consumeCall('view') ?? defaultView;
    const permute = consumeCall('permute') ?? [...shape].map((_dim, index) => index).join(', ');
    const finalView = consumeCall('view');
    const nextEditor = buildStep4Editor(previous, firstView, permute, finalView, product([...shape]));
    let slicedTokenKeys: string[] = [];
    let sliceValues: Record<string, number> = {};
    const bracket = rest.trim();
    if (bracket !== '') {
        if (bracket.startsWith('.') || !bracket.startsWith('[') || !bracket.endsWith(']')) {
            throw new Error('Tensor View input must be in form "tensor<.view(A)><.permute(B)><.view(C)><[D]> (text enclosed in <...> are optional)."');
        }
        const terms = bracket.slice(1, -1).split(',').map((part) => part.trim());
        const finalViewDims = finalView ? parseShapeSpec(finalView, product(nextEditor.baseDims.map((dim) => dim.size))) : [];
        terms.forEach((term, index) => {
            if (term === ':') return;
            const value = Number(term);
            if (!Number.isFinite(value)) throw new Error(`Invalid slice term "${term}".`);
            // implicit second views made simple edits fragile: changing the first
            // view left a stale hidden finalViewInput behind, and the next round-trip
            // failed because that old shape no longer matched the new base product
            const key = finalView
                ? `view:${finalViewDims[index]?.label ?? `*A${index}`}`
                : `group:${nextEditor.permutedDimIds[index] ?? `missing-${index}`}`;
            slicedTokenKeys.push(key);
            sliceValues[key] = Math.floor(value);
        });
    }
    return {
        ...nextEditor,
        slicedTokenKeys,
        sliceValues,
    };
}

function renderTensorViewWidget(snapshot: ViewerSnapshot): void {
    if (suspendTensorViewRender) return;
    const model = viewer.getInspectorModel();
    const tab = activeTab();
    const linearLayoutMeta = tab && isLinearLayoutTab(tab) ? composeLayoutMetaForTab(tab) : null;
    if (!model.handle) {
        tensorViewWidget.innerHTML = `${widgetTitle('tensor-view', 'Visualize tensor views, permutations, slices, or a combination of these ops.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }

    const error = viewErrors.get(model.handle.id);
    const editor = model.viewEditor;
    if (!editor) return;
    const selectionMap = tab && isLinearLayoutTab(tab) ? linearLayoutSelectionMapForTab(linearLayoutUi, tab) : null;
    const multiInput = selectionMap ? linearLayoutMultiInputModel(linearLayoutUi, selectionMap) : null;
    const tensorOptions = model.tensors.map((tensor) => `
      <option value="${tensor.id}" ${tensor.id === model.handle!.id ? 'selected' : ''}>${tensor.name || tensor.id}</option>
    `).join('');
    const sliceContent = model.viewTokens.map((token) => (
        token.kind === 'singleton'
            ? `<span class="dim-chip dim-chip-singleton">1</span>`
            : `<button class="dim-chip interactive-chip${token.sliced ? ' dim-chip-sliced dim-chip-active' : ''}" data-slice-token="${token.key}" type="button">${token.token}<span>=${token.size}</span></button>`
    )).join('');
    const originalAxisLabels = linearLayoutMeta?.tensors.find((tensor) => tensor.id === model.handle!.id)?.axisLabels ?? model.handle.axisLabels;
    const defaultLabeledShape = model.handle.shape.map((size, index) => `${originalAxisLabels[index] ?? `A${index}`}=${size}`).join(', ');
    tensorViewWidget.innerHTML = `
      ${widgetTitle('tensor-view', 'Visualize tensor views, permutations, slices, or a combination of these ops.')}
      <div class="widget-body">
        <div class="field">
          ${labelWithInfo('Tensor', 'Choose which loaded tensor the Permute/Slice editor controls.', 'tensor-select')}
          <select id="tensor-select">${tensorOptions}</select>
        </div>
        <div class="permute-slice-step">
          ${labelWithInfo('Tensor View', 'Edit the full tensor expression directly. Standard view, permute, and non-none indexing semantics apply.', 'tensor-view-input')}
          ${tensorViewHelpHtml(model.handle.shape, originalAxisLabels).replace('<details class="usage-guide">', `<details class="usage-guide"${tensorViewHelpOpen ? ' open' : ''}>`)}
          <textarea id="tensor-view-input" rows="3" placeholder="tensor">${model.preview}</textarea>
        </div>
        <div class="permute-slice-step">
          <div class="label-row"><span class="meta-label">Slice Dims</span>${infoButton('Convenience utility for inspecting different slices. Click a dimension to toggle between showing one/all elements at a time. If showing one element of a dimension, drag its slider to change the displayed index.')}</div>
          <div class="dim-chip-row dim-chip-row-compact" id="slice-dims">${sliceContent}</div>
        </div>
        ${error ? `<div class="error-box">${error}</div>` : ''}
        ${model.sliceTokens.length === 0 && !multiInput ? '' : '<div class="slider-list" id="slice-token-controls"></div>'}
        <div class="permute-slice-actions">
          <button class="reset-view-button interactive-chip" id="reset-view-button" type="button" title="Change tensor view to default view (original shape + dimension labels + no permutations)">Reset View</button>
        </div>
      </div>
    `;

    const tensorViewInput = tensorViewWidget.querySelector<HTMLTextAreaElement>('#tensor-view-input');
    const select = tensorViewWidget.querySelector<HTMLSelectElement>('#tensor-select');
    const sliceHost = tensorViewWidget.querySelector<HTMLElement>('#slice-token-controls');
    const usageGuide = tensorViewWidget.querySelector<HTMLDetailsElement>('.usage-guide');
    usageGuide?.addEventListener('toggle', () => {
        tensorViewHelpOpen = usageGuide.open;
    });
    select?.addEventListener('change', () => {
        logUi('tensor-select', select.value);
        viewer.setActiveTensor(select.value);
    });
    tensorViewInput?.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        tensorViewInput.blur();
    });
    tensorViewInput?.addEventListener('change', () => {
        logUi('tensor-view:change', { tensorId: model.handle!.id, value: tensorViewInput.value });
        try {
            applyTensorViewEditor(model.handle!.id, parseTensorViewExpressionInput(
                tensorViewInput.value,
                editor,
                model.handle!.shape,
            ));
        } catch (error) {
            viewErrors.set(model.handle!.id, error instanceof Error ? error.message : String(error));
            render(viewer.getSnapshot());
        }
    });
    tensorViewWidget.querySelectorAll<HTMLElement>('[data-slice-token]').forEach((button) => {
        button.addEventListener('click', () => {
            const key = button.dataset.sliceToken;
            if (!key) return;
            const sliced = editor.slicedTokenKeys.includes(key);
            applyTensorViewEditor(model.handle!.id, {
                ...editor,
                slicedTokenKeys: sliced ? editor.slicedTokenKeys.filter((entry) => entry !== key) : [...editor.slicedTokenKeys, key],
                sliceValues: sliced ? editor.sliceValues : { ...editor.sliceValues, [key]: editor.sliceValues[key] ?? 0 },
            });
        });
    });
    tensorViewWidget.querySelector<HTMLElement>('#reset-view-button')?.addEventListener('click', () => {
        applyTensorViewEditor(model.handle!.id, {
            version: 2,
            viewTensorInput: `[${defaultLabeledShape}]`,
            baseDims: [],
            permutedDimIds: [],
            flattenSeparators: [],
            singletons: [],
            slicedTokenKeys: [],
            sliceValues: {},
        });
    });

    const sliderRows = model.sliceTokens.map((token) => {
        const row = document.createElement('div');
        row.className = 'slider-row';
        const sliderId = `slice-${token.key.replace(/[^a-z0-9_-]/gi, '-')}`;
        row.innerHTML = `
          <label for="${sliderId}">${token.token}</label>
          <input id="${sliderId}" type="range" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
          <input id="${sliderId}-number" type="number" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
        `;
        const slider = row.querySelector<HTMLInputElement>(`#${sliderId}`);
        const number = row.querySelector<HTMLInputElement>(`#${sliderId}-number`);
        const syncTensorViewInput = (): void => {
            const tensorViewInput = tensorViewWidget.querySelector<HTMLTextAreaElement>('#tensor-view-input');
            if (tensorViewInput) tensorViewInput.value = viewer.getInspectorModel().preview;
        };
        const applyValue = (nextValue: number): void => {
            logUi('slice-token:update', { tensorId: model.handle!.id, token: token.token, value: nextValue });
            viewer.setSliceTokenValue(model.handle!.id, token.key, nextValue);
            syncLinearLayoutViewFilters(linearLayoutUi);
            // updating the whole widget during slider drag resets the active range input,
            // so keep the drag stable and only sync the expression text in place
            syncTensorViewInput();
            requestAnimationFrame(syncTensorViewInput);
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
            render(viewer.getSnapshot());
        });
        number?.addEventListener('change', () => {
            const clamped = Math.max(0, Math.min(token.size - 1, Number(number.value)));
            number.value = String(clamped);
            if (slider) slider.value = String(clamped);
            applyValue(clamped);
            suspendTensorViewRender = false;
            render(viewer.getSnapshot());
        });
        return row;
    });
    if (multiInput) {
        const row = document.createElement('div');
        row.className = 'slider-row';
        row.innerHTML = `
          <label for="multi-input-slider">Multi-Input</label>
          <input id="multi-input-slider" type="range" min="-1" max="${Math.max(0, multiInput.size - 1)}" value="${multiInput.value}" />
          <input id="multi-input-slider-number" type="number" min="-1" max="${Math.max(0, multiInput.size - 1)}" value="${multiInput.value}" />
        `;
        const slider = row.querySelector<HTMLInputElement>('#multi-input-slider');
        const number = row.querySelector<HTMLInputElement>('#multi-input-slider-number');
        const applyValue = (nextValue: number): void => {
            linearLayoutUiState.linearLayoutMultiInputState[multiInput.focusedTensorId] = nextValue;
            const activeTabId = linearLayoutUi.getActiveTabId();
            if (activeTabId) {
                linearLayoutUiState.linearLayoutMultiInputStates.set(activeTabId, cloneLinearLayoutMultiInputState(linearLayoutUiState.linearLayoutMultiInputState));
            }
            syncLinearLayoutViewFilters(linearLayoutUi);
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
            render(viewer.getSnapshot());
        });
        number?.addEventListener('change', () => {
            const clamped = Math.max(-1, Math.min(multiInput.size - 1, Number(number.value)));
            number.value = String(clamped);
            if (slider) slider.value = String(clamped);
            applyValue(clamped);
            suspendTensorViewRender = false;
            render(viewer.getSnapshot());
        });
        sliderRows.push(row);
    }
    sliceHost?.replaceChildren(...sliderRows);
}

function renderInspectorWidget(snapshot: ViewerSnapshot): void {
    const model = viewer.getInspectorModel();
    const dimensionMappingScheme = snapshot.dimensionMappingScheme ?? 'z-order';
    const tab = activeTab();
    const linearLayoutTab = tab && isLinearLayoutTab(tab) ? tab : null;
    if (!model.handle) {
        inspectorReady = false;
        inspectorRefs = null;
        inspectorWidget.innerHTML = `${widgetTitle('inspector', 'Shows metadata for the active tensor and hover data for the current cell.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }
    if (!inspectorReady) {
        inspectorWidget.innerHTML = `
          ${widgetTitle('inspector', 'Shows metadata for the active tensor and hover data for the current cell.')}
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

function renderLinearLayoutHoverPopup(): void {
    const tab = activeTab();
    const linearLayoutTab = tab && isLinearLayoutTab(tab) ? tab : null;
    const hover = viewer.getHover();
    const linearLayout = linearLayoutTab ? linearLayoutSelectionMapForTab(linearLayoutUi, linearLayoutTab) : null;
    const entries = linearLayoutHoverPopupEntries(linearLayoutUi, hover, linearLayout);
    if (entries.length === 0) {
        hoverPopup.classList.add('hidden');
        hoverPopup.innerHTML = '';
        return;
    }
    hoverPopup.innerHTML = `
      <div class="linear-layout-hover-popup-title">Input Cells</div>
      <div class="linear-layout-hover-popup-list">${entries.map((entry) => `
        <div class="linear-layout-hover-popup-item">
          <span class="linear-layout-hover-popup-swatch" style="--cell-color: ${escapeInfo(entry.color)};"></span>
          <span class="linear-layout-hover-popup-text">${escapeInfo(entry.text).replace(/\n/g, '<br />')}</span>
        </div>
      `).join('')}</div>
    `;
    hoverPopup.classList.remove('hidden');
    placeHoverPopup();
}

function placeHoverPopup(): void {
    if (hoverPopup.classList.contains('hidden')) return;
    const rect = viewport.getBoundingClientRect();
    const width = hoverPopup.offsetWidth;
    const height = hoverPopup.offsetHeight;
    const maxLeft = Math.max(12, rect.width - width - 12);
    const maxTop = Math.max(12, rect.height - height - 12);
    hoverPopup.style.left = `${Math.min(maxLeft, hoverPopupPointer.x + 18)}px`;
    hoverPopup.style.top = `${Math.min(maxTop, hoverPopupPointer.y + 18)}px`;
}

function updateHoverPopupPointer(event: PointerEvent): void {
    const rect = viewport.getBoundingClientRect();
    hoverPopupPointer = {
        x: Math.max(12, event.clientX - rect.left),
        y: Math.max(12, event.clientY - rect.top),
    };
    placeHoverPopup();
}

function renderSelectionWidget(snapshot: ViewerSnapshot): void {
    const model = viewer.getInspectorModel();
    const selectionModeActive = (snapshot.interactionMode ?? viewer.getInteractionMode()) === 'select';
    if (!model.handle) {
        selectionWidget.innerHTML = `${widgetTitle('selection', 'Shows how many cells are highlighted and summary statistics across their loaded numeric values. Selection is only available in 2D contiguous mapping.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }
    const summary = viewer.getSelectionSummary();
    const enabled = selectionEnabled(snapshot) && selectionModeActive;
    const note = !selectionModeActive
        ? 'Switch to Selection mode from the bottom toolbar to enable this panel.'
        : enabled
        ? 'Left-click and drag to draw a selection box, then release to apply it. Hold Shift to add cells, or hold Ctrl to remove cells.'
        : 'Selection is only available in 2D contiguous mapping.';
    selectionWidget.innerHTML = `
      ${widgetTitle('selection', 'Shows how many cells are highlighted and summary statistics across their loaded numeric values. Selection is only available in 2D contiguous mapping.')}
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
        <div class="colorbar-title">${range.name || range.id}</div>
        <div class="colorbar"></div>
        <div class="colorbar-labels">
          <span>${formatRangeValue(range.min)}</span>
          <span>${formatRangeValue(range.max)}</span>
        </div>
      </div>
    `).join('');
    colorbarWidget.innerHTML = `
      ${widgetTitle('colorbar', 'Shows the heatmap range for each loaded tensor using its current minimum and maximum values. The gradient can be linear or signed-log depending on Advanced Settings.')}
      <div class="widget-body">
        ${scaleMode}
        ${sections}
      </div>
    `;
}

function renderAdvancedSettingsWidget(snapshot: ViewerSnapshot): void {
    const currentValue = snapshot.dimensionBlockGapMultiple ?? 3;
    const displayGaps = snapshot.displayGaps ?? false;
    const logScale = snapshot.logScale ?? false;
    const collapseHiddenAxes = snapshot.collapseHiddenAxes ?? snapshot.showSlicesInSamePlace ?? false;
    const dimensionMappingScheme = snapshot.dimensionMappingScheme ?? 'z-order';
    advancedSettingsWidget.innerHTML = `
      ${widgetTitle('advanced-settings', 'Adjust lower-level layout tuning that changes how tensor dimension blocks are spaced and assigned to x, y, and z families.')}
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
    const tab = activeTab();
    const activeTensorId = tab && isLinearLayoutTab(tab) ? (snapshot.activeTensorId ?? null) : null;
    if (activeTensorId !== lastLinearLayoutActiveTensorId) {
        lastLinearLayoutActiveTensorId = activeTensorId;
        if (activeTensorId) {
            syncLinearLayoutViewFilters(linearLayoutUi);
            return;
        }
    }
    updateSidebar(snapshot);
    renderTabStrip();
    renderCellTextWidget(linearLayoutUi);
    renderControlDock(snapshot);
    renderTensorViewWidget(snapshot);
    renderInspectorWidget(snapshot);
    renderLinearLayoutHoverPopup();
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
    const storedMultiInputState = (tab.viewer as { linearLayoutMultiInputState?: unknown }).linearLayoutMultiInputState;
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
        if (isLinearLayoutMultiInputState(storedMultiInputState)) {
            linearLayoutUiState.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(storedMultiInputState));
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
        dimensionMappingScheme: isLinearLayout
            ? (tab.viewer.dimensionMappingScheme ?? 'contiguous')
            : tab.viewer.dimensionMappingScheme,
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
    if (isLinearLayout && isLinearLayoutMultiInputState(storedMultiInputState)) {
        linearLayoutUiState.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(storedMultiInputState));
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
    linearLayoutUiState.linearLayoutMultiInputStates.clear();
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
    linearLayoutUiState.linearLayoutMultiInputStates.clear();
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
    linearLayoutUiState.linearLayoutMultiInputStates.clear();
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
        const input = tensorViewWidget.querySelector<HTMLTextAreaElement>('#tensor-view-input');
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
    if (isPaletteInput && !event.ctrlKey && !event.metaKey && !event.altKey) return;
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
    } else if (!event.ctrlKey && !event.metaKey && !event.altKey && event.key.toLowerCase() === 'w') {
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
viewport.addEventListener('pointermove', updateHoverPopupPointer);
viewport.addEventListener('pointerleave', () => {
    hoverPopup.classList.add('hidden');
});
viewer.subscribeHover(() => {
    const snapshot = viewer.getSnapshot();
    renderInspectorWidget(snapshot);
    renderLinearLayoutHoverPopup();
});
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
