import {
    createTypedArray,
    serializeTensorViewEditor,
    TensorViewer,
    product,
    type BundleManifest,
    type DimensionMappingScheme,
    type InteractionMode,
    type LoadedBundleDocument,
    type NumericArray,
    type SessionBundleManifest,
    type TensorViewEditor,
    type ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import {
    formatAxisValues,
    formatRangeValue,
    infoButton,
    labelWithInfo,
    selectionEnabled,
} from './app-format.js';
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
} = mountAppShell(app);

const viewer = new TensorViewer(viewport);
const sidebar = tensorViewWidget.parentElement as HTMLElement;
const viewErrors = new Map<string, string>();
let suspendTensorViewRender = false;
let tensorViewHelpOpen = false;
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
let permuteDragDimId: string | null = null;
let permuteDropSlot: number | null = null;
let permuteDragPointerId: number | null = null;
let permuteSwapDimId: string | null = null;
let permuteDragStartX = 0;
let permuteDragMoved = false;

const MIN_VIEWPORT_WIDTH = 280;

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

type SidebarWidgetId = 'tensor-view' | 'inspector' | 'selection' | 'advanced-settings' | 'colorbar';

const sidebarWidgets: Record<SidebarWidgetId, HTMLElement> = {
    'tensor-view': tensorViewWidget,
    inspector: inspectorWidget,
    selection: selectionWidget,
    'advanced-settings': advancedSettingsWidget,
    colorbar: colorbarWidget,
};

const sidebarWidgetLabels: Record<SidebarWidgetId, string> = {
    'tensor-view': 'Permute/Slice',
    inspector: 'Hover Info',
    selection: 'Selection',
    'advanced-settings': 'Advanced Settings',
    colorbar: 'Heatmap',
};

sidebarHeader.classList.add('label-row');
sidebarHeader.innerHTML = `<span>Widgets</span>${infoButton('Extra settings to inspect/change the visible tensor(s). Click the arrows/widget header text on each widget to expand/collapse them. Change widget position by left-clicking + dragging on the grabber by the right of each widget.')}`;

let widgetOrder: SidebarWidgetId[] = ['tensor-view', 'inspector', 'selection', 'advanced-settings', 'colorbar'];
let draggedWidgetId: SidebarWidgetId | null = null;
let draggedWidgetSlot: number | null = null;
let draggedWidgetPointerId: number | null = null;
const collapsedWidgets = new Set<SidebarWidgetId>(widgetOrder);

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

function commandActions(): CommandAction[] {
    return [
        { action: 'command-palette', label: 'Command Palette', shortcut: '?', keywords: 'command palette search actions' },
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
        { action: 'tensor-view', label: 'Toggle Permute/Slice', shortcut: 'Ctrl+V', keywords: 'widgets permute slice tensor permutation slicing tensor view panel' },
        { action: 'inspector', label: 'Toggle Hover Info', shortcut: '', keywords: 'widgets hover info inspector panel' },
        { action: 'selection', label: 'Toggle Selection', shortcut: '', keywords: 'widgets selection panel stats highlighted cells' },
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
    return widgetOrder.filter((widgetId) => (
        (widgetId === 'tensor-view' && showTensorViewWidget)
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
    sidebar.replaceChildren(sidebarHeader, ...widgetOrder.map((widgetId) => sidebarWidgets[widgetId]));
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

function activeTab(): LoadedBundleDocument | undefined {
    return sessionTabs.find((tab) => tab.id === activeTabId);
}

function captureActiveTabSnapshot(): void {
    const tab = activeTab();
    if (!tab) return;
    tab.manifest.viewer = viewer.getSnapshot();
}

async function closeTab(tabId: string): Promise<void> {
    const index = sessionTabs.findIndex((tab) => tab.id === tabId);
    if (index < 0) return;
    const wasActive = activeTabId === tabId;
    if (wasActive) captureActiveTabSnapshot();
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
    tabStrip.classList.toggle('hidden', sessionTabs.length < 2);
    if (sessionTabs.length < 2) {
        tabStrip.replaceChildren();
        return;
    }
    tabStrip.replaceChildren(...sessionTabs.map((tab) => {
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
    }));
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
    if (!switchingTab) captureActiveTabSnapshot();
    switchingTab = true;
    activeTabId = tabId;
    viewer.loadBundleData(tab.manifest, tab.tensors);
    switchingTab = false;
    renderTabStrip();
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

sidebar.addEventListener('click', (event) => {
    const target = event.target as HTMLElement | null;
    const collapse = target?.closest<HTMLElement>('[data-widget-collapse]');
    if (collapse?.dataset.widgetCollapse) {
        const widgetId = collapse.dataset.widgetCollapse as SidebarWidgetId;
        if (collapsedWidgets.has(widgetId)) collapsedWidgets.delete(widgetId);
        else collapsedWidgets.add(widgetId);
        render(viewer.getSnapshot());
        return;
    }
});

sidebar.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const target = event.target as HTMLElement | null;
    const collapse = target?.closest<HTMLElement>('[data-widget-collapse]');
    if (!collapse?.dataset.widgetCollapse) return;
    event.preventDefault();
    const widgetId = collapse.dataset.widgetCollapse as SidebarWidgetId;
    if (collapsedWidgets.has(widgetId)) collapsedWidgets.delete(widgetId);
    else collapsedWidgets.add(widgetId);
    render(viewer.getSnapshot());
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

type EditorDisplayToken = {
    key: string;
    kind: 'axis_group' | 'singleton';
    label: string;
    size: number;
    dimIds: string[];
    sliced: boolean;
    value: number;
};

function editorGroups(editor: TensorViewEditor): EditorDisplayToken[] {
    const dimById = new Map(editor.baseDims.map((dim) => [dim.id, dim]));
    const groups: EditorDisplayToken[] = [];
    let current: typeof editor.baseDims = [];
    editor.permutedDimIds.forEach((dimId, index) => {
        const dim = dimById.get(dimId);
        if (!dim) return;
        current.push(dim);
        const split = editor.flattenSeparators[index] ?? true;
        if (!split) return;
        const dimIds = current.map((entry) => entry.id);
        const key = `group:${dimIds.join('+')}`;
        groups.push({
            key,
            kind: 'axis_group',
            label: current.map((entry) => entry.label).join(''),
            size: product(current.map((entry) => entry.size)),
            dimIds,
            sliced: editor.slicedTokenKeys.includes(key),
            value: editor.sliceValues[key] ?? 0,
        });
        current = [];
    });
    if (current.length !== 0) {
        const dimIds = current.map((entry) => entry.id);
        const key = `group:${dimIds.join('+')}`;
        groups.push({
            key,
            kind: 'axis_group',
            label: current.map((entry) => entry.label).join(''),
            size: product(current.map((entry) => entry.size)),
            dimIds,
            sliced: editor.slicedTokenKeys.includes(key),
            value: editor.sliceValues[key] ?? 0,
        });
    }
    return groups;
}

function editorStep3Tokens(editor: TensorViewEditor): EditorDisplayToken[] {
    const groups = editorGroups(editor);
    const tokens: EditorDisplayToken[] = groups.slice();
    editor.singletons
        .slice()
        .sort((left, right) => left.position - right.position || left.id.localeCompare(right.id))
        .forEach((singleton) => {
            tokens.splice(Math.max(0, Math.min(tokens.length, singleton.position)), 0, {
                key: `singleton:${singleton.id}`,
                kind: 'singleton',
                label: '1',
                size: 1,
                dimIds: [],
                sliced: editor.slicedTokenKeys.includes(`singleton:${singleton.id}`),
                value: editor.sliceValues[`singleton:${singleton.id}`] ?? 0,
            });
        });
    return tokens;
}

function applyTensorViewEditor(tensorId: string, editor: TensorViewEditor): void {
    try {
        logUi('tensor-view:apply:start', {
            tensorId,
            viewTensorInput: editor.viewTensorInput,
            baseDims: editor.baseDims,
            permutedDimIds: editor.permutedDimIds,
            flattenSeparators: editor.flattenSeparators,
            slicedTokenKeys: editor.slicedTokenKeys,
            sliceValues: editor.sliceValues,
        });
        viewer.setTensorView(tensorId, serializeTensorViewEditor(editor));
        const nextModel = viewer.getInspectorModel();
        logUi('tensor-view:apply:result', {
            tensorId,
            canonicalView: nextModel.viewInput,
            preview: nextModel.preview,
            viewEditor: nextModel.viewEditor,
            sliceTokens: nextModel.sliceTokens,
        });
        viewErrors.delete(tensorId);
    } catch (error) {
        logUi('tensor-view:apply:error', {
            tensorId,
            error: error instanceof Error ? error.message : String(error),
            editor,
        });
        viewErrors.set(tensorId, error instanceof Error ? error.message : String(error));
    }
    render(viewer.getSnapshot());
}

function tensorViewStepGuideHtml(title: string, sections: string[]): string {
    return `
      <details class="usage-guide">
        <summary>${title}</summary>
        <div class="usage-guide-body">${sections.join('')}</div>
      </details>
    `;
}

function tensorViewStepGuides(editor: TensorViewEditor, preview: string): {
    view: string;
    permute: string;
    flatten: string;
    singleton: string;
    slice: string;
} {
    const dims = editor.baseDims;
    const current = editor.viewTensorInput;
    const currentSizes = `[${dims.map((dim) => String(dim.size)).join(', ')}]`;
    const currentPreview = preview || 'tensor';
    const inferred = `[${dims.map((dim, index) => index === dims.length - 1 ? '-1' : String(dim.size)).join(', ')}]`;
    const merged = dims.length < 2
        ? currentSizes
        : `[${product(dims.slice(0, 2).map((dim) => dim.size))}${dims.slice(2).map((dim) => `, ${dim.size}`).join('')}]`;
    const wrongProduct = `[${dims.map((dim, index) => String(dim.size + (index === 0 ? 1 : 0))).join(', ')}]`;
    const permutedLabels = editor.permutedDimIds
        .map((dimId) => editor.baseDims.find((dim) => dim.id === dimId))
        .filter((dim): dim is TensorViewEditor['baseDims'][number] => dim !== undefined);
    const permuteOrder = permutedLabels.map((dim) => dim.label).join(', ');
    const permuteMoved = permutedLabels.length < 2
        ? permuteOrder || 'A'
        : [permutedLabels[permutedLabels.length - 1]!, ...permutedLabels.slice(0, -1)].map((dim) => dim.label).join(', ');
    const flattenBefore = permutedLabels.map((dim) => `${dim.label}=${dim.size}`).join(', ');
    const flattenAfter = permutedLabels.length < 2
        ? flattenBefore
        : [`${permutedLabels[0]!.label}${permutedLabels[1]!.label}=${permutedLabels[0]!.size * permutedLabels[1]!.size}`, ...permutedLabels.slice(2).map((dim) => `${dim.label}=${dim.size}`)].join(', ');
    const singletonAfter = dims.length === 0
        ? '[1]'
        : `[${dims[0]!.label}=${dims[0]!.size}, 1${dims.slice(1).map((dim) => `, ${dim.label}=${dim.size}`).join('')}]`;
    const sliceTarget = permutedLabels[1] ?? permutedLabels[0] ?? dims[0];
    const sliceExample = sliceTarget ? `${sliceTarget.label.toLowerCase()} fixed by slider` : 'first visible dimension fixed by slider';
    return {
        view: tensorViewStepGuideHtml('How do I use this?', [
            `<div class="usage-guide-step"><strong>What this step is for</strong><span>Define the logical dimensions that every later step will permute, flatten, expand, or slice. If this step is wrong, every later preview string is built on the wrong shape.</span></div>`,
            `<div class="usage-guide-step"><strong>Accepted input forms</strong><span>Use bracketed, comma-separated entries. If names are confusing, you can enter just sizes such as <code>[2, 3, 4]</code>, <code>[6, 4]</code>, or <code>[2, 3, -1]</code>. A bare number becomes an anonymous internal label, so the UI still has something to permute and slice later.</span></div>`,
            `<div class="usage-guide-step"><strong>Inference and validation</strong><span>You may use <code>-1</code> once to infer one size from the tensor element count. Labels must be unique. The final product must equal the original tensor element count. Inputs that change the product or repeat labels are rejected.</span></div>`,
            `<div class="usage-guide-columns">
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Valid examples</div>
                <div class="usage-guide-example"><code>${currentSizes}</code><span>The current logical shape written without dimension names.</span></div>
                <div class="usage-guide-example"><code>${inferred}</code><span>Uses one inferred size while preserving the total product.</span></div>
                <div class="usage-guide-example"><code>${merged}</code><span>Collapses adjacent dimensions into one explicit logical dimension before any permutation.</span></div>
              </div>
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Invalid examples</div>
                <div class="usage-guide-example"><code>${wrongProduct}</code><span>Changes the total element count, so the view no longer matches the tensor.</span></div>
                <div class="usage-guide-example"><code>[${dims.map(() => '-1').join(', ')}]</code><span>Uses <code>-1</code> more than once, so the missing sizes are underdetermined.</span></div>
                <div class="usage-guide-example"><code>[${dims.map((_dim, index) => index === 0 ? '0' : '1').join(', ')}]</code><span>Collapses the product instead of preserving it, so it cannot describe the original tensor.</span></div>
              </div>
            </div>`,
        ]),
        permute: tensorViewStepGuideHtml('How do I use this?', [
            `<div class="usage-guide-step"><strong>What this step is for</strong><span>Change axis order without changing axis sizes. This is the step that changes which logical dimension becomes the leftmost, topmost, or deepest visible axis in the preview.</span></div>`,
            `<div class="usage-guide-step"><strong>Ways to reorder</strong><span>Drag a box left or right to insert it at a new position. Or click one axis, then click another to swap them. Clicking the already selected axis cancels the pending swap. With only one dimension visible here, permutation is a no-op.</span></div>`,
            `<div class="usage-guide-step"><strong>What to expect in the preview</strong><span>If the order matches the underlying logical tensor, the preview omits <code>.permute(...)</code>. As soon as you move dimensions into a different order, the preview adds a <code>.permute(...)</code> term.</span></div>`,
            `<div class="usage-guide-columns">
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Current order</div>
                <div class="usage-guide-example"><code>${permuteOrder || 'A'}</code><span>Current logical order after the View Tensor step.</span></div>
                <div class="usage-guide-example"><code>${currentPreview}</code><span>Current preview before another permutation edit.</span></div>
              </div>
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Example reorder</div>
                <div class="usage-guide-example"><code>${permuteMoved}</code><span>Example order after dragging the last dimension to the front.</span></div>
                <div class="usage-guide-example"><code>click ${permutedLabels[0]?.label ?? 'A'} then ${permutedLabels[1]?.label ?? permutedLabels[0]?.label ?? 'A'}</code><span>Swaps two axes directly instead of dragging.</span></div>
              </div>
            </div>`,
        ]),
        flatten: tensorViewStepGuideHtml('How do I use this?', [
            `<div class="usage-guide-step"><strong>What this step is for</strong><span>Merge adjacent dimensions into one grouped axis after they are already in the order you want. Flattening changes the <code>.view(...)</code> part of the preview, not the <code>.permute(...)</code> part.</span></div>`,
            `<div class="usage-guide-step"><strong>Separator rules</strong><span>A comma keeps two adjacent dimensions separate. A multiply joins them into one grouped dimension. Flatten only works on adjacent dimensions in the current permuted order, so permute first if the pair you want is not adjacent yet.</span></div>`,
            `<div class="usage-guide-step"><strong>What cases matter</strong><span>Leaving every separator as a comma preserves one axis per logical dimension. Turning one separator into a multiply creates one grouped axis. Turning several separators into multiplies creates larger grouped axes.</span></div>`,
            `<div class="usage-guide-columns">
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Before flatten</div>
                <div class="usage-guide-example"><code>${flattenBefore || 'A=1'}</code><span>Every dimension is still separate.</span></div>
                <div class="usage-guide-example"><code>${currentPreview}</code><span>Current preview before toggling a separator.</span></div>
              </div>
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">After flatten</div>
                <div class="usage-guide-example"><code>${flattenAfter || 'A=1'}</code><span>Example result after flattening the first adjacent pair.</span></div>
                <div class="usage-guide-example"><code>comma keeps axes separate, multiply groups them</code><span>Use comma when you still need those axes to be permuted or sliced independently later.</span></div>
              </div>
            </div>`,
        ]),
        singleton: tensorViewStepGuideHtml('How do I use this?', [
            `<div class="usage-guide-step"><strong>What this step is for</strong><span>Insert explicit size-1 axes when you want the preview shape to match a broadcasted, expanded, or layout-sensitive form. Singletons do not change the element count.</span></div>`,
            `<div class="usage-guide-step"><strong>Where insertion matters</strong><span>Insert before the first axis to create a leading batch-like dimension, between axes to separate two existing dimensions, or at the end to create a trailing singleton. The insertion position changes the preview shape and the later slice positions.</span></div>`,
            `<div class="usage-guide-step"><strong>What cases to watch</strong><span>You can insert multiple singletons. They can stay visible as <code>1</code> axes, or later be sliced away. Removing a singleton deletes only that size-1 axis.</span></div>`,
            `<div class="usage-guide-columns">
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Current shape</div>
                <div class="usage-guide-example"><code>${current}</code><span>Current logical dimensions before inserting a singleton.</span></div>
                <div class="usage-guide-example"><code>${currentPreview}</code><span>Current preview before insertion.</span></div>
              </div>
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Example insertions</div>
                <div class="usage-guide-example"><code>${singletonAfter}</code><span>Example result after inserting one singleton after the first dimension.</span></div>
                <div class="usage-guide-example"><code>insert at start / middle / end</code><span>Use the position that makes the preview shape line up with the broadcasted form you need.</span></div>
              </div>
            </div>`,
        ]),
        slice: tensorViewStepGuideHtml('How do I use this?', [
            `<div class="usage-guide-step"><strong>What this step is for</strong><span>Hide one visible dimension from the layout and replace it with a slider-controlled fixed index. This changes the final bracketed slice term in the preview.</span></div>`,
            `<div class="usage-guide-step"><strong>How slicing behaves</strong><span>Click a dimension to toggle it into a slice. The visible layout loses that axis, and a slider appears below. Click the same dimension again to restore it as a visible axis. Multiple dimensions can be sliced at the same time.</span></div>`,
            `<div class="usage-guide-step"><strong>What cases matter</strong><span>Slicing a grouped axis fixes one flattened index, not each original sub-axis separately. If you need finer control, keep axes separate in Flatten Dims before slicing. Singleton slices are always fixed at <code>0</code>.</span></div>`,
            `<div class="usage-guide-columns">
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">Current state</div>
                <div class="usage-guide-example"><code>${currentPreview}</code><span>Current preview before toggling another slice.</span></div>
                <div class="usage-guide-example"><code>${sliceTarget?.label ?? 'A'}</code><span>Example dimension to slice next.</span></div>
              </div>
              <div class="usage-guide-column">
                <div class="usage-guide-subtitle">After slicing</div>
                <div class="usage-guide-example"><code>${sliceExample}</code><span>The chosen axis disappears from the visible layout and becomes slider-controlled.</span></div>
                <div class="usage-guide-example"><code>slice grouped axes only when one flattened index is enough</code><span>If you need separate control of the original component axes, undo the flatten first.</span></div>
              </div>
            </div>`,
        ]),
    };
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

function permuteDropSlotFor(clientX: number): number | null {
    const boxes = Array.from(tensorViewWidget.querySelectorAll<HTMLElement>('[data-permute-dim]'));
    if (boxes.length === 0) return null;
    for (let index = 0; index < boxes.length; index += 1) {
        const rect = boxes[index]!.getBoundingClientRect();
        if (clientX <= rect.left + rect.width / 2) return index;
    }
    return boxes.length;
}

function renderTensorViewWidget(snapshot: ViewerSnapshot): void {
    if (suspendTensorViewRender) return;
    const model = viewer.getInspectorModel();
    if (!model.handle) {
        tensorViewWidget.innerHTML = `${widgetTitle('tensor-view', 'Visualize tensor views, permutations, slices, or a combination of these ops.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }

    const error = viewErrors.get(model.handle.id);
    const editor = model.viewEditor;
    if (!editor) return;
    const tensorOptions = model.tensors.map((tensor) => `
      <option value="${tensor.id}" ${tensor.id === model.handle!.id ? 'selected' : ''}>${tensor.name || tensor.id}</option>
    `).join('');
    const sliceContent = model.viewTokens.map((token) => (
        token.kind === 'singleton'
            ? `<span class="dim-chip dim-chip-singleton">1</span>`
            : `<button class="dim-chip interactive-chip${token.sliced ? ' dim-chip-sliced dim-chip-active' : ''}" data-slice-token="${token.key}" type="button">${token.token}<span>=${token.size}</span></button>`
    )).join('');
    const defaultLabeledShape = model.handle.shape.map((size, index) => `${model.handle!.axisLabels[index] ?? `A${index}`}=${size}`).join(', ');
    tensorViewWidget.innerHTML = `
      ${widgetTitle('tensor-view', 'Visualize tensor views, permutations, slices, or a combination of these ops.')}
      <div class="widget-body">
        <div class="field">
          ${labelWithInfo('Tensor', 'Choose which loaded tensor the Permute/Slice editor controls.', 'tensor-select')}
          <select id="tensor-select">${tensorOptions}</select>
        </div>
        <div class="permute-slice-step">
          ${labelWithInfo('Tensor View', 'Edit the full tensor expression directly. Standard view, permute, and non-none indexing semantics apply.', 'tensor-view-input')}
          ${tensorViewHelpHtml(model.handle.shape, model.handle.axisLabels).replace('<details class="usage-guide">', `<details class="usage-guide"${tensorViewHelpOpen ? ' open' : ''}>`)}
          <textarea id="tensor-view-input" rows="3" placeholder="tensor">${model.preview}</textarea>
        </div>
        <div class="permute-slice-step">
          <div class="label-row"><span class="meta-label">Slice Dims</span>${infoButton('Convenience utility for inspecting different slices. Click a dimension to toggle between showing one/all elements at a time. If showing one element of a dimension, drag its slider to change the displayed index.')}</div>
          <div class="dim-chip-row dim-chip-row-compact" id="slice-dims">${sliceContent}</div>
        </div>
        ${error ? `<div class="error-box">${error}</div>` : ''}
        ${model.sliceTokens.length === 0 ? '' : '<div class="slider-list" id="slice-token-controls"></div>'}
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

    sliceHost?.replaceChildren(...model.sliceTokens.map((token) => {
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
    }));
}

function renderInspectorWidget(snapshot: ViewerSnapshot): void {
    const model = viewer.getInspectorModel();
    const dimensionMappingScheme = snapshot.dimensionMappingScheme ?? 'z-order';
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
    const displayGaps = snapshot.displayGaps ?? true;
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
    updateSidebar(snapshot);
    renderTabStrip();
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
    return {
        id: tab.id,
        title: tab.title,
        manifest: {
            version: 1,
            viewer: tab.viewer,
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
    sessionTabs = await Promise.all(manifest.tabs.map((tab) => loadSessionTab(tab)));
    activeTabId = sessionTabs[0]?.id ?? null;
    if (activeTabId) await loadTab(activeTabId);
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
        case 'save-svg':
            downloadSvg(svgFilename(), viewer.exportCurrentViewSvg());
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

window.addEventListener('keydown', async (event) => {
    const target = event.target as HTMLElement | null;
    const isEditing = Boolean(target && (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName) || target.isContentEditable));
    const isPaletteInput = target === commandPaletteInput;
    if (isPaletteInput && event.key === 'Escape') {
        event.preventDefault();
        closeCommandPalette();
        return;
    }
    if (isEditing) return;

    if (event.ctrlKey && event.key.toLowerCase() === 's') {
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
viewer.subscribeHover(() => renderInspectorWidget(viewer.getSnapshot()));
viewer.subscribeSelection(() => renderSelectionWidget(viewer.getSnapshot()));

tryLoadSession().then((loaded) => {
    if (!loaded) seedDemoTensor();
}).catch(() => seedDemoTensor());
}
