import {
    createTypedArray,
    defaultTensorViewEditor,
    parseTensorView,
    serializeTensorViewEditor,
    TensorViewer,
    expectedTensorByteLength,
    product,
    type BundleManifest,
    type DimensionMappingScheme,
    type LoadedBundleDocument,
    type NumericArray,
    type SessionBundleManifest,
    type TensorViewEditor,
    type ViewerSnapshot,
    VIEWER_LIMITS,
} from '@tensor-viz/viewer-core';
import {
    escapeInfo,
    formatAxisTokens,
    escapeHtml,
    formatAxisValues,
    formatNamedAxisValues,
    formatRangeValue,
    infoButton,
    labelWithInfo,
    selectionEnabled,
} from './app-format.js';
import type { CommandAction, DemoAppExtension, DemoExtensionContext, DemoWidgetSpec } from './app-extension.js';
import { getAppRoot, mountAppShell, renderWebglUnavailable, supportsWebGL } from './app-shell.js';
import { controlIcons, renderControlDockControls, type ControlSpec } from './control-dock.js';
import { createLinearLayoutExtension, type LinearLayoutExtensionRuntime } from './extensions/linear-layout/extension.js';
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
    widgets,
    tensorViewWidget,
    inspectorWidget,
    selectionWidget,
    advancedSettingsWidget,
    commandPalette,
    commandPaletteBackdrop,
    commandPaletteInput,
    commandPaletteList,
} = mountAppShell(app);

const viewer = new TensorViewer(viewport);
const infoTooltip = document.createElement('div');
infoTooltip.className = 'info-tooltip hidden';
app.appendChild(infoTooltip);
const controlTooltip = document.createElement('div');
controlTooltip.className = 'control-tooltip hidden';
app.appendChild(controlTooltip);
const sidebar = tensorViewWidget.parentElement as HTMLElement;
const sidebarScrollPad = document.createElement('div');
sidebarScrollPad.className = 'sidebar-scroll-pad';
const viewErrors = new Map<string, string>();
let suspendTensorViewRender = false;
let tensorViewHelpOpen = false;
let showTensorViewWidget = true;
let showAdvancedSettingsWidget = false;
let inspectorReady = false;
let sessionTabs: LoadedBundleDocument[] = [];
let activeTabId: string | null = null;
let editingTab: { id: string; title: string } | null = null;
let switchingTab = false;
let resizingSidebar = false;
let commandPaletteOpen = false;
let commandPaletteIndex = 0;
let commandPaletteMode: 'actions' | 'tabs' = 'actions';
let appliedStartupWidgetDefaults = false;
let activeTensorViewSliderPointerId: number | null = null;
let activeInfoTarget: HTMLElement | null = null;
let activeControlButton: HTMLButtonElement | null = null;

const MAX_SIDEBAR_WIDTH = 720;
const DATA_FILE_PATTERN = /^(?:tabs\/[a-z0-9_-]+\/)?tensors\/[a-z0-9_-]+\.bin$/i;
const TENSOR_CONTENT_TYPE = 'application/octet-stream';
const SESSION_MANIFEST_CONTENT_TYPE = 'application/json';
const SESSION_MANIFEST_MAX_BYTES = 8 * 1024 * 1024;
const SESSION_MAX_TENSORS = VIEWER_LIMITS.maxTensors;
const SESSION_MAX_TENSOR_BYTES = VIEWER_LIMITS.maxPayloadBytes;

function sessionApiToken(): string | null {
    return new URLSearchParams(window.location.search).get('token')
        ?? new URLSearchParams(window.location.hash.slice(1)).get('token');
}

const sessionToken = sessionApiToken();

type InspectorRefs = {
    hoveredTensor: HTMLDivElement;
    coordList: HTMLDivElement;
    hoveredTensorValue: HTMLSpanElement;
    tensorShapeValue: HTMLSpanElement;
    rankValue: HTMLSpanElement;
};

let inspectorRefs: InspectorRefs | null = null;
const extensionContext: DemoExtensionContext = {
    viewer,
    viewport,
    widgets,
    widgetTitle,
    getActiveTab: () => activeTab(),
    getActiveTabId: () => activeTabId,
    getSessionTabs: () => sessionTabs,
    setSessionTabs: (tabs) => {
        sessionTabs = tabs;
    },
    loadTab: async (tabId) => {
        await loadTab(tabId);
    },
    loadTabTensors: async (tensors) => loadTabTensors(tensors),
    render: () => {
        render(viewer.getSnapshot());
    },
};

type SidebarWidgetId = string;

const coreWidgetSpecs: DemoWidgetSpec[] = [
    {
        id: 'tensor-view',
        label: 'Permute/Slice',
        icon: '<span class="widget-title-text-icon widget-title-text-icon-wide">A<sup>T</sup>[i,:]</span>',
        defaultCollapsed: true,
        visible: () => showTensorViewWidget,
        render: (_ctx, snapshot) => { renderTensorViewWidget(snapshot); },
    },
    {
        id: 'inspector',
        label: 'Hover Info',
        icon: `
          <svg viewBox="0 0 24 24">
            <circle cx="11" cy="11" r="5.5" />
            <path d="M15 15l4 4" />
          </svg>
        `,
        defaultCollapsed: true,
        visible: (_ctx, snapshot) => snapshot.showInspectorPanel,
        render: (_ctx, snapshot) => { renderInspectorWidget(snapshot); },
    },
    {
        id: 'selection',
        label: 'Selection',
        icon: controlIcons.selection,
        defaultCollapsed: true,
        visible: (_ctx, snapshot) => (
            Boolean(snapshot.showSelectionPanel)
            && (snapshot.interactionMode ?? viewer.getInteractionMode()) === 'select'
        ),
        render: (_ctx, snapshot) => { renderSelectionWidget(snapshot); },
    },
    {
        id: 'advanced-settings',
        label: 'Advanced Settings',
        icon: `
          <svg viewBox="0 0 24 24">
            <path d="M5 6h14M8 12h11M5 18h14" />
            <circle cx="8" cy="6" r="1.7" fill="currentColor" stroke="none" />
            <circle cx="13" cy="12" r="1.7" fill="currentColor" stroke="none" />
            <circle cx="10" cy="18" r="1.7" fill="currentColor" stroke="none" />
          </svg>
        `,
        defaultCollapsed: true,
        visible: () => showAdvancedSettingsWidget,
        render: (_ctx, snapshot) => { renderAdvancedSettingsWidget(snapshot); },
    },
];

const linearLayoutExtension: LinearLayoutExtensionRuntime = createLinearLayoutExtension(extensionContext);
const extensions: DemoAppExtension[] = [linearLayoutExtension];
const widgetSpecs = [...extensions.flatMap((extension) => extension.widgets), ...coreWidgetSpecs];
const widgetSpecById = new Map(widgetSpecs.map((spec) => [spec.id, spec]));
const sidebarWidgets: Record<SidebarWidgetId, HTMLElement> = Object.fromEntries(widgetSpecs.map((spec) => [spec.id, widgets[spec.id]!]));
const sidebarWidgetLabels: Record<SidebarWidgetId, string> = Object.fromEntries(widgetSpecs.map((spec) => [spec.id, spec.label]));
const sidebarWidgetIcons: Record<SidebarWidgetId, string> = Object.fromEntries(widgetSpecs.map((spec) => [spec.id, spec.icon]));

let widgetOrder: SidebarWidgetId[] = widgetSpecs.map((spec) => spec.id);
sidebarHeader.classList.add('label-row');
sidebarHeader.innerHTML = `<span>Widgets</span>${infoButton('Extra settings to inspect/change the visible tensor(s). Click the arrows/widget header text on each widget to expand/collapse them. Change widget position by left-clicking + dragging on the grabber by the right of each widget.')}`;
let draggedWidgetId: SidebarWidgetId | null = null;
let draggedWidgetSlot: number | null = null;
let draggedWidgetPointerId: number | null = null;
const collapsedWidgets = new Set<SidebarWidgetId>(widgetSpecs.filter((spec) => spec.defaultCollapsed).map((spec) => spec.id));

function logUi(event: string, details?: unknown): void {
    if (details === undefined) console.log('[tensor-viz-ui]', event);
    else console.log('[tensor-viz-ui]', event, details);
}

function hideInfoTooltip(): void {
    activeInfoTarget = null;
    infoTooltip.classList.add('hidden');
}

function placeInfoTooltip(target: HTMLElement): void {
    const text = target.dataset.info?.trim();
    if (!text) {
        hideInfoTooltip();
        return;
    }
    activeInfoTarget = target;
    infoTooltip.textContent = text;
    infoTooltip.classList.remove('hidden');
    const margin = 12;
    const gap = 10;
    const targetRect = target.getBoundingClientRect();
    const tooltipRect = infoTooltip.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - tooltipRect.width - margin);
    const left = Math.min(maxLeft, Math.max(margin, targetRect.right - tooltipRect.width));
    const belowTop = targetRect.bottom + gap;
    const aboveTop = targetRect.top - gap - tooltipRect.height;
    const top = aboveTop >= margin || window.innerHeight - belowTop < tooltipRect.height
        ? Math.max(margin, aboveTop)
        : Math.min(window.innerHeight - tooltipRect.height - margin, belowTop);
    infoTooltip.style.left = `${left}px`;
    infoTooltip.style.top = `${top}px`;
}

function hideControlTooltip(): void {
    activeControlButton = null;
    controlTooltip.classList.add('hidden');
}

function placeControlTooltip(button: HTMLButtonElement): void {
    const label = button.dataset.tooltipLabel?.trim();
    const description = button.dataset.tooltipDescription?.trim();
    const shortcut = button.dataset.tooltipShortcut?.trim();
    if (!label || !description || !shortcut) {
        hideControlTooltip();
        return;
    }
    activeControlButton = button;
    controlTooltip.innerHTML = `<strong>${escapeInfo(label)}</strong><span>${escapeInfo(description)}</span><span class="control-tooltip-shortcut">Shortcut: ${escapeInfo(shortcut)}</span>`;
    controlTooltip.classList.remove('hidden');
    const margin = 12;
    const gap = 10;
    const buttonRect = button.getBoundingClientRect();
    const tooltipRect = controlTooltip.getBoundingClientRect();
    const centeredLeft = buttonRect.left + (buttonRect.width - tooltipRect.width) / 2;
    const maxLeft = Math.max(margin, window.innerWidth - tooltipRect.width - margin);
    const left = Math.min(maxLeft, Math.max(margin, centeredLeft));
    const aboveTop = buttonRect.top - gap - tooltipRect.height;
    const belowTop = buttonRect.bottom + gap;
    const top = aboveTop >= margin || window.innerHeight - belowTop < tooltipRect.height
        ? Math.max(margin, aboveTop)
        : Math.min(window.innerHeight - tooltipRect.height - margin, belowTop);
    controlTooltip.style.left = `${left}px`;
    controlTooltip.style.top = `${top}px`;
}

app.addEventListener('mouseover', (event) => {
    const target = (event.target as Element | null)?.closest<HTMLElement>('[data-info]');
    if (!target || target.classList.contains('control-button')) return;
    placeInfoTooltip(target);
});
app.addEventListener('mouseout', (event) => {
    if (!activeInfoTarget) return;
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && activeInfoTarget.contains(nextTarget)) return;
    hideInfoTooltip();
});
app.addEventListener('focusin', (event) => {
    const target = (event.target as Element | null)?.closest<HTMLElement>('[data-info]');
    if (!target || target.classList.contains('control-button')) return;
    placeInfoTooltip(target);
});
app.addEventListener('focusout', (event) => {
    if (!activeInfoTarget) return;
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && activeInfoTarget.contains(nextTarget)) return;
    hideInfoTooltip();
});
window.addEventListener('resize', () => {
    if (!activeInfoTarget?.isConnected) {
        hideInfoTooltip();
        return;
    }
    placeInfoTooltip(activeInfoTarget);
});
sidebar.addEventListener('scroll', () => {
    if (!activeInfoTarget?.isConnected) {
        hideInfoTooltip();
        return;
    }
    placeInfoTooltip(activeInfoTarget);
}, { passive: true });
controlDock.addEventListener('mouseover', (event) => {
    const button = (event.target as Element | null)?.closest<HTMLButtonElement>('.control-button');
    if (!button) return;
    placeControlTooltip(button);
});
controlDock.addEventListener('mouseout', (event) => {
    if (!activeControlButton) return;
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && activeControlButton.contains(nextTarget)) return;
    hideControlTooltip();
});
controlDock.addEventListener('focusin', (event) => {
    const button = (event.target as Element | null)?.closest<HTMLButtonElement>('.control-button');
    if (!button) return;
    placeControlTooltip(button);
});
controlDock.addEventListener('focusout', (event) => {
    if (!activeControlButton) return;
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && activeControlButton.contains(nextTarget)) return;
    hideControlTooltip();
});
window.addEventListener('resize', () => {
    if (!activeControlButton?.isConnected) {
        hideControlTooltip();
        return;
    }
    placeControlTooltip(activeControlButton);
});
controlDock.addEventListener('scroll', () => {
    if (!activeControlButton?.isConnected) {
        hideControlTooltip();
        return;
    }
    placeControlTooltip(activeControlButton);
}, { passive: true });

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
        { action: 'advanced-settings', label: 'Toggle Advanced Settings', shortcut: '', keywords: 'widgets advanced settings layout gap' },
        { action: 'view', label: 'Focus Permute/Slice Input', shortcut: '', keywords: 'focus permute slice tensor permutation slicing tensor view input field' },
        ...extensions.flatMap((extension) => extension.commands?.(extensionContext) ?? []),
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
    // widget visibility is derived from the active tab and viewer state instead
    // of unmounting widgets permanently, so drag order and collapsed state survive.
    return widgetOrder.filter((widgetId) => widgetSpecById.get(widgetId)?.visible(extensionContext, snapshot) ?? false);
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
    return sidebarWidgetIcons[widgetId] ?? '';
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
        const label = document.createElement('span');
        label.textContent = entry.label;
        const shortcut = document.createElement('span');
        shortcut.textContent = entry.shortcut;
        button.append(label, shortcut);
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
    const maxWidth = Math.max(0, app.clientWidth - sidebarSplitter.offsetWidth);
    const clamped = Math.max(0, Math.min(maxWidth, width));
    app.style.setProperty('--sidebar-width', `${clamped}px`);
    viewer.resize();
}

setSidebarWidth(MAX_SIDEBAR_WIDTH);

window.addEventListener('resize', () => {
    const currentWidth = Number.parseFloat(app.style.getPropertyValue('--sidebar-width')) || 0;
    setSidebarWidth(currentWidth);
});

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

function normalizedTensorViewSnapshot(
    tensor: BundleManifest['tensors'][number],
    ...views: Array<ViewerSnapshot['tensors'][number]['view'] | undefined>
): ViewerSnapshot['tensors'][number]['view'] {
    for (const view of [...views, { editor: defaultTensorViewEditor(tensor.shape, tensor.axisLabels), hiddenIndices: [] }]) {
        if (!view?.editor) continue;
        try {
            const parsed = parseTensorView(
                tensor.shape,
                serializeTensorViewEditor(view.editor),
                view.hiddenIndices,
                tensor.axisLabels,
            );
            if (parsed.ok) {
                return {
                    editor: parsed.spec.editor,
                    hiddenIndices: parsed.spec.hiddenIndices.slice(),
                };
            }
        } catch {
            continue;
        }
    }
    throw new Error('Tensor view editor state is invalid.');
}

function normalizeViewerSnapshot(tab: LoadedBundleDocument, snapshot: ViewerSnapshot): ViewerSnapshot {
    return {
        ...snapshot,
        tensors: snapshot.tensors.map((entry) => {
            const tensor = tab.manifest.tensors.find((candidate) => candidate.id === entry.id);
            if (!tensor) return entry;
            const previous = tab.manifest.viewer.tensors.find((candidate) => candidate.id === entry.id);
            return {
                ...entry,
                view: normalizedTensorViewSnapshot(tensor, entry.view, previous?.view, tensor.view),
            };
        }),
    };
}

function clearTabTitleEdit(): void {
    editingTab = null;
}

function captureActiveTabSnapshot(): void {
    const tab = activeTab();
    if (!tab) return;
    const snapshot = viewer.getSnapshot();
    extensions.forEach((extension) => {
        extension.captureSnapshot?.(extensionContext, tab, snapshot);
    });
    tab.manifest.viewer = normalizeViewerSnapshot(tab, snapshot);
}

async function closeTab(tabId: string): Promise<void> {
    const index = sessionTabs.findIndex((tab) => tab.id === tabId);
    if (index < 0) return;
    if (editingTab?.id === tabId) clearTabTitleEdit();
    const wasActive = activeTabId === tabId;
    if (wasActive) captureActiveTabSnapshot();
    extensions.forEach((extension) => {
        extension.clearTab?.(extensionContext, tabId);
    });
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
    tabStrip.classList.remove('hidden');
    if (sessionTabs.length === 0) {
        clearTabTitleEdit();
    }
    const tabs = sessionTabs.map((tab) => {
        const tabElement = document.createElement('div');
        const canSwitch = editingTab?.id !== tab.id && tab.id !== activeTabId;
        tabElement.className = `tab-button${tab.id === activeTabId ? ' active' : ''}`;
        tabElement.tabIndex = 0;
        tabElement.setAttribute('role', 'button');
        tabElement.setAttribute('aria-label', tab.title);
        tabElement.addEventListener('click', async () => {
            if (!canSwitch) return;
            await loadTab(tab.id);
        });
        tabElement.addEventListener('keydown', async (event) => {
            if (!['Enter', ' '].includes(event.key) || !canSwitch) return;
            event.preventDefault();
            await loadTab(tab.id);
        });
        tabElement.addEventListener('auxclick', async (event) => {
            if (event.button !== 1) return;
            event.preventDefault();
            await closeTab(tab.id);
        });

        const editing = editingTab?.id === tab.id;
        const label = editing ? document.createElement('input') : document.createElement('div');
        if (label instanceof HTMLInputElement) {
            label.type = 'text';
            label.className = 'tab-title-input';
            label.value = editingTab?.title ?? tab.title;
            label.setAttribute('aria-label', `Edit ${tab.title}`);
            let cancelled = false;
            queueMicrotask(() => {
                if (editingTab?.id !== tab.id) return;
                label.focus();
                label.select();
            });
            label.addEventListener('click', (event) => {
                event.stopPropagation();
            });
            label.addEventListener('input', () => {
                if (editingTab?.id === tab.id) editingTab.title = label.value;
            });
            label.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') label.blur();
                if (event.key !== 'Escape') return;
                event.preventDefault();
                cancelled = true;
                clearTabTitleEdit();
                renderTabStrip();
            });
            label.addEventListener('blur', () => {
                if (cancelled) return;
                const title = label.value.trim() || tab.title;
                tab.title = title;
                clearTabTitleEdit();
                renderTabStrip();
                logUi('tab:rename', { tabId: tab.id, title });
            });
        } else {
            label.className = 'tab-label';
            label.textContent = tab.title;
            label.addEventListener('dblclick', (event) => {
                event.preventDefault();
                event.stopPropagation();
                if (tab.id !== activeTabId) return;
                editingTab = { id: tab.id, title: tab.title };
                renderTabStrip();
            });
        }

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
        for (const extension of extensions) {
            const tab = await extension.createTab?.(extensionContext, id, title, viewer.getSnapshot());
            if (!tab) continue;
            sessionTabs = [tab];
            await loadTab(id);
            return;
        }
        seedDemoTensor();
        return;
    }
    captureActiveTabSnapshot();
    const nextTab = cloneTabDocument(currentTab, id, title);
    extensions.forEach((extension) => {
        extension.cloneTab?.(extensionContext, currentTab.id, id);
    });
    sessionTabs = [...sessionTabs, nextTab];
    await loadTab(id);
}

async function closeCurrentTab(): Promise<void> {
    if (activeTabId) await closeTab(activeTabId);
}

function renderControlDock(snapshot: ViewerSnapshot): void {
    const canSelect = selectionEnabled(snapshot);
    const canRotate = snapshot.displayMode === '3d';
    const interactionMode = snapshot.interactionMode ?? viewer.getInteractionMode();
    // app-entry decides control state because it owns viewer/tab context; the
    // control-dock module only renders this declarative list.
    const extensionControls = extensions.flatMap((extension) => extension.controls?.(extensionContext, snapshot) ?? []);
    const controls: ControlSpec[] = [
        {
            id: 'pan',
            label: 'Pan',
            description: 'Left click and drag to move the viewport without changing the tensor data.',
            shortcut: 'W',
            active: interactionMode === 'pan',
            content: controlIcons.pan,
            onClick: () => { viewer.setInteractionMode('pan'); },
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
            content: controlIcons.selection,
            onClick: () => { viewer.setInteractionMode('select'); },
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
            content: controlIcons.rotate,
            onClick: () => { viewer.setInteractionMode('rotate'); },
        },
        {
            id: '2d',
            label: '2D',
            description: 'Switch to the flat 2D viewer for grid inspection, panning, zooming, and box selection.',
            shortcut: 'Ctrl+2',
            active: snapshot.displayMode === '2d',
            content: '<span class="control-button-text" aria-hidden="true">2D</span>',
            onClick: () => { viewer.setDisplayMode('2d'); },
        },
        {
            id: '3d',
            label: '3D',
            description: 'Switch to the 3D viewer to orbit the scene and inspect stacked tensor layouts in depth.',
            shortcut: 'Ctrl+3',
            active: snapshot.displayMode === '3d',
            content: '<span class="control-button-text" aria-hidden="true">3D</span>',
            onClick: () => { viewer.setDisplayMode('3d'); },
        },
        ...extensionControls,
        {
            id: 'dim-lines',
            label: 'Dim Lines',
            description: 'Toggle dimension guide lines to show axis extents and family orientation in the current layout.',
            shortcut: 'Ctrl+D',
            active: snapshot.showDimensionLines,
            content: controlIcons.dimensionLines,
            onClick: () => { viewer.toggleDimensionLines(); },
        },
        {
            id: 'tensor-names',
            label: 'Tensor Names',
            description: 'Toggle tensor name labels above each rendered tensor in the current layout.',
            shortcut: 'Ctrl+N',
            active: snapshot.showTensorNames ?? true,
            content: controlIcons.tensorNames,
            onClick: () => { viewer.toggleTensorNames(); },
        },
        {
            id: 'gaps',
            label: 'Block Gaps',
            description: 'Toggle spacing inside higher-level tensor blocks so grouped dimensions appear either separated or packed.',
            shortcut: 'Ctrl+G',
            active: snapshot.displayGaps ?? false,
            content: controlIcons.gaps,
            onClick: () => { viewer.toggleDisplayGaps(); },
        },
        {
            id: 'mapping-contiguous',
            label: 'Contiguous Mapping',
            description: 'Use contiguous axis-family mapping so neighboring cells follow one continuous zig-zag traversal.',
            shortcut: 'Ctrl+M',
            active: snapshot.dimensionMappingScheme === 'contiguous',
            content: controlIcons.contiguousMapping,
            onClick: () => { viewer.setDimensionMappingScheme('contiguous'); },
        },
        {
            id: 'mapping-z-order',
            label: 'Z-Order Mapping',
            description: 'Use z-order axis-family mapping so traversal breaks into smaller interleaved zig-zag groups.',
            shortcut: 'Ctrl+M',
            active: snapshot.dimensionMappingScheme === 'z-order',
            content: controlIcons.zOrderMapping,
            onClick: () => { viewer.setDimensionMappingScheme('z-order'); },
        },
    ];
    renderControlDockControls(controlDock, controls);
}

async function loadTab(tabId: string): Promise<void> {
    const tab = sessionTabs.find((entry) => entry.id === tabId);
    if (!tab) return;
    if (!switchingTab && activeTabId && activeTabId !== tabId) captureActiveTabSnapshot();
    switchingTab = true;
    activeTabId = tabId;
    try {
        // normalize before loading so old or partial snapshots cannot hand the
        // viewer tensor-view state that no longer matches the manifest tensors.
        tab.manifest.viewer = normalizeViewerSnapshot(tab, tab.manifest.viewer);
        viewer.loadBundleData(tab.manifest, tab.tensors);
        if (tab.manifest.viewer.dimensionMappingScheme) {
            viewer.setDimensionMappingScheme(tab.manifest.viewer.dimensionMappingScheme);
        }
        if (!appliedStartupWidgetDefaults) {
            viewer.toggleSelectionPanel(false);
            appliedStartupWidgetDefaults = true;
        }
    } finally {
        switchingTab = false;
    }
    renderTabStrip();
    // extensions hydrate tab-local state after the viewer has accepted the
    // manifest, otherwise widget state can describe tensors that were rejected.
    extensions.forEach((extension) => {
        extension.afterLoadTab?.(extensionContext, tab);
    });
}

window.addEventListener('pointerup', () => {
    resizingSidebar = false;
    if (!suspendTensorViewRender || activeTensorViewSliderPointerId !== null) return;
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

function captureSidebarAnchor(element: HTMLElement | null, selector: string): { selector: string; top: number } | null {
    if (!element) return null;
    return { selector, top: element.getBoundingClientRect().top };
}

function renderPreservingSidebarScroll(anchor: { selector: string; top: number } | null = null): void {
    const previousScrollTop = sidebar.scrollTop;
    render(viewer.getSnapshot());
    sidebar.scrollTop = previousScrollTop;
    requestAnimationFrame(() => {
        if (!anchor) {
            sidebar.scrollTop = previousScrollTop;
            return;
        }
        const nextAnchor = sidebar.querySelector<HTMLElement>(anchor.selector);
        if (!nextAnchor) return;
        sidebar.scrollTop += nextAnchor.getBoundingClientRect().top - anchor.top;
    });
}

/** Keep compact textareas at their content height so widget fields stay visually aligned. */
function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

function beginTensorViewSliderDrag(slider: HTMLInputElement, pointerId: number): void {
    suspendTensorViewRender = true;
    activeTensorViewSliderPointerId = pointerId;
    slider.setPointerCapture(pointerId);
}

function endTensorViewSliderDrag(slider: HTMLInputElement, pointerId: number): void {
    if (activeTensorViewSliderPointerId !== pointerId) return;
    activeTensorViewSliderPointerId = null;
    suspendTensorViewRender = false;
    if (slider.hasPointerCapture(pointerId)) slider.releasePointerCapture(pointerId);
    renderPreservingSidebarScroll(captureSidebarAnchor(slider, `#${CSS.escape(slider.id)}`));
}

function applyTensorViewEditor(
    tensorId: string,
    editor: TensorViewEditor,
    anchor: { selector: string; top: number } | null = null,
): void {
    try {
        viewer.setTensorView(tensorId, serializeTensorViewEditor(editor));
        linearLayoutExtension.syncViewFilters();
        viewErrors.delete(tensorId);
    } catch (error) {
        viewErrors.set(tensorId, error instanceof Error ? error.message : String(error));
    }
    renderPreservingSidebarScroll(anchor);
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
    const shapeText = escapeHtml(shape.join(', '));
    const reversedRangeText = escapeHtml(shape.map((_dim, index) => shape.length - index - 1).join(', '));
    const labeledShapeText = escapeHtml(shape.map((size, index) => `${axisLabels[index] ?? `A${index}`}=${size}`).join(', '));
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
    const linearLayoutMeta = linearLayoutExtension.metaForTab(tab);
    if (!model.handle) {
        tensorViewWidget.innerHTML = `${widgetTitle('tensor-view', 'Visualize tensor views, permutations, slices, or a combination of these ops.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }

    const error = viewErrors.get(model.handle.id);
    const editor = model.viewEditor;
    if (!editor) return;
    const selectionMap = tab && linearLayoutExtension.isTab(tab) ? linearLayoutExtension.selectionMapForTab(tab) : null;
    const multiInput = selectionMap ? linearLayoutExtension.multiInputModel(selectionMap) : null;
    const tensorOptions = model.tensors.map((tensor) => `
      <option value="${escapeHtml(tensor.id)}" ${tensor.id === model.handle!.id ? 'selected' : ''}>${escapeHtml(tensor.name || tensor.id)}</option>
    `).join('');
    const sliceContent = model.viewTokens.map((token) => (
        token.kind === 'singleton'
            ? `<span class="dim-chip dim-chip-singleton">1</span>`
            : `<button class="dim-chip interactive-chip${token.sliced ? ' dim-chip-sliced dim-chip-active' : ''}" data-slice-token="${escapeHtml(token.key)}" type="button">${escapeHtml(token.token)}<span>=${token.size}</span></button>`
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
          <textarea id="tensor-view-input" class="compact-textarea" rows="1" placeholder="tensor" spellcheck="false">${escapeHtml(model.preview)}</textarea>
        </div>
        <div class="permute-slice-step">
          <div class="label-row"><span class="meta-label">Slice Dims</span>${infoButton('Convenience utility for inspecting different slices. Click a dimension to toggle between showing one/all elements at a time. If showing one element of a dimension, drag its slider to change the displayed index.')}</div>
          <div class="dim-chip-row dim-chip-row-compact" id="slice-dims">${sliceContent}</div>
        </div>
        ${error ? `<div class="error-box">${escapeHtml(error)}</div>` : ''}
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
    if (tensorViewInput) autosizeTextarea(tensorViewInput);
    tensorViewInput?.addEventListener('input', () => {
        autosizeTextarea(tensorViewInput);
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
            const anchor = captureSidebarAnchor(button, `[data-slice-token="${CSS.escape(key)}"]`);
            const sliced = editor.slicedTokenKeys.includes(key);
            applyTensorViewEditor(model.handle!.id, {
                ...editor,
                slicedTokenKeys: sliced ? editor.slicedTokenKeys.filter((entry) => entry !== key) : [...editor.slicedTokenKeys, key],
                sliceValues: sliced ? editor.sliceValues : { ...editor.sliceValues, [key]: editor.sliceValues[key] ?? 0 },
            }, anchor);
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
          <label for="${sliderId}">${escapeHtml(token.token)}</label>
          <input id="${sliderId}" type="range" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
          <input id="${sliderId}-number" type="number" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
        `;
        const slider = row.querySelector<HTMLInputElement>(`#${sliderId}`);
        const number = row.querySelector<HTMLInputElement>(`#${sliderId}-number`);
        const syncTensorViewInput = (): void => {
            const tensorViewInput = tensorViewWidget.querySelector<HTMLTextAreaElement>('#tensor-view-input');
            if (!tensorViewInput) return;
            tensorViewInput.value = viewer.getInspectorModel().preview;
            autosizeTextarea(tensorViewInput);
        };
        const applyValue = (nextValue: number): void => {
            logUi('slice-token:update', { tensorId: model.handle!.id, token: token.token, value: nextValue });
            viewer.setSliceTokenValue(model.handle!.id, token.key, nextValue);
            linearLayoutExtension.syncViewFilters();
            // updating the whole widget during slider drag resets the active range input,
            // so keep the drag stable and only sync the expression text in place
            syncTensorViewInput();
            requestAnimationFrame(syncTensorViewInput);
        };
        slider?.addEventListener('pointerdown', (event) => {
            beginTensorViewSliderDrag(slider, event.pointerId);
        });
        slider?.addEventListener('input', () => {
            if (!number || !slider) return;
            number.value = slider.value;
            applyValue(Number(slider.value));
        });
        slider?.addEventListener('pointerup', (event) => {
            endTensorViewSliderDrag(slider, event.pointerId);
        });
        slider?.addEventListener('pointercancel', (event) => {
            endTensorViewSliderDrag(slider, event.pointerId);
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
            linearLayoutExtension.setMultiInputValue(multiInput.focusedTensorId, nextValue);
        };
        slider?.addEventListener('pointerdown', (event) => {
            beginTensorViewSliderDrag(slider, event.pointerId);
        });
        slider?.addEventListener('input', () => {
            if (!number || !slider) return;
            number.value = slider.value;
            applyValue(Number(slider.value));
        });
        slider?.addEventListener('pointerup', (event) => {
            endTensorViewSliderDrag(slider, event.pointerId);
        });
        slider?.addEventListener('pointercancel', (event) => {
            endTensorViewSliderDrag(slider, event.pointerId);
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
    const linearLayoutTab = tab && linearLayoutExtension.isTab(tab) ? tab : null;
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
    const linearLayout = linearLayoutTab ? linearLayoutExtension.selectionMapForTab(linearLayoutTab) : null;
    const coordEntries = linearLayoutExtension.inspectorCoordEntries(linearLayoutExtension.ui, hover, hoveredStatus, linearLayout);
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
    if (suspendTensorViewRender) {
        renderInspectorWidget(snapshot);
        extensions.forEach((extension) => {
            extension.hover?.(extensionContext, snapshot);
        });
        return;
    }
    if (!switchingTab) captureActiveTabSnapshot();
    for (const extension of extensions) {
        if (extension.beforeRender?.(extensionContext, snapshot)) {
            return;
        }
    }
    updateSidebar(snapshot);
    renderTabStrip();
    renderControlDock(snapshot);
    widgetSpecs.forEach((spec) => {
        spec.render(extensionContext, snapshot);
    });
    extensions.forEach((extension) => {
        extension.afterRender?.(extensionContext, snapshot);
    });
}

function safeDataFile(dataFile: string): string {
    if (!DATA_FILE_PATTERN.test(dataFile) || dataFile.includes('..')) {
        throw new Error(`Unsafe tensor payload path ${dataFile}.`);
    }
    return dataFile;
}

function apiUrl(path: string): string {
    const url = new URL(path, window.location.href);
    if (sessionToken) url.searchParams.set('token', sessionToken);
    return `${url.pathname}${url.search}`;
}

async function boundedArrayBuffer(
    response: Response,
    options: {
        contentType: string;
        label: string;
        maxBytes: number;
        expectedBytes?: number;
    },
): Promise<ArrayBuffer> {
    const contentType = response.headers.get('content-type')?.split(';')[0]?.trim().toLowerCase();
    if (contentType !== options.contentType) throw new Error(`Unexpected ${options.label} content type ${contentType ?? 'unknown'}.`);
    const contentLength = response.headers.get('content-length');
    const limit = options.expectedBytes ?? options.maxBytes;
    if (contentLength !== null) {
        const declaredLength = Number(contentLength);
        if (options.expectedBytes !== undefined && declaredLength !== options.expectedBytes) {
            throw new Error(`${options.label} byte length ${contentLength} does not match expected ${options.expectedBytes}.`);
        }
        if (!Number.isFinite(declaredLength) || declaredLength < 0 || declaredLength > limit) {
            throw new Error(`${options.label} byte length ${contentLength} exceeds ${limit}.`);
        }
    }
    if (!response.body) {
        const buffer = await response.arrayBuffer();
        if (options.expectedBytes !== undefined && buffer.byteLength !== options.expectedBytes) {
            throw new Error(`${options.label} byte length ${buffer.byteLength} does not match expected ${options.expectedBytes}.`);
        }
        if (buffer.byteLength > limit) {
            throw new Error(`${options.label} exceeded ${limit} bytes.`);
        }
        return buffer;
    }
    const reader = response.body.getReader();
    const chunks: Uint8Array[] = [];
    let received = 0;
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        received += value.byteLength;
        if (received > limit) throw new Error(`${options.label} exceeded ${limit} bytes.`);
        chunks.push(value);
    }
    if (options.expectedBytes !== undefined && received !== options.expectedBytes) {
        throw new Error(`${options.label} byte length ${received} does not match expected ${options.expectedBytes}.`);
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    chunks.forEach((chunk) => {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
    });
    return bytes.buffer;
}

/** loads one tab's raw tensor payloads from the local python session server. */
async function loadTabTensors(tensors: BundleManifest['tensors']): Promise<Map<string, NumericArray>> {
    const entries: Array<readonly [string, NumericArray]> = [];
    for (const tensor of tensors.filter((entry) => entry.dataFile)) {
        const dataFile = safeDataFile(tensor.dataFile ?? '');
        const expectedBytes = expectedTensorByteLength(tensor.dtype, tensor.shape);
        const response = await fetch(apiUrl(`/api/${dataFile}`), { cache: 'no-store' });
        if (!response.ok) throw new Error(`Missing tensor payload ${dataFile}.`);
        entries.push([tensor.id, createTypedArray(tensor.dtype, await boundedArrayBuffer(response, {
            contentType: TENSOR_CONTENT_TYPE,
            expectedBytes,
            label: 'Tensor payload',
            maxBytes: expectedBytes,
        }))]);
    }
    return new Map(entries);
}

/** loads one session tab from the raw manifest plus tensor-byte endpoints. */
async function loadSessionTab(tab: SessionBundleManifest['tabs'][number]): Promise<LoadedBundleDocument> {
    for (const extension of extensions) {
        const loaded = await extension.loadSessionTab?.(extensionContext, tab);
        if (loaded) return loaded;
    }
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
    const response = await fetch(apiUrl('/api/session.json'), { cache: 'no-store' });
    if (!response.ok) return false;
    const manifest = JSON.parse(new TextDecoder().decode(await boundedArrayBuffer(response, {
        contentType: SESSION_MANIFEST_CONTENT_TYPE,
        label: 'Session manifest',
        maxBytes: SESSION_MANIFEST_MAX_BYTES,
    }))) as SessionBundleManifest;
    if (manifest.version !== 1) throw new Error(`Unsupported session version ${manifest.version}.`);
    if (!Array.isArray(manifest.tabs)) throw new Error('Session tabs must be an array.');
    if (manifest.tabs.length > VIEWER_LIMITS.maxTabs) throw new Error('Session has too many tabs.');
    const initialMapping = manifest.tabs[0]?.viewer.dimensionMappingScheme;
    if (initialMapping) viewer.setDimensionMappingScheme(initialMapping);
    extensions.forEach((extension) => {
        extension.beforeSessionLoad?.(extensionContext);
    });
    let totalTensors = 0;
    let totalTensorBytes = 0;
    for (const tab of manifest.tabs) {
        if (!Array.isArray(tab.tensors)) throw new Error('Session tab tensors must be an array.');
        totalTensors += tab.tensors.length;
        if (tab.tensors.length > VIEWER_LIMITS.maxTensors || totalTensors > SESSION_MAX_TENSORS) {
            throw new Error('Session has too many tensors.');
        }
        for (const tensor of tab.tensors.filter((entry) => entry.dataFile)) {
            safeDataFile(tensor.dataFile ?? '');
            totalTensorBytes += expectedTensorByteLength(tensor.dtype, tensor.shape);
            if (totalTensorBytes > SESSION_MAX_TENSOR_BYTES) {
                throw new Error('Session tensor payloads are too large.');
            }
        }
    }
    const loadedTabs: LoadedBundleDocument[] = [];
    for (const tab of manifest.tabs) {
        loadedTabs.push(await loadSessionTab(tab));
    }
    sessionTabs = loadedTabs;
    activeTabId = null;
    const initialTabId = sessionTabs[0]?.id ?? null;
    if (initialTabId) await loadTab(initialTabId);
    return true;
}

function seedDemoTensor(): void {
    sessionTabs.forEach((tab) => {
        extensions.forEach((extension) => {
            extension.clearTab?.(extensionContext, tab.id);
        });
    });
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

async function currentSvgDocument(): Promise<string> {
    if (viewer.getSnapshot().displayMode !== '2d') return viewer.exportCurrentViewSvg();
    return viewer.saveSvg().text();
}

async function loadFallbackTabs(): Promise<boolean> {
    for (const extension of extensions) {
        if (await extension.loadFallback?.(extensionContext)) return true;
    }
    return false;
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
    if (isPaletteInput && !event.ctrlKey && !event.metaKey && !event.altKey) return;
    if (isEditing && !isPaletteInput && !(event.ctrlKey && event.key.toLowerCase() === 's')) return;

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
viewport.addEventListener('pointermove', (event) => {
    extensions.forEach((extension) => {
        extension.pointerMove?.(extensionContext, event);
    });
});
viewport.addEventListener('pointerleave', () => {
    extensions.forEach((extension) => {
        extension.pointerLeave?.(extensionContext);
    });
});
viewer.subscribeHover(() => {
    const snapshot = viewer.getSnapshot();
    renderInspectorWidget(snapshot);
    extensions.forEach((extension) => {
        extension.hover?.(extensionContext, snapshot);
    });
});
viewer.subscribeSelectionPreview((selection) => {
    extensions.forEach((extension) => {
        extension.selectionPreview?.(extensionContext, selection);
    });
});
viewer.subscribeSelection((selection) => {
    renderSelectionWidget(viewer.getSnapshot());
    extensions.forEach((extension) => {
        extension.selection?.(extensionContext, selection);
    });
});
widgetSpecs.forEach((spec) => {
    spec.render(extensionContext, viewer.getSnapshot());
});

tryLoadSession().then(async (loaded) => {
    if (loaded) return;
    if (await loadFallbackTabs()) return;
    seedDemoTensor();
}).catch(async () => {
    if (await loadFallbackTabs()) return;
    seedDemoTensor();
});
}
