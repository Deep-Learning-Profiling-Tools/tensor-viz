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
import type { CommandAction, DemoAppExtension, DemoExtensionContext, DemoExtensionFactory, DemoTensorViewContribution, DemoWidgetSpec } from './app-extension.js';
import { getAppRoot, mountAppShell, renderWebglUnavailable, supportsWebGL, type AppShellWidgetSlot } from './app-shell.js';
import { controlIcons, renderControlDockControls, type ControlSpec } from './control-dock.js';
import './styles.css';

// this file owns the generic demo shell: tabs, widgets, command routing, and
// session loading. feature-specific behavior should enter through DemoAppExtension
// hooks so adding a preset family or widget does not require new shell branches.
/**
 * Runtime options for mounting the generic tensor-viz demo shell.
 *
 * @param root - Optional existing application root. When omitted, the shell uses the standard `#app` lookup and creates it if needed.
 * @param extensionFactories - Extension factories supplied by the host package. LL-viz passes its linear-layout factory here while standalone tensor-viz starts with an empty extension list.
 * @example
 * startDemoApp({
 *   root: document.querySelector<HTMLElement>('#app')!,
 *   extensionFactories: [linearLayoutExtensionFactory],
 * });
 */
export type DemoAppRuntimeOptions = {
    root?: HTMLDivElement;
    extensionFactories?: readonly DemoExtensionFactory[];
};

/**
 * Mount the generic tensor-viz demo shell into the document and wire any host-supplied extensions.
 *
 * @param options - Optional root element and extension factories supplied by the embedding package.
 * @returns Nothing. The call mutates the chosen DOM root by inserting the viewer shell and starts async session/fallback loading.
 * @noThrows Startup itself handles WebGL fallback rendering and catches asynchronous session-loading failures by seeding demo tensors; synchronous DOM failures still indicate an invalid host document.
 * @example
 * startDemoApp();
 * // The standalone tensor-viz app starts with no workflow-specific extension widgets.
 *
 * @example
 * startDemoApp({ extensionFactories: [linearLayoutExtensionFactory] });
 * // LL-viz receives the same shell plus its linear-layout widgets, controls, and tab hooks.
 */
export function startDemoApp(options: DemoAppRuntimeOptions = {}): void {
const app = options.root ?? getAppRoot();
const extensionFactories = [...(options.extensionFactories ?? [])];

if (!supportsWebGL()) {
    renderWebglUnavailable(app);
} else {
// startup and shell wiring
const CORE_WIDGET_SLOTS = [
    { id: 'tensor-view' },
    { id: 'inspector' },
    { id: 'selection' },
    { id: 'advanced-settings' },
] satisfies AppShellWidgetSlot[];
const EXTENSION_WIDGET_SLOTS = extensionFactories.flatMap((factory) => factory.widgetSlots);
const {
    viewport,
    tabStrip,
    controlDock,
    sidebarSplitter,
    sidebarHeader,
    widgets,
    commandPalette,
    commandPaletteBackdrop,
    commandPaletteInput,
    commandPaletteList,
} = mountAppShell(app, [...EXTENSION_WIDGET_SLOTS, ...CORE_WIDGET_SLOTS]);

const viewer = new TensorViewer(viewport);
const infoTooltip = document.createElement('div');
infoTooltip.className = 'info-tooltip hidden';
app.appendChild(infoTooltip);
const controlTooltip = document.createElement('div');
controlTooltip.className = 'control-tooltip hidden';
app.appendChild(controlTooltip);
const tensorViewWidget = widgets['tensor-view']!;
const inspectorWidget = widgets.inspector!;
const selectionWidget = widgets.selection!;
const advancedSettingsWidget = widgets['advanced-settings']!;
const sidebar = tensorViewWidget.parentElement as HTMLElement;
const sidebarScrollPad = document.createElement('div');
sidebarScrollPad.className = 'sidebar-scroll-pad';
const viewErrors = new Map<string, string>();
// slider drags update viewer state on every pointer move. full widget re-render
// during that stream steals focus/capture from the range input, so render is
// narrowed until pointerup; the e2e smoke test covers the visible widget path.
let suspendTensorViewRender = false;
let tensorViewHelpOpen = false;
let showTensorViewWidget = true;
let showAdvancedSettingsWidget = false;
let inspectorReady = false;
// sessionTabs is the in-memory source for tab buttons and extension metadata.
// active tabs are snapshotted before switching so unsaved slices/widget state do
// not disappear when a user clicks through several layouts.
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
// python sessions serve only manifest-declared binary blobs. keeping the path
// grammar tiny prevents apiUrl(`/api/${dataFile}`) from becoming arbitrary fetch.
const DATA_FILE_PATTERN = /^(?:tabs\/[a-z0-9_-]+\/)?tensors\/[a-z0-9_-]+\.bin$/i;
const TENSOR_CONTENT_TYPE = 'application/octet-stream';
const SESSION_MANIFEST_CONTENT_TYPE = 'application/json';
const SESSION_MANIFEST_MAX_BYTES = 8 * 1024 * 1024;
const SESSION_MAX_TENSORS = VIEWER_LIMITS.maxTensors;
const SESSION_MAX_TENSOR_BYTES = VIEWER_LIMITS.maxPayloadBytes;

/**
 * Reads the optional demo session API token from the browser URL, preferring the query string over
 * hash parameters so hosted links can pass `?token=...` or `#token=...`.
 *
 * @returns The token string from `window.location.search` or `window.location.hash`, or `null` when neither location contains a `token` parameter.
 * @noThrows URLSearchParams is constructed from the browser-provided search and hash strings and the function performs no network or storage access.
 * @example
 * history.replaceState(null, '', '/demo?token=abc123');
 * expect(sessionApiToken()).toBe('abc123');
 *
 * history.replaceState(null, '', '/demo#token=from-hash');
 * expect(sessionApiToken()).toBe('from-hash');
 *
 * history.replaceState(null, '', '/demo');
 * expect(sessionApiToken()).toBeNull();
 */
function sessionApiToken(): string | null {
    return new URLSearchParams(window.location.search).get('token')
        ?? new URLSearchParams(window.location.hash.slice(1)).get('token');
}

const sessionToken = sessionApiToken();

// extension host services
/**
 * DOM element handles for the sidebar hover inspector: tensor name and coordinate rows are written to
 * divs, while scalar metadata such as hovered value, tensor shape, and rank are written to spans.
 *
 * @example
 * const refs: InspectorRefs = {
 *   hoveredTensor: document.createElement('div'),
 *   coordList: document.createElement('div'),
 *   hoveredTensorValue: document.createElement('span'),
 *   tensorShapeValue: document.createElement('span'),
 *   rankValue: document.createElement('span'),
 * };
 * refs.hoveredTensorValue.textContent = '0.75';
 * expect(refs.hoveredTensorValue.textContent).toBe('0.75');
 */
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

/**
 * String key assigned by a registered sidebar widget spec and reused to look up that widget's DOM node, label, icon, order, collapse state, and drag state.
 *
 * @example
 * const widgetId: SidebarWidgetId = 'advanced-settings';
 * sidebarWidgets[widgetId].classList.toggle('collapsed', collapsedWidgets.has(widgetId));
 */
type SidebarWidgetId = string;

// core widget registry
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

const extensions: DemoAppExtension[] = extensionFactories.map((factory) => factory.create(extensionContext));
const widgetSpecs = [...extensions.flatMap((extension) => extension.widgets), ...coreWidgetSpecs];
const widgetSpecById = new Map(widgetSpecs.map((spec) => [spec.id, spec]));
// widgets are looked up once from shell slots, then driven by DemoWidgetSpec.
// missing slots are a registration bug and should fail during startup.
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

// tooltip plumbing
/**
 * Writes a tensor-viz demo UI telemetry line to the developer console with a stable `[tensor-viz-ui]` prefix.
 *
 * @param event - Colon-delimited UI action name such as `tab:rename`, `tensor-select`, or `advanced-settings:log-scale`.
 * @param details - Optional payload that describes the selected value or changed entity for the event; omitted events log only the prefix and event name.
 * @returns Nothing; callers observe the emitted `console.log` call.
 * @noThrows The function performs no validation and only forwards its arguments to `console.log`, so the demo code does not create an application-level error path.
 * @example
 * const calls: unknown[][] = [];
 * const originalLog = console.log;
 * console.log = (...args: unknown[]) => calls.push(args);
 *
 * logUi('tab:rename', { tabId: 'tab-1', title: 'Attention scores' });
 * logUi('tensor-select');
 *
 * console.log = originalLog;
 * calls;
 * // => [
 * //   ['[tensor-viz-ui]', 'tab:rename', { tabId: 'tab-1', title: 'Attention scores' }],
 * //   ['[tensor-viz-ui]', 'tensor-select'],
 * // ]
 */
function logUi(event: string, details?: unknown): void {
    if (details === undefined) console.log('[tensor-viz-ui]', event);
    else console.log('[tensor-viz-ui]', event, details);
}

/**
 * Clears the active info target and marks the shared info tooltip element as hidden.
 *
 * @returns Nothing; callers observe `activeInfoTarget` becoming `null` and `infoTooltip` receiving the `hidden` class.
 * @noThrows During normal app startup `infoTooltip` is a resolved shell element, and the function only assigns module state and adds a CSS class.
 * @example
 * activeInfoTarget = document.createElement('button');
 * infoTooltip.classList.remove('hidden');
 *
 * hideInfoTooltip();
 *
 * activeInfoTarget;
 * // => null
 * infoTooltip.classList.contains('hidden');
 * // => true
 */
function hideInfoTooltip(): void {
    activeInfoTarget = null;
    infoTooltip.classList.add('hidden');
}

/**
 * Shows the shared info tooltip for an element with `data-info`, copies the trimmed help text, and positions the tooltip inside the viewport near that element.
 *
 * @param target - Hovered or focused control/label element whose `data-info` attribute contains the tooltip copy and whose bounding box anchors the tooltip position.
 * @returns Nothing; callers observe `activeInfoTarget`, `infoTooltip.textContent`, visibility classes, and `left`/`top` styles being updated, or the tooltip being hidden when `data-info` is empty.
 * @noThrows The function treats missing or whitespace-only `data-info` as a hide request and otherwise uses standard DOM geometry/style APIs on the supplied element.
 * @example
 * const target = document.createElement('button');
 * target.dataset.info = 'Show tensor metadata';
 * target.getBoundingClientRect = () => ({ left: 20, top: 40, right: 120, bottom: 64, width: 100, height: 24, x: 20, y: 40, toJSON: () => ({}) });
 * infoTooltip.getBoundingClientRect = () => ({ left: 0, top: 0, right: 80, bottom: 30, width: 80, height: 30, x: 0, y: 0, toJSON: () => ({}) });
 *
 * placeInfoTooltip(target);
 *
 * activeInfoTarget === target;
 * // => true
 * infoTooltip.textContent;
 * // => 'Show tensor metadata'
 * infoTooltip.classList.contains('hidden');
 * // => false
 * infoTooltip.style.left;
 * // => '40px'
 * infoTooltip.style.top;
 * // => '74px'
 *
 * target.dataset.info = '   ';
 * placeInfoTooltip(target);
 * infoTooltip.classList.contains('hidden');
 * // => true
 */
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

/**
 * Clears the active control-button anchor and hides the shared control tooltip used by hover, focus, resize, and scroll handlers.
 *
 * @returns Void; after the call `activeControlButton` is `null` and `controlTooltip` has the `hidden` class.
 * @noThrows Only resets a module-local reference and adds a CSS class to the already-created tooltip element; those operations have no expected failure path in the mounted demo shell.
 * @example
 * activeControlButton = zoomInButton;
 * controlTooltip.classList.remove('hidden');
 *
 * hideControlTooltip();
 *
 * console.assert(activeControlButton === null);
 * console.assert(controlTooltip.classList.contains('hidden'));
 */
function hideControlTooltip(): void {
    activeControlButton = null;
    controlTooltip.classList.add('hidden');
}

/**
 * Shows the shared control tooltip for a dock button by reading its tooltip label, description, and shortcut metadata, then clamps the tooltip position inside the viewport.
 *
 * @param button - Control-dock button whose `data-tooltip-label`, `data-tooltip-description`, and `data-tooltip-shortcut` attributes supply the tooltip content.
 * @returns Void; populated buttons make `controlTooltip` visible, update its HTML, record `activeControlButton`, and write pixel `left`/`top` styles. Buttons missing any tooltip metadata hide the tooltip instead.
 * @noThrows Uses optional dataset reads, string escaping, class changes, and element geometry from the provided button and shared tooltip element; normal connected `HTMLButtonElement` controls do not create an expected exception path.
 * @example
 * const button = document.createElement('button');
 * button.dataset.tooltipLabel = 'Zoom in';
 * button.dataset.tooltipDescription = 'Increase the tensor canvas scale.';
 * button.dataset.tooltipShortcut = '+';
 * button.getBoundingClientRect = () => ({ left: 100, top: 80, right: 124, bottom: 104, width: 24, height: 24, x: 100, y: 80, toJSON: () => ({}) });
 * controlTooltip.getBoundingClientRect = () => ({ left: 0, top: 0, right: 160, bottom: 48, width: 160, height: 48, x: 0, y: 0, toJSON: () => ({}) });
 *
 * placeControlTooltip(button);
 *
 * console.assert(activeControlButton === button);
 * console.assert(!controlTooltip.classList.contains('hidden'));
 * console.assert(controlTooltip.textContent?.includes('Zoom in'));
 * console.assert(controlTooltip.textContent?.includes('Shortcut: +'));
 * console.assert(controlTooltip.style.left.endsWith('px'));
 * console.assert(controlTooltip.style.top.endsWith('px'));
 *
 * const incompleteButton = document.createElement('button');
 * placeControlTooltip(incompleteButton);
 * console.assert(controlTooltip.classList.contains('hidden'));
 */
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

// command palette
/**
 * Formats the Selection widget's highlighted-cell count from the viewer selection summary.
 *
 * @param summary - Selection summary returned by `TensorViewer.getSelectionSummary()`, including total highlighted cells and how many of those cells have loaded numeric values.
 * @param enabled - Whether selection statistics are available for the current tensor mapping; false means the widget should display an unavailable state.
 * @returns The text shown in the Highlighted Cells row: `Unavailable` when disabled, `0` for an empty selection, the total count when every selected cell has a loaded value, or `N (M with values)` when some selected cells lack values.
 * @noThrows Reads numeric fields from a viewer-produced summary and performs deterministic string formatting without parsing or DOM access.
 * @example
 * const summary = { count: 8, availableCount: 5, stats: null } as ReturnType<TensorViewer['getSelectionSummary']>;
 *
 * console.assert(selectionCountValue(summary, true) === '8 (5 with values)');
 * console.assert(selectionCountValue({ ...summary, count: 5, availableCount: 5 }, true) === '5');
 * console.assert(selectionCountValue(summary, false) === 'Unavailable');
 */
function selectionCountValue(summary: ReturnType<TensorViewer['getSelectionSummary']>, enabled: boolean): string {
    if (!enabled) return 'Unavailable';
    if (summary.count === 0) return '0';
    return summary.availableCount === summary.count ? String(summary.count) : `${summary.count} (${summary.availableCount} with values)`;
}

/**
 * Formats one numeric statistic for the Selection widget's min, percentile, max, mean, or standard-deviation rows.
 *
 * @param summary - Selection summary returned by `TensorViewer.getSelectionSummary()`, with an optional `stats` object computed from selected cells that have loaded numeric values.
 * @param enabled - Whether selection statistics are supported for the current tensor mapping; false suppresses statistic values.
 * @param key - Statistic field to display, such as `min`, `p50`, `max`, `mean`, or `std`.
 * @returns An em dash when statistics are disabled or absent; otherwise the selected statistic formatted with the same range-value formatter used elsewhere in the sidebar.
 * @noThrows For viewer-produced summaries and the compile-time-limited statistic key, the function only checks field presence and delegates formatting for an existing numeric statistic.
 * @example
 * const summary = {
 *   count: 4,
 *   availableCount: 4,
 *   stats: { min: 1, p25: 2, p50: 3, p75: 4, max: 5, mean: 3, std: 1.25 }
 * } as ReturnType<TensorViewer['getSelectionSummary']>;
 *
 * console.assert(selectionStatValue(summary, true, 'p50') === formatRangeValue(3));
 * console.assert(selectionStatValue(summary, false, 'p50') === '—');
 * console.assert(selectionStatValue({ ...summary, stats: null }, true, 'p50') === '—');
 */
function selectionStatValue(summary: ReturnType<TensorViewer['getSelectionSummary']>, enabled: boolean, key: keyof NonNullable<ReturnType<TensorViewer['getSelectionSummary']>['stats']>): string {
    if (!enabled || !summary.stats) return '—';
    return formatRangeValue(summary.stats[key]);
}

/**
 * Builds the command-palette entries for the demo shell's built-in viewer controls, then appends any actions contributed by registered extensions.
 *
 * @returns Command actions whose `action` IDs are dispatched by the palette, with display labels, keyboard shortcut hints, and searchable keywords for viewer display, tab, widget, and export commands.
 * @noThrows Builds plain command descriptors from in-memory extension registrations without parsing user input or touching the DOM.
 * @example
 * const actions = commandActions();
 * actions.some((entry) => entry.action === 'save-svg' && entry.label === 'Save as SVG'); // true
 * actions.some((entry) => entry.action === 'heatmap' && entry.shortcut === 'Ctrl+H'); // true
 */
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

/**
 * Converts the current session tab list into command-palette entries that switch directly to a tab and mark the active tab as current.
 *
 * @returns One `tab:<id>` action per session tab, using the tab title as the label, `Current` as the shortcut hint for the active tab, and tab-title keywords for palette search.
 * @noThrows Maps already-loaded tab state into plain action objects without parsing user input, touching the DOM, or performing I/O.
 * @example
 * // Given sessionTabs = [{ id: 'main', title: 'Main' }, { id: 'compare', title: 'Comparison' }]
 * // and activeTabId = 'main':
 * tabActions();
 * // [
 * //   { action: 'tab:main', label: 'Main', shortcut: 'Current', keywords: 'tab Main' },
 * //   { action: 'tab:compare', label: 'Comparison', shortcut: '', keywords: 'tab Comparison' },
 * // ]
 */
function tabActions(): CommandAction[] {
    return sessionTabs.map((tab) => ({
        action: `tab:${tab.id}`,
        label: tab.title,
        shortcut: tab.id === activeTabId ? 'Current' : '',
        keywords: `tab ${tab.title}`,
    }));
}

/**
 * Selects the action source for the command palette, returning tab-navigation entries in tab mode and the full viewer command list otherwise.
 *
 * @returns The tab action list when `commandPaletteMode` is `'tabs'`; otherwise the built-in and extension command actions shown by the normal command palette.
 * @noThrows Reads current palette mode and already-loaded action arrays; it does not perform DOM work or validate external input.
 * @example
 * // When commandPaletteMode === 'tabs', the palette contains tab-switching actions such as:
 * paletteActions().every((entry) => entry.action.startsWith('tab:')); // true
 *
 * // In the default command mode, entries include viewer commands such as Save as SVG.
 * paletteActions().some((entry) => entry.action === 'save-svg'); // true
 */
function paletteActions(): CommandAction[] {
    return commandPaletteMode === 'tabs' ? tabActions() : commandActions();
}

/**
 * Scores how well normalized command-palette search text matches a candidate label and keyword string as an in-order character subsequence.
 *
 * @param candidate - Lower-cased searchable text built from a command action's label and keywords.
 * @param query - Lower-cased, whitespace-normalized palette query whose characters must appear in `candidate` in order.
 * @returns A numeric relevance score when every query character is found; higher scores favor word-boundary and consecutive matches, while `null` tells callers to filter out non-matching actions.
 * @noThrows Scans JavaScript strings with bounded index arithmetic only; unmatched queries return `null` instead of raising an error.
 * @example
 * fuzzyScore('save as svg file save export svg vector image 2d', 'svg') !== null; // true
 * fuzzyScore('toggle heatmap display heatmap colors', 'xyz'); // null
 */
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

/**
 * Builds the command-palette action list for the text currently typed in the palette search box.
 *
 * The search text is trimmed, lowercased, and whitespace-normalized before matching against each action's label and keywords.
 * An empty search returns the full palette action list; a non-empty search returns fuzzy matches sorted by descending score
 * and then alphabetically by label.
 *
 * @returns Command actions that should be rendered or executed for the current palette query, in display order.
 * @noThrows Reads the palette input value and filters in-memory action metadata; empty and non-matching queries are represented as arrays rather than errors.
 * @example
 * // With a palette containing labels "Open Tensor", "Toggle Sidebar", and "Reset View":
 * commandPaletteInput.value = '  tog   side ';
 * const actions = filteredCommandActions();
 * actions.map((action) => action.label);
 * // => ['Toggle Sidebar']
 */
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

// sidebar widget lifecycle
/**
 * Selects the sidebar widgets whose registered visibility predicates allow them for a viewer snapshot.
 *
 * The returned IDs keep the user's current sidebar drag order so widgets can be hidden without losing their order,
 * collapsed state, or DOM nodes.
 *
 * @param snapshot - Snapshot from the viewer containing the active tensor/viewer state passed to extension widget visibility callbacks.
 * @returns Sidebar widget IDs that should be shown for the supplied snapshot, ordered according to the current widgetOrder array.
 * @noThrows Missing widget specs and predicates that decline visibility are treated as hidden through the optional-chain/nullish fallback.
 * @example
 * const snapshot = viewer.getSnapshot();
 * const visible = visibleSidebarWidgets(snapshot);
 * // If only the tensor view and inspector predicates accept this snapshot:
 * visible;
 * // => ['tensor-view', 'inspector']
 */
function visibleSidebarWidgets(snapshot: ViewerSnapshot): SidebarWidgetId[] {
    // widget visibility is derived from the active tab and viewer state instead
    // of unmounting widgets permanently, so drag order and collapsed state survive.
    return widgetOrder.filter((widgetId) => widgetSpecById.get(widgetId)?.visible(extensionContext, snapshot) ?? false);
}

/**
 * Renders the shared header markup used at the top of a sidebar widget.
 *
 * The header includes the localized widget label, collapse/expand button state, the widget icon, an info button using
 * the provided help text, and the drag handle attributes used by sidebar reordering.
 *
 * @param widgetId - Registered sidebar widget ID used to look up the display label/icon and to stamp collapse and drag data attributes.
 * @param info - Help text displayed by the widget header's info button tooltip/popover.
 * @returns HTML string that widget renderers prepend to their widget body to get the standard title, info, collapse, and drag controls.
 * @noThrows For registered sidebar widget IDs, the function only reads in-memory label/collapse state and interpolates a markup string.
 * @example
 * collapsedWidgets.add('tensor-view');
 * const html = widgetTitle('tensor-view', 'Visualize tensor views, permutations, and slices.');
 * html.includes('aria-label="Expand Tensor View"');
 * // => true
 * html.includes('data-widget-handle="tensor-view"');
 * // => true
 */
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

/**
 * Reorders the sidebar DOM to match the current widgetOrder array.
 *
 * The sidebar header stays first, every widget element is reinserted in widgetOrder order, and the scroll pad stays last.
 * After replacing the children, drag affordances are synchronized with the newly applied order.
 *
 * @returns Nothing; mutates the sidebar element's child order and refreshes sidebar drag state.
 * @noThrows Uses already-created sidebar/header/widget/scroll-pad DOM nodes and does not validate external input; DOM replacement represents the state change.
 * @example
 * widgetOrder = ['inspector', 'tensor-view'];
 * applySidebarOrder();
 * Array.from(sidebar.children).map((child) => child.id);
 * // => [sidebarHeader.id, sidebarWidgets.inspector.id, sidebarWidgets['tensor-view'].id, sidebarScrollPad.id]
 */
function applySidebarOrder(): void {
    sidebar.replaceChildren(sidebarHeader, ...widgetOrder.map((widgetId) => sidebarWidgets[widgetId]), sidebarScrollPad);
    syncSidebarDragState();
}

/**
 * Updates one sidebar widget header so its collapse button matches the widget's stored collapsed state.
 *
 * @param widgetId - Sidebar widget key used in collapsedWidgets, sidebarWidgetLabels, and the header data attributes.
 * @param widget - Rendered sidebar widget element that may contain matching data-widget-collapse and data-widget-chevron controls.
 * @returns No value; when both controls are present, mutates the chevron text plus the button's aria-label and aria-expanded attributes.
 * @noThrows Missing header controls are treated as a no-op, and the function only reads the collapsed set and writes DOM text/attributes for a known widget id.
 * @example
 * collapsedWidgets.add('inspector');
 * syncWidgetHeaderState('inspector', sidebarWidgets.inspector);
 * expect(sidebarWidgets.inspector.querySelector('[data-widget-chevron="inspector"]')?.textContent).toBe('▸');
 * expect(sidebarWidgets.inspector.querySelector('[data-widget-collapse="inspector"]')?.getAttribute('aria-label')).toBe('Expand Inspector');
 * expect(sidebarWidgets.inspector.querySelector('[data-widget-collapse="inspector"]')?.getAttribute('aria-expanded')).toBe('false');
 */
function syncWidgetHeaderState(widgetId: SidebarWidgetId, widget: HTMLElement): void {
    const collapsed = collapsedWidgets.has(widgetId);
    const button = widget.querySelector<HTMLElement>(`[data-widget-collapse="${widgetId}"]`);
    const chevron = widget.querySelector<HTMLElement>(`[data-widget-chevron="${widgetId}"]`);
    if (!button || !chevron) return;
    chevron.textContent = collapsed ? '▸' : '▾';
    button.setAttribute('aria-label', `${collapsed ? 'Expand' : 'Collapse'} ${sidebarWidgetLabels[widgetId]}`);
    button.setAttribute('aria-expanded', String(!collapsed));
}

/**
 * Refreshes sidebar drag-and-drop CSS markers for the widget currently being reordered.
 *
 * @returns No value; removes stale drop-marker classes from every sidebar widget, marks the dragged widget, and adds the before/after drop class to the visible target slot.
 * @noThrows The function exits when there is no drop slot or no visible widget, and clamps the pending slot before indexing the visible widget list.
 * @example
 * draggedWidgetId = 'inspector';
 * draggedWidgetSlot = 0;
 * syncSidebarDragState();
 * expect(sidebarWidgets.inspector.classList.contains('widget-dragging')).toBe(true);
 * expect(sidebarWidgets[visibleSidebarWidgets(viewer.getSnapshot())[0]].classList.contains('widget-drop-before')).toBe(true);
 */
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

/**
 * Toggles a sidebar widget between expanded and collapsed, re-renders the sidebar, and keeps the widget header anchored in the scroll viewport.
 *
 * @param widgetId - Sidebar widget key whose entry in collapsedWidgets should be added or removed.
 * @returns No value; mutates collapsedWidgets, triggers render, and may adjust the sidebar scroll padding and scrollTop to preserve the header position.
 * @noThrows Missing header elements before or after render are guarded with early returns; normal calls use an existing sidebar widget id from the rendered sidebar controls.
 * @example
 * collapsedWidgets.delete('inspector');
 * toggleWidgetCollapse('inspector');
 * expect(collapsedWidgets.has('inspector')).toBe(true);
 * expect(sidebarWidgets.inspector.classList.contains('collapsed')).toBe(true);
 * expect(sidebarWidgets.inspector.querySelector('[data-widget-collapse="inspector"]')?.getAttribute('aria-expanded')).toBe('false');
 */
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

/**
 * Converts a pointer's viewport Y coordinate into the insertion slot for reordering visible sidebar widgets.
 *
 * @param clientY - PointerEvent.clientY value measured in viewport pixels during a sidebar drag.
 * @returns Zero-based insertion index before the first visible widget whose vertical midpoint is at or below clientY, visible.length for a drop after the last widget, or null when no widgets are visible.
 * @noThrows The function returns null for an empty visible-widget list and otherwise only reads bounding rectangles from registered visible sidebar widget elements.
 * @example
 * vi.spyOn(sidebarWidgets.inspector, 'getBoundingClientRect').mockReturnValue({ top: 100, height: 40 } as DOMRect);
 * expect(sidebarWidgetSlot(110)).toBe(0); // before inspector because 110 is above its midpoint at 120
 * expect(sidebarWidgetSlot(500)).toBe(visibleSidebarWidgets(viewer.getSnapshot()).length);
 */
function sidebarWidgetSlot(clientY: number): number | null {
    const visible = visibleSidebarWidgets(viewer.getSnapshot());
    if (visible.length === 0) return null;
    for (let index = 0; index < visible.length; index += 1) {
        const rect = sidebarWidgets[visible[index]!].getBoundingClientRect();
        if (clientY <= rect.top + rect.height / 2) return index;
    }
    return visible.length;
}

/**
 * Reorders a visible sidebar widget to the requested drop slot and reapplies the sidebar order.
 *
 * @param widgetId - Sidebar widget id from the current visible widget list; ids that are not currently visible leave the order unchanged.
 * @param slot - Requested zero-based insertion slot among visible widgets; values below zero move to the front and values beyond the list move to the end.
 * @returns Nothing; updates the module-level widget order and refreshes the sidebar layout through `applySidebarOrder()`.
 * @noThrows Missing widgets are ignored and out-of-range slots are clamped before the order array is rewritten, so normal drag-drop inputs have no expected throw path.
 * @example
 * // With visible widgets ordered as ['summary', 'inspector', 'settings']:
 * moveSidebarWidgetToSlot('summary', 2);
 * // The visible order becomes ['inspector', 'summary', 'settings'].
 *
 * moveSidebarWidgetToSlot('settings', -1);
 * // The visible order becomes ['settings', 'inspector', 'summary'] because the slot is clamped to 0.
 */
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

/**
 * Cancels the active sidebar drag bookkeeping and refreshes the sidebar drag styling.
 *
 * @returns Nothing; sets the dragged widget id, drop slot, and pointer id to `null`, then calls `syncSidebarDragState()`.
 * @noThrows The function only clears nullable module-level drag fields and runs the existing drag-state synchronizer, so ending or cancelling a pointer drag has no expected throw path.
 * @example
 * // During a sidebar drag:
 * // draggedWidgetId = 'inspector'; draggedWidgetSlot = 1; draggedWidgetPointerId = 42;
 * clearSidebarDragState();
 * // draggedWidgetId === null
 * // draggedWidgetSlot === null
 * // draggedWidgetPointerId === null
 */
function clearSidebarDragState(): void {
    draggedWidgetId = null;
    draggedWidgetSlot = null;
    draggedWidgetPointerId = null;
    syncSidebarDragState();
}

/**
 * Looks up the text icon displayed beside a sidebar widget title.
 *
 * @param widgetId - Sidebar widget id used as the key into the registered sidebar icon map.
 * @returns The configured icon text for the widget, or an empty string when the widget has no registered icon.
 * @noThrows Unknown or iconless widget ids use the empty-string fallback instead of throwing for a missing map entry.
 * @example
 * // When sidebarWidgetIcons.inspector is 'ⓘ':
 * widgetIcon('inspector');
 * // => 'ⓘ'
 *
 * // When a widget has no icon mapping:
 * widgetIcon('custom-widget' as SidebarWidgetId);
 * // => ''
 */
function widgetIcon(widgetId: SidebarWidgetId): string {
    return sidebarWidgetIcons[widgetId] ?? '';
}

/**
 * Rebuilds the command palette list for the current filter text and selected command index.
 *
 * @returns Nothing; when the palette is open, replaces the list DOM with matching command buttons or an empty-state message and clamps `commandPaletteIndex` to a valid action.
 * @noThrows A closed palette returns before touching the DOM, empty results render a fixed empty message, and out-of-range selection indexes are clamped before item classes are applied.
 * @example
 * // With the palette open, filter text matching "Open file", and commandPaletteIndex = 0:
 * renderCommandPalette();
 * // commandPaletteList contains a `.command-palette-item.active` button labeled "Open file".
 *
 * // With the palette open and no matching actions:
 * renderCommandPalette();
 * // commandPaletteIndex === 0
 * // commandPaletteList.innerHTML === '<div class="command-palette-empty">No matching commands.</div>'
 */
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

/**
 * Shows the command palette in action-search mode so keyboard shortcuts and menu commands can be chosen.
 *
 * @returns No value; the palette is made visible, reset to the first action result, cleared, rendered, and focused.
 * @noThrows Uses already-created palette DOM nodes and synchronous state assignments; invalid command text is not read during opening.
 * @example
 * openCommandPalette();
 *
 * console.assert(commandPaletteOpen === true);
 * console.assert(commandPaletteMode === 'actions');
 * console.assert(commandPaletteInput.placeholder === 'Type a command');
 * console.assert(!commandPalette.classList.contains('hidden'));
 */
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

/**
 * Shows the command palette in tab-selection mode so the user can switch to another open demo tab.
 *
 * @returns No value; the palette is made visible, reset to the first tab result, cleared, rendered, and focused.
 * @noThrows Uses already-created palette DOM nodes and synchronous state assignments; tab lookup happens later when a result is chosen.
 * @example
 * openTabPalette();
 *
 * console.assert(commandPaletteOpen === true);
 * console.assert(commandPaletteMode === 'tabs');
 * console.assert(commandPaletteInput.placeholder === 'Select a tab');
 * console.assert(!commandPalette.classList.contains('hidden'));
 */
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

/**
 * Hides the command palette and clears its transient search UI after backdrop clicks, Escape, or action selection.
 *
 * @returns No value; an open palette is hidden, reset to action mode, emptied, and its rendered result list is removed.
 * @noThrows Closing is idempotent: when the palette is already closed the function returns before touching the DOM, otherwise it only mutates existing palette elements.
 * @example
 * commandPaletteOpen = true;
 * commandPaletteInput.value = 'save';
 * commandPaletteList.replaceChildren(document.createElement('li'));
 *
 * closeCommandPalette();
 *
 * console.assert(commandPaletteOpen === false);
 * console.assert(commandPaletteMode === 'actions');
 * console.assert(commandPalette.classList.contains('hidden'));
 * console.assert(commandPaletteInput.value === '');
 * console.assert(commandPaletteList.children.length === 0);
 */
function closeCommandPalette(): void {
    if (!commandPaletteOpen) return;
    commandPaletteOpen = false;
    commandPaletteMode = 'actions';
    commandPalette.classList.add('hidden');
    commandPaletteInput.value = '';
    commandPaletteList.replaceChildren();
}

/**
 * Applies a requested sidebar width to the app shell, clamping it to the space left beside the splitter and resizing the viewer canvas.
 *
 * @param width - Requested sidebar width in CSS pixels, such as the drag distance from the right edge or the default maximum width.
 * @returns No value; updates the `--sidebar-width` CSS custom property and notifies the viewer to recompute its layout.
 * @noThrows Negative and oversized widths are clamped into the available app width instead of being rejected; the remaining work is synchronous style mutation and viewer resizing.
 * @example
 * // With app.clientWidth = 800 and sidebarSplitter.offsetWidth = 8:
 * setSidebarWidth(1000);
 *
 * console.assert(app.style.getPropertyValue('--sidebar-width') === '792px');
 * console.assert(viewer.resizeCalls === 1);
 */
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

// tab documents
/**
 * Finds the loaded session tab that is currently selected in the demo tab strip.
 *
 * @returns The `LoadedBundleDocument` whose `id` matches `activeTabId`, or `undefined` when no tab is active or the active id no longer exists in `sessionTabs`.
 * @noThrows Performs only an in-memory array lookup against `sessionTabs` and does not parse, clone, or load tab contents.
 * @example
 * sessionTabs = [layoutATab, layoutBTab];
 * activeTabId = layoutBTab.id;
 *
 * const tab = activeTab();
 *
 * console.assert(tab === layoutBTab);
 */
function activeTab(): LoadedBundleDocument | undefined {
    return sessionTabs.find((tab) => tab.id === activeTabId);
}

/**
 * Chooses the first available default title for a newly created layout tab.
 *
 * @returns A `Layout N` title using the lowest positive integer that is not already present in `sessionTabs`, suitable for displaying in the tab strip.
 * @noThrows Reads existing tab titles into a `Set` and increments a local counter; normal session-tab records already contain string titles.
 * @example
 * sessionTabs = [
 *     { ...loadedTab, id: 'tab-1', title: 'Layout 1' },
 *     { ...loadedTab, id: 'tab-2', title: 'Layout 3' },
 * ];
 *
 * console.assert(nextTabTitle() === 'Layout 2');
 */
function nextTabTitle(): string {
    const used = new Set(sessionTabs.map((tab) => tab.title));
    let index = 1;
    while (used.has(`Layout ${index}`)) index += 1;
    return `Layout ${index}`;
}

/**
 * Creates an independent copy of a loaded tab document for the Add Tab workflow.
 *
 * @param tab - Source session tab whose manifest and tensor byte arrays should be duplicated.
 * @param id - Identifier to assign to the new tab entry.
 * @param title - Display title to show for the new tab in the tab strip.
 * @returns A `LoadedBundleDocument` with the supplied `id` and `title`, a structured-cloned manifest, and a new tensor map whose array values are sliced copies of the source tensors.
 * @noThrows Has no validation branch; it expects the source tab to contain structured-cloneable manifest data and tensor values that support `slice()`.
 * @example
 * const source = {
 *     id: 'tab-1',
 *     title: 'Layout 1',
 *     manifest,
 *     tensors: new Map([['weights', new Float32Array([1, 2, 3])]]),
 * };
 *
 * const clone = cloneTabDocument(source, 'tab-2', 'Layout 2');
 *
 * console.assert(clone.id === 'tab-2');
 * console.assert(clone.title === 'Layout 2');
 * console.assert(clone.manifest !== source.manifest);
 * console.assert(clone.tensors.get('weights') !== source.tensors.get('weights'));
 * console.assert(clone.tensors.get('weights')?.[0] === 1);
 */
function cloneTabDocument(tab: LoadedBundleDocument, id: string, title: string): LoadedBundleDocument {
    return {
        id,
        title,
        manifest: structuredClone(tab.manifest),
        tensors: new Map(Array.from(tab.tensors.entries(), ([tensorId, data]) => [tensorId, data.slice()])),
    };
}

/**
 * Selects a loadable tensor-view snapshot for a tensor before restoring it into the viewer.
 *
 * @param tensor - Manifest tensor entry whose `shape` and optional `axisLabels` define the valid dimensions and labels for tensor-view parsing.
 * @param views - Candidate saved views, tried in order, such as the incoming snapshot view, the previous tab view, and the manifest's default view.
 * @returns A normalized view containing the parsed editor spec and a copied `hiddenIndices` array; callers can store it in the viewer snapshot without sharing the candidate array instance.
 * @throws Error with message `Tensor view editor state is invalid.` when every supplied candidate view is missing or unparsable and the generated default editor also cannot be parsed for the tensor shape.
 * @example
 * const tensor = { id: 'activations', shape: [2, 3], axisLabels: [['row0', 'row1'], ['col0', 'col1', 'col2']] };
 * const savedView = {
 *     editor: { mode: 'expression', expression: '[:, col1]' },
 *     hiddenIndices: [0],
 * };
 *
 * const view = normalizedTensorViewSnapshot(tensor, savedView);
 *
 * console.assert(view.hiddenIndices !== savedView.hiddenIndices);
 * console.assert(view.hiddenIndices[0] === 0);
 *
 * @example
 * const tensor = { id: 'broken', shape: [], axisLabels: [] };
 *
 * try {
 *     normalizedTensorViewSnapshot(tensor, { editor: { mode: 'expression', expression: '[' }, hiddenIndices: [] });
 * } catch (error) {
 *     console.assert(error instanceof Error);
 *     console.assert(error.message === 'Tensor view editor state is invalid.');
 * }
 */
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

/**
 * Reconciles a saved or freshly captured viewer snapshot with the tab manifest before the
 * snapshot is persisted or loaded back into the viewer.
 *
 * Each tensor entry keeps its non-view fields, but entries whose id exists in the tab
 * manifest receive a normalized tensor-view snapshot using the manifest tensor definition,
 * the tab's previous saved view, and the tensor's default view. Entries for tensors that are
 * no longer present in the manifest are preserved unchanged.
 *
 * @param tab - Loaded session tab whose manifest supplies the tensor definitions, previous viewer tensor views, and tensor defaults used for normalization.
 * @param snapshot - Viewer snapshot captured from the live viewer or read from the tab manifest, including the tensor entries to reconcile by id.
 * @returns A new viewer snapshot object suitable for storing on `tab.manifest.viewer` or passing to the viewer; matching tensor entries contain normalized `view` values.
 * @noThrows The function performs no explicit validation and leaves snapshot entries without matching manifest tensors unchanged instead of raising an error.
 * @example
 * const normalized = normalizeViewerSnapshot(tab, {
 *   ...tab.manifest.viewer,
 *   tensors: [{ id: 'activation', view: { expression: '0:4, :' } }],
 * });
 *
 * expect(normalized.tensors[0]?.id).toBe('activation');
 * expect(normalized.tensors[0]?.view).toEqual(
 *   normalizedTensorViewSnapshot(
 *     tab.manifest.tensors.find((tensor) => tensor.id === 'activation')!,
 *     { expression: '0:4, :' },
 *     tab.manifest.viewer.tensors.find((tensor) => tensor.id === 'activation')?.view,
 *     tab.manifest.tensors.find((tensor) => tensor.id === 'activation')!.view,
 *   ),
 * );
 */
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

/**
 * Ends any in-progress tab title rename so the tab strip returns to its normal label view.
 *
 * @returns Nothing; `editingTab` is reset to `null`.
 * @noThrows Clearing the edit marker is a single assignment and does not inspect DOM state or validate tab data.
 * @example
 * editingTab = sessionTabs[0]!;
 * clearTabTitleEdit();
 *
 * expect(editingTab).toBeNull();
 */
function clearTabTitleEdit(): void {
    editingTab = null;
}

/**
 * Saves the live viewer state into the active session tab before switching, cloning, closing,
 * or rendering tab-specific UI.
 *
 * The raw viewer snapshot is offered to extensions first so extension-owned metadata can be
 * captured while it still matches the visible viewer. The tab manifest then receives the
 * normalized snapshot used for later reloads. If no tab is active, the function is a no-op.
 *
 * @returns Nothing; when an active tab exists, `activeTab().manifest.viewer` is replaced with the normalized viewer snapshot.
 * @noThrows The app-level control flow has no explicit error branch: a missing active tab returns early, and the remaining work delegates to optional extension hooks before assigning the normalized snapshot.
 * @example
 * activeTabId = 'tab-a';
 * viewer.getSnapshot = () => ({ tensors: [{ id: 'weights', view: { expression: ':' } }] });
 *
 * captureActiveTabSnapshot();
 *
 * expect(sessionTabs.find((tab) => tab.id === 'tab-a')!.manifest.viewer.tensors[0]?.id).toBe('weights');
 */
function captureActiveTabSnapshot(): void {
    const tab = activeTab();
    if (!tab) return;
    const snapshot = viewer.getSnapshot();
    // extensions capture their metadata before the viewer snapshot is normalized,
    // otherwise tab-specific state like compose-layout tensor views can lag behind
    // the visible viewer after a user edits a slice and immediately switches tabs.
    extensions.forEach((extension) => {
        extension.captureSnapshot?.(extensionContext, tab, snapshot);
    });
    tab.manifest.viewer = normalizeViewerSnapshot(tab, snapshot);
}

/**
 * Removes a session tab and updates the surrounding viewer shell state.
 *
 * Closing the active tab first captures its current viewer snapshot, notifies extensions to
 * discard tab-scoped data, removes the tab from `sessionTabs`, and loads the previous
 * neighboring tab when one remains. Closing the final tab clears the active id, seeds the
 * fallback demo tensor, and renders the empty-tab state. Unknown tab ids are ignored.
 *
 * @param tabId - Identifier of the session tab to remove from `sessionTabs`; ids not present in the current session are treated as no-ops.
 * @returns A promise that resolves with no value after any replacement active tab has finished loading and the tab strip/viewer state has been refreshed.
 * @noThrows The function performs no explicit validation: missing tab ids return immediately, and the final-tab case is handled by reseeding and rendering instead of throwing.
 * @example
 * sessionTabs = [firstTab, secondTab];
 * activeTabId = secondTab.id;
 *
 * await closeTab(secondTab.id);
 *
 * expect(sessionTabs.map((tab) => tab.id)).toEqual([firstTab.id]);
 * expect(activeTabId).toBe(firstTab.id);
 *
 * @example
 * sessionTabs = [firstTab];
 * await closeTab('missing-tab');
 *
 * expect(sessionTabs).toEqual([firstTab]);
 */
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

/**
 * Rebuilds the demo tab bar from `sessionTabs`, showing the active tab, close buttons,
 * the inline title editor for `editingTab`, and the trailing "Add New Tab" action.
 *
 * @returns Nothing. Callers observe the updated `tabStrip` children and the event handlers
 * wired to load, close, rename, or create tabs.
 * @noThrows Does not intentionally throw for an empty tab list; it clears any pending title edit
 * and renders only the action area. DOM API failures or rejected async tab handlers are outside
 * this renderer's expected synchronous path.
 * @example
 * sessionTabs = [{ id: 'tab-1', title: 'Weights' }, { id: 'tab-2', title: 'Activations' }];
 * activeTabId = 'tab-2';
 * editingTab = null;
 * renderTabStrip();
 *
 * console.assert(tabStrip.querySelectorAll('.tab-button').length === 2);
 * console.assert(tabStrip.querySelector('.tab-button.active')?.getAttribute('aria-label') === 'Activations');
 * console.assert(tabStrip.querySelector('.tab-strip-action')?.textContent === 'Add New Tab');
 */
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

/**
 * Creates a new demo tab: extensions may provide the first tab for an empty session, otherwise
 * the active tab's document is captured, cloned under a fresh tab id, registered with extensions,
 * and loaded as the active tab.
 *
 * @returns A promise that settles after the new or seeded tab has been loaded, so command handlers
 * can wait before refreshing UI that depends on `activeTabId` or `sessionTabs`.
 * @noThrows Has no validation branch of its own; expected operation uses the existing active tab,
 * extension hooks, and viewer snapshot. Rejections from extension hooks or tab loading propagate
 * to the caller.
 * @example
 * sessionTabs = [{ id: 'tab-1', title: 'Tensor 1', document: currentDocument }];
 * activeTabId = 'tab-1';
 * await addNewTab();
 *
 * console.assert(sessionTabs.length === 2);
 * console.assert(sessionTabs[1].id.startsWith('tab-'));
 * console.assert(sessionTabs[1].title === 'Tensor 2');
 * console.assert(activeTabId === sessionTabs[1].id);
 */
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

/**
 * Closes the tab referenced by `activeTabId`, delegating fallback selection, session updates,
 * and demo reseeding to `closeTab`.
 *
 * @returns A promise that resolves after the active tab has been closed, or immediately when no
 * tab is active.
 * @noThrows The missing-active-tab case is guarded by `if (activeTabId)`, making it a no-op;
 * any rejection comes from the delegated `closeTab` workflow rather than from this guard.
 * @example
 * sessionTabs = [{ id: 'tab-1', title: 'Weights' }, { id: 'tab-2', title: 'Activations' }];
 * activeTabId = 'tab-2';
 * await closeCurrentTab();
 *
 * console.assert(!sessionTabs.some((tab) => tab.id === 'tab-2'));
 * console.assert(activeTabId !== 'tab-2');
 *
 * activeTabId = null;
 * await closeCurrentTab(); // no tab is closed and the session remains unchanged
 */
async function closeCurrentTab(): Promise<void> {
    if (activeTabId) await closeTab(activeTabId);
}

/**
 * Builds the control dock for the supplied viewer snapshot, including pan/select/rotate modes,
 * 2D and 3D display toggles, extension-contributed controls, dimension-line and tensor-name
 * toggles, gap display, and dimension-mapping buttons.
 *
 * @param snapshot - Viewer state snapshot whose `displayMode`, `interactionMode`, selection
 * availability, label visibility, gap setting, and mapping scheme determine which dock buttons
 * are active or disabled.
 * @returns Nothing. Callers observe `controlDock` being replaced with buttons that call the
 * corresponding viewer setters or extension handlers when clicked.
 * @noThrows Does not intentionally reject unsupported modes; unavailable actions such as select
 * outside a selectable 2D layout or rotate outside 3D are rendered disabled. DOM failures or
 * exceptions from extension `controls` providers are allowed to propagate.
 * @example
 * renderControlDock({
 *   ...viewer.getSnapshot(),
 *   displayMode: '2d',
 *   interactionMode: 'select',
 *   showDimensionLines: true,
 *   showTensorNames: true,
 *   displayGaps: false,
 *   dimensionMappingScheme: 'contiguous',
 * });
 *
 * console.assert(controlDock.querySelector('[data-control-id="select"]')?.classList.contains('active'));
 * console.assert(controlDock.querySelector('[data-control-id="rotate"]')?.hasAttribute('disabled'));
 * console.assert(controlDock.querySelector('[data-control-id="mapping-contiguous"]')?.classList.contains('active'));
 */
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

/**
 * Switches the demo shell to the session tab with the given id, loads that tab's manifest and tensors into the viewer, applies saved viewer options, refreshes the tab strip, and lets extensions hydrate tab-local state.
 *
 * @param tabId - Id of an entry in `sessionTabs`; ids that are not present are ignored without changing the active tab.
 * @returns Promise that resolves after the viewer has accepted the tab bundle, startup widget defaults have been applied if needed, the tab strip has rendered, and extension `afterLoadTab` hooks have run.
 * @noThrows Unknown tab ids return early, and this function does not intentionally validate or throw before delegating to viewer and extension hooks.
 * @example
 * sessionTabs = [linearLayoutTab, profilerTab];
 * await loadTab(linearLayoutTab.id);
 * expect(activeTabId).toBe(linearLayoutTab.id);
 * expect(viewer.getSnapshot().dimensionMappingScheme).toBe(linearLayoutTab.manifest.viewer.dimensionMappingScheme);
 *
 * @example
 * const previousActiveTabId = activeTabId;
 * await loadTab('missing-tab');
 * expect(activeTabId).toBe(previousActiveTabId);
 */
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

// sidebar rendering
/**
 * Reconciles the sidebar DOM with a viewer snapshot by showing only the widgets that are visible for that snapshot, applying collapsed widget state, syncing widget headers, and refreshing drag affordances.
 *
 * @param snapshot - Snapshot returned by the viewer render pipeline; extension visibility rules inspect it to decide which registered sidebar widgets should be displayed.
 * @returns Nothing; the registered sidebar widget elements are updated in place with `hidden` and `collapsed` classes and synchronized header state.
 * @noThrows The function only iterates registered widget elements and toggles DOM state for the supplied snapshot; normal visibility decisions are represented as class changes rather than errors.
 * @example
 * updateSidebar(snapshotWithSelection);
 * expect(sidebarWidgets.inspector.classList.contains('hidden')).toBe(false);
 * expect(sidebarWidgets['linear-layout'].classList.contains('collapsed')).toBe(collapsedWidgets.has('linear-layout'));
 */
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

/**
 * Captures a sidebar control's current vertical position so a later render can keep the same control aligned after widgets expand, collapse, or change content above it.
 *
 * @param element - Sidebar element currently under interaction, such as a tensor-view slider or slice-token button; `null` means there is no control to anchor.
 * @param selector - CSS selector that can find the same sidebar element after the sidebar is re-rendered.
 * @returns The selector and the element's current viewport `top` coordinate, or `null` when no element was supplied.
 * @noThrows A missing element is handled by returning `null`; otherwise the helper only reads `getBoundingClientRect()` and stores the caller-provided selector.
 * @example
 * const slider = document.querySelector<HTMLElement>('#tensor-view-slider-0')!;
 * vi.spyOn(slider, 'getBoundingClientRect').mockReturnValue({ top: 120 } as DOMRect);
 * expect(captureSidebarAnchor(slider, '#tensor-view-slider-0')).toEqual({
 *   selector: '#tensor-view-slider-0',
 *   top: 120,
 * });
 * expect(captureSidebarAnchor(null, '#tensor-view-slider-0')).toBeNull();
 */
function captureSidebarAnchor(element: HTMLElement | null, selector: string): { selector: string; top: number } | null {
    if (!element) return null;
    return { selector, top: element.getBoundingClientRect().top };
}

/**
 * Re-renders the viewer and sidebar while preserving the sidebar scroll position, optionally adjusting after the next animation frame so a captured control stays at the same viewport height.
 *
 * @param anchor - Optional selector/top pair from `captureSidebarAnchor`; pass `null` when only the previous `sidebar.scrollTop` should be restored.
 * @returns Nothing; the function calls the normal render path and mutates `sidebar.scrollTop` immediately and again after the browser has laid out the re-rendered sidebar.
 * @noThrows Null anchors and selectors that no longer match an element are treated as no-op scroll adjustments rather than errors.
 * @example
 * sidebar.scrollTop = 200;
 * renderPreservingSidebarScroll(null);
 * await nextAnimationFrame();
 * expect(sidebar.scrollTop).toBe(200);
 *
 * @example
 * sidebar.scrollTop = 200;
 * renderPreservingSidebarScroll({ selector: '#slice-token-k', top: 80 });
 * await nextAnimationFrame();
 * // If the re-rendered control moved down by 12px, scrollTop is increased to keep it under the cursor.
 * expect(sidebar.scrollTop).toBe(212);
 */
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
        // expanding/collapsing widgets changes content above active controls; anchoring
        // by selector keeps the control under the cursor instead of jumping vertically.
        sidebar.scrollTop += nextAnchor.getBoundingClientRect().top - anchor.top;
    });
}

// tensor-view editor
/**
 * Resizes a sidebar textarea to exactly fit its current content so tensor-view and widget fields do not show an inner scrollbar while editing.
 *
 * @param textarea - Rendered textarea whose `scrollHeight` reflects the current text after the DOM has been laid out.
 * @returns Nothing; sets `textarea.style.height` to `0` before measuring and then to the textarea's current `scrollHeight` in pixels.
 * @noThrows Reading `scrollHeight` and assigning inline `style.height` on an existing `HTMLTextAreaElement` are synchronous DOM property operations with no expected application-level error path.
 * @example
 * const textarea = document.createElement('textarea');
 * textarea.value = 'view: [batch, head, token]';
 * Object.defineProperty(textarea, 'scrollHeight', { value: 48, configurable: true });
 *
 * autosizeTextarea(textarea);
 *
 * console.assert(textarea.style.height === '48px');
 */
function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

/**
 * Marks a tensor-view range slider as actively dragged so preview rendering can pause until the pointer is released.
 *
 * @param slider - Tensor-view `input[type="range"]` element that received the pointerdown event.
 * @param pointerId - `PointerEvent.pointerId` from that pointerdown event; stored so only the matching pointer can end the drag.
 * @returns Nothing; records the active pointer id, suspends tensor-view rendering, and asks the slider to capture subsequent pointer events.
 * @noThrows Stores drag bookkeeping and delegates pointer capture to the slider element supplied by the pointer event.
 * @example
 * const slider = document.createElement('input');
 * slider.type = 'range';
 * let capturedPointer: number | undefined;
 * slider.setPointerCapture = (pointerId) => { capturedPointer = pointerId; };
 *
 * beginTensorViewSliderDrag(slider, 7);
 *
 * console.assert(capturedPointer === 7);
 */
function beginTensorViewSliderDrag(slider: HTMLInputElement, pointerId: number): void {
    suspendTensorViewRender = true;
    activeTensorViewSliderPointerId = pointerId;
    slider.setPointerCapture(pointerId);
}

/**
 * Completes the active tensor-view slider drag, resumes rendering, releases pointer capture, and rerenders while keeping the slider's sidebar position anchored.
 *
 * @param slider - Tensor-view `input[type="range"]` element that may currently hold pointer capture for the drag.
 * @param pointerId - `PointerEvent.pointerId` from pointerup or pointercancel; ignored when it does not match the active slider drag pointer.
 * @returns Nothing; clears the active pointer when it matches, resumes tensor-view rendering, releases capture if present, and refreshes the sidebar around the slider.
 * @noThrows Ignores non-matching pointers and checks pointer capture before releasing it, so normal drag completion does not raise validation errors.
 * @example
 * const slider = document.createElement('input');
 * slider.id = 'slice-token-0';
 * slider.type = 'range';
 * let releasedPointer: number | undefined;
 * slider.hasPointerCapture = (pointerId) => pointerId === 7;
 * slider.releasePointerCapture = (pointerId) => { releasedPointer = pointerId; };
 *
 * beginTensorViewSliderDrag(slider, 7);
 * endTensorViewSliderDrag(slider, 7);
 *
 * console.assert(releasedPointer === 7);
 */
function endTensorViewSliderDrag(slider: HTMLInputElement, pointerId: number): void {
    if (activeTensorViewSliderPointerId !== pointerId) return;
    activeTensorViewSliderPointerId = null;
    suspendTensorViewRender = false;
    if (slider.hasPointerCapture(pointerId)) slider.releasePointerCapture(pointerId);
    renderPreservingSidebarScroll(captureSidebarAnchor(slider, `#${CSS.escape(slider.id)}`));
}

/**
 * Commits a parsed tensor-view editor model to the viewer, lets extensions react to the changed view, and rerenders the sidebar with any validation message updated.
 *
 * @param tensorId - Viewer tensor handle id whose tensor-view expression should be replaced.
 * @param editor - Parsed tensor-view editor state, including the view input text and selected sliced token keys to serialize for the viewer.
 * @param anchor - Optional sidebar scroll anchor captured before a button or slider edit so the rerender returns to the same control.
 * @returns Nothing; on success clears the tensor's view error, and on serialization or viewer rejection stores the error message for display before rerendering.
 * @noThrows Catches tensor-view serialization and viewer validation failures, stores their messages for the sidebar, and still rerenders.
 * @example
 * const editor = {
 *   version: 2,
 *   viewTensorInput: '[batch, token]',
 *   slicedTokenKeys: ['batch'],
 * };
 * const anchor = { selector: '[data-slice-token="batch"]', top: 120 };
 *
 * applyTensorViewEditor('logits', editor, anchor);
 *
 * // The next sidebar render shows the serialized tensor view for `logits`; if the
 * // viewer rejects it, the tensor's validation message is rendered instead.
 */
function applyTensorViewEditor(
    tensorId: string,
    editor: TensorViewEditor,
    anchor: { selector: string; top: number } | null = null,
): void {
    try {
        viewer.setTensorView(tensorId, serializeTensorViewEditor(editor));
        // tensor-view edits can change which linear-layout roots are visible, so
        // extensions get one post-change hook instead of the shell importing their
        // filtering logic directly.
        extensions.forEach((extension) => {
            extension.afterTensorViewChange?.(extensionContext, tensorId);
        });
        viewErrors.delete(tensorId);
    } catch (error) {
        viewErrors.set(tensorId, error instanceof Error ? error.message : String(error));
    }
    renderPreservingSidebarScroll(anchor);
}

/**
 * Removes a single leading `[` and single trailing `]` from tensor-call text before comparing it with the editor's view input.
 *
 * @param value - Raw tensor call input text from the view editor, such as `[batch, head]` or `batch, head`.
 * @returns The same text without one outer bracket pair; inner brackets or unpaired brackets are otherwise preserved.
 * @noThrows Uses only string replacement on the provided string, so malformed bracket text is returned as normalized text instead of raising an error.
 * @example
 * tensorCallInputValue('[batch, head]'); // 'batch, head'
 * tensorCallInputValue('batch, head'); // 'batch, head'
 */
function tensorCallInputValue(value: string): string {
    return value.replace(/^\[/, '').replace(/\]$/, '');
}

/**
 * Parses one integer expression from tensor-view controls, including `*`-separated products used in dimension or permutation fields.
 *
 * @param value - Text for a single integer term, such as `3`, `2 * 4`, `-1`, or a blank/invalid token from an input field.
 * @returns The parsed integer or product; returns `-1` for the inference sentinel and `NaN` for blank or non-finite terms.
 * @noThrows Invalid numeric tokens are converted to `NaN`, and no error is raised for empty or malformed input.
 * @example
 * parseIntegerTerm('2 * 4'); // 8
 * parseIntegerTerm('-1'); // -1
 * Number.isNaN(parseIntegerTerm('width')); // true
 */
function parseIntegerTerm(value: string): number {
    const term = value.trim();
    if (term === '') return Number.NaN;
    if (term === '-1') return -1;
    const parts = term.split('*').map((part) => Number(part.trim()));
    if (parts.some((part) => !Number.isFinite(part))) return Number.NaN;
    return parts.reduce((acc, part) => acc * part, 1);
}

/**
 * Parses the comma-separated tensor view shape field into labeled dimensions and resolves a single `-1` size from the tensor element count.
 *
 * @param value - Shape specification text such as `batch=2, head=4`, `2, 4, -1`, or anonymous labels like `*0=8`.
 * @param totalElements - Number of elements in the tensor view, used to infer the size of the dimension written as `-1`.
 * @returns Dimension descriptors in input order, with generated `*A#` labels for bare numeric dimensions and concrete sizes after inference.
 * @throws Error when a comma-separated term contains unsupported syntax, or when a `-1` dimension cannot be inferred because the known product is zero or does not divide `totalElements`.
 * @example
 * parseShapeSpec('batch=2, head=4, width=-1', 24);
 * // [
 * //   { label: 'batch', size: 2 },
 * //   { label: 'head', size: 4 },
 * //   { label: 'width', size: 3 },
 * // ]
 *
 * @example
 * parseShapeSpec('2, 3', 6);
 * // [
 * //   { label: '*A0', size: 2 },
 * //   { label: '*A1', size: 3 },
 * // ]
 *
 * @example
 * parseShapeSpec('batch=-1, width=5', 12);
 * // throws Error: Could not infer a valid -1 dimension.
 */
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

/**
 * Parses a comma-separated control value into finite integer terms for tensor-view operations such as permutation indices.
 *
 * @param value - Comma-separated integer-term text, where each entry may be a number or multiplication expression like `0, 2, 3 * 4`.
 * @returns Finite parsed numbers in their original order; blank entries and malformed terms are omitted.
 * @noThrows Each entry is parsed with `parseIntegerTerm`, which reports invalid tokens as `NaN`, and this helper filters those values out.
 * @example
 * parseIntegerListInput('0, 2, 3 * 4, width, , -1'); // [0, 2, 12, -1]
 */
function parseIntegerListInput(value: string): number[] {
    return value.split(',').map(parseIntegerTerm).filter((part) => Number.isFinite(part));
}

/**
 * Rebuilds the Tensor View editor after the user edits the first view, permutation,
 * or optional final view text.
 *
 * The returned editor keeps the previous editor metadata, refreshes the displayed
 * `tensor.view(...)` input, derives base-dimension ids and labels from the first
 * view, converts the permutation indices into permuted dimension ids, resets
 * flatten separators for the new order, and clears singleton dimensions.
 *
 * @param previous - Existing Tensor View editor state whose dimension ids are reused when the first view shape did not change.
 * @param viewInput - Comma-separated shape expression for the first `tensor.view(...)`, without surrounding brackets.
 * @param permuteInput - Comma-separated zero-based dimension indices from `tensor.permute(...)`.
 * @param finalViewInput - Optional comma-separated shape expression for the final chained `.view(...)`, or null when no final view is present.
 * @param totalElements - Product of the source tensor shape, used to validate and infer the first view dimensions.
 * @returns A TensorViewEditor ready to render the updated view/permutation controls and serialize the edited tensor expression.
 * @throws Error when `viewInput` is not a valid shape for `totalElements` or `permuteInput` contains an invalid integer list.
 * @example
 * const next = buildStep4Editor(previous, 'B=2, C=3', '1, 0', null, 6);
 *
 * console.assert(next.viewTensorInput === '[B=2, C=3]');
 * console.assert(next.finalViewInput === undefined);
 * console.assert(next.baseDims.map((dim) => dim.label).join(',') === 'B,C');
 * console.assert(next.permutedDimIds.length === 2);
 * console.assert(next.flattenSeparators.length === 1);
 *
 * @example
 * try {
 *   buildStep4Editor(previous, '2, 3', 'not-an-index', null, 6);
 * } catch (error) {
 *   console.assert(error instanceof Error);
 * }
 */
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

/**
 * Builds the expandable Tensor View usage guide shown beside the direct tensor
 * expression textarea.
 *
 * The examples in the fragment are tailored to the loaded tensor: they show the
 * current shape, a reversed permutation order, and a labeled shape using supplied
 * axis labels with `A0`, `A1`, ... fallbacks.
 *
 * @param shape - Source tensor dimension sizes used in generated `tensor.view(...)` and reversed `tensor.permute(...)` examples.
 * @param axisLabels - Optional labels for each tensor axis; missing labels are rendered as `A${index}` fallbacks.
 * @returns An escaped HTML `<details>` fragment that callers insert into the Tensor View panel.
 * @noThrows This formatter only maps arrays, joins strings, and escapes interpolated text; it does not parse user input or touch the DOM.
 * @example
 * const html = tensorViewHelpHtml([2, 3], ['batch', 'feature']);
 *
 * console.assert(html.includes('tensor.view(2, 3)'));
 * console.assert(html.includes('tensor.permute(1, 0)'));
 * console.assert(html.includes('tensor.view(batch=2, feature=3)'));
 */
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

/**
 * Parses the text from the Tensor View textarea into the editor model used by the
 * view, permute, final-view, and slice controls.
 *
 * Supported input starts with `tensor` and may include chained `.view(A)`,
 * `.permute(B)`, a second `.view(C)`, and a trailing `[D]` slice. Missing first
 * view or permutation calls default to the source tensor shape and identity
 * permutation.
 *
 * @param value - User-entered tensor expression, such as `tensor.view(B=2, C=3).permute(1, 0)[0, :]`.
 * @param previous - Existing Tensor View editor state used to preserve dimension ids when compatible with the new first view.
 * @param shape - Source tensor shape used for default view text, identity permutation, and element-count validation.
 * @returns Updated TensorViewEditor state that callers pass to `applyTensorViewEditor` to refresh the Tensor View UI.
 * @throws Error when the text does not start with `tensor`, a `.view(...)` or `.permute(...)` call is unclosed, trailing text is not a bracket slice, or a slice term is neither `:` nor a finite number.
 * @example
 * const editor = parseTensorViewExpressionInput(
 *   'tensor.view(B=2, C=3).permute(1, 0)[0, :]',
 *   previous,
 *   [2, 3],
 * );
 *
 * console.assert(editor.viewTensorInput === '[B=2, C=3]');
 * console.assert(editor.slicedTokenKeys.length === 1);
 * console.assert(Object.values(editor.sliceValues)[0] === 0);
 *
 * @example
 * try {
 *   parseTensorViewExpressionInput('image.view(2, 3)', previous, [2, 3]);
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'Tensor View must start with "tensor".');
 * }
 */
function parseTensorViewExpressionInput(
    value: string,
    previous: TensorViewEditor,
    shape: readonly number[],
): TensorViewEditor {
    const text = value.trim();
    if (!text.startsWith('tensor')) throw new Error('Tensor View must start with "tensor".');
    let rest = text.slice('tensor'.length);
    /**
 * Consumes the next chained `.view(...)` or `.permute(...)` call from the captured
 * `rest` suffix of the Tensor View expression.
 *
 * When the next characters match the requested call name, this helper returns the
 * trimmed text between the matching parentheses and advances `rest` past the
 * consumed call. Parentheses inside the argument text are balanced before the call
 * is considered closed.
 *
 * @param name - Chained call kind to consume at the current `rest` cursor: either `view` or `permute`.
 * @returns The raw argument text from the matched call, or null when `rest` does not start with `.${name}(`.
 * @throws Error when the matching call starts but its closing parenthesis is missing.
 * @example
 * // With rest initially set to '.view(B=2, C=3).permute(1, 0)':
 * const viewArgs = consumeCall('view');
 * console.assert(viewArgs === 'B=2, C=3');
 * console.assert(rest === '.permute(1, 0)');
 *
 * @example
 * // With rest initially set to '.view(2, 3':
 * try {
 *   consumeCall('view');
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'Unclosed view(...) in Tensor View.');
 * }
 */
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

/**
 * Rebuilds the Tensor View sidebar from the active inspector model, including the tensor selector,
 * editable tensor-view expression, slice chips, validation error message, stock slice sliders, and
 * extension-contributed tensor-view controls.
 *
 * @param snapshot - Viewer render snapshot supplied by the widget host for the current render pass; the renderer uses the global viewer inspector model derived from this pass and leaves the widget unchanged when tensor-view rendering is suspended or no editor is available.
 * @returns No value; replaces `tensorViewWidget.innerHTML` and attaches event handlers that update the active tensor, apply tensor-view expressions, and drive slice controls.
 * @noThrows The renderer has no intentional validation throw path: missing tensor handles and missing editors are handled with an empty-state message or early return, while invalid tensor-view text is caught in the textarea change handler and displayed in the widget.
 * @example
 * // With no active tensor, the widget is rebuilt as an empty Tensor View panel.
 * renderTensorViewWidget(viewer.getSnapshot());
 * console.assert(tensorViewWidget.textContent?.includes('No tensor loaded.'));
 *
 * @example
 * // With a loaded tensor, the editable expression and tensor selector are rendered for the active tensor.
 * renderTensorViewWidget(viewer.getSnapshot());
 * console.assert(tensorViewWidget.querySelector('#tensor-select'));
 * console.assert(tensorViewWidget.querySelector<HTMLTextAreaElement>('#tensor-view-input')?.value === viewer.getInspectorModel().preview);
 */
function renderTensorViewWidget(snapshot: ViewerSnapshot): void {
    if (suspendTensorViewRender) return;
    const model = viewer.getInspectorModel();
    const tab = activeTab();
    if (!model.handle) {
        tensorViewWidget.innerHTML = `${widgetTitle('tensor-view', 'Visualize tensor views, permutations, slices, or a combination of these ops.')}<div class="widget-body">No tensor loaded.</div>`;
        return;
    }

    const error = viewErrors.get(model.handle.id);
    const editor = model.viewEditor;
    if (!editor) return;
    // extensions can add domain-specific tensor-view affordances without the shell
    // learning their metadata. linear-layout uses this for original axis labels and
    // the many-to-one multi-input slider.
    const tensorViewContribution = extensions.reduce<DemoTensorViewContribution>((merged, extension) => {
        const contribution = extension.tensorView?.(extensionContext, { tab, tensorId: model.handle!.id });
        if (!contribution) return merged;
        return {
            axisLabels: merged.axisLabels ?? contribution.axisLabels,
            sliders: [...(merged.sliders ?? []), ...(contribution.sliders ?? [])],
        };
    }, {});
    const extensionSliders = tensorViewContribution.sliders ?? [];
    const tensorOptions = model.tensors.map((tensor) => `
      <option value="${escapeHtml(tensor.id)}" ${tensor.id === model.handle!.id ? 'selected' : ''}>${escapeHtml(tensor.name || tensor.id)}</option>
    `).join('');
    const sliceContent = model.viewTokens.map((token) => (
        token.kind === 'singleton'
            ? `<span class="dim-chip dim-chip-singleton">1</span>`
            : `<button class="dim-chip interactive-chip${token.sliced ? ' dim-chip-sliced dim-chip-active' : ''}" data-slice-token="${escapeHtml(token.key)}" type="button">${escapeHtml(token.token)}<span>=${token.size}</span></button>`
    )).join('');
    const originalAxisLabels = tensorViewContribution.axisLabels ?? model.handle.axisLabels;
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
        ${model.sliceTokens.length === 0 && extensionSliders.length === 0 ? '' : '<div class="slider-list" id="slice-token-controls"></div>'}
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
        /**
 * Copies the inspector model's latest tensor-view preview into the already-rendered textarea and
 * resizes that textarea so slider drags can update the expression without replacing the focused widget.
 *
 * @returns No value; mutates only `#tensor-view-input.value` and its autosized height when the textarea exists.
 * @noThrows The closure is a no-op when the Tensor View textarea is absent, and otherwise performs only a DOM value assignment followed by textarea autosizing.
 * @example
 * tensorViewWidget.innerHTML = '<textarea id="tensor-view-input">tensor[:, 0]</textarea>';
 * syncTensorViewInput();
 * console.assert(tensorViewWidget.querySelector<HTMLTextAreaElement>('#tensor-view-input')?.value === viewer.getInspectorModel().preview);
 */
        const syncTensorViewInput = (): void => {
            const tensorViewInput = tensorViewWidget.querySelector<HTMLTextAreaElement>('#tensor-view-input');
            if (!tensorViewInput) return;
            tensorViewInput.value = viewer.getInspectorModel().preview;
            autosizeTextarea(tensorViewInput);
        };
        /**
 * Applies a stock slice-slider position to the current tensor-view token, notifies extensions that
 * the tensor view changed, and refreshes the expression textarea without rerendering the active range input.
 *
 * @param nextValue - Clamped slice index selected for the current `token.key` of the active tensor.
 * @returns No value; records the new slice-token value in the viewer, invokes `afterTensorViewChange` hooks, and synchronizes the visible tensor-view expression immediately and on the next animation frame.
 * @noThrows The closure does not validate or throw for slider values itself; pointer and number-input handlers clamp values before calling it, and this body only forwards the value to the viewer and extension hooks.
 * @example
 * // Moving the slice slider for token "i" to 3 updates that token on the active tensor.
 * applyValue(3);
 * console.assert(viewer.getInspectorModel().preview.includes('3'));
 */
        const applyValue = (nextValue: number): void => {
            logUi('slice-token:update', { tensorId: model.handle!.id, token: token.token, value: nextValue });
            viewer.setSliceTokenValue(model.handle!.id, token.key, nextValue);
            extensions.forEach((extension) => {
                extension.afterTensorViewChange?.(extensionContext, model.handle!.id);
            });
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
    extensionSliders.forEach((sliderSpec) => {
        const row = document.createElement('div');
        row.className = 'slider-row';
        const sliderId = `extension-slider-${sliderSpec.id.replace(/[^a-z0-9_-]/gi, '-')}`;
        row.innerHTML = `
          <label for="${sliderId}">${escapeHtml(sliderSpec.label)}</label>
          <input id="${sliderId}" type="range" min="${sliderSpec.min}" max="${sliderSpec.max}" value="${sliderSpec.value}" />
          <input id="${sliderId}-number" type="number" min="${sliderSpec.min}" max="${sliderSpec.max}" value="${sliderSpec.value}" />
        `;
        const slider = row.querySelector<HTMLInputElement>(`#${sliderId}`);
        const number = row.querySelector<HTMLInputElement>(`#${sliderId}-number`);
        /**
 * Forwards an extension-owned tensor-view slider value to the callback supplied with that slider spec.
 *
 * @param nextValue - Clamped numeric value read from the extension slider's range or number input.
 * @returns No value; the extension's `sliderSpec.onChange` callback owns any resulting state change or rerender request.
 * @noThrows This closure only delegates to `sliderSpec.onChange`; it has no local validation branch or explicit throw, so any failure would come from the extension callback itself.
 * @example
 * const observed: number[] = [];
 * sliderSpec.onChange = (value) => observed.push(value);
 * applyValue(2);
 * console.assert(observed[0] === 2);
 */
        const applyValue = (nextValue: number): void => {
            sliderSpec.onChange(nextValue);
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
            const clamped = Math.max(sliderSpec.min, Math.min(sliderSpec.max, Number(number.value)));
            number.value = String(clamped);
            if (slider) slider.value = String(clamped);
            applyValue(clamped);
            suspendTensorViewRender = false;
            render(viewer.getSnapshot());
        });
        sliderRows.push(row);
    });
    sliceHost?.replaceChildren(...sliderRows);
}

/**
 * Rebuilds the inspector sidebar so it shows the active tensor metadata and, when a cell is hovered,
 * the hovered tensor name plus coordinate rows supplied by extensions or the hovered tensor fallback.
 *
 * @param snapshot - Viewer snapshot whose display mode and dimension-mapping scheme control how axis labels,
 * tensor coordinates, and binary coordinate tokens are formatted in the inspector.
 * @returns Nothing. The function writes the empty-tensor message or inspector metadata grid into
 * `inspectorWidget` and updates cached inspector element references for later hover refreshes.
 * @noThrows The renderer reads viewer state and writes deterministic HTML; absent tensor handles and missing
 * cached refs are handled with early returns instead of reported as caller errors.
 * @example
 * // With no loaded tensor, the inspector explains that there is nothing to inspect.
 * renderInspectorWidget({ displayMode: 'axis-labels', dimensionMappingScheme: 'z-order' } as ViewerSnapshot);
 * console.assert(inspectorWidget.textContent?.includes('No tensor loaded.'));
 *
 * @example
 * // After the viewer reports a hovered cell, the panel contains the hovered tensor name and coordinate rows.
 * renderInspectorWidget({ displayMode: 'axis-labels', dimensionMappingScheme: 'contiguous' } as ViewerSnapshot);
 * console.assert(inspectorWidget.querySelector('#inspector-hovered-tensor-value')?.textContent === 'weights');
 */
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
    /**
 * Formats one visible tensor coordinate as fixed-width binary axis tokens for the inspector's monospace row.
 *
 * @param coord - Per-axis tensor coordinate for the hovered or extension-supplied cell; `null` means the row has
 * no coordinate to display.
 * @param shape - Per-axis tensor extents used to choose each binary token width, so axis values line up with the
 * largest possible index for that axis.
 * @returns A display-mode-aware string of padded binary coordinate tokens, or an empty string when either the
 * coordinate or shape is unavailable.
 * @noThrows Missing coordinate and shape data are treated as an empty inspector value, and numeric axes are only
 * converted with `toString(2)` and padding.
 * @example
 * const text = binaryCoord([3, 1], [8, 2]);
 * console.assert(text.includes('011'));
 * console.assert(text.includes('1'));
 *
 * @example
 * console.assert(binaryCoord(null, [8, 2]) === '');
 */
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
    // inspector coordinate rows are extension-supplied when a workflow can relate
    // several tensors. otherwise the shell falls back to the hovered tensor only.
    const extensionCoordEntries = extensions.flatMap((extension) => (
        extension.inspectorCoords?.(extensionContext, { snapshot, hover, hoveredStatus }) ?? []
    ));
    const coordEntries = extensionCoordEntries.length > 0 || !hover
        ? extensionCoordEntries
        : [{
            title: hover.tensorName,
            labels: hoveredStatus?.axisLabels.slice() ?? [],
            shape: hoveredStatus?.shape.slice() ?? [],
            coord: hover.tensorCoord,
            hovered: true,
        }];
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

/**
 * Replaces the selection sidebar with the highlighted-cell count and numeric summary statistics, or with guidance
 * explaining why selection is unavailable for the current interaction mode or dimension mapping.
 *
 * @param snapshot - Viewer snapshot whose interaction mode and layout settings determine whether selection
 * statistics are enabled for the current tensor view.
 * @returns Nothing. The function writes the selection widget markup, including count, min/percentile/max/mean/std
 * fields and mode-specific help text, into `selectionWidget`.
 * @noThrows Missing tensor handles are rendered as a "No tensor loaded" message, and unavailable selection modes
 * are represented as disabled panel text rather than thrown errors.
 * @example
 * renderSelectionWidget({ interactionMode: 'pan' } as ViewerSnapshot);
 * console.assert(selectionWidget.textContent?.includes('Switch to Selection mode'));
 *
 * @example
 * renderSelectionWidget({ interactionMode: 'select', dimensionMappingScheme: 'contiguous' } as ViewerSnapshot);
 * console.assert(selectionWidget.textContent?.includes('Highlighted Cells'));
 * console.assert(selectionWidget.textContent?.includes('Mean'));
 */
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

/**
 * Builds the advanced-settings sidebar controls for tensor layout spacing, axis-family mapping, block gaps,
 * collapsed hidden axes, and signed-log color scaling, then wires each control back to the viewer.
 *
 * @param snapshot - Viewer snapshot that seeds the control values for block-gap scale, gap visibility,
 * log scaling, hidden-axis collapse, and dimension-mapping scheme.
 * @returns Nothing. The function replaces `advancedSettingsWidget` contents and attaches change handlers that
 * apply accepted values to the viewer and mirror normalized values back into the form controls.
 * @noThrows Optional controls are queried defensively and skipped when absent; the required block-gap input is
 * checked before event listeners are attached.
 * @example
 * renderAdvancedSettingsWidget({
 *   dimensionBlockGapMultiple: 2,
 *   displayGaps: true,
 *   logScale: false,
 *   collapseHiddenAxes: true,
 *   dimensionMappingScheme: 'contiguous',
 * } as ViewerSnapshot);
 * console.assert(advancedSettingsWidget.querySelector<HTMLInputElement>('#dimension-block-gap-multiple')?.value === '2');
 * console.assert(advancedSettingsWidget.querySelector<HTMLInputElement>('#display-gaps')?.checked === true);
 * console.assert(advancedSettingsWidget.querySelector<HTMLSelectElement>('#dimension-mapping-scheme')?.value === 'contiguous');
 */
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

// render cycle
/**
 * Rebuild the demo shell for a viewer snapshot and run extension render hooks.
 *
 * The full pass captures tab UI state, allows extensions to intercept rendering,
 * refreshes the sidebar, tab strip, control dock, registered widgets, and then
 * notifies `afterRender` hooks. During suspended tensor-view slider drags it
 * refreshes only the inspector and extension hover hooks so the active range
 * input keeps pointer capture.
 *
 * @param snapshot - Viewer snapshot containing the active tensor, view, hover, selection, and display state to mirror into app chrome and extension widgets.
 * @returns Nothing; callers observe updated DOM chrome, refreshed widget contents, and invoked extension lifecycle hooks.
 * @noThrows The coordinator performs no parsing or validation itself; it only forwards the already-produced snapshot to DOM renderers and optional extension callbacks.
 * @example
 * const snapshot = viewer.snapshot();
 * render(snapshot);
 * // The sidebar, tab strip, control dock, and registered widget bodies now
 * // reflect `snapshot`, unless an extension `beforeRender` hook intercepted the pass.
 */
function render(snapshot: ViewerSnapshot): void {
    if (suspendTensorViewRender) {
        // live slider drags still need hover/inspector freshness, but rebuilding
        // the full sidebar steals pointer capture from the range input.
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

// python session loading
/**
 * Validate a tensor payload asset path before it is used to fetch bundle data.
 *
 * @param dataFile - Manifest-provided payload path that must match the allowed data-file pattern and must not contain `..` traversal segments.
 * @returns The same payload path after it has passed the safety checks, suitable for constructing a fetch URL.
 * @throws Error when `dataFile` fails the allowed payload-path pattern or includes `..` path traversal.
 * @example
 * safeDataFile('tensors/layer-0.bin');
 * // => 'tensors/layer-0.bin'
 *
 * @example
 * safeDataFile('../secrets.bin');
 * // throws Error: Unsafe tensor payload path ../secrets.bin.
 */
function safeDataFile(dataFile: string): string {
    if (!DATA_FILE_PATTERN.test(dataFile) || dataFile.includes('..')) {
        throw new Error(`Unsafe tensor payload path ${dataFile}.`);
    }
    return dataFile;
}

/**
 * Resolve an app API path against the current page and append the active session token.
 *
 * @param path - Relative or absolute URL path requested by demo fetch code, such as a manifest or tensor payload endpoint.
 * @returns A same-origin pathname plus query string that callers can pass to `fetch`; when a session token is active it is included as the `token` query parameter.
 * @noThrows The browser URL constructor receives `window.location.href` as its base, so ordinary app-relative endpoint paths do not require caller-side validation here.
 * @example
 * // With the page at https://example.test/demo/ and sessionToken set to 'abc123':
 * apiUrl('/api/manifest.json');
 * // => '/api/manifest.json?token=abc123'
 */
function apiUrl(path: string): string {
    const url = new URL(path, window.location.href);
    if (sessionToken) url.searchParams.set('token', sessionToken);
    return `${url.pathname}${url.search}`;
}

/**
 * Read a fetch response into one ArrayBuffer while enforcing payload type and size limits.
 *
 * The check protects manifest and tensor-byte downloads by validating the declared
 * MIME type, honoring an exact byte count when one is known, and stopping streamed
 * reads as soon as they exceed the configured ceiling.
 *
 * @param response - Fetch `Response` whose `content-type`, optional `content-length`, and body bytes are validated before use.
 * @param options - Validation rules: required MIME `contentType`, human-readable error `label`, maximum accepted `maxBytes`, and optional exact `expectedBytes` count.
 * @returns A promise for the complete validated response body as an `ArrayBuffer` that callers can parse as manifest or tensor data.
 * @throws Error when the response MIME type differs from `options.contentType`, `content-length` is non-finite, negative, larger than the allowed limit, differs from `expectedBytes`, the streamed body exceeds the limit, or the final body length differs from `expectedBytes`.
 * @example
 * const response = new Response(new Uint8Array([1, 2, 3]), {
 *   headers: { 'content-type': 'application/octet-stream', 'content-length': '3' },
 * });
 * const buffer = await boundedArrayBuffer(response, {
 *   contentType: 'application/octet-stream',
 *   label: 'tensor payload',
 *   maxBytes: 1024,
 *   expectedBytes: 3,
 * });
 * new Uint8Array(buffer);
 * // => Uint8Array [1, 2, 3]
 *
 * @example
 * const html = new Response('<!doctype html>', {
 *   headers: { 'content-type': 'text/html' },
 * });
 * await boundedArrayBuffer(html, {
 *   contentType: 'application/octet-stream',
 *   label: 'tensor payload',
 *   maxBytes: 1024,
 * });
 * // rejects with Error: Unexpected tensor payload content type text/html.
 */
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
        // streaming responses may omit content-length, so enforce the same byte
        // ceiling while reading instead of trusting headers alone.
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

/**
 * Fetches the binary payload files referenced by a tab manifest and decodes them
 * into typed numeric arrays for viewer rendering.
 *
 * Manifest tensor entries without a `dataFile` are skipped because their bytes
 * are supplied by an extension or another loader path.
 *
 * @param tensors - Tensor manifest entries for one tab, including each tensor id,
 * dtype, shape, and optional local-session `dataFile` path.
 * @returns A map from tensor id to the decoded typed array for every manifest
 * entry that referenced a payload file.
 * @throws Error when a payload endpoint returns a non-OK response, when a
 * `dataFile` path fails the safe local-file check, or when the payload response
 * does not match the expected tensor content type or byte length.
 * @example
 * ```ts
 * const tensors = [
 *   { id: 'activations', dtype: 'float32', shape: [2, 2], dataFile: 'activations.bin' },
 * ] satisfies BundleManifest['tensors'];
 *
 * const loaded = await loadTabTensors(tensors);
 * loaded.get('activations') instanceof Float32Array; // true
 * loaded.get('activations')?.length; // 4
 * ```
 * @example
 * ```ts
 * await expect(loadTabTensors([
 *   { id: 'missing', dtype: 'float32', shape: [1], dataFile: 'missing.bin' },
 * ])).rejects.toThrow('Missing tensor payload missing.bin.');
 * ```
 */
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

/**
 * Converts one session-manifest tab into the loaded document shape used by the
 * demo shell, giving registered extensions the first chance to migrate or
 * hydrate extension-owned viewer state.
 *
 * If no extension returns a document, the fallback loader preserves the tab id,
 * title, viewer state, and tensor manifest, then fetches the tab's tensor bytes.
 *
 * @param tab - One tab object from `session.json`, including its id, title,
 * serialized viewer state, and tensor manifest entries.
 * @returns The loaded bundle document that can be stored in `sessionTabs` and
 * later activated by `loadTab`.
 * @throws Error when an extension loader rejects or when the fallback tensor
 * payload loading fails for an unsafe path, missing payload, or invalid payload
 * response.
 * @example
 * ```ts
 * const tab = {
 *   id: 'tab-1',
 *   title: 'Python session',
 *   viewer: { dimensionMappingScheme: 'rows' },
 *   tensors: [],
 * } satisfies SessionBundleManifest['tabs'][number];
 *
 * const loaded = await loadSessionTab(tab);
 * loaded.id; // 'tab-1'
 * loaded.title; // 'Python session'
 * loaded.tensors.size; // 0
 * ```
 */
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

/**
 * Attempts to restore the browser demo from the local Python server's
 * `/api/session.json` manifest and its referenced tensor payload endpoints.
 *
 * A missing session manifest is not an error; startup callers use the `false`
 * result to continue with baked fallback tabs or the generated sample tensor.
 *
 * @returns `true` after a valid session manifest has been loaded into
 * `sessionTabs` and the first tab has been activated; `false` when the session
 * manifest endpoint is absent or returns a non-OK response.
 * @throws Error when the manifest has an unsupported version, `tabs` is not an
 * array, the tab or tensor count exceeds viewer limits, a tab's `tensors` field
 * is not an array, tensor payload byte totals exceed the session limit, or a
 * referenced tensor file fails path, fetch, content-type, or size validation.
 * @example
 * ```ts
 * const loaded = await tryLoadSession();
 * if (!loaded) {
 *   seedDemoTensor();
 * }
 * ```
 * @example
 * ```ts
 * // Given /api/session.json responds with { "version": 2, "tabs": [] }:
 * await expect(tryLoadSession()).rejects.toThrow('Unsupported session version 2.');
 * ```
 */
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
    // validate every tab before fetching any payload so one malicious manifest
    // cannot partially hydrate tensors before size/path checks fail.
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

/**
 * Resets the demo out of session-tab mode and adds a deterministic 4×4×4
 * `Float64Array` tensor named `Sample` so the viewer has data to display when
 * no saved or fallback session is available.
 *
 * Existing tabs are offered to extension cleanup hooks before the tab list and
 * active tab id are cleared.
 *
 * @returns Nothing; the function clears `sessionTabs`, resets `activeTabId`, and
 * mutates the viewer by adding the generated sample tensor.
 * @noThrows The function performs only synchronous in-memory cleanup and sample
 * generation; under normal extension contracts, `clearTab` hooks are cleanup
 * callbacks and are not expected to reject or report recoverable errors.
 * @example
 * ```ts
 * seedDemoTensor();
 *
 * sessionTabs.length; // 0
 * activeTabId; // null
 * viewer.getSnapshot().tensors.some((tensor) => tensor.name === 'Sample'); // true
 * ```
 */
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

/**
 * Starts a browser download for serialized SVG markup by wrapping it in an
 * `image/svg+xml` Blob, assigning the generated object URL to a temporary
 * anchor, clicking that anchor, and then revoking the object URL.
 *
 * @param filename - Name assigned to the downloaded file through the anchor's
 * `download` attribute, typically the sanitized active-tab title with an
 * `.svg` extension.
 * @param svg - Complete serialized SVG document to write into the downloaded
 * file.
 * @returns Nothing; the observable effect is the browser download request.
 * @noThrows Creates a Blob URL and temporary anchor from already-generated SVG markup without validating user input.
 * @example
 * ```ts
 * const clickedLinks: Array<{ download: string; href: string }> = [];
 * const originalCreateElement = document.createElement.bind(document);
 * document.createElement = ((tagName: string) => {
 *   const element = originalCreateElement(tagName);
 *   if (tagName === 'a') {
 *     element.click = () => clickedLinks.push({
 *       download: (element as HTMLAnchorElement).download,
 *       href: (element as HTMLAnchorElement).href,
 *     });
 *   }
 *   return element;
 * }) as typeof document.createElement;
 *
 * downloadSvg('attention-head.svg', '<svg viewBox="0 0 1 1"></svg>');
 *
 * console.assert(clickedLinks[0].download === 'attention-head.svg');
 * console.assert(clickedLinks[0].href.startsWith('blob:'));
 * document.createElement = originalCreateElement;
 * ```
 */
function downloadSvg(filename: string, svg: string): void {
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
}

/**
 * Builds the default SVG export filename from the active tab title by trimming
 * whitespace, lowercasing it, replacing non-alphanumeric runs with hyphens, and
 * falling back to `tensor-viz.svg` when there is no usable title.
 *
 * @returns A filesystem-friendly `.svg` filename for the current export action.
 * @noThrows Reads the active tab title and applies string normalization only; missing or punctuation-only titles use the fallback filename.
 * @example
 * ```ts
 * // When the active tab title is "Attention Head #3":
 * console.assert(svgFilename() === 'attention-head-3.svg');
 *
 * // When there is no active tab, or the title contains no letters or digits:
 * console.assert(svgFilename() === 'tensor-viz.svg');
 * ```
 */
function svgFilename(): string {
    const title = activeTab()?.title ?? 'tensor-viz';
    const base = title.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    return `${base || 'tensor-viz'}.svg`;
}

/**
 * Serializes the currently displayed viewer contents as SVG for the Save SVG
 * command, using the viewer's 2D SVG snapshot path in 2D mode and the current
 * view export path for other display modes.
 *
 * @returns Promise resolving to the complete SVG markup that will be written to
 * the downloaded `.svg` file.
 * @noThrows Chooses the appropriate viewer SVG export path for the current display mode and returns that export promise to the caller.
 * @example
 * ```ts
 * const svg = await currentSvgDocument();
 *
 * console.assert(svg.startsWith('<svg'));
 * console.assert(svg.includes('</svg>'));
 * ```
 */
async function currentSvgDocument(): Promise<string> {
    if (viewer.getSnapshot().displayMode !== '2d') return viewer.exportCurrentViewSvg();
    return viewer.saveSvg().text();
}

/**
 * Gives registered demo extensions a chance to create their built-in fallback
 * tabs during startup when restoring a saved session did not populate the app.
 * The first extension whose `loadFallback` hook resolves to `true` stops the
 * search.
 *
 * @returns Promise resolving to `true` when an extension loaded fallback tabs;
 * otherwise `false`, allowing startup to seed the generic demo tensor.
 * @noThrows Iterates registered extension fallback hooks and returns a boolean startup decision; extensions own their own failure handling.
 * @example
 * ```ts
 * const loaded = await loadFallbackTabs();
 *
 * if (loaded) {
 *   console.assert(activeTab() !== undefined);
 * } else {
 *   seedDemoTensor();
 * }
 * ```
 */
async function loadFallbackTabs(): Promise<boolean> {
    for (const extension of extensions) {
        if (await extension.loadFallback?.(extensionContext)) return true;
    }
    return false;
}

// command execution and global events
/**
 * Dispatches a command-palette, menu, keyboard-shortcut, or tab-selection action into the demo shell.
 *
 * Recognized ids update viewer display options, toggle shell widgets, focus the tensor-view editor,
 * download the current SVG, add or close tabs, or load a tab when the id has the `tab:<tabId>` form.
 * Unknown action ids are logged and otherwise ignored after the command palette is closed.
 *
 * @param action - Command id from a `CommandAction.action`, a menu button `data-action`, a keyboard shortcut, or a `tab:<sessionTab.id>` entry.
 * @returns Promise that settles after any asynchronous side effect for the command, such as loading a tab, saving the SVG, or adding/closing a tab, has completed.
 * @noThrows The dispatcher performs no validation that throws for unrecognized ids; ids that do not match a switch case simply leave the viewer unchanged after logging and closing the palette.
 * @example
 * await runAction('3d');
 * // The active viewer is switched to 3D display mode.
 *
 * @example
 * await runAction(`tab:${sessionTabs[0].id}`);
 * // The tab whose id matches `sessionTabs[0].id` is loaded.
 *
 * @example
 * await runAction('not-a-command');
 * // No command-specific viewer setting is changed; the action is only logged and the command palette is closed.
 */
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
}
