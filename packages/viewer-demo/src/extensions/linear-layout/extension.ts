import type {
    LoadedBundleDocument,
    TensorViewSnapshot,
    ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import type {
    DemoAppExtension,
    DemoExtensionContext,
    DemoExtensionFactory,
    DemoWidgetSpec,
    LoadedSessionTab,
} from '../../app-extension.js';
import type { AppShellWidgetSlot } from '../../app-shell.js';
import { controlIcons, type ControlSpec } from '../../control-dock.js';
import { escapeInfo } from '../../app-format.js';
import {
    composeLayoutStateFromLegacySpec,
    createComposeLayoutDocument,
    isComposeLayoutMeta,
    type ComposeLayoutMeta,
} from './linear-layout.js';
import {
    applyLinearLayoutCellText,
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
    renderLinearLayoutEditorWidgets,
    renderLinearLayoutColorWidget,
    renderLinearLayoutVisibleTensorsWidget,
    renderLinearLayoutWidget,
    renderLinearLayoutPresetWidget,
    snapshotTensorViews,
    syncLinearLayoutCellTextState,
    syncLinearLayoutMultiInputState,
    syncLinearLayoutSelection,
    syncLinearLayoutSelectionPreview,
    syncLinearLayoutState,
    syncLinearLayoutViewFilters,
    toggleLinearLayoutPropagateOutputs,
    type LinearLayoutCellTextState,
    type LinearLayoutFormState,
    type LinearLayoutMultiInputState,
    type LinearLayoutSelectionMap,
    type LinearLayoutTensorViewsState,
    type LinearLayoutUiContext,
    type LinearLayoutUiState,
} from './linear-layout-ui.js';
import { linearLayoutPropagateOutputsInfo } from './widgets/linear-layout-color-widget.js';

// extension.ts is the bridge between the generic demo shell and the
// linear-layout workflow.
// model files parse layouts and compute tensor data; widget files render forms.
// this file wires those pieces into the host extension lifecycle: widgets,
// tab creation, session load/save, hover/selection synchronization, and control
// dock commands.
// keep new linear-layout behavior data-driven in model/widget files whenever
// possible.  Code added here should usually mean the host application needs a
// new lifecycle hook or a new connection between existing lifecycle hooks.
// tab-local maps below mirror the host tab ids because a single app session can
// switch between ordinary tensor tabs and compose-layout tabs without tearing
// down the extension.
// hover, selection, and tensor-view hooks all call back into state/viewer-sync
// modules so this lifecycle file does not duplicate mapping math.
// session load supports both current compose-layout snapshots and older
// linearLayoutSpec snapshots; remove neither path without a migration.
// widget ids must stay aligned with LINEAR_LAYOUT_WIDGET_SLOTS or the app shell
// cannot hand the extension real DOM hosts during startup.
// controls are optional host commands, while widgets are always declared here.
// use runtime.widgets for re-render loops so future widget additions do not
// need another hard-coded render list.

/**
 * shape of linear layout extension runtime data used by the viewer.
 */
export type LinearLayoutExtensionRuntime = DemoAppExtension & {
    state: LinearLayoutUiState;
    ui: LinearLayoutUiContext;
    isTab: (tab: LoadedBundleDocument | undefined) => boolean;
};

const LINEAR_LAYOUT_WIDGETS = [
    'linear-layout-preset',
    'linear-layout',
    'linear-layout-visible-tensors',
    'linear-layout-color',
    'cell-text',
] as const;

export const LINEAR_LAYOUT_WIDGET_SLOTS = [
    { id: 'linear-layout-preset', beforeHeader: true },
    { id: 'linear-layout', beforeHeader: true },
    { id: 'linear-layout-visible-tensors', beforeHeader: true },
    { id: 'linear-layout-color', beforeHeader: true },
    { id: 'cell-text', beforeHeader: true },
] satisfies AppShellWidgetSlot[];

/**
 * return require widget for the current viewer state.
 */
function requireWidget(ctx: DemoExtensionContext, widgetId: typeof LINEAR_LAYOUT_WIDGETS[number]): HTMLElement {
    const widget = ctx.widgets[widgetId];
    if (!widget) throw new Error(`Missing ${widgetId} widget.`);
    return widget;
}

/**
 * return linear layout widget icon for the current viewer state.
 */
function linearLayoutWidgetIcon(widgetId: string): string {
    switch (widgetId) {
        case 'linear-layout-preset':
            return `
              <svg viewBox="0 0 24 24">
                <path d="M5 7h14M5 12h14M5 17h9" />
                <path d="M16.5 15.5l2 2 3.5-4" />
              </svg>
            `;
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
              <svg viewBox="0 0 200 200">
                <defs>
                  <linearGradient id="cell-color-widget-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color: #ff0000;" />
                    <stop offset="20%" style="stop-color: #ffff00;" />
                    <stop offset="50%" style="stop-color: #00ff00;" />
                    <stop offset="75%" style="stop-color: #00ffff;" />
                    <stop offset="100%" style="stop-color: #0000ff;" />
                  </linearGradient>
                </defs>
                <rect x="10" y="10" width="180" height="180" style="fill: url(#cell-color-widget-gradient); stroke: #000000; stroke-width: 8;" />
                <text x="100" y="145" text-anchor="middle" style="fill: #000000; stroke: none; font-family: sans-serif; font-size: 140px; font-weight: 700;">T</text>
              </svg>
            `;
        case 'cell-text':
            return `
              <svg viewBox="0 0 24 24">
                <rect x="2.5" y="4" width="19" height="16" style="fill: #ffffff; stroke: #111827; stroke-width: 1.5;" />
                <text x="12" y="14.2" text-anchor="middle" dominant-baseline="middle" style="fill: #111827; stroke: none; font: 700 7px 'IBM Plex Mono', monospace;">T:0</text>
              </svg>
            `;
        default:
            return '';
    }
}

/**
 * return linear layout widgets for the current viewer state.
 */
function linearLayoutWidgets(ui: LinearLayoutUiContext): DemoWidgetSpec[] {
    /** return whether a host tab should show linear-layout widgets. */
    const active = (ctx: DemoExtensionContext): boolean => {
        const tab = ctx.getActiveTab();
        return Boolean(tab && isLinearLayoutTab(tab));
    };
    return [
        {
            id: 'linear-layout-preset',
            label: 'Preset',
            icon: linearLayoutWidgetIcon('linear-layout-preset'),
            defaultCollapsed: false,
            visible: active,
            render: () => { renderLinearLayoutPresetWidget(ui); },
        },
        {
            id: 'linear-layout',
            label: 'Linear Layout Specifications',
            icon: linearLayoutWidgetIcon('linear-layout'),
            defaultCollapsed: false,
            visible: active,
            render: () => { renderLinearLayoutWidget(ui); },
        },
        {
            id: 'linear-layout-visible-tensors',
            label: 'Visible Tensors',
            icon: linearLayoutWidgetIcon('linear-layout-visible-tensors'),
            defaultCollapsed: true,
            visible: active,
            render: () => { renderLinearLayoutVisibleTensorsWidget(ui); },
        },
        {
            id: 'linear-layout-color',
            label: 'Cell Color/Text',
            icon: linearLayoutWidgetIcon('linear-layout-color'),
            defaultCollapsed: true,
            visible: active,
            render: () => { renderLinearLayoutColorWidget(ui); },
        },
        {
            id: 'cell-text',
            label: 'Cell Text',
            icon: linearLayoutWidgetIcon('cell-text'),
            defaultCollapsed: true,
            visible: active,
            render: () => { renderCellTextWidget(ui); },
        },
    ];
}

/**
 * create linear layout extension for the current viewer state.
 */
export function createLinearLayoutExtension(ctx: DemoExtensionContext): LinearLayoutExtensionRuntime {
    const hoverPopup = document.createElement('div');
    hoverPopup.className = 'linear-layout-hover-popup hidden';
    ctx.viewport.appendChild(hoverPopup);
    // popup placement is tracked in viewport-local pixels so scrolling the page
    // does not move the popup away from the hovered canvas cell.
    let hoverPopupPointer = { x: 16, y: 16 };
    let lastActiveTensorId: string | null = null;
    const state: LinearLayoutUiState = {
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
    const ui: LinearLayoutUiContext = {
        viewer: ctx.viewer,
        viewport: ctx.viewport,
        linearLayoutPresetWidget: requireWidget(ctx, 'linear-layout-preset'),
        linearLayoutWidget: requireWidget(ctx, 'linear-layout'),
        linearLayoutVisibleTensorsWidget: requireWidget(ctx, 'linear-layout-visible-tensors'),
        cellTextWidget: requireWidget(ctx, 'cell-text'),
        linearLayoutColorWidget: requireWidget(ctx, 'linear-layout-color'),
        state,
        widgetTitle: ctx.widgetTitle,
        getActiveTab: ctx.getActiveTab,
        getActiveTabId: ctx.getActiveTabId,
        getSessionTabs: ctx.getSessionTabs,
        setSessionTabs: ctx.setSessionTabs,
        loadTab: ctx.loadTab,
        renderLinearLayoutEditorWidgets: () => { renderLinearLayoutEditorWidgets(ui); },
    };
    /** render the hover popup from the current viewer hover and selection map. */
    const renderHoverPopup = (): void => {
        const tab = ctx.getActiveTab();
        const linearLayoutTab = tab && isLinearLayoutTab(tab) ? tab : null;
        const hover = ctx.viewer.getLiveHover();
        const selectionMap = linearLayoutTab ? linearLayoutSelectionMapForTab(ui, linearLayoutTab) : null;
        const entries = linearLayoutHoverPopupEntries(ui, hover, selectionMap);
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
    };
    /** keep the hover popup inside the viewport even near the bottom/right edge. */
    const placeHoverPopup = (): void => {
        if (hoverPopup.classList.contains('hidden')) return;
        const rect = ctx.viewport.getBoundingClientRect();
        const width = hoverPopup.offsetWidth;
        const height = hoverPopup.offsetHeight;
        const maxLeft = Math.max(12, rect.width - width - 12);
        const maxTop = Math.max(12, rect.height - height - 12);
        hoverPopup.style.left = `${Math.min(maxLeft, hoverPopupPointer.x + 18)}px`;
        hoverPopup.style.top = `${Math.min(maxTop, hoverPopupPointer.y + 18)}px`;
    };
    const runtime: LinearLayoutExtensionRuntime = {
        id: 'linear-layout',
        widgets: linearLayoutWidgets(ui),
        state,
        ui,
        isTab: (tab) => Boolean(tab && isLinearLayoutTab(tab)),
        tensorView: (_tensorViewCtx, { tab, tensorId }) => {
            if (!tab || !isLinearLayoutTab(tab)) return null;
            const meta = composeLayoutMetaForTab(tab);
            const selectionMap = linearLayoutSelectionMapForTab(ui, tab);
            const multiInput = selectionMap ? linearLayoutMultiInputModel(ui, selectionMap) : null;
            const axisLabels = meta?.tensors.find((tensor) => tensor.id === tensorId)?.axisLabels;
            // multi-input sliders appear only for focused non-injective cells;
            // the model returns null for ordinary one-root cells.
            return {
                axisLabels,
                sliders: multiInput ? [{
                    id: 'linear-layout-multi-input',
                    label: 'Multi-Input',
                    min: -1,
                    max: Math.max(0, multiInput.size - 1),
                    value: multiInput.value,
                    onChange: (value) => {
                        state.linearLayoutMultiInputState[multiInput.focusedTensorId] = value;
                        const activeTabId = ctx.getActiveTabId();
                        if (activeTabId) state.linearLayoutMultiInputStates.set(activeTabId, cloneLinearLayoutMultiInputState(state.linearLayoutMultiInputState));
                        syncLinearLayoutViewFilters(ui);
                    },
                }] : [],
            };
        },
        afterTensorViewChange: () => { syncLinearLayoutViewFilters(ui); },
        inspectorCoords: (_inspectorCtx, { hover, hoveredStatus }) => {
            const tab = ctx.getActiveTab();
            const linearLayoutTab = tab && isLinearLayoutTab(tab) ? tab : null;
            if (!linearLayoutTab) return [];
            return inspectorCoordEntries(ui, hover, hoveredStatus, linearLayoutSelectionMapForTab(ui, linearLayoutTab));
        },
        controls: (controlCtx, snapshot): ControlSpec[] => {
            const tab = controlCtx.getActiveTab();
            const active = Boolean(tab && isLinearLayoutTab(tab));
            const injective = tab && isLinearLayoutTab(tab)
                ? (composeLayoutMetaForTab(tab)?.injective ?? true)
                : true;
            return [{
                id: 'propagate-outputs',
                label: 'Propagate Outputs',
                description: active
                    ? linearLayoutPropagateOutputsInfo(injective)
                    : 'Propagate Outputs is available for linear-layout tabs.',
                shortcut: 'N/A',
                active: state.linearLayoutState.propagateOutputs,
                disabled: !active,
                content: controlIcons.propagateOutputs,
                onClick: async () => {
                    await toggleLinearLayoutPropagateOutputs(ui);
                },
            }];
        },
        createTab: (_tabCtx, id, title, snapshot) => {
            state.linearLayoutState = emptyLinearLayoutState();
            const document = createComposeLayoutDocument(state.linearLayoutState, snapshot, title);
            const meta = composeLayoutMetaForTab(document);
            // new tabs snapshot the viewer immediately so later tensor-view
            // edits can be restored when switching away and back.
            state.linearLayoutCellTextState = defaultLinearLayoutCellTextState(meta?.rootInputLabels ?? []);
            state.linearLayoutMultiInputState = defaultLinearLayoutMultiInputState();
            state.linearLayoutStates.set(id, cloneLinearLayoutState(state.linearLayoutState));
            state.linearLayoutCellTextStates.set(id, cloneLinearLayoutCellTextState(state.linearLayoutCellTextState));
            state.linearLayoutMultiInputStates.set(id, cloneLinearLayoutMultiInputState(state.linearLayoutMultiInputState));
            state.linearLayoutTensorViewsStates.set(id, snapshotTensorViews(document.manifest.viewer));
            return { ...document, id, title };
        },
        captureSnapshot: (_tabCtx, tab, snapshot) => {
            if (!isLinearLayoutTab(tab)) return;
            const extendedSnapshot = snapshot as ViewerSnapshot & {
                composeLayoutMeta?: ComposeLayoutMeta;
                composeLayoutState?: LinearLayoutFormState;
                linearLayoutCellTextState?: LinearLayoutCellTextState;
                linearLayoutMultiInputState?: LinearLayoutMultiInputState;
                composeLayoutTensorViews?: LinearLayoutTensorViewsState;
            };
            const cloned = cloneLinearLayoutState(state.linearLayoutState);
            const clonedCellText = cloneLinearLayoutCellTextState(state.linearLayoutCellTextState);
            const clonedMultiInput = cloneLinearLayoutMultiInputState(state.linearLayoutMultiInputState);
            const tensorViews = preservedLinearLayoutTensorViews(ui, tab.id);
            // write both tab-local caches and the serialized snapshot because a
            // save can be followed by either in-session tab switching or reload.
            state.linearLayoutStates.set(tab.id, cloned);
            state.linearLayoutCellTextStates.set(tab.id, clonedCellText);
            state.linearLayoutMultiInputStates.set(tab.id, clonedMultiInput);
            state.linearLayoutTensorViewsStates.set(tab.id, tensorViews);
            extendedSnapshot.composeLayoutState = cloned;
            extendedSnapshot.linearLayoutCellTextState = clonedCellText;
            extendedSnapshot.linearLayoutMultiInputState = clonedMultiInput;
            extendedSnapshot.composeLayoutTensorViews = cloneLinearLayoutTensorViewsState(tensorViews);
            const composeLayoutMeta = composeLayoutMetaForTab(tab);
            if (composeLayoutMeta) extendedSnapshot.composeLayoutMeta = composeLayoutMeta;
        },
        clearTab: (_tabCtx, tabId) => {
            state.linearLayoutStates.delete(tabId);
            state.linearLayoutCellTextStates.delete(tabId);
            state.linearLayoutMultiInputStates.delete(tabId);
            state.linearLayoutTensorViewsStates.delete(tabId);
            state.linearLayoutSelectionMaps.delete(tabId);
        },
        cloneTab: (_tabCtx, fromTabId, toTabId) => {
            const linearLayoutState = state.linearLayoutStates.get(fromTabId);
            if (linearLayoutState) state.linearLayoutStates.set(toTabId, cloneLinearLayoutState(linearLayoutState));
            const cellTextState = state.linearLayoutCellTextStates.get(fromTabId);
            if (cellTextState) state.linearLayoutCellTextStates.set(toTabId, cloneLinearLayoutCellTextState(cellTextState));
            const multiInputState = state.linearLayoutMultiInputStates.get(fromTabId);
            if (multiInputState) state.linearLayoutMultiInputStates.set(toTabId, cloneLinearLayoutMultiInputState(multiInputState));
            const tensorViewsState = state.linearLayoutTensorViewsStates.get(fromTabId);
            if (tensorViewsState) state.linearLayoutTensorViewsStates.set(toTabId, cloneLinearLayoutTensorViewsState(tensorViewsState));
            state.linearLayoutSelectionMaps.delete(toTabId);
        },
        beforeSessionLoad: () => {
            state.linearLayoutMultiInputStates.clear();
            state.linearLayoutSelectionMaps.clear();
        },
        loadSessionTab: async (tabCtx, tab: LoadedSessionTab) => {
            const legacySpec = (tab.viewer as { linearLayoutSpec?: unknown }).linearLayoutSpec;
            const storedComposeState = (tab.viewer as { composeLayoutState?: unknown }).composeLayoutState;
            const storedTensorViews = (tab.viewer as { composeLayoutTensorViews?: unknown }).composeLayoutTensorViews;
            const composeMeta = (tab.viewer as { composeLayoutMeta?: unknown }).composeLayoutMeta;
            const storedMultiInputState = (tab.viewer as { linearLayoutMultiInputState?: unknown }).linearLayoutMultiInputState;
            if (legacySpec) {
                // older demos stored one linearLayoutSpec field instead of the
                // compose-layout state object; keep that path so saved examples
                // and external links remain loadable.
                const linearLayoutState = isLinearLayoutState(storedComposeState)
                    ? cloneLinearLayoutState(storedComposeState)
                    : composeLayoutStateFromLegacySpec(legacySpec, tab.title);
                const document = createComposeLayoutDocument(linearLayoutState, {
                    ...tab.viewer,
                    showSelectionPanel: false,
                }, tab.title);
                state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(linearLayoutState));
                if (storedTensorViews && typeof storedTensorViews === 'object') {
                    state.linearLayoutTensorViewsStates.set(tab.id, cloneLinearLayoutTensorViewsState(storedTensorViews as Record<string, TensorViewSnapshot>));
                } else {
                    state.linearLayoutTensorViewsStates.set(tab.id, snapshotTensorViews(document.manifest.viewer));
                }
                if (isLinearLayoutMultiInputState(storedMultiInputState)) {
                    state.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(storedMultiInputState));
                }
                return { ...document, id: tab.id, title: tab.title };
            }
            const isLinearLayout = isComposeLayoutMeta(composeMeta);
            if (!isLinearLayout) return null;
            // loaded compose-layout tabs intentionally hide the generic
            // selection panel because selection is mirrored across all tensors.
            const viewerState = {
                ...tab.viewer,
                dimensionMappingScheme: tab.viewer.dimensionMappingScheme ?? 'contiguous',
                showSelectionPanel: false,
            };
            const storedLinearLayoutState = (viewerState as { composeLayoutState?: unknown }).composeLayoutState;
            if (isLinearLayoutState(storedLinearLayoutState)) {
                state.linearLayoutStates.set(tab.id, cloneLinearLayoutState(storedLinearLayoutState));
            }
            if (storedTensorViews && typeof storedTensorViews === 'object') {
                state.linearLayoutTensorViewsStates.set(tab.id, cloneLinearLayoutTensorViewsState(storedTensorViews as LinearLayoutTensorViewsState));
            } else {
                state.linearLayoutTensorViewsStates.set(tab.id, snapshotTensorViews(viewerState));
            }
            const storedCellTextState = (viewerState as { linearLayoutCellTextState?: unknown }).linearLayoutCellTextState;
            if (isLinearLayoutCellTextState(storedCellTextState)) {
                state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(storedCellTextState));
            }
            if (isLinearLayoutMultiInputState(storedMultiInputState)) {
                state.linearLayoutMultiInputStates.set(tab.id, cloneLinearLayoutMultiInputState(storedMultiInputState));
            }
            return {
                id: tab.id,
                title: tab.title,
                manifest: { version: 1, viewer: viewerState, tensors: tab.tensors },
                tensors: await tabCtx.loadTabTensors(tab.tensors),
            };
        },
        afterLoadTab: (_tabCtx, tab) => {
            // tab load is the single place where form state, cell text,
            // multi-input sliders, and viewer filters are all rehydrated.
            syncLinearLayoutState(ui, tab);
            syncLinearLayoutCellTextState(ui, tab);
            syncLinearLayoutMultiInputState(ui, tab);
            runtime.widgets.forEach((widget) => widget.render(ctx, ctx.viewer.getSnapshot()));
            syncLinearLayoutViewFilters(ui);
            applyLinearLayoutCellText(ui);
            syncLinearLayoutSelectionPreview(ui, new Map());
        },
        beforeRender: (_renderCtx, snapshot) => {
            const tab = ctx.getActiveTab();
            const activeTensorId = tab && isLinearLayoutTab(tab) ? (snapshot.activeTensorId ?? null) : null;
            if (activeTensorId === lastActiveTensorId) return false;
            lastActiveTensorId = activeTensorId;
            if (!activeTensorId) return false;
            // active tensor changes can expose a different multi-input slider
            // without changing the underlying layout document.
            syncLinearLayoutViewFilters(ui);
            return true;
        },
        afterRender: () => {
            renderHoverPopup();
        },
        loadFallback: async () => loadBakedLinearLayoutTabs(ui),
        pointerMove: (_pointerCtx, event) => {
            const rect = ctx.viewport.getBoundingClientRect();
            hoverPopupPointer = {
                x: Math.max(12, event.clientX - rect.left),
                y: Math.max(12, event.clientY - rect.top),
            };
            placeHoverPopup();
        },
        pointerLeave: () => {
            hoverPopup.classList.add('hidden');
        },
        hover: () => { renderHoverPopup(); },
        selectionPreview: (_selectionCtx, selection) => {
            syncLinearLayoutSelectionPreview(ui, selection);
        },
        selection: (_selectionCtx, selection) => {
            syncLinearLayoutSelection(ui, selection);
        },
    };
    return runtime;
}

export const linearLayoutExtensionFactory = {
    widgetSlots: LINEAR_LAYOUT_WIDGET_SLOTS,
    create: createLinearLayoutExtension,
} satisfies DemoExtensionFactory;
