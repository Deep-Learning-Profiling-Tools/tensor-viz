import type {
    BundleManifest,
    LoadedBundleDocument,
    HoverInfo,
    NumericArray,
    SelectionCoords,
    SessionBundleManifest,
    TensorStatus,
    TensorViewer,
    ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import type { AppShellWidgetSlot } from './app-shell.js';
import type { ControlSpec } from './control-dock.js';

/**
 * one searchable command palette action contributed by the shell or an extension.
 *
 * @example
 * const saveSvgAction: CommandAction = {
 *     action: 'save-svg',
 *     label: 'Save SVG',
 *     shortcut: 'Ctrl+S',
 *     keywords: 'export download vector image',
 * };
 * // The command palette displays "Save SVG", matches searches for "download",
 * // and passes "save-svg" to the action dispatcher when selected.
 */
export type CommandAction = {
    action: string;
    label: string;
    shortcut: string;
    keywords: string;
};

/**
 * one sidebar widget contract rendered by the shared demo shell.
 *
 * widgets are deliberately data-driven: the shell owns collapse, drag, and
 * visibility plumbing, while each widget owns only its rendered body and event
 * bindings. for example, the core tensor-view widget and the linear-layout
 * preset widget both use this shape even though they live in different modules.
 *
 * @example
 * const tensorStatsWidget: DemoWidgetSpec = {
 *     id: 'tensor-stats',
 *     label: 'Tensor Stats',
 *     icon: 'Σ',
 *     defaultCollapsed: false,
 *     visible: (_ctx, snapshot) => snapshot.tensors.length > 0,
 *     render: (ctx, snapshot) => {
 *         ctx.widgets['tensor-stats'].textContent = `${snapshot.tensors.length} tensors loaded`;
 *     },
 * };
 * // The shell creates a sidebar panel labeled "Tensor Stats" and asks the widget
 * // to render only while the current viewer snapshot contains tensors.
 */
export type DemoWidgetSpec = {
    id: string;
    label: string;
    icon: string;
    defaultCollapsed: boolean;
    visible: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => boolean;
    render: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => void;
};

/**
 * extra controls an extension can place under the core tensor-view slice sliders.
 *
 * @example
 * let selectedHead = 0;
 * const attentionHeadSlider: DemoTensorViewSliderSpec = {
 *     id: 'attention-head',
 *     label: 'Attention head',
 *     min: 0,
 *     max: 7,
 *     value: selectedHead,
 *     onChange: (value) => {
 *         selectedHead = value;
 *     },
 * };
 * attentionHeadSlider.onChange(3);
 * // selectedHead is now 3, and the extension can re-render its tensor-view contribution for head 3.
 */
export type DemoTensorViewSliderSpec = {
    id: string;
    label: string;
    min: number;
    max: number;
    value: number;
    onChange: (value: number) => void;
};

/**
 * tensor-view metadata supplied by an extension for one active tensor.
 *
 * the linear-layout extension uses this to restore original axis labels and add
 * a multi-input slider. a future sparsity extension could contribute a mask
 * slider through the same shape without changing the core tensor-view widget.
 *
 * @example
 * const contribution: DemoTensorViewContribution = {
 *   axisLabels: ['batch', 'row', 'col'],
 *   sliders: [{
 *     id: 'linear-layout-input',
 *     label: 'Input tensor',
 *     min: 0,
 *     max: 2,
 *     value: 1,
 *     step: 1,
 *   }],
 * };
 *
 * // The tensor-view panel can display the original axis names and render one
 * // extension-provided slider for choosing the visible linear-layout input.
 * contribution.axisLabels?.join(' / '); // 'batch / row / col'
 * contribution.sliders?.[0]?.label; // 'Input tensor'
 */
export type DemoTensorViewContribution = {
    axisLabels?: readonly string[];
    sliders?: DemoTensorViewSliderSpec[];
};

/**
 * one row in the hover inspector's coordinate list.
 *
 * @example
 * const row: DemoInspectorCoordEntry = {
 *   title: 'Output activation',
 *   labels: ['batch', 'token', 'channel'],
 *   shape: [1, 128, 64],
 *   coord: [0, 12, 7],
 *   hovered: true,
 * };
 *
 * // The inspector can render the coordinate as a labeled location in the
 * // hovered tensor.
 * row.labels.map((label, index) => `${label}=${row.coord?.[index]}`).join(', ');
 * // 'batch=0, token=12, channel=7'
 */
export type DemoInspectorCoordEntry = {
    title: string;
    labels: string[];
    shape: number[];
    coord: number[] | null;
    hovered: boolean;
};

/**
 * host services exposed to demo extensions.
 *
 * extensions use this object instead of importing app-entry internals. the
 * linear-layout extension uses `loadTabTensors` for baked examples and `render`
 * after widget edits; a future profiler extension could use the same active-tab
 * and widget services without adding new app-entry branches.
 *
 * @example
 * const sidebar = document.createElement('section');
 * const context: DemoExtensionContext = {
 *   viewer: viewerInstance,
 *   viewport: document.createElement('canvas'),
 *   widgets: { 'linear-layout-controls': sidebar },
 *   widgetTitle: (widgetId, info) => `${widgetId}: ${info}`,
 *   getActiveTab: () => loadedTab,
 *   getActiveTabId: () => 'linear-layout-demo',
 *   getSessionTabs: () => [loadedTab],
 *   setSessionTabs: tabs => { sessionTabs = tabs; },
 *   loadTab: async id => { activeTabId = id; },
 *   loadTabTensors: async tensors => new Map([['weights', weightsArray]]),
 *   render: () => { renderCount += 1; },
 * };
 *
 * context.render();
 * renderCount; // 1
 * context.widgetTitle('linear-layout-controls', '2 tensors');
 * // 'linear-layout-controls: 2 tensors'
 */
export type DemoExtensionContext = {
    viewer: TensorViewer;
    viewport: HTMLElement;
    widgets: Record<string, HTMLElement>;
    widgetTitle: (widgetId: string, info: string) => string;
    getActiveTab: () => LoadedBundleDocument | undefined;
    getActiveTabId: () => string | null;
    getSessionTabs: () => LoadedBundleDocument[];
    setSessionTabs: (tabs: LoadedBundleDocument[]) => void;
    loadTab: (id: string) => Promise<void>;
    loadTabTensors: (tensors: BundleManifest['tensors']) => Promise<Map<string, NumericArray>>;
    render: () => void;
};

/**
 * raw session-tab shape before an extension normalizes it into a loaded document.
 *
 * @example
 * const savedTab = {
 *   id: 'linear-layout-demo',
 *   title: 'Linear layout preset',
 *   viewer: {
 *     expression: 'output[row, col]',
 *     linearLayoutSpec: { presetId: 'matmul-basic' },
 *   },
 * } as LoadedSessionTab;
 *
 * // A session loader can inspect extension-owned viewer metadata before it
 * // returns a normalized LoadedBundleDocument for the app shell.
 * const legacySpec = (savedTab.viewer as { linearLayoutSpec?: unknown }).linearLayoutSpec;
 * Boolean(legacySpec); // true
 */
export type LoadedSessionTab = SessionBundleManifest['tabs'][number];

/**
 * Describes one demo-shell extension and the optional hooks it contributes to
 * tabs, widgets, commands, rendering, pointer interaction, session loading, and
 * tensor-view UI.
 *
 * The app shell collects these hooks from registered feature packages. A small
 * extension can provide only widgets and commands, while the linear-layout
 * extension also handles session migration, hover state, inspector rows,
 * controls, and fallback tabs.
 *
 * @example
 * const annotationExtension: DemoAppExtension = {
 *   id: 'annotations',
 *   widgets: [{ id: 'annotation-list', slot: 'left', render: () => undefined }],
 *   commands: () => [{ id: 'annotations.clear', label: 'Clear annotations', run: () => undefined }],
 *   beforeRender: (_ctx, snapshot) => snapshot.displayMode === '2d',
 * };
 *
 * annotationExtension.beforeRender?.({} as DemoExtensionContext, { displayMode: '2d' } as ViewerSnapshot);
 * // true; the shell may continue rendering after the extension accepts the 2D snapshot.
 */
export type DemoAppExtension = {
    id: string;
    widgets: DemoWidgetSpec[];
    commands?: (ctx: DemoExtensionContext) => CommandAction[];
    controls?: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => ControlSpec[];
    tensorView?: (ctx: DemoExtensionContext, args: {
        tab: LoadedBundleDocument | undefined;
        tensorId: string;
    }) => DemoTensorViewContribution | null;
    afterTensorViewChange?: (ctx: DemoExtensionContext, tensorId: string) => void;
    inspectorCoords?: (ctx: DemoExtensionContext, args: {
        snapshot: ViewerSnapshot;
        hover: HoverInfo | null;
        hoveredStatus: TensorStatus | null;
    }) => DemoInspectorCoordEntry[];
    createTab?: (ctx: DemoExtensionContext, id: string, title: string, snapshot: ViewerSnapshot) => LoadedBundleDocument | Promise<LoadedBundleDocument> | null;
    captureSnapshot?: (ctx: DemoExtensionContext, tab: LoadedBundleDocument, snapshot: ViewerSnapshot) => void;
    clearTab?: (ctx: DemoExtensionContext, tabId: string) => void;
    cloneTab?: (ctx: DemoExtensionContext, fromTabId: string, toTabId: string) => void;
    beforeSessionLoad?: (ctx: DemoExtensionContext) => void;
    loadSessionTab?: (ctx: DemoExtensionContext, tab: LoadedSessionTab) => LoadedBundleDocument | Promise<LoadedBundleDocument | null> | null;
    afterLoadTab?: (ctx: DemoExtensionContext, tab: LoadedBundleDocument) => void;
    beforeRender?: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => boolean;
    afterRender?: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => void;
    loadFallback?: (ctx: DemoExtensionContext) => Promise<boolean>;
    pointerMove?: (ctx: DemoExtensionContext, event: PointerEvent) => void;
    pointerLeave?: (ctx: DemoExtensionContext) => void;
    hover?: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => void;
    selectionPreview?: (ctx: DemoExtensionContext, selection: SelectionCoords) => void;
    selection?: (ctx: DemoExtensionContext, selection: SelectionCoords) => void;
};

/**
 * Registry entry that advertises an extension's app-shell widget slots before
 * the shell context is available, then creates the runtime extension after the
 * shell has initialized shared services.
 *
 * @example
 * const factory: DemoExtensionFactory = {
 *   widgetSlots: ['left-sidebar' as AppShellWidgetSlot],
 *   create: (ctx) => ({
 *     id: 'annotations',
 *     widgets: [{ id: 'annotation-list', slot: 'left-sidebar' as AppShellWidgetSlot, render: () => undefined }],
 *     commands: () => [{ id: 'annotations.clear', label: 'Clear annotations', run: () => undefined }],
 *   }),
 * };
 *
 * const extension = factory.create({} as DemoExtensionContext);
 * extension.id;
 * // 'annotations'
 */
export type DemoExtensionFactory = {
    widgetSlots: AppShellWidgetSlot[];
    create: (ctx: DemoExtensionContext) => DemoAppExtension;
};
