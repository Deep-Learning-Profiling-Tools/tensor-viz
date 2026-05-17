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
 * const value: CommandAction = {} as CommandAction;
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
 * const value: DemoWidgetSpec = {} as DemoWidgetSpec;
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
 * const value: DemoTensorViewSliderSpec = {} as DemoTensorViewSliderSpec;
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
 * const value: DemoTensorViewContribution = {} as DemoTensorViewContribution;
 */
export type DemoTensorViewContribution = {
    axisLabels?: readonly string[];
    sliders?: DemoTensorViewSliderSpec[];
};

/**
 * one row in the hover inspector's coordinate list.
 *
 * @example
 * const value: DemoInspectorCoordEntry = {} as DemoInspectorCoordEntry;
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
 * const value: DemoExtensionContext = {} as DemoExtensionContext;
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
 * const value: LoadedSessionTab = {} as LoadedSessionTab;
 */
export type LoadedSessionTab = SessionBundleManifest['tabs'][number];

/**
 * behavior hooks for one demo feature package.
 *
 * hooks are optional so a feature can expose only the surfaces it needs. for
 * example, linear-layout implements session migration, hover, controls, and
 * widgets; a simple annotation extension might only contribute widgets and
 * commands while sharing the same lifecycle attributes.
 *
 * @example
 * const value: DemoAppExtension = {} as DemoAppExtension;
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
 * factory shape used by the static registry before the shell context exists.
 *
 * @example
 * const value: DemoExtensionFactory = {} as DemoExtensionFactory;
 */
export type DemoExtensionFactory = {
    widgetSlots: AppShellWidgetSlot[];
    create: (ctx: DemoExtensionContext) => DemoAppExtension;
};
