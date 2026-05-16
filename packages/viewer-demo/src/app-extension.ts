import type {
    BundleManifest,
    LoadedBundleDocument,
    NumericArray,
    SelectionCoords,
    SessionBundleManifest,
    TensorViewer,
    ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import type { ControlSpec } from './control-dock.js';

export type CommandAction = {
    action: string;
    label: string;
    shortcut: string;
    keywords: string;
};

export type DemoWidgetSpec = {
    id: string;
    label: string;
    icon: string;
    defaultCollapsed: boolean;
    visible: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => boolean;
    render: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => void;
};

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

export type LoadedSessionTab = SessionBundleManifest['tabs'][number];

export type DemoAppExtension = {
    id: string;
    widgets: DemoWidgetSpec[];
    commands?: (ctx: DemoExtensionContext) => CommandAction[];
    controls?: (ctx: DemoExtensionContext, snapshot: ViewerSnapshot) => ControlSpec[];
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
