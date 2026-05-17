/**
 * DOM handles for the mounted demo shell, including the viewer viewport, sidebar chrome, command palette, and registered widget containers.
 *
 * @example
 * const refs = mountAppShell(appRoot, [
 *   { id: 'selection' },
 *   { id: 'linear-layout-color', beforeHeader: true },
 * ]);
 * refs.viewport.id;
 * // 'viewport'
 * refs.widgets.selection.dataset.widgetId;
 * // 'selection'
 */
export type AppShellRefs = {
    app: HTMLDivElement;
    viewport: HTMLDivElement;
    tabStrip: HTMLDivElement;
    controlDock: HTMLDivElement;
    sidebarSplitter: HTMLDivElement;
    sidebar: HTMLElement;
    widgets: Record<string, HTMLElement>;
    sidebarHeader: HTMLDivElement;
    commandPalette: HTMLDivElement;
    commandPaletteBackdrop: HTMLDivElement;
    commandPaletteInput: HTMLInputElement;
    commandPaletteList: HTMLDivElement;
};

/**
 * Declares a shell-managed widget container that core app code or an extension wants `mountAppShell` to create.
 *
 * @example
 * const sidebarSlot: AppShellWidgetSlot = { id: 'selection' };
 * const commandPanelSlot: AppShellWidgetSlot = { id: 'linear-layout-color', beforeHeader: true };
 *
 * sidebarSlot.id;
 * // 'selection'
 * commandPanelSlot.beforeHeader;
 * // true
 */
export type AppShellWidgetSlot = {
    id: string;
    beforeHeader?: boolean;
};

/**
 * Finds a required element inside the mounted shell so startup fails immediately when expected markup is missing.
 *
 * @param root - DOM subtree to search, usually the application root after `mountAppShell` has written its template.
 * @param selector - CSS selector for the required shell element, such as `#viewport` or `.sidebar-header`.
 * @param name - Human-readable element name included in the startup error message when the selector has no match.
 * @returns The first element matching `selector`, typed as the element kind requested by the caller for later event binding or rendering.
 * @throws Error when `root.querySelector(selector)` returns `null`; the message is `Missing ${name}.`.
 * @example
 * const root = document.createElement('div');
 * root.innerHTML = '<div id="viewport"></div>';
 *
 * requireElement<HTMLDivElement>(root, '#viewport', 'viewport').id;
 * // 'viewport'
 *
 * expect(() => requireElement(root, '#command-palette', 'command palette'))
 *   .toThrow(new Error('Missing command palette.'));
 */
function requireElement<T extends Element>(root: ParentNode, selector: string, name: string): T {
    const element = root.querySelector<T>(selector);
    if (!element) throw new Error(`Missing ${name}.`);
    return element;
}

/**
 * Finds the Vite `#app` mount element that the demo shell replaces during startup.
 *
 * @returns The `HTMLDivElement` with id `app`, used as the root for the WebGL fallback view or the full tensor-viewer shell.
 * @throws {Error} When `document` does not contain an element matching `#app`; the error message is `Missing app root.`.
 * @example
 * ```ts
 * document.body.innerHTML = '<div id="app"></div>';
 *
 * const app = getAppRoot();
 *
 * console.assert(app.id === 'app');
 * ```
 *
 * @example
 * ```ts
 * document.body.innerHTML = '';
 *
 * try {
 *   getAppRoot();
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'Missing app root.');
 * }
 * ```
 */
export function getAppRoot(): HTMLDivElement {
    const app = document.querySelector<HTMLDivElement>('#app');
    if (!app) throw new Error('Missing app root.');
    return app;
}

/**
 * Probes the browser for a WebGL2, WebGL, or experimental WebGL canvas context before the demo creates its three.js renderer.
 *
 * @returns `true` when a canvas context can be created for WebGL rendering; `false` when WebGL globals are unavailable, context creation returns `null`, or context probing fails.
 * @noThrows Canvas context creation is wrapped in a `try`/`catch`, so browser probing failures are reported as `false` instead of escaping to startup code.
 * @example
 * ```ts
 * if (!supportsWebGL()) {
 *   renderWebglUnavailable(app);
 * }
 * ```
 */
export function supportsWebGL(): boolean {
    const canvas = document.createElement('canvas');
    try {
        return Boolean(
            (typeof WebGL2RenderingContext !== 'undefined' && canvas.getContext('webgl2'))
            || (typeof WebGLRenderingContext !== 'undefined'
                && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl'))),
        );
    } catch {
        return false;
    }
}

/**
 * Replaces the demo mount point with the startup notice shown when tensors cannot be rendered because WebGL is unavailable.
 *
 * @param app - The demo `#app` root element whose existing contents should be replaced by the WebGL-disabled message.
 * @returns No value; callers observe the fallback view through `app.innerHTML`.
 * @noThrows The function only assigns fixed fallback markup to the provided element and performs no DOM queries or renderer initialization.
 * @example
 * ```ts
 * const app = document.createElement('div');
 *
 * renderWebglUnavailable(app);
 *
 * console.assert(app.querySelector('.startup-note') !== null);
 * console.assert(app.textContent?.includes('This viewer needs WebGL to render tensors'));
 * ```
 */
export function renderWebglUnavailable(app: HTMLDivElement): void {
    app.innerHTML = `
      <main class="startup-note">
        <p>This viewer needs WebGL to render tensors, but WebGL appears disabled or unavailable in this browser.</p>
        <p>Enable WebGL or hardware acceleration in your browser settings, then reload this page.</p>
      </main>
    `;
}

/**
 * Builds the demo application's reusable chrome: ribbon menus, tab strip, viewport host, control dock, sidebar widgets, and command palette.
 *
 * Widget slots are supplied before extension lifecycle hooks run so extensions receive real DOM hosts. Slots with `beforeHeader` are rendered above the sidebar header for extension-owned command panels; all other slots are rendered in the normal widgets section.
 *
 * @param app - The demo `#app` root element whose contents should be replaced with the full shell markup.
 * @param widgetSlots - Widget host declarations; each slot id becomes a `[data-widget-id="..."]` section and `beforeHeader` controls whether it appears above the sidebar header.
 * @returns References to the generated shell elements, including the viewport, tab strip, control dock, sidebar, command palette nodes, and a `widgets` map keyed by widget slot id.
 * @noThrows The shell markup is generated in one assignment and includes the fixed elements that are looked up before returning; with valid widget slot ids, there is no expected startup throw path.
 * @example
 * ```ts
 * const app = document.createElement('div');
 * const refs = mountAppShell(app, [
 *   { id: 'preset-selector', beforeHeader: true },
 *   { id: 'inspector' },
 * ]);
 *
 * console.assert(refs.viewport.id === 'viewport');
 * console.assert(refs.widgets['preset-selector'].dataset.widgetId === 'preset-selector');
 * console.assert(refs.widgets.inspector.closest('.sidebar') === refs.sidebar);
 * ```
 */
export function mountAppShell(app: HTMLDivElement, widgetSlots: AppShellWidgetSlot[]): AppShellRefs {
    // widgets before the header are extension-owned command panels; sidebar
    // widgets are ordinary viewer panels that share the default layout chrome.
    const primaryWidgets = widgetSlots
        .filter((slot) => slot.beforeHeader)
        .map((slot) => `<section class="widget" id="${slot.id}-widget" data-widget-id="${slot.id}"></section>`)
        .join('\n');
    const sidebarWidgets = widgetSlots
        .filter((slot) => !slot.beforeHeader)
        .map((slot) => `<section class="widget" id="${slot.id}-widget" data-widget-id="${slot.id}"></section>`)
        .join('\n');
    app.innerHTML = `
      <div class="ribbon">
        <div class="menu">
          <button class="menu-trigger" type="button">File</button>
          <div class="menu-list">
            <button data-action="save-svg" type="button">Save SVG <span>Ctrl+S</span></button>
          </div>
        </div>
        <div class="menu">
          <button class="menu-trigger" type="button">Display</button>
          <div class="menu-list">
            <button data-action="2d" type="button">Display as 2D <span>Ctrl+2</span></button>
            <button data-action="3d" type="button">Display as 3D <span>Ctrl+3</span></button>
            <button data-action="heatmap" type="button">Toggle Heatmap <span>Ctrl+H</span></button>
            <button data-action="dims" type="button">Toggle Dimension Lines <span>Ctrl+D</span></button>
            <button data-action="tensor-names" type="button">Toggle Tensor Names <span></span></button>
            <div class="menu-submenu">
              <button class="menu-submenu-trigger" type="button">Advanced <span>&gt;</span></button>
              <div class="menu-list menu-submenu-list">
                <button data-action="mapping-contiguous" type="button">Set Contiguous Axis Family Mapping <span></span></button>
                <button data-action="mapping-z-order" type="button">Set Z-Order Axis Family Mapping <span></span></button>
                <button data-action="display-gaps" type="button">Toggle Block Gaps <span></span></button>
                <button data-action="collapse-hidden-axes" type="button">Toggle Collapse Hidden Axes <span></span></button>
                <button data-action="log-scale" type="button">Toggle Log Scale <span></span></button>
              </div>
            </div>
          </div>
        </div>
        <div class="menu">
          <button class="menu-trigger" type="button">Widgets</button>
          <div class="menu-list">
            <button data-action="tensor-view" type="button">Toggle Permute/Slice <span>Ctrl+V</span></button>
            <button data-action="inspector" type="button">Toggle Hover Info <span></span></button>
            <button data-action="selection" type="button">Toggle Selection <span></span></button>
            <button data-action="advanced-settings" type="button">Toggle Advanced Settings <span></span></button>
          </div>
        </div>
      </div>
      <div class="tab-strip" id="tab-strip"></div>
      <main class="viewport-wrap">
        <div id="viewport"></div>
        <div class="control-dock" id="control-dock"></div>
      </main>
      <div class="sidebar-splitter" id="sidebar-splitter" role="separator" aria-orientation="vertical" aria-label="Resize widgets sidebar"></div>
      <aside class="sidebar" id="sidebar">
        ${primaryWidgets}
        <div class="sidebar-header">Widgets</div>
        ${sidebarWidgets}
      </aside>
      <div class="command-palette hidden" id="command-palette">
        <div class="command-palette-backdrop" id="command-palette-backdrop"></div>
        <div class="command-palette-dialog">
          <input id="command-palette-input" type="text" placeholder="Type a command" autocomplete="off" />
          <div class="command-palette-list" id="command-palette-list"></div>
        </div>
      </div>
    `;
    const widgets = Object.fromEntries(widgetSlots.map((slot) => [
        slot.id,
        requireElement<HTMLElement>(app, `[data-widget-id="${slot.id}"]`, `${slot.id} widget`),
    ]));

    return {
        app,
        viewport: requireElement(app, '#viewport', 'viewport'),
        tabStrip: requireElement(app, '#tab-strip', 'tab strip'),
        controlDock: requireElement(app, '#control-dock', 'control dock'),
        sidebarSplitter: requireElement(app, '#sidebar-splitter', 'sidebar splitter'),
        sidebar: requireElement(app, '#sidebar', 'sidebar'),
        widgets,
        sidebarHeader: requireElement(app, '.sidebar-header', 'sidebar header'),
        commandPalette: requireElement(app, '#command-palette', 'command palette'),
        commandPaletteBackdrop: requireElement(app, '#command-palette-backdrop', 'command palette backdrop'),
        commandPaletteInput: requireElement(app, '#command-palette-input', 'command palette input'),
        commandPaletteList: requireElement(app, '#command-palette-list', 'command palette list'),
    };
}
