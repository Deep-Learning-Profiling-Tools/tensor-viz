export type AppShellRefs = {
    app: HTMLDivElement;
    viewport: HTMLDivElement;
    tabStrip: HTMLDivElement;
    sidebarSplitter: HTMLDivElement;
    tensorViewWidget: HTMLElement;
    inspectorWidget: HTMLElement;
    selectionWidget: HTMLElement;
    advancedSettingsWidget: HTMLElement;
    colorbarWidget: HTMLElement;
    commandPalette: HTMLDivElement;
    commandPaletteBackdrop: HTMLDivElement;
    commandPaletteInput: HTMLInputElement;
    commandPaletteList: HTMLDivElement;
};

function requireElement<T extends Element>(root: ParentNode, selector: string, name: string): T {
    const element = root.querySelector<T>(selector);
    if (!element) throw new Error(`Missing ${name}.`);
    return element;
}

export function getAppRoot(): HTMLDivElement {
    const app = document.querySelector<HTMLDivElement>('#app');
    if (!app) throw new Error('Missing app root.');
    return app;
}

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

export function renderWebglUnavailable(app: HTMLDivElement): void {
    app.innerHTML = `
      <main class="startup-note">
        <p>This viewer needs WebGL to render tensors, but WebGL appears disabled or unavailable in this browser.</p>
        <p>Enable WebGL or hardware acceleration in your browser settings, then reload this page.</p>
      </main>
    `;
}

export function mountAppShell(app: HTMLDivElement): AppShellRefs {
    app.innerHTML = `
      <div class="ribbon">
        <div class="menu">
          <button class="menu-trigger" type="button">File</button>
          <div class="menu-list">
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
            <button data-action="tensor-view" type="button">Toggle Tensor View <span>Ctrl+V</span></button>
            <button data-action="inspector" type="button">Toggle Inspector <span></span></button>
            <button data-action="selection" type="button">Toggle Selection <span></span></button>
            <button data-action="advanced-settings" type="button">Toggle Advanced Settings <span></span></button>
          </div>
        </div>
      </div>
      <div class="tab-strip hidden" id="tab-strip"></div>
      <main class="viewport-wrap">
        <div id="viewport"></div>
      </main>
      <div class="sidebar-splitter" id="sidebar-splitter" role="separator" aria-orientation="vertical" aria-label="Resize widgets sidebar"></div>
      <aside class="sidebar" id="sidebar">
        <section class="widget" id="tensor-view-widget"></section>
        <section class="widget" id="inspector-widget"></section>
        <section class="widget" id="selection-widget"></section>
        <section class="widget" id="advanced-settings-widget"></section>
        <section class="widget" id="colorbar-widget"></section>
      </aside>
      <div class="command-palette hidden" id="command-palette">
        <div class="command-palette-backdrop" id="command-palette-backdrop"></div>
        <div class="command-palette-dialog">
          <input id="command-palette-input" type="text" placeholder="Type a command" autocomplete="off" />
          <div class="command-palette-list" id="command-palette-list"></div>
        </div>
      </div>
    `;

    return {
        app,
        viewport: requireElement(app, '#viewport', 'viewport'),
        tabStrip: requireElement(app, '#tab-strip', 'tab strip'),
        sidebarSplitter: requireElement(app, '#sidebar-splitter', 'sidebar splitter'),
        tensorViewWidget: requireElement(app, '#tensor-view-widget', 'tensor view widget'),
        inspectorWidget: requireElement(app, '#inspector-widget', 'inspector widget'),
        selectionWidget: requireElement(app, '#selection-widget', 'selection widget'),
        advancedSettingsWidget: requireElement(app, '#advanced-settings-widget', 'advanced settings widget'),
        colorbarWidget: requireElement(app, '#colorbar-widget', 'colorbar widget'),
        commandPalette: requireElement(app, '#command-palette', 'command palette'),
        commandPaletteBackdrop: requireElement(app, '#command-palette-backdrop', 'command palette backdrop'),
        commandPaletteInput: requireElement(app, '#command-palette-input', 'command palette input'),
        commandPaletteList: requireElement(app, '#command-palette-list', 'command palette list'),
    };
}
