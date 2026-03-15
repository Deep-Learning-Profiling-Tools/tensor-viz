import {
    loadSessionBundle,
    TensorViewer,
    product,
    type LoadedBundleDocument,
    type ViewerSnapshot,
} from '@tensor-viz/viewer-core';
import './styles.css';

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) throw new Error('Missing app root.');

app.innerHTML = `
  <div class="ribbon">
    <div class="menu">
      <button class="menu-trigger" type="button">File</button>
      <div class="menu-list">
        <button data-action="open" type="button">Open Tensor <span>Ctrl+O</span></button>
        <button data-action="save" type="button">Save Tensor <span>Ctrl+S</span></button>
      </div>
    </div>
    <div class="menu">
      <button class="menu-trigger" type="button">Display</button>
      <div class="menu-list">
        <button data-action="2d" type="button">Display as 2D <span>Ctrl+2</span></button>
        <button data-action="3d" type="button">Display as 3D <span>Ctrl+3</span></button>
        <button data-action="heatmap" type="button">Toggle Heatmap <span>Ctrl+H</span></button>
        <button data-action="dims" type="button">Toggle Dimension Lines <span>Ctrl+D</span></button>
      </div>
    </div>
    <div class="menu">
      <button class="menu-trigger" type="button">Widgets</button>
      <div class="menu-list">
        <button data-action="tensor-view" type="button">Toggle Tensor View <span>Ctrl+V</span></button>
        <button data-action="inspector" type="button">Toggle Inspector <span></span></button>
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
    <section class="widget" id="colorbar-widget"></section>
  </aside>
  <div class="command-palette hidden" id="command-palette">
    <div class="command-palette-backdrop" id="command-palette-backdrop"></div>
    <div class="command-palette-dialog">
      <input id="command-palette-input" type="text" placeholder="Type a command" autocomplete="off" />
      <div class="command-palette-list" id="command-palette-list"></div>
    </div>
  </div>
  <input class="hidden" id="file-input" type="file" accept=".npy" />
`;

const viewport = document.querySelector<HTMLDivElement>('#viewport');
const tabStrip = document.querySelector<HTMLDivElement>('#tab-strip');
const sidebarSplitter = document.querySelector<HTMLDivElement>('#sidebar-splitter');
const tensorViewWidget = document.querySelector<HTMLElement>('#tensor-view-widget');
const inspectorWidget = document.querySelector<HTMLElement>('#inspector-widget');
const colorbarWidget = document.querySelector<HTMLElement>('#colorbar-widget');
const commandPalette = document.querySelector<HTMLDivElement>('#command-palette');
const commandPaletteBackdrop = document.querySelector<HTMLDivElement>('#command-palette-backdrop');
const commandPaletteInput = document.querySelector<HTMLInputElement>('#command-palette-input');
const commandPaletteList = document.querySelector<HTMLDivElement>('#command-palette-list');
const fileInput = document.querySelector<HTMLInputElement>('#file-input');
if (
    !viewport
    || !tabStrip
    || !sidebarSplitter
    || !tensorViewWidget
    || !inspectorWidget
    || !colorbarWidget
    || !commandPalette
    || !commandPaletteBackdrop
    || !commandPaletteInput
    || !commandPaletteList
    || !fileInput
) throw new Error('Missing app elements.');

const viewer = new TensorViewer(viewport);
const viewErrors = new Map<string, string>();
let suspendTensorViewRender = false;
let showTensorViewWidget = true;
let inspectorReady = false;
let sessionTabs: LoadedBundleDocument[] = [];
let activeTabId: string | null = null;
let switchingTab = false;
let resizingSidebar = false;
let commandPaletteOpen = false;
let commandPaletteIndex = 0;
let commandPaletteMode: 'actions' | 'tabs' = 'actions';

const MIN_VIEWPORT_WIDTH = 280;
const MAX_SIDEBAR_WIDTH = 720;

type InspectorRefs = {
    hoverTensor: HTMLDivElement;
    fullCoord: HTMLDivElement;
    value: HTMLDivElement;
    hoverTensorValue: HTMLSpanElement;
    fullCoordValue: HTMLSpanElement;
    valueField: HTMLSpanElement;
    dtypeValue: HTMLSpanElement;
    shapeValue: HTMLSpanElement;
    rankValue: HTMLSpanElement;
};

let inspectorRefs: InspectorRefs | null = null;

type CommandAction = {
    action: string;
    label: string;
    shortcut: string;
    keywords: string;
};

function logUi(event: string, details?: unknown): void {
    if (details === undefined) console.log('[tensor-viz-ui]', event);
    else console.log('[tensor-viz-ui]', event, details);
}

function formatRangeValue(value: number): string {
    return Number.isInteger(value) ? String(value) : value.toPrecision(6);
}

function commandActions(): CommandAction[] {
    return [
        { action: 'command-palette', label: 'Command Palette', shortcut: '?', keywords: 'command palette search actions' },
        { action: 'open', label: 'Open Tensor', shortcut: 'Ctrl+O', keywords: 'file open load tensor npy' },
        { action: 'save', label: 'Save Tensor', shortcut: 'Ctrl+S', keywords: 'file save export tensor npy' },
        { action: '2d', label: 'Display as 2D', shortcut: 'Ctrl+2', keywords: 'display 2d orthographic' },
        { action: '3d', label: 'Display as 3D', shortcut: 'Ctrl+3', keywords: 'display 3d perspective' },
        { action: 'heatmap', label: 'Toggle Heatmap', shortcut: 'Ctrl+H', keywords: 'display heatmap colors' },
        { action: 'dims', label: 'Toggle Dimension Lines', shortcut: 'Ctrl+D', keywords: 'display dimensions guides labels' },
        { action: 'tensor-view', label: 'Toggle Tensor View', shortcut: 'Ctrl+V', keywords: 'widgets tensor view panel' },
        { action: 'inspector', label: 'Toggle Inspector', shortcut: '', keywords: 'widgets inspector panel' },
        { action: 'view', label: 'Focus Tensor View Input', shortcut: '', keywords: 'focus tensor view input field' },
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

function filteredCommandActions(): CommandAction[] {
    const query = commandPaletteInput.value.trim().toLowerCase();
    const actions = paletteActions();
    if (!query) return actions;
    return actions.filter((entry) => `${entry.label} ${entry.keywords}`.toLowerCase().includes(query));
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
    const maxWidth = Math.max(0, Math.min(MAX_SIDEBAR_WIDTH, app.clientWidth - MIN_VIEWPORT_WIDTH));
    const clamped = Math.max(0, Math.min(maxWidth, width));
    app.style.setProperty('--sidebar-width', `${clamped}px`);
}

function activeTab(): LoadedBundleDocument | undefined {
    return sessionTabs.find((tab) => tab.id === activeTabId);
}

function captureActiveTabSnapshot(): void {
    const tab = activeTab();
    if (!tab) return;
    tab.manifest.viewer = viewer.getSnapshot();
}

function renderTabStrip(): void {
    tabStrip.classList.toggle('hidden', sessionTabs.length < 2);
    if (sessionTabs.length < 2) {
        tabStrip.replaceChildren();
        return;
    }
    tabStrip.replaceChildren(...sessionTabs.map((tab) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = `tab-button${tab.id === activeTabId ? ' active' : ''}`;
        button.textContent = tab.title;
        button.addEventListener('click', async () => {
            if (tab.id === activeTabId) return;
            await loadTab(tab.id);
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

function updateSidebar(snapshot: ViewerSnapshot): void {
    tensorViewWidget.classList.toggle('hidden', !showTensorViewWidget);
    inspectorWidget.classList.toggle('hidden', !snapshot.showInspectorPanel);
    const model = viewer.getInspectorModel();
    colorbarWidget.classList.toggle('hidden', !snapshot.heatmap || !model.colorRange);
}

function renderTensorViewWidget(snapshot: ViewerSnapshot): void {
    if (suspendTensorViewRender) return;
    const model = viewer.getInspectorModel();
    if (!model.handle) {
        tensorViewWidget.innerHTML = '<h2>Tensor View</h2><div class="widget-body">No tensor loaded.</div>';
        return;
    }

    const error = viewErrors.get(model.handle.id);
    const tensorOptions = model.tensors.map((tensor) => `
      <option value="${tensor.id}" ${tensor.id === model.handle!.id ? 'selected' : ''}>${tensor.id} (${tensor.name})</option>
    `).join('');
    tensorViewWidget.innerHTML = `
      <h2>Tensor View</h2>
      <div class="widget-body">
        <div class="field">
          <label for="tensor-select">Tensor</label>
          <select id="tensor-select">${tensorOptions}</select>
        </div>
        <div class="field">
          <label for="view-input">Visible Dimensions</label>
          <input id="view-input" type="text" value="${model.viewInput}" placeholder="empty resets to default" />
        </div>
        ${error ? `<div class="error-box">${error}</div>` : ''}
        <div class="field">
          <label>Preview</label>
          <div class="mono-block">${model.preview}</div>
        </div>
        <div class="slider-list" id="hidden-token-controls"></div>
      </div>
    `;

    const input = tensorViewWidget.querySelector<HTMLInputElement>('#view-input');
    const select = tensorViewWidget.querySelector<HTMLSelectElement>('#tensor-select');
    const hiddenHost = tensorViewWidget.querySelector<HTMLElement>('#hidden-token-controls');
    select?.addEventListener('change', () => {
        logUi('tensor-select', select.value);
        viewer.setActiveTensor(select.value);
    });
    if (input) {
        input.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter') return;
            input.blur();
        });
        input.addEventListener('change', () => {
            try {
                logUi('tensor-view:change', { tensorId: model.handle!.id, value: input.value });
                viewer.setTensorView(model.handle!.id, input.value);
                viewErrors.delete(model.handle!.id);
            } catch (error) {
                viewErrors.set(model.handle!.id, error instanceof Error ? error.message : String(error));
            }
            render(snapshot);
        });
    }

    hiddenHost?.replaceChildren(...model.hiddenTokens.map((token) => {
        const row = document.createElement('div');
        row.className = 'slider-row';
        row.innerHTML = `
          <label for="hidden-${token.token}">${token.token}</label>
          <input id="hidden-${token.token}" type="range" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
          <input id="hidden-${token.token}-number" type="number" min="0" max="${Math.max(0, token.size - 1)}" value="${token.value}" />
        `;
        const slider = row.querySelector<HTMLInputElement>(`#hidden-${token.token}`);
        const number = row.querySelector<HTMLInputElement>(`#hidden-${token.token}-number`);
        const applyValue = (nextValue: number): void => {
            logUi('hidden-token:update', { tensorId: model.handle!.id, token: token.token, value: nextValue });
            viewer.setHiddenTokenValue(model.handle!.id, token.token, nextValue);
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
            render(snapshot);
        });
        number?.addEventListener('change', () => {
            const clamped = Math.max(0, Math.min(token.size - 1, Number(number.value)));
            number.value = String(clamped);
            if (slider) slider.value = String(clamped);
            applyValue(clamped);
            suspendTensorViewRender = false;
            render(snapshot);
        });
        return row;
    }));
}

function renderInspectorWidget(): void {
    const model = viewer.getInspectorModel();
    if (!model.handle) {
        inspectorReady = false;
        inspectorRefs = null;
        inspectorWidget.innerHTML = '<h2>Inspector</h2><div class="widget-body">No tensor loaded.</div>';
        return;
    }
    if (!inspectorReady) {
        inspectorWidget.innerHTML = `
          <h2>Inspector</h2>
          <div class="widget-body meta-grid">
            <div id="inspector-hover-tensor"><span class="meta-label">Hover Tensor</span><span class="meta-value" id="inspector-hover-tensor-value"></span></div>
            <div id="inspector-full-coord"><span class="meta-label">Full Coord</span><span class="meta-value" id="inspector-full-coord-value"></span></div>
            <div id="inspector-value"><span class="meta-label">Value</span><span class="meta-value" id="inspector-value-field"></span></div>
            <div><span class="meta-label">DType</span><span class="meta-value" id="inspector-dtype"></span></div>
            <div><span class="meta-label">Shape</span><span class="meta-value" id="inspector-shape"></span></div>
            <div><span class="meta-label">Rank</span><span class="meta-value" id="inspector-rank"></span></div>
          </div>
        `;
        inspectorRefs = {
            hoverTensor: inspectorWidget.querySelector<HTMLDivElement>('#inspector-hover-tensor')!,
            fullCoord: inspectorWidget.querySelector<HTMLDivElement>('#inspector-full-coord')!,
            value: inspectorWidget.querySelector<HTMLDivElement>('#inspector-value')!,
            hoverTensorValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-hover-tensor-value')!,
            fullCoordValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-full-coord-value')!,
            valueField: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-value-field')!,
            dtypeValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-dtype')!,
            shapeValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-shape')!,
            rankValue: inspectorWidget.querySelector<HTMLSpanElement>('#inspector-rank')!,
        };
        inspectorReady = true;
    }
    if (!inspectorRefs) return;
    const hover = viewer.getHover();
    inspectorRefs.hoverTensor.classList.toggle('hidden', !hover);
    inspectorRefs.fullCoord.classList.toggle('hidden', !hover);
    inspectorRefs.value.classList.toggle('hidden', !hover);
    inspectorRefs.hoverTensorValue.textContent = hover?.tensorName ?? '';
    inspectorRefs.fullCoordValue.textContent = hover ? `[${hover.fullCoord.join(', ')}]` : '';
    inspectorRefs.valueField.textContent = hover ? String(hover.value) : '';
    inspectorRefs.dtypeValue.textContent = model.handle.dtype;
    inspectorRefs.shapeValue.textContent = `[${model.handle.shape.join(', ')}]`;
    inspectorRefs.rankValue.textContent = String(model.handle.rank);
}

function renderColorbarWidget(snapshot: ViewerSnapshot): void {
    const model = viewer.getInspectorModel();
    if (!snapshot.heatmap || !model.colorRange) {
        colorbarWidget.innerHTML = '';
        return;
    }
    const sections = model.colorRanges.map((range) => `
      <div class="colorbar-section">
        <div class="colorbar-title">${range.id} (${range.name})</div>
        <div class="colorbar"></div>
        <div class="colorbar-labels">
          <span>${formatRangeValue(range.min)}</span>
          <span>${formatRangeValue(range.max)}</span>
        </div>
      </div>
    `).join('');
    colorbarWidget.innerHTML = `
      <h2>Colorbar</h2>
      <div class="widget-body">
        ${sections}
      </div>
    `;
}

function render(snapshot: ViewerSnapshot): void {
    if (!switchingTab) captureActiveTabSnapshot();
    updateSidebar(snapshot);
    renderTabStrip();
    renderTensorViewWidget(snapshot);
    renderInspectorWidget();
    renderColorbarWidget(snapshot);
}

function sanitizeFilename(name: string): string {
    return name.replace(/[^a-z0-9._-]+/gi, '_').replace(/^_+|_+$/g, '') || 'tensor';
}

async function saveTensorToDisk(): Promise<void> {
    const blob = await viewer.saveFile();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${sanitizeFilename(viewer.getInspectorModel().handle?.name ?? 'tensor')}.npy`;
    anchor.click();
    URL.revokeObjectURL(url);
}

async function tryLoadSession(): Promise<boolean> {
    const response = await fetch('/api/session.viz', { cache: 'no-store' });
    if (!response.ok) return false;
    sessionTabs = await loadSessionBundle(await response.blob());
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

async function openLocalFile(file: File): Promise<void> {
    sessionTabs = [];
    activeTabId = null;
    renderTabStrip();
    await viewer.openFile(file);
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
        case 'save':
            await saveTensorToDisk();
            return;
        case '2d':
            viewer.setDisplayMode('2d');
            return;
        case '3d':
            viewer.setDisplayMode('3d');
            return;
        case 'heatmap':
            viewer.toggleHeatmap();
            return;
        case 'tensor-view':
            showTensorViewWidget = !showTensorViewWidget;
            render(viewer.getSnapshot());
            return;
        case 'view': {
        const input = tensorViewWidget.querySelector<HTMLInputElement>('#view-input');
        input?.focus();
        input?.select();
        logUi('tensor-view:focus');
        return;
        }
        case 'dims':
            viewer.toggleDimensionLines();
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
    const isEditing = target && ['INPUT', 'TEXTAREA'].includes(target.tagName);
    const isPaletteInput = target === commandPaletteInput;
    if (isPaletteInput && event.key === 'Escape') {
        event.preventDefault();
        closeCommandPalette();
        return;
    }
    if (isEditing && !isPaletteInput && !(event.ctrlKey && event.key.toLowerCase() === 's')) return;

    if (event.ctrlKey && event.key.toLowerCase() === 'o') {
        event.preventDefault();
        await runAction('open');
    } else if (event.ctrlKey && event.key.toLowerCase() === 's') {
        event.preventDefault();
        await runAction('save');
    } else if (event.ctrlKey && event.key === '2') {
        event.preventDefault();
        await runAction('2d');
    } else if (event.ctrlKey && event.key === '3') {
        event.preventDefault();
        await runAction('3d');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'h') {
        event.preventDefault();
        await runAction('heatmap');
    } else if (event.ctrlKey && event.key.toLowerCase() === 'd') {
        event.preventDefault();
        await runAction('dims');
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
viewer.subscribeHover(() => renderInspectorWidget());

tryLoadSession().then((loaded) => {
    if (!loaded) seedDemoTensor();
}).catch(() => seedDemoTensor());
