import {
    TensorViewer,
    product,
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
      </div>
    </div>
    <div class="menu">
      <button class="menu-trigger" type="button">Widgets</button>
      <div class="menu-list">
        <button data-action="tensor-view" type="button">Toggle Tensor View <span>Ctrl+V</span></button>
        <button data-action="dims" type="button">Toggle Dimension Lines <span>Ctrl+D</span></button>
        <button data-action="inspector" type="button">Toggle Inspector <span></span></button>
      </div>
    </div>
  </div>
  <main class="viewport-wrap">
    <div id="viewport"></div>
  </main>
  <aside class="sidebar" id="sidebar">
    <section class="widget" id="tensor-view-widget"></section>
    <section class="widget" id="inspector-widget"></section>
    <section class="widget" id="colorbar-widget"></section>
  </aside>
  <input class="hidden" id="file-input" type="file" accept=".viz" />
`;

const viewport = document.querySelector<HTMLDivElement>('#viewport');
const tensorViewWidget = document.querySelector<HTMLElement>('#tensor-view-widget');
const inspectorWidget = document.querySelector<HTMLElement>('#inspector-widget');
const colorbarWidget = document.querySelector<HTMLElement>('#colorbar-widget');
const fileInput = document.querySelector<HTMLInputElement>('#file-input');
if (!viewport || !tensorViewWidget || !inspectorWidget || !colorbarWidget || !fileInput) throw new Error('Missing app elements.');

const viewer = new TensorViewer(viewport);
const viewErrors = new Map<string, string>();
let suspendTensorViewRender = false;
let showTensorViewWidget = true;

function logUi(event: string, details?: unknown): void {
    if (details === undefined) console.log('[tensor-viz-ui]', event);
    else console.log('[tensor-viz-ui]', event, details);
}

function formatRangeValue(value: number): string {
    return Number.isInteger(value) ? String(value) : value.toPrecision(6);
}

window.addEventListener('pointerup', () => {
    if (!suspendTensorViewRender) return;
    suspendTensorViewRender = false;
    render(viewer.getSnapshot());
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
        inspectorWidget.innerHTML = '<h2>Inspector</h2><div class="widget-body">No tensor loaded.</div>';
        return;
    }

    const error = viewErrors.get(model.handle.id);
    const hover = viewer.getHover();
    const inspectorDetail = hover
        ? `
          <div><span class="meta-label">Hover Tensor</span><span class="meta-value">${hover.tensorName}</span></div>
          <div><span class="meta-label">Full Coord</span><span class="meta-value">[${hover.fullCoord.join(', ')}]</span></div>
          <div><span class="meta-label">Value</span><span class="meta-value">${hover.value}</span></div>
        `
        : '<div><span class="meta-label">Hover</span><span class="meta-value">No cell hovered.</span></div>';
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

    inspectorWidget.innerHTML = `
      <h2>Inspector</h2>
      <div class="widget-body meta-grid">
        ${inspectorDetail}
        <div><span class="meta-label">DType</span><span class="meta-value">${model.handle.dtype}</span></div>
        <div><span class="meta-label">Shape</span><span class="meta-value">[${model.handle.shape.join(', ')}]</span></div>
        <div><span class="meta-label">Rank</span><span class="meta-value">${model.handle.rank}</span></div>
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
    updateSidebar(snapshot);
    renderTensorViewWidget(snapshot);
    renderColorbarWidget(snapshot);
}

async function saveBundleToDisk(): Promise<void> {
    const blob = await viewer.saveFile();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'tensor.viz';
    anchor.click();
    URL.revokeObjectURL(url);
}

async function tryLoadSession(): Promise<boolean> {
    const response = await fetch('/api/session.viz', { cache: 'no-store' });
    if (!response.ok) return false;
    const blob = await response.blob();
    await viewer.openFile(new File([blob], 'session.viz', { type: 'application/zip' }));
    return true;
}

function seedDemoTensor(): void {
    const shape = [4, 4, 4];
    const data = new Float64Array(product(shape));
    for (let index = 0; index < data.length; index += 1) {
        data[index] = Math.sin(index / 3) * 10;
    }
    viewer.addTensor(shape, data, 'Sample');
}

async function openLocalFile(file: File): Promise<void> {
    await viewer.openFile(file);
}

async function runAction(action: string): Promise<void> {
    logUi('action', action);
    switch (action) {
        case 'open':
            fileInput.click();
            return;
        case 'save':
            await saveBundleToDisk();
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
    if (isEditing && !(event.ctrlKey && event.key.toLowerCase() === 's')) return;

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
    }
});

viewer.subscribe(render);

tryLoadSession().then((loaded) => {
    if (!loaded) seedDemoTensor();
}).catch(() => seedDemoTensor());
