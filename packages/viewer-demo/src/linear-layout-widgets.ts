import { escapeInfo, infoButton, labelWithInfo } from './app-format.js';
import { autoColorLayoutState, bakedComposeLayoutExamples, buildComposeRuntime, createComposeLayoutDocument, propagationLabels } from './linear-layout.js';
import {
    cloneLinearLayoutCellTextState,
    cloneLinearLayoutMultiInputState,
    cloneLinearLayoutState,
    composeLayoutMetaForTab,
    defaultLinearLayoutCellTextState,
    defaultLinearLayoutMultiInputState,
    isLinearLayoutTab,
    refreshLinearLayoutMatrixPreview,
    snapshotTensorViews,
    storeLinearLayoutState,
    type LinearLayoutChannel,
    type LinearLayoutUiContext,
} from './linear-layout-state.js';
import { applyLinearLayoutCellText, preservedLinearLayoutTensorViews } from './linear-layout-viewer-sync.js';

const LINEAR_LAYOUT_CHANNELS: LinearLayoutChannel[] = ['H', 'S', 'L'];
const VISIBLE_TENSORS_ERROR = 'At least one tensor in the render chain must stay visible.';

function linearLayoutSpecsHelpHtml(): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Add one or more named layout blocks in the form <strong>name: [inputs] -> [outputs]</strong>.</span>
          </div>
          <div class="usage-guide-step">
            <span>Each block must include exactly one labeled basis row per input, such as <strong>T: [[1,0],[0,1]]</strong>.</span>
          </div>
          <div class="usage-guide-step">
            <span>Separate multiple layout definitions with a blank line.</span>
          </div>
          <div class="usage-guide-step">
            <span>Input/output labels and basis-row labels must look like <strong>T</strong>, <strong>A0</strong>, or <strong>B12</strong>: one leading letter followed by optional digits only.</span>
          </div>
          <div class="usage-guide-step">
            <span>Layout names may use letters, digits, and underscores, but must start with a letter or underscore.</span>
          </div>
          <div class="usage-guide-column">
            <div class="usage-guide-subtitle">Single Layout</div>
            <div class="usage-guide-example">
              <code>Tile2x1: [T,W] -&gt; [Y,X]</code>
              <code>T: [[0,1],[0,2]]</code>
              <code>W: [[1,0]]</code>
            </div>
            <div class="usage-guide-example">
              <code>Swizzle: [Y,X] -&gt; [Y,X]</code>
              <code>Y: [[1,1],[2,2]]</code>
              <code>X: [[0,1],[0,2]]</code>
            </div>
            <div class="usage-guide-example">
              <code>Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[2,0],[4,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
            </div>
            <div class="usage-guide-subtitle">Multiple Layouts</div>
            <div class="usage-guide-example">
              <code>Tile2x1: [T,W] -&gt; [Y,X]</code>
              <code>T: [[0,1],[0,2]]</code>
              <code>W: [[1,0]]</code>
              <code></code>
              <code>Swizzle: [Y,X] -&gt; [Y,X]</code>
              <code>Y: [[1,1],[2,2]]</code>
              <code>X: [[0,1],[0,2]]</code>
              <code></code>
              <code>GetT: [Y1,X1] -&gt; [T]</code>
              <code>Y1: [[1],[2],[16]]</code>
              <code>X1: [[4],[8]]</code>
              <code></code>
              <code>GetW: [Y2,X2] -&gt; [W]</code>
              <code>Y2: [[1]]</code>
              <code>X2: [[2]]</code>
              <code></code>
              <code>GetR: [X3] -&gt; [R]</code>
              <code>X3: [[1]]</code>
            </div>
            <div class="usage-guide-subtitle">Non-Surjective Layout</div>
            <div class="usage-guide-example">
              <code>Sparse_Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[4,0],[8,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
            </div>
          </div>
        </div>
      </details>
    `;
}

function linearLayoutOperationHelpHtml(): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Use the names you defined above to choose which layout chain to render.</span>
          </div>
          <div class="usage-guide-step">
            <span>Supported operators are parentheses, inverse like <strong>inv(A)</strong>, composition like <strong>A(B)</strong>, and products like <strong>A * B</strong>.</span>
          </div>
          <div class="usage-guide-step">
            <span>Precedence is <strong>parentheses</strong>, then <strong>inv(...)</strong>, then <strong>composition A(B)</strong>, then <strong>product A * B</strong>.</span>
          </div>
          <div class="usage-guide-column">
            <div class="usage-guide-subtitle">Reference Specs</div>
            <div class="usage-guide-step">
              <span>Copy this entire block into the <strong>Layouts</strong> textbox above.</span>
            </div>
            <div class="usage-guide-example">
              <code>Tile2x1: [T,W] -&gt; [Y,X]</code>
              <code>T: [[0,1],[0,2]]</code>
              <code>W: [[1,0]]</code>
              <code></code>
              <code>Swizzle: [Y,X] -&gt; [S,B]</code>
              <code>Y: [[1,1],[2,2]]</code>
              <code>X: [[0,1],[0,2]]</code>
              <code></code>
              <code>GetT: [Y1,X1] -&gt; [T]</code>
              <code>Y1: [[1],[2],[16]]</code>
              <code>X1: [[4],[8]]</code>
              <code></code>
              <code>GetW: [Y2,X2] -&gt; [W]</code>
              <code>Y2: [[1]]</code>
              <code>X2: [[2]]</code>
              <code></code>
              <code>GetR: [X3] -&gt; [R]</code>
              <code>X3: [[1]]</code>
              <code></code>
              <code>Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[2,0],[4,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
              <code></code>
              <code>Sparse_Block: [T,W,R] -&gt; [A,B]</code>
              <code>T: [[4,0],[8,0],[0,1],[0,2],[0,4]]</code>
              <code>W: [[0,8],[0,16]]</code>
              <code>R: [[1,0]]</code>
            </div>
            <div class="usage-guide-subtitle">Examples</div>
            <div class="usage-guide-step">
              <span>Copy any one pale-box example below into <strong>Layout Operation</strong>.</span>
            </div>
            <div class="usage-guide-subtitle">Single Layout</div>
            <div class="usage-guide-example"><code>Tile2x1</code></div>
            <div class="usage-guide-example"><code>Swizzle</code></div>
            <div class="usage-guide-example"><code>Sparse_Block</code></div>
            <div class="usage-guide-subtitle">Composition</div>
            <div class="usage-guide-example"><code>Swizzle(Tile2x1)</code></div>
            <div class="usage-guide-subtitle">Inverse</div>
            <div class="usage-guide-example"><code>inv(GetT)</code></div>
            <div class="usage-guide-example"><code>inv(Swizzle(Tile2x1))</code></div>
            <div class="usage-guide-example"><code>inv(Swizzle)(Tile2x1)</code></div>
            <div class="usage-guide-subtitle">Product</div>
            <div class="usage-guide-example"><code>GetT * GetW * GetR</code></div>
            <div class="usage-guide-example"><code>Swizzle * Swizzle</code></div>
            <div class="usage-guide-subtitle">Needs Parentheses</div>
            <div class="usage-guide-example"><code>(Swizzle * Swizzle)(Tile2x1)</code></div>
            <div class="usage-guide-subtitle">Composition + Inverse + Product</div>
            <div class="usage-guide-example"><code>Sparse_Block(inv(inv(GetT)) * GetW * GetR)</code></div>
            <div class="usage-guide-example"><code>inv(GetT * GetW * GetR)(inv(Block))</code></div>
          </div>
        </div>
      </details>
    `;
}

function linearLayoutColorHelpHtml(): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Drag a propagated axis from <strong>Available Axes</strong> onto H, S, or L to control that channel.</span>
          </div>
          <div class="usage-guide-step">
            <span>Drag an assigned chip back to the pool to clear it, or drag between channels to swap assignments.</span>
          </div>
          <div class="usage-guide-step">
            <span>Toggle <strong>Propagate Outputs</strong> to switch between input-driven and output-driven labels/colors, then click <strong>Recolor Layout</strong> to apply the new mapping.</span>
          </div>
        </div>
      </details>
    `;
}

export function renderCellTextWidget(ctx: LinearLayoutUiContext): void {
    ctx.cellTextWidget.innerHTML = '';
    ctx.cellTextWidget.classList.add('hidden');
}

export function renderLinearLayoutWidget(ctx: LinearLayoutUiContext): void {
    const showLocalStatus = ctx.state.linearLayoutNotice?.text !== VISIBLE_TENSORS_ERROR;
    const statusClass = ctx.state.linearLayoutNotice?.tone === 'success' ? 'success-box' : 'error-box';
    const status = showLocalStatus && ctx.state.linearLayoutNotice ? `<div class="${statusClass}">${escapeInfo(ctx.state.linearLayoutNotice.text)}</div>` : '';
    const matrixBlock = !ctx.state.showLinearLayoutMatrix
        ? ''
        : `<div class="mono-block linear-layout-matrix-preview">${ctx.state.linearLayoutMatrixPreview}</div>`;
    ctx.linearLayoutWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout', 'Define one or more named injective layouts, then use Layout Operation to build the rendered tensor chain.').replace('<h2>Linear Layout Specifications</h2>', '<h2>Layout Specs</h2>')}
      <div class="widget-body">
        <p class="widget-copy">Each specification starts with <span class="inline-code">name: [inputs] -> [outputs]</span>, followed by exactly one labeled basis row per input such as <span class="inline-code">T: [[1,0],[0,1]]</span>. Separate specifications with blank lines. Layout Operation supports names, <span class="inline-code">inv(...)</span>, <span class="inline-code">*</span>, and parentheses.</p>
        <div class="field">
          ${labelWithInfo('Layouts', 'Enter one or more specification blocks. Each block has one signature line plus one labeled basis row per input label.', 'linear-layout-specs')}
          ${linearLayoutSpecsHelpHtml()}
          <textarea id="linear-layout-specs" class="compact-textarea" rows="8" spellcheck="false">${escapeInfo(ctx.state.linearLayoutState.specsText)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Layout Operation', 'Enter the layout expression to visualize, such as A(B), inv(A), A * B, or parenthesized combinations.', 'linear-layout-operation')}
          ${linearLayoutOperationHelpHtml()}
          <textarea id="linear-layout-operation" class="compact-textarea" rows="2" spellcheck="false">${escapeInfo(ctx.state.linearLayoutState.operationText)}</textarea>
        </div>
        <div class="field">
          ${labelWithInfo('Input Tensor Name', 'Sets the display name of the root tensor at the start of the rendered chain.', 'linear-layout-input-name')}
          <input id="linear-layout-input-name" type="text" value="${escapeInfo(ctx.state.linearLayoutState.inputName)}" />
        </div>
        <div class="button-row linear-layout-action-row">
          <button class="primary-button" id="linear-layout-apply" type="button" title="Build the current layout chain and render the resulting tensors.">Render Layout</button>
          <button class="secondary-button" id="linear-layout-matrix" type="button" title="${ctx.state.showLinearLayoutMatrix ? 'Hide the matrix blocks for the current layout chain.' : 'Show the matrix blocks for the current layout chain.'}">${ctx.state.showLinearLayoutMatrix ? 'Hide Matrix' : 'Show Matrix'}</button>
          <button class="secondary-button" id="linear-layout-copy" type="button" title="Copy Python initialization code for the current layout definitions and operation.">Copy Init Code</button>
        </div>
        ${status}
        ${matrixBlock}
      </div>
    `;

    const specsInput = ctx.linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-specs');
    const operationInput = ctx.linearLayoutWidget.querySelector<HTMLTextAreaElement>('#linear-layout-operation');
    const inputNameInput = ctx.linearLayoutWidget.querySelector<HTMLInputElement>('#linear-layout-input-name');
    const apply = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-apply');
    const copy = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-copy');
    const matrix = ctx.linearLayoutWidget.querySelector<HTMLButtonElement>('#linear-layout-matrix');
    if (specsInput) autosizeTextarea(specsInput);
    if (operationInput) autosizeTextarea(operationInput);
    specsInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.specsText = specsInput.value;
        autosizeTextarea(specsInput);
    });
    operationInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.operationText = operationInput.value;
        autosizeTextarea(operationInput);
    });
    inputNameInput?.addEventListener('input', () => {
        ctx.state.linearLayoutState.inputName = inputNameInput.value;
    });
    apply?.addEventListener('click', async () => {
        await applyLinearLayoutSpec(ctx);
    });
    copy?.addEventListener('click', async () => {
        try {
            await copyText(buildComposeRuntime(ctx.state.linearLayoutState).pythonCode);
            ctx.state.linearLayoutNotice = { tone: 'success', text: 'Copied Python init code.' };
        } catch (error) {
            ctx.state.linearLayoutNotice = { tone: 'error', text: error instanceof Error ? error.message : String(error) };
        }
        renderLinearLayoutWidget(ctx);
    });
    matrix?.addEventListener('click', () => {
        ctx.state.showLinearLayoutMatrix = !ctx.state.showLinearLayoutMatrix;
        renderLinearLayoutWidget(ctx);
    });
    renderLinearLayoutVisibleTensorsWidget(ctx);
    renderLinearLayoutColorWidget(ctx);
}

export function renderLinearLayoutVisibleTensorsWidget(ctx: LinearLayoutUiContext): void {
    const tab = activeLinearLayoutTab(ctx);
    const meta = tab ? composeLayoutMetaForTab(tab) : null;
    if (!meta) {
        ctx.linearLayoutVisibleTensorsWidget.classList.add('hidden');
        ctx.linearLayoutVisibleTensorsWidget.innerHTML = '';
        return;
    }
    ctx.linearLayoutVisibleTensorsWidget.classList.remove('hidden');
    const status = ctx.state.linearLayoutNotice?.text === VISIBLE_TENSORS_ERROR
        ? `<div class="error-box">${escapeInfo(ctx.state.linearLayoutNotice.text)}</div>`
        : '';
    ctx.linearLayoutVisibleTensorsWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout-visible-tensors', 'Toggle which tensors in the render chain stay visible for the current tab.')}
      <div class="widget-body">
        <div class="checklist-field">
          ${meta.tensors.map((tensor) => `
            <label class="checklist-row" for="linear-layout-visible-${tensor.id}">
              <span>${escapeInfo(tensor.title)}</span>
              <input id="linear-layout-visible-${tensor.id}" type="checkbox" ${ctx.state.linearLayoutState.visibleTensors[tensor.id] !== false ? 'checked' : ''} />
            </label>
          `).join('')}
        </div>
        ${status}
      </div>
    `;
    meta.tensors.forEach((tensor) => {
        ctx.linearLayoutVisibleTensorsWidget.querySelector<HTMLInputElement>(`#linear-layout-visible-${CSS.escape(tensor.id)}`)?.addEventListener('change', async (event) => {
            const target = event.currentTarget as HTMLInputElement;
            ctx.state.linearLayoutState.visibleTensors[tensor.id] = target.checked;
            await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
        });
    });
}

export async function applyLinearLayoutSpec(
    ctx: LinearLayoutUiContext,
    options: { replaceTabs?: boolean; silent?: boolean; preserveTensorViews?: boolean } = {},
): Promise<boolean> {
    try {
        const activeTab = activeLinearLayoutTab(ctx);
        const activeMeta = activeTab ? composeLayoutMetaForTab(activeTab) : null;
        const layoutChanged = !activeMeta
            || activeMeta.specsText !== ctx.state.linearLayoutState.specsText
            || activeMeta.operationText !== ctx.state.linearLayoutState.operationText;
        const runtime = buildComposeRuntime(ctx.state.linearLayoutState);
        if (layoutChanged || !mappingMatchesLabels(
            ctx.state.linearLayoutState.mapping,
            propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0],
        )) {
            const autoColor = autoColorLayoutState(
                ctx.state.linearLayoutState.specsText,
                ctx.state.linearLayoutState.operationText,
                ctx.state.linearLayoutState.propagateOutputs,
            );
            ctx.state.linearLayoutState.mapping = autoColor.mapping;
            ctx.state.linearLayoutState.ranges = autoColor.ranges;
        }
        refreshLinearLayoutMatrixPreview(ctx);
        const document = createComposeLayoutDocument(
            ctx.state.linearLayoutState,
            ctx.viewer.getSnapshot(),
            undefined,
            options.preserveTensorViews ? preservedLinearLayoutTensorViews(ctx) : undefined,
        );
        ctx.state.linearLayoutCellTextState = normalizeCellTextState(
            ctx.state.linearLayoutCellTextState,
            propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0],
        );
        storeLinearLayoutState(ctx.state.linearLayoutState);
        await upsertLinearLayoutTab(ctx, document, options.replaceTabs);
        const activeTitle = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId())?.title ?? document.title;
        ctx.state.linearLayoutNotice = options.silent ? null : { tone: 'success', text: `Rendered ${activeTitle}.` };
        renderLinearLayoutWidget(ctx);
        return true;
    } catch (error) {
        ctx.state.linearLayoutNotice = { tone: 'error', text: error instanceof Error ? error.message : String(error) };
        renderLinearLayoutWidget(ctx);
        return false;
    }
}

export async function loadBakedLinearLayoutTabs(ctx: LinearLayoutUiContext): Promise<boolean> {
    const examples = bakedComposeLayoutExamples();
    if (examples.length === 0) return false;
    ctx.state.linearLayoutStates.clear();
    ctx.state.linearLayoutCellTextStates.clear();
    ctx.state.linearLayoutMultiInputStates.clear();
    ctx.state.linearLayoutTensorViewsStates.clear();
    ctx.setSessionTabs(examples.map(({ state, title }, index) => {
        const document = createComposeLayoutDocument(state, undefined, title);
        const id = `tab-${index + 1}`;
        ctx.state.linearLayoutStates.set(id, cloneLinearLayoutState(state));
        ctx.state.linearLayoutCellTextStates.set(id, defaultLinearLayoutCellTextState());
        ctx.state.linearLayoutMultiInputStates.set(id, defaultLinearLayoutMultiInputState());
        ctx.state.linearLayoutTensorViewsStates.set(id, snapshotTensorViews(document.manifest.viewer));
        return { ...document, id, title };
    }));
    ctx.state.linearLayoutSelectionMaps.clear();
    const initialTabId = ctx.getSessionTabs()[0]?.id ?? null;
    if (!initialTabId) return false;
    await ctx.loadTab(initialTabId);
    await settleInitialLayout(ctx);
    await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
    return true;
}

async function upsertLinearLayoutTab(
    ctx: LinearLayoutUiContext,
    document: ReturnType<typeof createComposeLayoutDocument>,
    replaceTabs = false,
): Promise<void> {
    const activeTab = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId());
    const targetId = activeTab?.id ?? document.id;
    const targetTitle = activeTab?.title ?? document.title;
    const nextDocument = { ...document, id: targetId, title: targetTitle };
    ctx.state.linearLayoutStates.set(targetId, cloneLinearLayoutState(ctx.state.linearLayoutState));
    ctx.state.linearLayoutCellTextStates.set(targetId, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
    ctx.state.linearLayoutMultiInputStates.set(targetId, cloneLinearLayoutMultiInputState(ctx.state.linearLayoutMultiInputState));
    ctx.state.linearLayoutSelectionMaps.delete(targetId);
    if (replaceTabs || ctx.getSessionTabs().length === 0) {
        ctx.setSessionTabs([nextDocument]);
    } else {
        const index = ctx.getSessionTabs().findIndex((tab) => tab.id === targetId);
        ctx.setSessionTabs(index === -1
            ? [...ctx.getSessionTabs(), nextDocument]
            : ctx.getSessionTabs().map((tab, tabIndex) => tabIndex === index ? nextDocument : tab));
    }
    await ctx.loadTab(targetId);
}

function activeLinearLayoutTab(ctx: LinearLayoutUiContext) {
    const tab = ctx.getActiveTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

function linearLayoutPropagationLabels(ctx: LinearLayoutUiContext): { labels: string[]; injective: boolean } {
    const tab = activeLinearLayoutTab(ctx);
    const meta = tab ? composeLayoutMetaForTab(tab) : null;
    try {
        const runtime = buildComposeRuntime(ctx.state.linearLayoutState);
        return { labels: propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0], injective: runtime.injective };
    } catch {
        return {
            labels: ctx.state.linearLayoutState.propagateOutputs
                ? meta?.finalOutputLabels.slice() ?? []
                : meta?.rootInputLabels.slice() ?? [],
            injective: meta?.injective ?? true,
        };
    }
}

function normalizeCellTextState(state: Record<string, boolean>, labels: string[]): Record<string, boolean> {
    return Object.fromEntries(labels.map((label) => [label, state[label] ?? false]));
}

function mappingMatchesLabels(mapping: Record<LinearLayoutChannel, string>, labels: string[]): boolean {
    const allowed = new Set(labels);
    return LINEAR_LAYOUT_CHANNELS.every((channel) => mapping[channel] === 'none' || allowed.has(mapping[channel]));
}

function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

async function copyText(text: string): Promise<void> {
    if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return;
    }
    const input = document.createElement('textarea');
    input.value = text;
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.select();
    document.execCommand('copy');
    document.body.removeChild(input);
}

async function settleInitialLayout(ctx: LinearLayoutUiContext): Promise<void> {
    if ('fonts' in document) {
        try {
            await (document as Document & { fonts: { ready: Promise<unknown> } }).fonts.ready;
        } catch {
            // ignore font-settlement errors
        }
    }
    let stableFrames = 0;
    let previousSize = '';
    while (stableFrames < 2) {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        const nextSize = `${ctx.viewport.clientWidth}x${ctx.viewport.clientHeight}`;
        stableFrames = nextSize === previousSize ? stableFrames + 1 : 0;
        previousSize = nextSize;
    }
    ctx.viewer.resize();
    ctx.viewer.refitView();
}

function renderLinearLayoutColorWidget(ctx: LinearLayoutUiContext): void {
    const activeElement = document.activeElement;
    const focusedInput = activeElement instanceof HTMLInputElement && ctx.linearLayoutColorWidget.contains(activeElement)
        ? { id: activeElement.id, start: activeElement.selectionStart, end: activeElement.selectionEnd }
        : null;
    const channelLabels: Record<LinearLayoutChannel, string> = { H: 'Hue', S: 'Sat', L: 'Light' };
    const { labels, injective } = linearLayoutPropagationLabels(ctx);
    const assignedLabels = new Set(
        LINEAR_LAYOUT_CHANNELS
            .map((channel) => ctx.state.linearLayoutState.mapping[channel])
            .filter((label): label is string => label !== 'none' && labels.includes(label)),
    );
    const availableLabels = labels.filter((label) => !assignedLabels.has(label));
    ctx.linearLayoutColorWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout-color', 'Configure propagated cell labels, H/S/L color mapping, and whether labels/colors follow inputs forward or outputs backward.')}
      <div class="widget-body">
        ${linearLayoutColorHelpHtml()}
        <div class="field">
          <label class="checklist-row" for="linear-layout-propagate-outputs">
            <span class="label-row"><span class="meta-label">Propagate Outputs</span>${infoButton(injective
                ? 'When off, colors and cell text come from the input space and flow forward. When on, they come from the final output space and flow backward.'
                : 'When off, non-injective layouts keep the current popup, ghost-layer, and multi-input behavior. When on, colors and cell text come from the final output space.')}</span>
            <input id="linear-layout-propagate-outputs" type="checkbox" ${ctx.state.linearLayoutState.propagateOutputs ? 'checked' : ''} />
          </label>
        </div>
        <div class="field">
          <div class="label-row"><span class="meta-label">Cell Text</span>${infoButton('Choose which propagated axes are drawn as per-cell labels. The available labels follow the current Propagate Outputs mode.')}</div>
          <div class="checklist-field">
            ${labels.map((label) => `
              <label class="checklist-row" for="cell-text-${label}">
                <span>${label}</span>
                <input id="cell-text-${label}" type="checkbox" ${ctx.state.linearLayoutCellTextState[label] ? 'checked' : ''} />
              </label>
            `).join('')}
          </div>
        </div>
        <div class="field">
          ${labelWithInfo('Available Axes', 'Drag one propagated axis onto H, S, or L. Drag a colored axis back here to clear that channel.')}
          <div class="mapping-pool mapping-drop-zone" data-pool="true">
            ${availableLabels.map((label) => `<button class="mapping-chip" type="button" draggable="true" data-axis="${label}">${label}</button>`).join('')}
            ${availableLabels.length === 0 ? '<span class="mapping-empty">all axes assigned</span>' : ''}
          </div>
        </div>
        ${LINEAR_LAYOUT_CHANNELS.map((channel) => `
          <div class="inline-row mapping-row">
            <span class="range-label">${channelLabels[channel]}</span>
            <div class="mapping-drop-zone" data-channel="${channel}">
              ${ctx.state.linearLayoutState.mapping[channel] !== 'none' && labels.includes(ctx.state.linearLayoutState.mapping[channel] as string)
        ? `<button class="mapping-chip mapping-chip-assigned" type="button" draggable="true" data-channel="${channel}" data-axis="${ctx.state.linearLayoutState.mapping[channel]}">${ctx.state.linearLayoutState.mapping[channel]}</button>`
        : '<span class="mapping-empty">none</span>'}
            </div>
            <input id="linear-layout-${channel.toLowerCase()}-min" type="number" step="0.01" value="${escapeInfo(ctx.state.linearLayoutState.ranges[channel][0])}" />
            <span class="range-separator">to</span>
            <input id="linear-layout-${channel.toLowerCase()}-max" type="number" step="0.01" value="${escapeInfo(ctx.state.linearLayoutState.ranges[channel][1])}" />
          </div>
        `).join('')}
        <div class="button-row">
          <button class="primary-button" id="linear-layout-recolor" type="button" title="Apply the current H/S/L axis assignments and numeric ranges to recolor the layout.">Recolor Layout</button>
        </div>
      </div>
    `;

    ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-propagate-outputs')?.addEventListener('change', async () => {
        ctx.state.linearLayoutState.propagateOutputs = ctx.linearLayoutColorWidget
            .querySelector<HTMLInputElement>('#linear-layout-propagate-outputs')?.checked ?? false;
        const autoColor = autoColorLayoutState(
            ctx.state.linearLayoutState.specsText,
            ctx.state.linearLayoutState.operationText,
            ctx.state.linearLayoutState.propagateOutputs,
        );
        ctx.state.linearLayoutState.mapping = autoColor.mapping;
        ctx.state.linearLayoutState.ranges = autoColor.ranges;
        ctx.state.linearLayoutCellTextState = normalizeCellTextState(
            ctx.state.linearLayoutCellTextState,
            linearLayoutPropagationLabels(ctx).labels,
        );
        const tab = activeLinearLayoutTab(ctx);
        if (tab) ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
        await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
    });
    const syncCellText = (): void => {
        ctx.state.linearLayoutCellTextState = Object.fromEntries(labels.map((label) => [
            label,
            ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#cell-text-${CSS.escape(label)}`)?.checked ?? false,
        ]));
        const tab = activeLinearLayoutTab(ctx);
        if (tab) ctx.state.linearLayoutCellTextStates.set(tab.id, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
        applyLinearLayoutCellText(ctx);
    };
    labels.forEach((label) => {
        ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#cell-text-${CSS.escape(label)}`)?.addEventListener('change', syncCellText);
    });

    const writeDragPayload = (event: DragEvent, payload: Record<string, string>): void => {
        event.dataTransfer?.setData('application/x-linear-layout-mapping', JSON.stringify(payload));
        event.dataTransfer?.setData('text/plain', JSON.stringify(payload));
        if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
    };
    const readDragPayload = (event: DragEvent): Record<string, string> | null => {
        const raw = event.dataTransfer?.getData('application/x-linear-layout-mapping') || event.dataTransfer?.getData('text/plain');
        if (!raw) return null;
        try {
            return JSON.parse(raw) as Record<string, string>;
        } catch {
            return null;
        }
    };
    ctx.linearLayoutColorWidget.querySelectorAll<HTMLElement>('[draggable="true"]').forEach((element) => {
        element.addEventListener('dragstart', (event) => {
            const channel = element.dataset.channel;
            const axis = element.dataset.axis;
            if (axis) writeDragPayload(event, channel ? { kind: 'channel', channel, axis } : { kind: 'axis', axis });
        });
    });
    ctx.linearLayoutColorWidget.querySelectorAll<HTMLElement>('.mapping-drop-zone').forEach((element) => {
        element.addEventListener('dragover', (event) => {
            event.preventDefault();
            element.classList.add('drag-over');
        });
        element.addEventListener('dragleave', () => {
            element.classList.remove('drag-over');
        });
        element.addEventListener('drop', (event) => {
            event.preventDefault();
            element.classList.remove('drag-over');
            const payload = readDragPayload(event);
            const targetChannel = element.dataset.channel as LinearLayoutChannel | undefined;
            if (!payload) return;
            if (element.dataset.pool === 'true') {
                if (payload.kind === 'channel' && payload.channel) {
                    ctx.state.linearLayoutState.mapping[payload.channel as LinearLayoutChannel] = 'none';
                    renderLinearLayoutColorWidget(ctx);
                }
                return;
            }
            if (!targetChannel) return;
            if (payload.kind === 'channel') {
                const sourceChannel = payload.channel as LinearLayoutChannel;
                if (!sourceChannel || sourceChannel === targetChannel) return;
                const sourceAxis = ctx.state.linearLayoutState.mapping[sourceChannel];
                ctx.state.linearLayoutState.mapping[sourceChannel] = ctx.state.linearLayoutState.mapping[targetChannel];
                ctx.state.linearLayoutState.mapping[targetChannel] = sourceAxis;
            } else if (payload.axis) {
                LINEAR_LAYOUT_CHANNELS.forEach((channel) => {
                    if (ctx.state.linearLayoutState.mapping[channel] === payload.axis) ctx.state.linearLayoutState.mapping[channel] = 'none';
                });
                ctx.state.linearLayoutState.mapping[targetChannel] = payload.axis;
            }
            renderLinearLayoutColorWidget(ctx);
        });
    });
    ([
        ['H', 'h'],
        ['S', 's'],
        ['L', 'l'],
    ] as const).forEach(([channel, key]) => {
        ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#linear-layout-${key}-min`)?.addEventListener('input', (event) => {
            ctx.state.linearLayoutState.ranges[channel][0] = (event.currentTarget as HTMLInputElement).value;
        });
        ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#linear-layout-${key}-max`)?.addEventListener('input', (event) => {
            ctx.state.linearLayoutState.ranges[channel][1] = (event.currentTarget as HTMLInputElement).value;
        });
    });
    ctx.linearLayoutColorWidget.querySelector<HTMLButtonElement>('#linear-layout-recolor')?.addEventListener('click', async () => {
        await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
    });
    if (focusedInput) {
        const nextInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#${focusedInput.id}`);
        nextInput?.focus();
        if (nextInput && focusedInput.start !== null && focusedInput.end !== null) {
            nextInput.setSelectionRange(focusedInput.start, focusedInput.end);
        }
    }
}
