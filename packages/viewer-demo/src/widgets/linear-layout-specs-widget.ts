import { escapeInfo, labelWithInfo } from '../app-format.js';
import {
    bakedComposeLayoutExamples,
    buildComposeRuntime,
    createComposeLayoutDocument,
} from '../linear-layout.js';
import {
    cloneLinearLayoutMultiInputState,
    cloneLinearLayoutState,
    composeLayoutMetaForTab,
    defaultLinearLayoutCellTextState,
    defaultLinearLayoutMultiInputState,
    snapshotTensorViews,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';
import {
    VISIBLE_TENSORS_ERROR,
    autosizeTextarea,
    copyText,
    settleInitialLayout,
} from './linear-layout-widget-shared.js';

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
        ctx.renderLinearLayoutEditorWidgets();
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
        const meta = composeLayoutMetaForTab(document);
        ctx.state.linearLayoutStates.set(id, cloneLinearLayoutState(state));
        ctx.state.linearLayoutCellTextStates.set(id, defaultLinearLayoutCellTextState(meta?.rootInputLabels ?? []));
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
    ctx.renderLinearLayoutEditorWidgets();
    return true;
}
