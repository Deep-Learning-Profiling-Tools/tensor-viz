import { escapeInfo } from '../../../app-format.js';
import { composeLayoutMetaForTab, type LinearLayoutUiContext } from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';
import { VISIBLE_TENSORS_ERROR, activeLinearLayoutTab } from './linear-layout-widget-shared.js';

/**
 * render linear layout visible tensors widget for the current viewer state.
 */
export function renderLinearLayoutVisibleTensorsWidget(ctx: LinearLayoutUiContext): void {
    const tab = activeLinearLayoutTab(ctx);
    const meta = tab ? composeLayoutMetaForTab(tab) : null;
    if (!meta) {
        ctx.linearLayoutVisibleTensorsWidget.classList.add('hidden');
        ctx.linearLayoutVisibleTensorsWidget.innerHTML = '';
        return;
    }
    ctx.linearLayoutVisibleTensorsWidget.classList.remove('hidden');
    // surface apply errors in this widget because toggles are the only control
    // that can hide the tensor whose spec currently fails to rebuild.
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
        // visibility edits rebuild the current layout while preserving any
        // per-tensor view strings the user has already typed.
        ctx.linearLayoutVisibleTensorsWidget.querySelector<HTMLInputElement>(`#linear-layout-visible-${CSS.escape(tensor.id)}`)?.addEventListener('change', async (event) => {
            const target = event.currentTarget as HTMLInputElement;
            ctx.state.linearLayoutState.visibleTensors[tensor.id] = target.checked;
            await applyLinearLayoutSpec(ctx, { silent: true, preserveTensorViews: true });
            ctx.renderLinearLayoutEditorWidgets();
        });
    });
}
