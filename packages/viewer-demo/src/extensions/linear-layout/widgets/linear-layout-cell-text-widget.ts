import type { LinearLayoutUiContext } from '../linear-layout-state.js';

/**
 * render cell text widget for the current viewer state.
 */
export function renderCellTextWidget(ctx: LinearLayoutUiContext): void {
    ctx.cellTextWidget.innerHTML = '';
    ctx.cellTextWidget.classList.add('hidden');
}
