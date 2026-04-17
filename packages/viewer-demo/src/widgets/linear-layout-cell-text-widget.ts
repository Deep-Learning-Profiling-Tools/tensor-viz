import type { LinearLayoutUiContext } from '../linear-layout-state.js';

export function renderCellTextWidget(ctx: LinearLayoutUiContext): void {
    ctx.cellTextWidget.innerHTML = '';
    ctx.cellTextWidget.classList.add('hidden');
}
