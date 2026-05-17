import type { LinearLayoutUiContext } from '../linear-layout-state.js';

/**
 * render cell text widget for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * renderCellTextWidget(ctx);
 */
export function renderCellTextWidget(ctx: LinearLayoutUiContext): void {
    ctx.cellTextWidget.innerHTML = '';
    ctx.cellTextWidget.classList.add('hidden');
}
