import type { LinearLayoutUiContext } from '../linear-layout-state.js';
import { renderLinearLayoutColorWidget } from './linear-layout-color-widget.js';
import { renderLinearLayoutWidget } from './linear-layout-specs-widget.js';
import { renderLinearLayoutVisibleTensorsWidget } from './linear-layout-visible-tensors-widget.js';

/**
 * render linear layout editor widgets for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * renderLinearLayoutEditorWidgets(ctx);
 */
export function renderLinearLayoutEditorWidgets(ctx: LinearLayoutUiContext): void {
    renderLinearLayoutWidget(ctx);
    renderLinearLayoutVisibleTensorsWidget(ctx);
    renderLinearLayoutColorWidget(ctx);
}
