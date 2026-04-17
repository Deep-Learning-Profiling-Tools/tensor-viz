import type { LinearLayoutUiContext } from '../linear-layout-state.js';
import { renderLinearLayoutColorWidget } from './linear-layout-color-widget.js';
import { renderLinearLayoutWidget } from './linear-layout-specs-widget.js';
import { renderLinearLayoutVisibleTensorsWidget } from './linear-layout-visible-tensors-widget.js';

export function renderLinearLayoutEditorWidgets(ctx: LinearLayoutUiContext): void {
    renderLinearLayoutWidget(ctx);
    renderLinearLayoutVisibleTensorsWidget(ctx);
    renderLinearLayoutColorWidget(ctx);
}
