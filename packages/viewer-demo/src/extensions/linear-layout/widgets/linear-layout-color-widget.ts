import { escapeInfo, infoButton, labelWithInfo } from '../../../app-format.js';
import { autoColorLayoutState } from '../linear-layout.js';
import {
    cloneLinearLayoutCellTextState,
    type LinearLayoutChannel,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';
import { applyLinearLayoutCellText } from '../linear-layout-viewer-sync.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';
import {
    LINEAR_LAYOUT_CHANNELS,
    activeLinearLayoutTab,
    linearLayoutPropagationLabels,
    normalizeCellTextState,
} from './linear-layout-widget-shared.js';

/**
 * return linear layout color help html for the current viewer state.
 */
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

/**
 * return linear layout propagate outputs info for the current viewer state.
 */
export function linearLayoutPropagateOutputsInfo(injective: boolean): string {
    return injective
        ? 'When off, colors and cell text come from the input space and flow forward. When on, they come from the final output space and flow backward.'
        : 'When off, non-injective layouts keep the current popup, ghost-layer, and multi-input behavior. When on, colors and cell text come from the final output space.';
}

/**
 * toggle linear layout propagate outputs for the current viewer state.
 */
export async function toggleLinearLayoutPropagateOutputs(ctx: LinearLayoutUiContext): Promise<void> {
    ctx.state.linearLayoutState.propagateOutputs = !ctx.state.linearLayoutState.propagateOutputs;
    // propagation changes the coordinate space that H/S/L channels reference,
    // so recolor from the current spec instead of trying to reinterpret ranges.
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
    ctx.renderLinearLayoutEditorWidgets();
}

/**
 * render linear layout color widget for the current viewer state.
 */
export function renderLinearLayoutColorWidget(ctx: LinearLayoutUiContext): void {
    const activeElement = document.activeElement;
    const focusedInput = activeElement instanceof HTMLInputElement && ctx.linearLayoutColorWidget.contains(activeElement)
        ? { id: activeElement.id, start: activeElement.selectionStart, end: activeElement.selectionEnd }
        : null;
    const channelLabels: Record<LinearLayoutChannel, string> = { H: 'Hue', S: 'Sat', L: 'Light' };
    const { labels, injective } = linearLayoutPropagationLabels(ctx);
    // assigned labels disappear from the source pool; dragging a chip back to
    // the pool clears that color channel without needing a separate reset button.
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
            <span class="label-row"><span class="meta-label">Propagate Outputs</span>${infoButton(linearLayoutPropagateOutputsInfo(injective))}</span>
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
        ${LINEAR_LAYOUT_CHANNELS.map((channel) => {
            const assignedAxis = ctx.state.linearLayoutState.mapping[channel];
            const assigned = assignedAxis !== 'none' && labels.includes(assignedAxis);
            return `
          <div class="inline-row mapping-row">
            <span class="range-label">${channelLabels[channel]}</span>
            <div class="mapping-drop-zone" data-channel="${channel}">
              ${assigned
        ? `<button class="mapping-chip mapping-chip-assigned" type="button" draggable="true" data-channel="${channel}" data-axis="${assignedAxis}">${assignedAxis}</button>`
        : '<span class="mapping-empty">none</span>'}
            </div>
            <input id="linear-layout-${channel.toLowerCase()}-min" type="number" step="0.01" value="${escapeInfo(ctx.state.linearLayoutState.ranges[channel][0])}" />
            <span class="range-separator${assigned ? '' : ' range-separator-unused'}">to</span>
            <input
              id="linear-layout-${channel.toLowerCase()}-max"
              class="${assigned ? '' : 'unused-range-input'}"
              type="${assigned ? 'number' : 'text'}"
              ${assigned ? 'step="0.01"' : ''}
              value="${assigned ? escapeInfo(ctx.state.linearLayoutState.ranges[channel][1]) : 'N/A'}"
              ${assigned ? '' : 'readonly aria-readonly="true" title="This upper bound is unused while no axis is mapped to this color channel."'}
            />
          </div>
        `;
        }).join('')}
        <div class="button-row">
          <button class="primary-button" id="linear-layout-recolor" type="button" title="Apply the current H/S/L axis assignments and numeric ranges to recolor the layout.">Recolor Layout</button>
        </div>
      </div>
    `;

    ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>('#linear-layout-propagate-outputs')?.addEventListener('change', async () => {
        // checkbox state is read from the DOM because the click may come from a
        // label activation rather than from a direct input event target.
        const checked = ctx.linearLayoutColorWidget
            .querySelector<HTMLInputElement>('#linear-layout-propagate-outputs')?.checked ?? false;
        if (checked === ctx.state.linearLayoutState.propagateOutputs) return;
        await toggleLinearLayoutPropagateOutputs(ctx);
    });
    /** write checkbox state into tab-local cell-text settings and repaint labels. */
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
        // each checkbox writes to the persisted tab state immediately so a
        // later spec re-apply keeps the user's current label visibility.
        ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#cell-text-${CSS.escape(label)}`)?.addEventListener('change', syncCellText);
    });

    /** store a drag payload under a private type plus text fallback for browsers. */
    const writeDragPayload = (event: DragEvent, payload: Record<string, string>): void => {
        event.dataTransfer?.setData('application/x-linear-layout-mapping', JSON.stringify(payload));
        event.dataTransfer?.setData('text/plain', JSON.stringify(payload));
        if (event.dataTransfer) event.dataTransfer.effectAllowed = 'move';
    };
    /** parse one color-channel drag payload, returning null for unrelated drags. */
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
            // chips dragged from a channel carry both source channel and axis so
            // drops can distinguish swaps from pool-to-channel assignments.
            const channel = element.dataset.channel;
            const axis = element.dataset.axis;
            if (axis) writeDragPayload(event, channel ? { kind: 'channel', channel, axis } : { kind: 'axis', axis });
        });
    });
    ctx.linearLayoutColorWidget.querySelectorAll<HTMLElement>('.mapping-drop-zone').forEach((element) => {
        element.addEventListener('dragover', (event) => {
            // this widget accepts every internal chip drag; the payload is
            // validated again on drop before mutating mapping state.
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
        ctx.renderLinearLayoutEditorWidgets();
    });
    if (focusedInput) {
        const nextInput = ctx.linearLayoutColorWidget.querySelector<HTMLInputElement>(`#${focusedInput.id}`);
        nextInput?.focus();
        if (nextInput && focusedInput.start !== null && focusedInput.end !== null) {
            nextInput.setSelectionRange(focusedInput.start, focusedInput.end);
        }
    }
}
