import type { LoadedBundleDocument } from '@tensor-viz/viewer-core';
import { buildComposeRuntime, propagationLabels } from '../linear-layout.js';
import {
    composeLayoutMetaForTab,
    isLinearLayoutTab,
    type LinearLayoutChannel,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';

export const LINEAR_LAYOUT_CHANNELS: LinearLayoutChannel[] = ['H', 'S', 'L'];
export const VISIBLE_TENSORS_ERROR = 'At least one tensor in the render chain must stay visible.';

/**
 * return active linear layout tab for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * activeLinearLayoutTab(ctx);
 */
export function activeLinearLayoutTab(ctx: LinearLayoutUiContext): LoadedBundleDocument | null {
    const tab = ctx.getActiveTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

/**
 * return linear layout propagation labels for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @returns Object containing computed state for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * linearLayoutPropagationLabels(ctx);
 */
export function linearLayoutPropagationLabels(ctx: LinearLayoutUiContext): { labels: string[]; injective: boolean } {
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

/**
 * normalize cell text state for the current viewer state.
 *
 * @param state - State object read or updated by this operation.
 * @param labels - labels input used by this operation (string[]).
 * @returns Computed Record<string, boolean> value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizeCellTextState(state, labels);
 */
export function normalizeCellTextState(state: Record<string, boolean>, labels: string[]): Record<string, boolean> {
    return Object.fromEntries(labels.map((label) => [label, state[label] ?? true]));
}

/**
 * return mapping matches labels for the current viewer state.
 *
 * @param mapping - Mapping data used by this operation.
 * @param labels - labels input used by this operation (string[]).
 * @returns Whether the requested condition holds.
 * @noThrows This function has no direct throw path.
 * @example
 * mappingMatchesLabels(mapping, labels);
 */
export function mappingMatchesLabels(mapping: Record<LinearLayoutChannel, string>, labels: string[]): boolean {
    const allowed = new Set(labels);
    return LINEAR_LAYOUT_CHANNELS.every((channel) => mapping[channel] === 'none' || allowed.has(mapping[channel]));
}

/**
 * return autosize textarea for the current viewer state.
 *
 * @param textarea - textarea input used by this operation (HTMLTextAreaElement).
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * autosizeTextarea(textarea);
 */
export function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

/**
 * return copy text for the current viewer state.
 *
 * @param text - Text supplied by the caller.
 * @returns Promise that resolves to the computed value.
 * @noThrows This function has no direct throw path.
 * @example
 * copyText(text);
 */
export async function copyText(text: string): Promise<void> {
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

/**
 * return settle initial layout for the current viewer state.
 *
 * @param ctx - Context object that supplies viewer state and DOM references.
 * @returns Promise that resolves to the computed value.
 * @noThrows This function has no direct throw path.
 * @example
 * settleInitialLayout(ctx);
 */
export async function settleInitialLayout(ctx: LinearLayoutUiContext): Promise<void> {
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
