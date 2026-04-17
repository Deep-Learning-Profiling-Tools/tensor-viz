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

export function activeLinearLayoutTab(ctx: LinearLayoutUiContext): LoadedBundleDocument | null {
    const tab = ctx.getActiveTab();
    return tab && isLinearLayoutTab(tab) ? tab : null;
}

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

export function normalizeCellTextState(state: Record<string, boolean>, labels: string[]): Record<string, boolean> {
    return Object.fromEntries(labels.map((label) => [label, state[label] ?? false]));
}

export function mappingMatchesLabels(mapping: Record<LinearLayoutChannel, string>, labels: string[]): boolean {
    const allowed = new Set(labels);
    return LINEAR_LAYOUT_CHANNELS.every((channel) => mapping[channel] === 'none' || allowed.has(mapping[channel]));
}

export function autosizeTextarea(textarea: HTMLTextAreaElement): void {
    textarea.style.height = '0';
    textarea.style.height = `${textarea.scrollHeight}px`;
}

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
