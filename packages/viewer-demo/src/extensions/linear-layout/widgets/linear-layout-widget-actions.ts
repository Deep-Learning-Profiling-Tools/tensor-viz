import {
    autoColorLayoutState,
    buildComposeRuntime,
    createComposeLayoutDocument,
    propagationLabels,
} from '../linear-layout.js';
import {
    cloneLinearLayoutCellTextState,
    cloneLinearLayoutMultiInputState,
    cloneLinearLayoutState,
    composeLayoutMetaForTab,
    refreshLinearLayoutMatrixPreview,
    storeLinearLayoutState,
    type LinearLayoutUiContext,
} from '../linear-layout-state.js';
import { preservedLinearLayoutTensorViews } from '../linear-layout-viewer-sync.js';
import {
    activeLinearLayoutTab,
    mappingMatchesLabels,
    normalizeCellTextState,
} from './linear-layout-widget-shared.js';

/**
 * apply linear layout spec for the current viewer state.
 */
export async function applyLinearLayoutSpec(
    ctx: LinearLayoutUiContext,
    options: { replaceTabs?: boolean; silent?: boolean; preserveTensorViews?: boolean } = {},
): Promise<boolean> {
    try {
        const activeTab = activeLinearLayoutTab(ctx);
        const activeMeta = activeTab ? composeLayoutMetaForTab(activeTab) : null;
        // recolor only when the evaluated label space changed or the current
        // mapping references labels that no longer exist after an edit.
        const layoutChanged = !activeMeta
            || activeMeta.specsText !== ctx.state.linearLayoutState.specsText
            || activeMeta.operationText !== ctx.state.linearLayoutState.operationText;
        const runtime = buildComposeRuntime(ctx.state.linearLayoutState);
        if (layoutChanged || !mappingMatchesLabels(
            ctx.state.linearLayoutState.mapping,
            propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0],
        )) {
            const autoColor = autoColorLayoutState(
                ctx.state.linearLayoutState.specsText,
                ctx.state.linearLayoutState.operationText,
                ctx.state.linearLayoutState.propagateOutputs,
            );
            ctx.state.linearLayoutState.mapping = autoColor.mapping;
            ctx.state.linearLayoutState.ranges = autoColor.ranges;
        }
        refreshLinearLayoutMatrixPreview(ctx);
        const document = createComposeLayoutDocument(
            ctx.state.linearLayoutState,
            ctx.viewer.getSnapshot(),
            undefined,
            options.preserveTensorViews ? preservedLinearLayoutTensorViews(ctx) : undefined,
        );
        ctx.state.linearLayoutCellTextState = normalizeCellTextState(
            ctx.state.linearLayoutCellTextState,
            propagationLabels(runtime, ctx.state.linearLayoutState.propagateOutputs)[0],
        );
        storeLinearLayoutState(ctx.state.linearLayoutState);
        await upsertLinearLayoutTab(ctx, document, options.replaceTabs);
        const activeTitle = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId())?.title ?? document.title;
        ctx.state.linearLayoutNotice = options.silent ? null : { tone: 'success', text: `Rendered ${activeTitle}.` };
        return true;
    } catch (error) {
        ctx.state.linearLayoutNotice = { tone: 'error', text: error instanceof Error ? error.message : String(error) };
        return false;
    }
}

/**
 * upsert linear layout tab for the current viewer state.
 */
async function upsertLinearLayoutTab(
    ctx: LinearLayoutUiContext,
    document: ReturnType<typeof createComposeLayoutDocument>,
    replaceTabs = false,
): Promise<void> {
    const activeTab = ctx.getSessionTabs().find((tab) => tab.id === ctx.getActiveTabId());
    const targetId = activeTab?.id ?? document.id;
    const targetTitle = activeTab?.title ?? document.title;
    const nextDocument = { ...document, id: targetId, title: targetTitle };
    // tab-local widget state is saved before loadTab swaps viewer data; without
    // this, applying a layout would reset cell text, multi-input, and selection caches.
    ctx.state.linearLayoutStates.set(targetId, cloneLinearLayoutState(ctx.state.linearLayoutState));
    ctx.state.linearLayoutCellTextStates.set(targetId, cloneLinearLayoutCellTextState(ctx.state.linearLayoutCellTextState));
    ctx.state.linearLayoutMultiInputStates.set(targetId, cloneLinearLayoutMultiInputState(ctx.state.linearLayoutMultiInputState));
    ctx.state.linearLayoutSelectionMaps.delete(targetId);
    if (replaceTabs || ctx.getSessionTabs().length === 0) {
        ctx.setSessionTabs([nextDocument]);
    } else {
        const index = ctx.getSessionTabs().findIndex((tab) => tab.id === targetId);
        ctx.setSessionTabs(index === -1
            ? [...ctx.getSessionTabs(), nextDocument]
            : ctx.getSessionTabs().map((tab, tabIndex) => tabIndex === index ? nextDocument : tab));
    }
    await ctx.loadTab(targetId);
}
