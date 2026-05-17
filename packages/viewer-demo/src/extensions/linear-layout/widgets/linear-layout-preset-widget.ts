import { escapeInfo, infoButton } from '../../../app-format.js';
import {
    cloneComposeLayoutPresetSelection,
    composeLayoutPresetFields,
    composeLayoutPresets,
    composeLayoutPresetForSelection,
    composeLayoutPresetOptions,
    normalizeComposeLayoutPresetSelection,
    type ComposeLayoutPresetField,
    type ComposeLayoutPresetOptions,
    type ComposeLayoutPresetSelection,
} from '../linear-layout.js';
import type { LinearLayoutUiContext } from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';

let clearPresetOutsideClickHandler: (() => void) | null = null;

/**
 * return linear layout preset help html for the current viewer state.
 */
function linearLayoutPresetHelpHtml(): string {
    const instructions = composeLayoutPresetOptions(undefined).instruction.join(', ');
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Load a preset layout by typing into text fields and selecting values from the dropdowns. Once the text <strong>No preset matches the current selection yet.</strong> is replaced with <strong>Selected preset: &lt;preset&gt;</strong>, click <strong>Load Preset</strong> to visualize the specified layout.</span>
          </div>
          <div class="usage-guide-step">
            <span>Currently supported instructions: <strong>${escapeInfo(instructions || 'none')}</strong>.</span>
          </div>
          <div class="usage-guide-subtitle">Examples</div>
          <div class="usage-guide-examples">
            <div class="usage-guide-example">
              <code>GPU Arch: sm_80
Instruction: mma
Matrix Size: m16n8k16
DType: b16
Operand: A</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_100a
Instruction: swizzle
Matrix Size: 128B
DType: b128
Major: K-major</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_75
Instruction: ldmatrix
Matrix Size: m8n8.x1
DType: b16
Transpose: no</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_100
Instruction: stmatrix
Matrix Size: m16n16.x2
DType: b8
Transpose: yes</code>
            </div>
            <div class="usage-guide-example">
              <code>GPU Arch: sm_90a
Instruction: wgmma
Matrix Size: m64n64
DType: b32
Operand: D</code>
            </div>
          </div>
        </div>
      </details>
    `;
}

/**
 * return preset search field for the current viewer state.
 */
function presetSearchField(
    field: ComposeLayoutPresetField,
    value: string,
    validOptions: string[],
    invalidOptions: string[],
    selection: ComposeLayoutPresetSelection,
): string {
    const validMatches = filteredPresetOptions(validOptions, value);
    const invalidMatches = filteredPresetOptions(invalidOptions, value);
    return `
      <div class="preset-field" data-preset-field="${field.id}">
        <span class="meta-label">${field.label}</span>
        <input id="${field.id}" type="text" value="${escapeInfo(value)}" placeholder="${escapeInfo(field.placeholder)}" autocomplete="off" />
        <div class="preset-option-list">
          ${presetOptionsHtml(field.key, field.id, validMatches, invalidMatches, selection)}
        </div>
      </div>
    `;
}

/**
 * return filtered preset options for the current viewer state.
 */
function filteredPresetOptions(options: string[], query: string): string[] {
    const normalizedQuery = query.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (!normalizedQuery) return options;
    return options.filter((option) => fuzzyPresetMatch(option, normalizedQuery));
}

/**
 * return fuzzy preset match for the current viewer state.
 */
function fuzzyPresetMatch(option: string, normalizedQuery: string): boolean {
    const normalizedOption = option.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (normalizedOption.includes(normalizedQuery)) return true;
    let queryIndex = 0;
    for (const char of normalizedOption) {
        if (char === normalizedQuery[queryIndex]) queryIndex += 1;
        if (queryIndex === normalizedQuery.length) return true;
    }
    return false;
}

/**
 * bind preset input for the current viewer state.
 */
function bindPresetInput(
    ctx: LinearLayoutUiContext,
    input: HTMLInputElement | null,
    field: string,
): void {
    input?.addEventListener('focus', () => {
        syncPresetControls(ctx, input.id);
    });
    input?.addEventListener('input', () => {
        ctx.state.linearLayoutState.presetSelection = {
            ...ctx.state.linearLayoutState.presetSelection,
            [field]: input.value,
        };
        syncPresetControls(ctx, input.id);
    });
    input?.addEventListener('blur', () => {
        // blur is the commit point for typed text.  Normalizing here lets users
        // type partial values without immediately clearing their in-progress input.
        ctx.state.linearLayoutState.presetSelection = normalizeComposeLayoutPresetSelection(
            ctx.state.linearLayoutState.presetSelection,
        );
        syncPresetControls(ctx, null);
    });
    input?.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter') return;
        const firstOption = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>(
            `[data-preset-input="${input.id}"][data-preset-value]`,
        );
        if (!firstOption) return;
        event.preventDefault();
        firstOption.click();
    });
}

/**
 * bind preset options for the current viewer state.
 */
function bindPresetOptions(ctx: LinearLayoutUiContext): void {
    ctx.linearLayoutPresetWidget.querySelectorAll<HTMLButtonElement>('[data-preset-input][data-preset-value]').forEach((button) => {
        button.addEventListener('mousedown', (event) => {
            event.preventDefault();
        });
        button.addEventListener('click', () => {
            const inputId = button.dataset.presetInput ?? '';
            const field = composeLayoutPresetFields().find((candidate) => candidate.id === inputId)?.key ?? null;
            if (!field) return;
            ctx.state.linearLayoutState.presetSelection = presetSelectionForOption(
                ctx.state.linearLayoutState.presetSelection,
                field,
                button.dataset.presetValue ?? '',
                button.dataset.presetValidity === 'invalid',
            );
            syncPresetControls(ctx, null);
        });
    });
}

/**
 * sync preset controls for the current viewer state.
 */
function syncPresetControls(ctx: LinearLayoutUiContext, activeInputId: string | null): void {
    const presetOptions = composeLayoutPresetOptions(ctx.state.linearLayoutState.presetSelection);
    if (activeInputId === null && renderedPresetFieldIds(ctx.linearLayoutPresetWidget) !== visiblePresetFields(ctx.state.linearLayoutState.presetSelection, presetOptions).map((field) => field.id).join(',')) {
        // field visibility depends on selected facets, so re-render only when the
        // field set changes; otherwise update inputs in place to preserve focus.
        renderLinearLayoutPresetWidget(ctx);
        return;
    }
    const preset = composeLayoutPresetForSelection(ctx.state.linearLayoutState.presetSelection);
    const renderedFields = composeLayoutPresetFields().filter((field) => (
        ctx.linearLayoutPresetWidget.querySelector<HTMLElement>(`[data-preset-field="${field.id}"]`) !== null
    ));
    renderedFields.forEach((field) => {
        const input = ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${CSS.escape(field.id)}`);
        const list = ctx.linearLayoutPresetWidget.querySelector<HTMLElement>(`[data-preset-field="${field.id}"] .preset-option-list`);
        if (!input || !list) return;
        if (field.id !== activeInputId) input.value = ctx.state.linearLayoutState.presetSelection[field.key] ?? '';
        const validOptions = presetFieldOptions(presetOptions, field.key);
        list.innerHTML = presetOptionsHtml(
            field.key,
            field.id,
            filteredPresetOptions(validOptions, input.value),
            filteredPresetOptions(invalidPresetFieldOptions(field.key, validOptions), input.value),
            {
                ...ctx.state.linearLayoutState.presetSelection,
                [field.key]: input.value,
            },
        );
    });
    bindPresetOptions(ctx);
    setPresetDropdownVisibility(ctx.linearLayoutPresetWidget, activeInputId);
    const summary = ctx.linearLayoutPresetWidget.querySelector<HTMLElement>('#linear-layout-preset-summary');
    if (summary) {
        summary.innerHTML = preset
            ? `Selected preset: <span class="inline-code">${escapeInfo(preset.title)}</span>`
            : 'No preset matches the current selection yet.';
    }
    const loadPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-load-preset');
    if (loadPreset) loadPreset.disabled = preset === null;
}

/**
 * return invalid preset option info for the current viewer state.
 */
function invalidPresetOptionInfo(field: string, value: string, selection: ComposeLayoutPresetSelection): string {
    const nextSelection = presetSelectionForOption(selection, field, value, true);
    const clearedFields = composeLayoutPresetFields()
        .filter(({ key }) => key !== field && selection[key] && !nextSelection[key])
        .map(({ label }) => label);
    if (clearedFields.length === 0) {
        return 'Selecting this clears conflicting fields.';
    }
    return `Selecting this clears conflicting fields: ${clearedFields.join(', ')}`;
}

/**
 * return preset options html for the current viewer state.
 */
function presetOptionsHtml(
    field: string,
    inputId: string,
    validOptions: string[],
    invalidOptions: string[],
    selection: ComposeLayoutPresetSelection,
): string {
    if (validOptions.length === 0 && invalidOptions.length === 0) return '<span class="mapping-empty">no matches</span>';
    return [
        ...validOptions.map((option) => (
            `<button class="preset-option" type="button" data-preset-input="${inputId}" data-preset-value="${escapeInfo(option)}" data-preset-validity="valid">${escapeInfo(option)}</button>`
        )),
        ...(invalidOptions.length > 0
            ? [`<div class="preset-option-divider"><span>conflicts with current selection</span>${infoButton("You didn't make an error. You can still choose one of these options, and the preset widget will clear any conflicting fields automatically.")}</div>`]
            : []),
        ...invalidOptions.map((option) => (
            `<button class="preset-option preset-option-invalid" type="button" data-preset-input="${inputId}" data-preset-value="${escapeInfo(option)}" data-preset-validity="invalid" data-info="${escapeInfo(invalidPresetOptionInfo(field, option, selection))}">${escapeInfo(option)}</button>`
        )),
    ].join('');
}

/**
 * return invalid preset field options for the current viewer state.
 */
function invalidPresetFieldOptions(field: string, validOptions: string[]): string[] {
    const allOptions = presetFieldOptions(composeLayoutPresetOptions(undefined), field);
    return allOptions.filter((option) => !validOptions.includes(option));
}

/** choose an option and keep as many compatible existing fields as possible. */
function presetSelectionForOption(
    selection: ComposeLayoutPresetSelection,
    field: string,
    value: string,
    invalid: boolean,
) {
    if (!invalid) {
        return normalizeComposeLayoutPresetSelection({
            ...selection,
            [field]: value,
        });
    }
    const next = cloneComposeLayoutPresetSelection(undefined);
    next[field] = value;
    // invalid options are shown intentionally: selecting one starts a new valid
    // path and copies over only fields that still match some preset.
    composeLayoutPresetFields().forEach(({ key }) => {
        if (key === field) return;
        const candidate = selection[key];
        if (!candidate) return;
        if (presetMatches({
            ...next,
            [key]: candidate,
        })) next[key] = candidate;
    });
    const visibleKeys = new Set(visiblePresetFields(next, composeLayoutPresetOptions(next)).map(({ key }) => key));
    // hidden fields must be cleared or they can silently block preset matching
    // after an instruction switch such as mma -> swizzle.
    composeLayoutPresetFields().forEach(({ key }) => {
        if (!visibleKeys.has(key)) next[key] = '';
    });
    return normalizeComposeLayoutPresetSelection(next);
}

/**
 * return preset matches for the current viewer state.
 */
function presetMatches(filters: ComposeLayoutPresetSelection): boolean {
    return composeLayoutPresets().some((preset) => Object.keys(filters).every((key) => {
        const value = filters[key];
        if (!value) return true;
        return preset.facets[key]?.includes(value) ?? false;
    }));
}

/**
 * set preset dropdown visibility for the current viewer state.
 */
function setPresetDropdownVisibility(root: HTMLElement, activeInputId: string | null): void {
    root.querySelectorAll<HTMLElement>('.preset-option-list').forEach((list) => {
        list.classList.toggle('is-open', list.closest<HTMLElement>('[data-preset-field]')?.dataset.presetField === activeInputId);
    });
}

/**
 * return preset field options for the current viewer state.
 */
function presetFieldOptions(options: ComposeLayoutPresetOptions, field: string): string[] {
    return options[field] ?? [];
}

/**
 * return visible preset fields for the current viewer state.
 */
function visiblePresetFields(
    selection: ComposeLayoutPresetSelection,
    options: ComposeLayoutPresetOptions,
): ComposeLayoutPresetField[] {
    // required fields always show.  optional fields appear once their dependency
    // path is active and the filtered catalog has values for them.
    return composeLayoutPresetFields().filter((field) => field.required
        || Boolean(selection[field.key])
        || (field.dependsOn.every((key) => Boolean(selection[key])) && presetFieldOptions(options, field.key).length > 0));
}

/**
 * return rendered preset field ids for the current viewer state.
 */
function renderedPresetFieldIds(root: HTMLElement): string {
    return Array.from(root.querySelectorAll<HTMLElement>('[data-preset-field]'))
        .map((field) => field.dataset.presetField ?? '')
        .join(',');
}

/**
 * render linear layout preset widget for the current viewer state.
 */
export function renderLinearLayoutPresetWidget(ctx: LinearLayoutUiContext): void {
    clearPresetOutsideClickHandler?.();
    const presetSelection = cloneComposeLayoutPresetSelection(ctx.state.linearLayoutState.presetSelection);
    ctx.state.linearLayoutState.presetSelection = presetSelection;
    const presetOptions = composeLayoutPresetOptions(presetSelection);
    const preset = composeLayoutPresetForSelection(presetSelection);
    ctx.linearLayoutPresetWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout-preset', 'Choose a preset layout family and load its saved spec into Layout Specs.')}
        <div class="widget-body">
        ${linearLayoutPresetHelpHtml()}
        <div class="preset-stack">
          ${visiblePresetFields(presetSelection, presetOptions).map((field) => presetSearchField(
            field,
            presetSelection[field.key] ?? '',
            presetFieldOptions(presetOptions, field.key),
            invalidPresetFieldOptions(field.key, presetFieldOptions(presetOptions, field.key)),
            presetSelection,
        )).join('')}
        </div>
        <div class="button-row linear-layout-action-row">
          <button class="primary-button" id="linear-layout-load-preset" type="button" ${preset ? '' : 'disabled'} title="Overwrite the editor with the selected preset and render it.">Load Preset</button>
          <button class="secondary-button" id="linear-layout-clear-preset" type="button" title="Clear the preset search fields without changing the loaded layout spec.">Clear Preset</button>
        </div>
        <div class="widget-copy preset-summary" id="linear-layout-preset-summary">${preset
            ? `Selected preset: <span class="inline-code">${escapeInfo(preset.title)}</span>`
            : 'No preset matches the current selection yet.'}</div>
      </div>
    `;
    const loadPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-load-preset');
    const clearPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-clear-preset');
    composeLayoutPresetFields().forEach((field) => {
        bindPresetInput(ctx, ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${field.id}`), field.key);
    });
    bindPresetOptions(ctx);
    /** close open preset dropdowns when focus moves outside the widget. */
    const outsideClickHandler = (event: PointerEvent) => {
        const target = event.target;
        if (!(target instanceof Node)) return;
        if (ctx.linearLayoutPresetWidget.contains(target)) return;
        setPresetDropdownVisibility(ctx.linearLayoutPresetWidget, null);
    };
    document.addEventListener('pointerdown', outsideClickHandler, true);
    clearPresetOutsideClickHandler = () => {
        // the widget can be re-rendered often while typing, so remove the old
        // capture listener before installing a replacement to avoid duplicate closes.
        document.removeEventListener('pointerdown', outsideClickHandler, true);
    };
    loadPreset?.addEventListener('click', async () => {
        const nextPreset = composeLayoutPresetForSelection(ctx.state.linearLayoutState.presetSelection);
        if (!nextPreset) return;
        ctx.state.linearLayoutState.specsText = nextPreset.state.specsText;
        ctx.state.linearLayoutState.operationText = nextPreset.state.operationText;
        ctx.state.linearLayoutState.inputName = nextPreset.state.inputName;
        await applyLinearLayoutSpec(ctx);
        ctx.renderLinearLayoutEditorWidgets();
    });
    clearPreset?.addEventListener('click', () => {
        ctx.state.linearLayoutState.presetSelection = normalizeComposeLayoutPresetSelection(undefined);
        syncPresetControls(ctx, null);
    });
}
