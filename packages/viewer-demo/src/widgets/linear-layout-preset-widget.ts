import { escapeInfo, infoButton } from '../app-format.js';
import {
    cloneComposeLayoutPresetSelection,
    composeLayoutPresets,
    composeLayoutPresetForSelection,
    composeLayoutPresetOptions,
    normalizeComposeLayoutPresetSelection,
} from '../linear-layout.js';
import type { LinearLayoutUiContext } from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';

let clearPresetOutsideClickHandler: (() => void) | null = null;

type PresetFieldKey = 'gpuArch' | 'instruction' | 'matrixSize' | 'dtype' | 'operand' | 'trans' | 'major';

const PRESET_FIELDS: Array<{
    key: PresetFieldKey;
    id: string;
    label: string;
    placeholder: string;
    valuesId: string;
    valuesLabel: string;
}> = [
    {
        key: 'gpuArch',
        id: 'linear-layout-preset-gpu-arch',
        label: 'GPU Arch',
        placeholder: 'Type GPU arch',
        valuesId: 'linear-layout-preset-values-gpu-arch',
        valuesLabel: 'GPU Arch',
    },
    {
        key: 'instruction',
        id: 'linear-layout-preset-instruction',
        label: 'Instruction',
        placeholder: 'Type instruction',
        valuesId: 'linear-layout-preset-values-instruction',
        valuesLabel: 'Instruction',
    },
    {
        key: 'matrixSize',
        id: 'linear-layout-preset-matrix-size',
        label: 'Matrix Size',
        placeholder: 'Type matrix size',
        valuesId: 'linear-layout-preset-values-matrix-size',
        valuesLabel: 'Matrix Size',
    },
    {
        key: 'dtype',
        id: 'linear-layout-preset-dtype',
        label: 'DType',
        placeholder: 'Type dtype',
        valuesId: 'linear-layout-preset-values-dtype',
        valuesLabel: 'DType',
    },
    {
        key: 'operand',
        id: 'linear-layout-preset-operand',
        label: 'Operand',
        placeholder: 'Type operand',
        valuesId: 'linear-layout-preset-values-operand',
        valuesLabel: 'Operand',
    },
    {
        key: 'trans',
        id: 'linear-layout-preset-trans',
        label: 'Transpose',
        placeholder: 'Type transpose',
        valuesId: 'linear-layout-preset-values-trans',
        valuesLabel: 'Transpose',
    },
    {
        key: 'major',
        id: 'linear-layout-preset-major',
        label: 'Major',
        placeholder: 'Type major',
        valuesId: 'linear-layout-preset-values-major',
        valuesLabel: 'Major',
    },
];

function linearLayoutPresetHelpHtml(): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Load a PTX layout by typing into text fields and selecting values from the dropdowns. Once the text <strong>No preset matches the current selection yet.</strong> is replaced with <strong>Selected preset: &lt;preset&gt;</strong>, click <strong>Load Preset</strong> to visualize the specified layout.</span>
          </div>
          <div class="usage-guide-step">
            <span>Currently, <strong>mma</strong>, <strong>ldmatrix</strong>, <strong>stmatrix</strong>, <strong>swizzle</strong>, and <strong>wgmma</strong> instructions are supported.</span>
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

function presetSearchField(
    field: typeof PRESET_FIELDS[number],
    value: string,
    validOptions: string[],
    invalidOptions: string[],
    selection: Record<PresetFieldKey, string>,
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

function filteredPresetOptions(options: string[], query: string): string[] {
    const normalizedQuery = normalizePresetSearch(query);
    if (!normalizedQuery) return options;
    return options.filter((option) => fuzzyPresetMatch(option, normalizedQuery));
}

function normalizePresetSearch(value: string): string {
    return value.toLowerCase().replace(/[^a-z0-9]/g, '');
}

function fuzzyPresetMatch(option: string, normalizedQuery: string): boolean {
    const normalizedOption = normalizePresetSearch(option);
    if (normalizedOption.includes(normalizedQuery)) return true;
    let queryIndex = 0;
    for (const char of normalizedOption) {
        if (char === normalizedQuery[queryIndex]) queryIndex += 1;
        if (queryIndex === normalizedQuery.length) return true;
    }
    return false;
}

function bindPresetInput(
    ctx: LinearLayoutUiContext,
    input: HTMLInputElement | null,
    field: PresetFieldKey,
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

function bindPresetOptions(ctx: LinearLayoutUiContext): void {
    ctx.linearLayoutPresetWidget.querySelectorAll<HTMLButtonElement>('[data-preset-input][data-preset-value]').forEach((button) => {
        button.addEventListener('mousedown', (event) => {
            event.preventDefault();
        });
        button.addEventListener('click', () => {
            const inputId = button.dataset.presetInput ?? '';
            const field = presetFieldForInputId(inputId);
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

function syncPresetControls(ctx: LinearLayoutUiContext, activeInputId: string | null): void {
    if (activeInputId === null && renderedPresetFieldIds(ctx.linearLayoutPresetWidget) !== visiblePresetFieldIds(ctx.state.linearLayoutState.presetSelection)) {
        renderLinearLayoutPresetWidget(ctx);
        return;
    }
    const presetOptions = composeLayoutPresetOptions(ctx.state.linearLayoutState.presetSelection);
    const preset = composeLayoutPresetForSelection(ctx.state.linearLayoutState.presetSelection);
    const renderedFields = PRESET_FIELDS.filter((field) => (
        ctx.linearLayoutPresetWidget.querySelector<HTMLElement>(`[data-preset-field="${field.id}"]`) !== null
    ));
    renderedFields.forEach((field) => {
        const input = ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${CSS.escape(field.id)}`);
        const list = ctx.linearLayoutPresetWidget.querySelector<HTMLElement>(`[data-preset-field="${field.id}"] .preset-option-list`);
        if (!input || !list) return;
        if (field.id !== activeInputId) input.value = ctx.state.linearLayoutState.presetSelection[field.key];
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
    setPresetValuesText(ctx.linearLayoutPresetWidget, presetOptions);
    const summary = ctx.linearLayoutPresetWidget.querySelector<HTMLElement>('#linear-layout-preset-summary');
    if (summary) {
        summary.innerHTML = preset
            ? `Selected preset: <span class="inline-code">${escapeInfo(preset.title)}</span>`
            : 'No preset matches the current selection yet.';
    }
    const loadPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-load-preset');
    if (loadPreset) loadPreset.disabled = preset === null;
}

function invalidPresetOptionInfo(field: PresetFieldKey, value: string, selection: Record<PresetFieldKey, string>): string {
    const nextSelection = presetSelectionForOption(selection, field, value, true);
    const clearedFields = PRESET_FIELDS
        .filter(({ key }) => key !== field && selection[key] && !nextSelection[key])
        .map(({ label }) => label);
    if (clearedFields.length === 0) {
        return 'Selecting this clears conflicting fields.';
    }
    return `Selecting this clears conflicting fields: ${clearedFields.join(', ')}`;
}

function presetOptionsHtml(
    field: PresetFieldKey,
    inputId: string,
    validOptions: string[],
    invalidOptions: string[],
    selection: Record<PresetFieldKey, string>,
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

function invalidPresetFieldOptions(field: PresetFieldKey, validOptions: string[]): string[] {
    const allOptions = presetFieldOptions(composeLayoutPresetOptions(undefined), field);
    return allOptions.filter((option) => !validOptions.includes(option));
}

function presetSelectionForOption(
    selection: {
        gpuArch: string;
        instruction: string;
        matrixSize: string;
        dtype: string;
        operand: string;
        trans: string;
        major: string;
    },
    field: PresetFieldKey,
    value: string,
    invalid: boolean,
) {
    if (!invalid) {
        return normalizeComposeLayoutPresetSelection({
            ...selection,
            [field]: value,
        });
    }
    const next = {
        gpuArch: '',
        instruction: '',
        matrixSize: '',
        dtype: '',
        operand: '',
        trans: '',
        major: '',
    };
    next[field] = value;
    PRESET_FIELDS.forEach(({ key }) => {
        if (key === field) return;
        const candidate = selection[key];
        if (!candidate) return;
        if (presetMatches({
            ...next,
            [key]: candidate,
        })) next[key] = candidate;
    });
    const visibleKeys = new Set(visiblePresetFields(next).map(({ key }) => key));
    PRESET_FIELDS.forEach(({ key }) => {
        if (!visibleKeys.has(key)) next[key] = '';
    });
    return normalizeComposeLayoutPresetSelection(next);
}

function presetMatches(filters: Partial<Record<PresetFieldKey, string>>): boolean {
    return composeLayoutPresets().some((preset) => PRESET_FIELDS.every(({ key }) => {
        const value = filters[key];
        if (!value) return true;
        return key === 'gpuArch' ? preset.gpuArchs.includes(value) : preset[key] === value;
    }));
}

function setPresetDropdownVisibility(root: HTMLElement, activeInputId: string | null): void {
    root.querySelectorAll<HTMLElement>('.preset-option-list').forEach((list) => {
        list.classList.toggle('is-open', list.closest<HTMLElement>('[data-preset-field]')?.dataset.presetField === activeInputId);
    });
}

function setPresetValuesText(
    root: HTMLElement,
    options: {
        gpuArchs: string[];
        instructions: string[];
        matrixSizes: string[];
        dtypes: string[];
        operands: string[];
        transes: string[];
        majors: string[];
    },
): void {
    PRESET_FIELDS.forEach((field) => {
        const element = root.querySelector<HTMLElement>(`#${field.valuesId}`);
        if (element) {
            element.textContent = `${field.valuesLabel}: ${presetFieldOptions(options, field.key).join(', ') || 'none'}`;
        }
    });
}

function presetFieldOptions(
    options: {
        gpuArchs: string[];
        instructions: string[];
        matrixSizes: string[];
        dtypes: string[];
        operands: string[];
        transes: string[];
        majors: string[];
    },
    field: PresetFieldKey,
): string[] {
    if (field === 'gpuArch') return options.gpuArchs;
    if (field === 'instruction') return options.instructions;
    if (field === 'matrixSize') return options.matrixSizes;
    if (field === 'dtype') return options.dtypes;
    if (field === 'operand') return options.operands;
    if (field === 'trans') return options.transes;
    return options.majors;
}

function presetFieldForInputId(inputId: string): PresetFieldKey | null {
    return PRESET_FIELDS.find((field) => field.id === inputId)?.key ?? null;
}

function visiblePresetFields(selection: { instruction: string }): typeof PRESET_FIELDS {
    const instruction = selection.instruction;
    return PRESET_FIELDS.filter((field) => {
        if (field.key === 'operand') return instruction === 'mma' || instruction === 'wgmma';
        if (field.key === 'trans') return instruction === 'ldmatrix' || instruction === 'stmatrix';
        if (field.key === 'major') return instruction === 'swizzle';
        return true;
    });
}

function visiblePresetFieldIds(selection: { instruction: string }): string {
    return visiblePresetFields(selection).map((field) => field.id).join(',');
}

function renderedPresetFieldIds(root: HTMLElement): string {
    return Array.from(root.querySelectorAll<HTMLElement>('[data-preset-field]'))
        .map((field) => field.dataset.presetField ?? '')
        .join(',');
}

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
          ${visiblePresetFields(presetSelection).map((field) => presetSearchField(
            field,
            presetSelection[field.key],
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
    PRESET_FIELDS.forEach((field) => {
        bindPresetInput(ctx, ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${field.id}`), field.key);
    });
    bindPresetOptions(ctx);
    const outsideClickHandler = (event: PointerEvent) => {
        const target = event.target;
        if (!(target instanceof Node)) return;
        if (ctx.linearLayoutPresetWidget.contains(target)) return;
        setPresetDropdownVisibility(ctx.linearLayoutPresetWidget, null);
    };
    document.addEventListener('pointerdown', outsideClickHandler, true);
    clearPresetOutsideClickHandler = () => {
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
