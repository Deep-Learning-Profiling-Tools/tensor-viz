import { escapeInfo } from '../app-format.js';
import {
    composeLayoutPresetForSelection,
    composeLayoutPresetOptions,
    normalizeComposeLayoutPresetSelection,
} from '../linear-layout.js';
import type { LinearLayoutUiContext } from '../linear-layout-state.js';
import { applyLinearLayoutSpec } from './linear-layout-widget-actions.js';

let clearPresetOutsideClickHandler: (() => void) | null = null;

type PresetFieldKey = 'gpuArch' | 'category' | 'matrixSize' | 'dtype' | 'operand';

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
        key: 'category',
        id: 'linear-layout-preset-category',
        label: 'Category',
        placeholder: 'Type category',
        valuesId: 'linear-layout-preset-values-category',
        valuesLabel: 'Category',
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
];

function linearLayoutPresetHelpHtml(options: {
    gpuArchs: string[];
    categories: string[];
    matrixSizes: string[];
    dtypes: string[];
    operands: string[];
}): string {
    return `
      <details class="usage-guide">
        <summary>How do I use this?</summary>
        <div class="usage-guide-body">
          <div class="usage-guide-step">
            <span>Choose a supported combination, then click <strong>Load Preset</strong> to overwrite the current Layouts, Layout Operation, and Input Tensor Name fields.</span>
          </div>
          <div class="usage-guide-step">
            <span>This currently includes <strong>mma-v2</strong>, <strong>mma</strong>, <strong>wgmma</strong>, and <strong>tcgen05</strong> presets. The dropdowns are structured so more PTX layout families can be added without changing the editor flow.</span>
          </div>
          <div class="usage-guide-subtitle">Valid Values</div>
          <div class="usage-guide-example">
            ${PRESET_FIELDS.map((field) => `
              <code id="${field.valuesId}">${field.valuesLabel}: ${escapeInfo(presetFieldOptions(options, field.key).join(', ') || 'none')}</code>
            `).join('')}
          </div>
        </div>
      </details>
    `;
}

function presetSearchField(
    field: typeof PRESET_FIELDS[number],
    value: string,
    options: string[],
): string {
    const matches = filteredPresetOptions(options, value);
    return `
      <div class="preset-field" data-preset-field="${field.id}">
        <label class="meta-label" for="${field.id}">${field.label}</label>
        <input id="${field.id}" type="text" value="${escapeInfo(value)}" placeholder="${escapeInfo(field.placeholder)}" autocomplete="off" />
        <div class="preset-option-list">
          ${matches.length > 0
        ? matches.map((option) => `<button class="preset-option" type="button" data-preset-input="${field.id}" data-preset-value="${escapeInfo(option)}">${escapeInfo(option)}</button>`).join('')
        : '<span class="mapping-empty">no matches</span>'}
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
    input?.addEventListener('input', () => {
        ctx.state.linearLayoutState.presetSelection = {
            ...ctx.state.linearLayoutState.presetSelection,
            [field]: input.value,
        };
        syncPresetControls(ctx, input.id);
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
            ctx.state.linearLayoutState.presetSelection = normalizeComposeLayoutPresetSelection({
                ...ctx.state.linearLayoutState.presetSelection,
                [field]: button.dataset.presetValue ?? '',
            });
            syncPresetControls(ctx, null);
        });
    });
}

function syncPresetControls(ctx: LinearLayoutUiContext, activeInputId: string | null): void {
    const presetOptions = composeLayoutPresetOptions(ctx.state.linearLayoutState.presetSelection);
    const preset = composeLayoutPresetForSelection(ctx.state.linearLayoutState.presetSelection);
    PRESET_FIELDS.forEach((field) => {
        const input = ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${CSS.escape(field.id)}`);
        const list = ctx.linearLayoutPresetWidget.querySelector<HTMLElement>(`[data-preset-field="${field.id}"] .preset-option-list`);
        if (!input || !list) return;
        input.value = ctx.state.linearLayoutState.presetSelection[field.key];
        list.innerHTML = presetOptionsHtml(field.id, filteredPresetOptions(presetFieldOptions(presetOptions, field.key), input.value));
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
    if (activeInputId) {
        const activeInput = ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${CSS.escape(activeInputId)}`);
        activeInput?.focus();
    }
}

function presetOptionsHtml(inputId: string, options: string[]): string {
    return options.length > 0
        ? options.map((option) => `<button class="preset-option" type="button" data-preset-input="${inputId}" data-preset-value="${escapeInfo(option)}">${escapeInfo(option)}</button>`).join('')
        : '<span class="mapping-empty">no matches</span>';
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
        categories: string[];
        matrixSizes: string[];
        dtypes: string[];
        operands: string[];
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
        categories: string[];
        matrixSizes: string[];
        dtypes: string[];
        operands: string[];
    },
    field: PresetFieldKey,
): string[] {
    if (field === 'gpuArch') return options.gpuArchs;
    if (field === 'category') return options.categories;
    if (field === 'matrixSize') return options.matrixSizes;
    if (field === 'dtype') return options.dtypes;
    return options.operands;
}

function presetFieldForInputId(inputId: string): PresetFieldKey | null {
    return PRESET_FIELDS.find((field) => field.id === inputId)?.key ?? null;
}

export function renderLinearLayoutPresetWidget(ctx: LinearLayoutUiContext): void {
    clearPresetOutsideClickHandler?.();
    const presetSelection = ctx.state.linearLayoutState.presetSelection;
    const presetOptions = composeLayoutPresetOptions(presetSelection);
    const preset = composeLayoutPresetForSelection(presetSelection);
    ctx.linearLayoutPresetWidget.innerHTML = `
      ${ctx.widgetTitle('linear-layout-preset', 'Choose a preset layout family and load its saved spec into Layout Specs.')}
      <div class="widget-body">
        ${linearLayoutPresetHelpHtml(presetOptions)}
        <div class="preset-stack">
          ${PRESET_FIELDS.map((field) => presetSearchField(
            field,
            presetSelection[field.key],
            presetFieldOptions(presetOptions, field.key),
        )).join('')}
        </div>
        <div class="button-row linear-layout-action-row">
          <button class="secondary-button" id="linear-layout-load-preset" type="button" ${preset ? '' : 'disabled'} title="Overwrite the editor with the selected preset and render it.">Load Preset</button>
        </div>
        <div class="widget-copy preset-summary" id="linear-layout-preset-summary">${preset
            ? `Selected preset: <span class="inline-code">${escapeInfo(preset.title)}</span>`
            : 'No preset matches the current selection yet.'}</div>
      </div>
    `;
    const loadPreset = ctx.linearLayoutPresetWidget.querySelector<HTMLButtonElement>('#linear-layout-load-preset');
    PRESET_FIELDS.forEach((field) => {
        bindPresetInput(ctx, ctx.linearLayoutPresetWidget.querySelector<HTMLInputElement>(`#${field.id}`), field.key);
    });
    bindPresetOptions(ctx);
    const outsideClickHandler = (event: PointerEvent) => {
        const target = event.target;
        if (!(target instanceof Node)) return;
        const element = target instanceof Element ? target : target.parentElement;
        const presetInput = element?.closest<HTMLInputElement>('input[id^="linear-layout-preset-"]');
        if (presetInput && ctx.linearLayoutPresetWidget.contains(presetInput)) {
            setPresetDropdownVisibility(ctx.linearLayoutPresetWidget, presetInput.id);
            return;
        }
        if (element?.closest('.preset-option') && ctx.linearLayoutPresetWidget.contains(target)) return;
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
}
