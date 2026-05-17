import { COMPOSE_LAYOUT_PRESET_FAMILIES } from './presets/index.js';
import type {
    ComposeLayoutPresetDefinition,
    ComposeLayoutPresetFacetValue,
    ComposeLayoutPresetFieldDefinition,
} from './presets/types.js';
import {
    formatSpecsText,
    parseLayoutSpecs,
    parseSignature,
    stripLayoutComment,
} from './linear-layout-parser.js';

/**
 * shape of compose layout preset selection data used by the viewer.
 *
 * @example
 * const value: ComposeLayoutPresetSelection = {} as ComposeLayoutPresetSelection;
 */
export type ComposeLayoutPresetSelection = Record<string, string>;

/**
 * normalized field metadata consumed directly by the preset widget.
 *
 * @example
 * const value: ComposeLayoutPresetField = {} as ComposeLayoutPresetField;
 */
export type ComposeLayoutPresetField = Required<Omit<ComposeLayoutPresetFieldDefinition, 'values'>> & {
    id: string;
    values: string[];
};

/**
 * editor state inserted when a preset selection resolves to one layout.
 *
 * @example
 * const value: ComposeLayoutPresetState = {} as ComposeLayoutPresetState;
 */
export type ComposeLayoutPresetState = {
    specsText: string;
    operationText: string;
    inputName: string;
};

/**
 * normalized preset with compatibility aliases for older widget call sites.
 *
 * @example
 * const value: ComposeLayoutPreset = {} as ComposeLayoutPreset;
 */
export type ComposeLayoutPreset = {
    title: string;
    facets: Record<string, string[]>;
    gpuArchs: string[];
    instruction: string;
    matrixSize: string;
    dtype: string;
    operand: string;
    trans: string;
    major: string;
    state: ComposeLayoutPresetState;
};

/**
 * options keyed by field name plus legacy plural aliases.
 *
 * @example
 * const value: ComposeLayoutPresetOptions = {} as ComposeLayoutPresetOptions;
 */
export type ComposeLayoutPresetOptions = Record<string, string[]> & {
    gpuArchs: string[];
    instructions: string[];
    matrixSizes: string[];
    dtypes: string[];
    operands: string[];
    transes: string[];
    majors: string[];
};

const LEGACY_PRESET_FIELD_KEYS = ['gpuArch', 'instruction', 'matrixSize', 'dtype', 'operand', 'trans', 'major'] as const;

const PRESET_FIELD_OPTION_ALIASES = {
    gpuArch: 'gpuArchs',
    instruction: 'instructions',
    matrixSize: 'matrixSizes',
    dtype: 'dtypes',
    operand: 'operands',
    trans: 'transes',
    major: 'majors',
} as const;

const PRESET_DEFINITIONS: readonly ComposeLayoutPresetDefinition[] = COMPOSE_LAYOUT_PRESET_FAMILIES.flatMap((family) => family.presets);

// field metadata is merged from family declarations plus facet keys found in
// preset data; this lets contributed families add fields without widget edits.
const COMPOSE_LAYOUT_PRESET_FIELDS = mergedPresetFields([
    ...COMPOSE_LAYOUT_PRESET_FAMILIES.flatMap((family) => family.fields ?? []),
    ...PRESET_DEFINITIONS.flatMap((definition) => (
        Object.keys(definition.facets ?? {}).map((key) => inferredPresetFieldDefinition(key))
    )),
]);

const COMPOSE_LAYOUT_PRESET_CATALOG = PRESET_DEFINITIONS.map((definition) => composeLayoutPreset(definition));

/**
 * build an empty selection that includes every registered field key.
 *
 * @returns Computed ComposeLayoutPresetSelection value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * emptyComposeLayoutPresetSelection();
 */
export function emptyComposeLayoutPresetSelection(): ComposeLayoutPresetSelection {
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [field.key, '']));
}

/**
 * clone external selection data while dropping unknown/non-string values.
 *
 * @param selection - Selection data used by this operation.
 * @returns Computed ComposeLayoutPresetSelection value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * cloneComposeLayoutPresetSelection(selection);
 */
export function cloneComposeLayoutPresetSelection(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPresetSelection {
    const record = selection as (Record<string, unknown> & { category?: unknown }) | undefined;
    const cloned = emptyComposeLayoutPresetSelection();
    COMPOSE_LAYOUT_PRESET_FIELDS.forEach((field) => {
        const value = record?.[field.key];
        cloned[field.key] = typeof value === 'string' ? value : '';
    });
    if (!cloned.instruction && typeof record?.category === 'string') cloned.instruction = record.category;
    return cloned;
}

/**
 * validate persisted preset selection state before it is copied into the editor.
 *
 * @param value - Value supplied by the caller.
 * @returns Computed value is ComposeLayoutPresetSelection value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * isComposeLayoutPresetSelection(value);
 */
export function isComposeLayoutPresetSelection(value: unknown): value is ComposeLayoutPresetSelection {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => typeof entry === 'string');
}

/**
 * return field metadata in render order.
 *
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * composeLayoutPresetFields();
 */
export function composeLayoutPresetFields(): ComposeLayoutPresetField[] {
    return COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => ({
        ...field,
        dependsOn: [...field.dependsOn],
        values: [...field.values],
    }));
}

/**
 * return the internal catalog for invariant tests and model-level matching.
 *
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * composeLayoutPresetCatalog();
 */
export function composeLayoutPresetCatalog(): ComposeLayoutPreset[] {
    return COMPOSE_LAYOUT_PRESET_CATALOG;
}

/**
 * return a cloned catalog for UI code that may derive temporary state from presets.
 *
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * composeLayoutPresets();
 */
export function composeLayoutPresets(): ComposeLayoutPreset[] {
    return composeLayoutPresetCatalog().map((preset) => ({
        ...preset,
        facets: Object.fromEntries(Object.entries(preset.facets).map(([key, values]) => [key, [...values]])),
        gpuArchs: [...preset.gpuArchs],
        state: { ...preset.state },
    }));
}

/**
 * infer the selector values for an editor state that exactly matches a preset.
 *
 * @param state - State object read or updated by this operation.
 * @returns Computed ComposeLayoutPresetSelection value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * matchedComposeLayoutPresetSelection(state);
 */
export function matchedComposeLayoutPresetSelection(
    state: ComposeLayoutPresetState,
): ComposeLayoutPresetSelection {
    const canonicalSpecsText = canonicalLayoutSpecsText(state.specsText);
    // presets are matched by canonicalized text, operation, and input name so
    // adding comments to a preset does not strand an already-loaded editor state.
    const preset = composeLayoutPresetCatalog().find((entry) => canonicalLayoutSpecsText(entry.state.specsText) === canonicalSpecsText
        && entry.state.operationText === state.operationText
        && entry.state.inputName === state.inputName);
    if (!preset) return emptyComposeLayoutPresetSelection();
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [
        field.key,
        preset.facets[field.key]?.[0] ?? '',
    ]));
}

/**
 * compute the valid dropdown/text options after applying the current selection.
 *
 * @param selection - Selection data used by this operation.
 * @returns Computed ComposeLayoutPresetOptions value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * composeLayoutPresetOptions(selection);
 */
export function composeLayoutPresetOptions(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPresetOptions {
    const current = cloneComposeLayoutPresetSelection(selection);
    const presets = composeLayoutPresetCatalog();
    const options: ComposeLayoutPresetOptions = {
        gpuArchs: [],
        instructions: [],
        matrixSizes: [],
        dtypes: [],
        operands: [],
        transes: [],
        majors: [],
    };
    COMPOSE_LAYOUT_PRESET_FIELDS.forEach((field) => {
        // when computing choices for one field, ignore that field's current
        // value; otherwise a typo would hide the valid options needed to recover.
        const values = uniquePresetFacetValues(filteredPresets(
            presets,
            Object.fromEntries(Object.entries(current).map(([key, value]) => [key, key === field.key ? '' : value])),
        ), field);
        options[field.key] = values;
        const alias = PRESET_FIELD_OPTION_ALIASES[field.key as keyof typeof PRESET_FIELD_OPTION_ALIASES];
        if (alias) options[alias] = values;
    });
    return options;
}

/**
 * clear invalid field values and autofill singleton options.
 *
 * @param selection - Selection data used by this operation.
 * @returns Computed ComposeLayoutPresetSelection value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizeComposeLayoutPresetSelection(selection);
 */
export function normalizeComposeLayoutPresetSelection(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPresetSelection {
    const current = cloneComposeLayoutPresetSelection(selection);
    const normalized = { ...current };
    COMPOSE_LAYOUT_PRESET_FIELDS.forEach((field) => {
        normalized[field.key] = normalizedPresetField(current[field.key] ?? '', composeLayoutPresetOptions(normalized)[field.key] ?? []);
    });
    return normalized;
}

/**
 * resolve a fully specified selection to exactly one preset.
 *
 * @param selection - Selection data used by this operation.
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * composeLayoutPresetForSelection(selection);
 */
export function composeLayoutPresetForSelection(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPreset | null {
    const current = cloneComposeLayoutPresetSelection(selection);
    const matches = filteredPresets(composeLayoutPresetCatalog(), current).filter((preset) => (
        COMPOSE_LAYOUT_PRESET_FIELDS.every((field) => presetFieldIsComplete(preset, current, field.key))
    ));
    return matches.length === 1 ? matches[0]! : null;
}

/**
 * return merged preset fields for the current viewer state.
 *
 * @param definitions - definitions input used by this operation (readonly ComposeLayoutPresetFieldDefinition[]).
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * mergedPresetFields(definitions);
 */
function mergedPresetFields(definitions: readonly ComposeLayoutPresetFieldDefinition[]): ComposeLayoutPresetField[] {
    const fields = new Map<string, ComposeLayoutPresetField>();
    definitions.forEach((definition) => {
        fields.set(definition.key, normalizePresetFieldDefinition(definition, fields.get(definition.key)));
    });
    return Array.from(fields.values()).sort((left, right) => left.order - right.order || left.key.localeCompare(right.key));
}

/**
 * normalize preset field definition for the current viewer state.
 *
 * @param definition - definition input used by this operation (ComposeLayoutPresetFieldDefinition).
 * @param current - current input used by this operation (ComposeLayoutPresetField).
 * @returns Computed ComposeLayoutPresetField value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizePresetFieldDefinition(definition, current);
 */
function normalizePresetFieldDefinition(
    definition: ComposeLayoutPresetFieldDefinition,
    current?: ComposeLayoutPresetField,
): ComposeLayoutPresetField {
    const values = [...new Set([...(current?.values ?? []), ...(definition.values ?? [])])];
    const dependsOn = [...new Set([...(current?.dependsOn ?? []), ...(definition.dependsOn ?? [])])];
    return {
        key: definition.key,
        id: current?.id ?? `linear-layout-preset-${definition.key.replace(/([a-z0-9])([A-Z])/g, '$1-$2').replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`,
        label: current?.label ?? definition.label,
        placeholder: current?.placeholder ?? definition.placeholder,
        order: Math.min(current?.order ?? definition.order, definition.order),
        required: Boolean(current?.required || definition.required),
        dependsOn,
        values,
    };
}

/**
 * return inferred preset field definition for the current viewer state.
 *
 * @param key - key input used by this operation (string).
 * @returns Computed ComposeLayoutPresetFieldDefinition value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * inferredPresetFieldDefinition(key);
 */
function inferredPresetFieldDefinition(key: string): ComposeLayoutPresetFieldDefinition {
    const label = key
        .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
        .replace(/[_-]+/g, ' ')
        .replace(/\b\w/g, (char) => char.toUpperCase());
    return {
        key,
        label,
        placeholder: `Type ${label.toLowerCase()}`,
        order: 100,
    };
}

/**
 * return preset operation text for the current viewer state.
 *
 * @param specsText - Text supplied by the caller.
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * presetOperationText(specsText);
 */
function presetOperationText(specsText: string): string {
    const signature = specsText.split('\n', 1)[0]?.trim() ?? '';
    const colonIndex = signature.indexOf(':');
    return (colonIndex === -1 ? signature : signature.slice(0, colonIndex)).trim();
}

/**
 * return axis comment for the current viewer state.
 *
 * @param label - label input used by this operation (string).
 * @param signature - signature input used by this operation ({ inputs: string[]; outputs: string[] }).
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * axisComment(label, signature);
 */
function axisComment(label: string, signature: { inputs: string[]; outputs: string[] }): string {
    if (label === 'T') return 'T = thread (AKA lane)';
    if (label === 'R') {
        return signature.inputs.includes('C') || signature.outputs.includes('R32')
            ? 'R = row'
            : 'R = register';
    }
    if (label === 'R32') return 'R32 = packed 32-bit register';
    if (label === 'C') return 'C = column';
    if (label === 'W') return 'W = warp';
    if (label === 'Y') return 'Y = y-position';
    if (label === 'X') return 'X = x-position';
    if (label === 'M') return 'M = row';
    if (label === 'N') return 'N = column';
    if (label === 'K') return 'K = reduction dimension';
    if (label === 'O') return 'O = logical offset';
    if (label === 'H') return 'H = higher-order tile axis';
    if (label === 'L') return 'L = line';
    if (label === 'B') return 'B = byte offset';
    return `${label} = ${label} axis`;
}

/**
 * return annotated layout specs text for the current viewer state.
 *
 * @param specsText - Text supplied by the caller.
 * @param comments - comments input used by this operation (string[]).
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * annotatedLayoutSpecsText(specsText, comments);
 */
function annotatedLayoutSpecsText(specsText: string, comments: string[] = []): string {
    const lines = specsText.replace(/\r\n/g, '\n').split('\n');
    const signature = parseSignature(stripLayoutComment(lines[0] ?? '').trim());
    const labelComments = [...signature.inputs, ...signature.outputs].map((label) => axisComment(label, signature));
    // comments become part of the loaded editor text, so keep generated axis
    // notes deterministic and deduplicated to avoid preset matching churn.
    return [
        lines[0] ?? '',
        ...Array.from(new Set([...labelComments, ...comments])).map((comment) => `# ${comment}`),
        ...lines.slice(1),
    ].join('\n');
}

/**
 * return preset definition facets for the current viewer state.
 *
 * @param definition - definition input used by this operation (ComposeLayoutPresetDefinition).
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * presetDefinitionFacets(definition);
 */
function presetDefinitionFacets(definition: ComposeLayoutPresetDefinition): Record<string, string[]> {
    // every normalized preset receives every known field key.  Empty arrays mean
    // the field is irrelevant for that preset and should remain blank.
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [
        field.key,
        presetDefinitionFacetValues(definition, field.key),
    ]));
}

/**
 * return preset definition facet values for the current viewer state.
 *
 * @param definition - definition input used by this operation (ComposeLayoutPresetDefinition).
 * @param key - key input used by this operation (string).
 * @returns Text entries formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * presetDefinitionFacetValues(definition, key);
 */
function presetDefinitionFacetValues(definition: ComposeLayoutPresetDefinition, key: string): string[] {
    const facet = definition.facets?.[key];
    if (facet !== undefined) return normalizePresetFacetValue(facet);
    // legacy scalar fields keep old NVIDIA presets working while new families
    // can describe selector behavior entirely through facets.
    if (LEGACY_PRESET_FIELD_KEYS.includes(key as typeof LEGACY_PRESET_FIELD_KEYS[number])) {
        return normalizePresetFacetValue(definition[key as typeof LEGACY_PRESET_FIELD_KEYS[number]] ?? '');
    }
    return [];
}

/**
 * normalize preset facet value for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @returns Text entries formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizePresetFacetValue(value);
 */
function normalizePresetFacetValue(value: ComposeLayoutPresetFacetValue): string[] {
    const values = Array.isArray(value) ? value : [value];
    return values.map((entry) => String(entry)).filter(Boolean);
}

/**
 * return preset facet scalar for the current viewer state.
 *
 * @param facets - facets input used by this operation (Record<string, string[]>).
 * @param key - key input used by this operation (string).
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * presetFacetScalar(facets, key);
 */
function presetFacetScalar(facets: Record<string, string[]>, key: string): string {
    return facets[key]?.[0] ?? '';
}

/**
 * compose layout preset for the current viewer state.
 *
 * @param definition - definition input used by this operation (ComposeLayoutPresetDefinition).
 * @returns Computed ComposeLayoutPreset value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * composeLayoutPreset(definition);
 */
function composeLayoutPreset(definition: ComposeLayoutPresetDefinition): ComposeLayoutPreset {
    const inputName = definition.inputName ?? 'Hardware Layout';
    const facets = presetDefinitionFacets(definition);
    const gpuArchs = facets.gpuArch ?? [];
    if ('specsText' in definition) {
        const specsText = annotatedLayoutSpecsText(definition.specsText, definition.comments);
        const operationText = presetOperationText(specsText);
        return {
            title: definition.title ?? operationText,
            facets,
            gpuArchs,
            instruction: presetFacetScalar(facets, 'instruction'),
            matrixSize: presetFacetScalar(facets, 'matrixSize'),
            dtype: presetFacetScalar(facets, 'dtype'),
            operand: presetFacetScalar(facets, 'operand'),
            trans: presetFacetScalar(facets, 'trans'),
            major: presetFacetScalar(facets, 'major'),
            state: { specsText, operationText, inputName },
        };
    }
    // named definitions are the compact path for ISA-table-style presets: the
    // source file stores row data, and this model builds the editor notation.
    const specsText = annotatedLayoutSpecsText(
        [`${definition.name}: ${definition.signature}`, ...definition.rows.map(([label, bases]) => `${label}: ${bases}`)].join('\n'),
        definition.comments,
    );
    return {
        title: definition.title ?? definition.name,
        facets,
        gpuArchs,
        instruction: presetFacetScalar(facets, 'instruction'),
        matrixSize: presetFacetScalar(facets, 'matrixSize'),
        dtype: presetFacetScalar(facets, 'dtype'),
        operand: presetFacetScalar(facets, 'operand'),
        trans: presetFacetScalar(facets, 'trans'),
        major: presetFacetScalar(facets, 'major'),
        state: {
            specsText,
            operationText: definition.name,
            inputName,
        },
    };
}

/**
 * return filtered presets for the current viewer state.
 *
 * @param presets - presets input used by this operation (ComposeLayoutPreset[]).
 * @param filters - filters input used by this operation (ComposeLayoutPresetSelection).
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * filteredPresets(presets, filters);
 */
function filteredPresets(
    presets: ComposeLayoutPreset[],
    filters: ComposeLayoutPresetSelection,
): ComposeLayoutPreset[] {
    return presets.filter((preset) => Object.entries(filters).every(([key, value]) => {
        if (!value) return true;
        return preset.facets[key]?.includes(value) ?? false;
    }));
}

/**
 * return normalized preset field for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @param options - Options that tune this operation.
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizedPresetField(value, options);
 */
function normalizedPresetField(value: string, options: string[]): string {
    if (options.includes(value)) return value;
    return options.length === 1 ? options[0] ?? '' : '';
}

/**
 * return unique preset facet values for the current viewer state.
 *
 * @param presets - presets input used by this operation (ComposeLayoutPreset[]).
 * @param field - Preset field metadata used by this operation.
 * @returns Text entries formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * uniquePresetFacetValues(presets, field);
 */
function uniquePresetFacetValues(presets: ComposeLayoutPreset[], field: ComposeLayoutPresetField): string[] {
    const values = new Set(presets.flatMap((preset) => preset.facets[field.key] ?? []));
    // catalog-provided values define display order; contributed values still
    // appear, but after known values so existing UX stays stable.
    return [
        ...field.values.filter((value) => values.has(value)),
        ...Array.from(values).filter((value) => !field.values.includes(value)),
    ];
}

/**
 * return preset field is complete for the current viewer state.
 *
 * @param preset - Preset data used by this operation.
 * @param selection - Selection data used by this operation.
 * @param key - key input used by this operation (string).
 * @returns Whether the requested condition holds.
 * @noThrows This function has no direct throw path.
 * @example
 * presetFieldIsComplete(preset, selection, key);
 */
function presetFieldIsComplete(
    preset: ComposeLayoutPreset,
    selection: ComposeLayoutPresetSelection,
    key: string,
): boolean {
    const values = preset.facets[key] ?? [];
    const selected = selection[key] ?? '';
    return values.length === 0 ? selected === '' : values.includes(selected);
}

/**
 * return canonical layout specs text for the current viewer state.
 *
 * @param specsText - Text supplied by the caller.
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * canonicalLayoutSpecsText(specsText);
 */
function canonicalLayoutSpecsText(specsText: string): string {
    try {
        return formatSpecsText(parseLayoutSpecs(specsText));
    } catch {
        return specsText.replace(/\r\n/g, '\n').trim();
    }
}
