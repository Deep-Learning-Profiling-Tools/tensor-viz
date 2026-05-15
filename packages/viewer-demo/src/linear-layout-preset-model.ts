import { COMPOSE_LAYOUT_PRESET_FAMILIES } from './linear-layout-presets/index.js';
import type {
    ComposeLayoutPresetDefinition,
    ComposeLayoutPresetFacetValue,
    ComposeLayoutPresetFieldDefinition,
} from './linear-layout-presets/types.js';
import {
    formatSpecsText,
    parseLayoutSpecs,
    parseSignature,
    stripLayoutComment,
} from './linear-layout-parser.js';

export type ComposeLayoutPresetSelection = Record<string, string>;

/** normalized field metadata consumed directly by the preset widget. */
export type ComposeLayoutPresetField = Required<Omit<ComposeLayoutPresetFieldDefinition, 'values'>> & {
    id: string;
    values: string[];
};

/** editor state inserted when a preset selection resolves to one layout. */
export type ComposeLayoutPresetState = {
    specsText: string;
    operationText: string;
    inputName: string;
};

/** normalized preset with compatibility aliases for older widget call sites. */
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

/** options keyed by field name plus legacy plural aliases. */
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

/** build an empty selection that includes every registered field key. */
export function emptyComposeLayoutPresetSelection(): ComposeLayoutPresetSelection {
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [field.key, '']));
}

/** clone external selection data while dropping unknown/non-string values. */
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

/** validate persisted preset selection state before it is copied into the editor. */
export function isComposeLayoutPresetSelection(value: unknown): value is ComposeLayoutPresetSelection {
    if (!value || typeof value !== 'object') return false;
    return Object.values(value as Record<string, unknown>).every((entry) => typeof entry === 'string');
}

/** return field metadata in render order. */
export function composeLayoutPresetFields(): ComposeLayoutPresetField[] {
    return COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => ({
        ...field,
        dependsOn: [...field.dependsOn],
        values: [...field.values],
    }));
}

/** return the internal catalog for invariant tests and model-level matching. */
export function composeLayoutPresetCatalog(): ComposeLayoutPreset[] {
    return COMPOSE_LAYOUT_PRESET_CATALOG;
}

/** return a cloned catalog for UI code that may derive temporary state from presets. */
export function composeLayoutPresets(): ComposeLayoutPreset[] {
    return composeLayoutPresetCatalog().map((preset) => ({
        ...preset,
        facets: Object.fromEntries(Object.entries(preset.facets).map(([key, values]) => [key, [...values]])),
        gpuArchs: [...preset.gpuArchs],
        state: { ...preset.state },
    }));
}

/** infer the selector values for an editor state that exactly matches a preset. */
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

/** compute the valid dropdown/text options after applying the current selection. */
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
        const values = uniquePresetFacetValues(filteredPresets(presets, withoutPresetField(current, field.key)), field);
        options[field.key] = values;
        const alias = PRESET_FIELD_OPTION_ALIASES[field.key as keyof typeof PRESET_FIELD_OPTION_ALIASES];
        if (alias) options[alias] = values;
    });
    return options;
}

/** clear invalid field values and autofill singleton options. */
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

/** resolve a fully specified selection to exactly one preset. */
export function composeLayoutPresetForSelection(
    selection: ComposeLayoutPresetSelection | undefined,
): ComposeLayoutPreset | null {
    const current = cloneComposeLayoutPresetSelection(selection);
    const matches = filteredPresets(composeLayoutPresetCatalog(), current).filter((preset) => (
        COMPOSE_LAYOUT_PRESET_FIELDS.every((field) => presetFieldIsComplete(preset, current, field.key))
    ));
    return matches.length === 1 ? matches[0]! : null;
}

function layoutSpecText(signature: string, rows: string[]): string {
    return [signature, ...rows].join('\n');
}

function mergedPresetFields(definitions: readonly ComposeLayoutPresetFieldDefinition[]): ComposeLayoutPresetField[] {
    const fields = new Map<string, ComposeLayoutPresetField>();
    definitions.forEach((definition) => {
        fields.set(definition.key, normalizePresetFieldDefinition(definition, fields.get(definition.key)));
    });
    return Array.from(fields.values()).sort((left, right) => left.order - right.order || left.key.localeCompare(right.key));
}

function normalizePresetFieldDefinition(
    definition: ComposeLayoutPresetFieldDefinition,
    current?: ComposeLayoutPresetField,
): ComposeLayoutPresetField {
    const values = [...new Set([...(current?.values ?? []), ...(definition.values ?? [])])];
    const dependsOn = [...new Set([...(current?.dependsOn ?? []), ...(definition.dependsOn ?? [])])];
    return {
        key: definition.key,
        id: current?.id ?? `linear-layout-preset-${kebabPresetFieldKey(definition.key)}`,
        label: current?.label ?? definition.label,
        placeholder: current?.placeholder ?? definition.placeholder,
        order: Math.min(current?.order ?? definition.order, definition.order),
        required: Boolean(current?.required || definition.required),
        dependsOn,
        values,
    };
}

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

function kebabPresetFieldKey(key: string): string {
    return key.replace(/([a-z0-9])([A-Z])/g, '$1-$2').replace(/[^a-z0-9]+/gi, '-').toLowerCase();
}

function presetOperationText(specsText: string): string {
    const signature = specsText.split('\n', 1)[0]?.trim() ?? '';
    const colonIndex = signature.indexOf(':');
    return (colonIndex === -1 ? signature : signature.slice(0, colonIndex)).trim();
}

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

function presetDefinitionFacets(definition: ComposeLayoutPresetDefinition): Record<string, string[]> {
    // every normalized preset receives every known field key.  Empty arrays mean
    // the field is irrelevant for that preset and should remain blank.
    return Object.fromEntries(COMPOSE_LAYOUT_PRESET_FIELDS.map((field) => [
        field.key,
        presetDefinitionFacetValues(definition, field.key),
    ]));
}

function presetDefinitionFacetValues(definition: ComposeLayoutPresetDefinition, key: string): string[] {
    const facet = definition.facets?.[key];
    if (facet !== undefined) return normalizePresetFacetValue(facet);
    // legacy scalar fields keep old NVIDIA presets working while new families
    // can describe selector behavior entirely through facets.
    if (isLegacyPresetFieldKey(key)) return normalizePresetFacetValue(definition[key] ?? '');
    return [];
}

function normalizePresetFacetValue(value: ComposeLayoutPresetFacetValue): string[] {
    const values = Array.isArray(value) ? value : [value];
    return values.map((entry) => String(entry)).filter(Boolean);
}

function isLegacyPresetFieldKey(key: string): key is typeof LEGACY_PRESET_FIELD_KEYS[number] {
    return LEGACY_PRESET_FIELD_KEYS.includes(key as typeof LEGACY_PRESET_FIELD_KEYS[number]);
}

function presetFacetScalar(facets: Record<string, string[]>, key: string): string {
    return facets[key]?.[0] ?? '';
}

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
        layoutSpecText(`${definition.name}: ${definition.signature}`, definition.rows.map(([label, bases]) => `${label}: ${bases}`)),
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

function filteredPresets(
    presets: ComposeLayoutPreset[],
    filters: ComposeLayoutPresetSelection,
): ComposeLayoutPreset[] {
    return presets.filter((preset) => Object.entries(filters).every(([key, value]) => {
        if (!value) return true;
        return preset.facets[key]?.includes(value) ?? false;
    }));
}

function normalizedPresetField(value: string, options: string[]): string {
    if (options.includes(value)) return value;
    return options.length === 1 ? options[0] ?? '' : '';
}

function withoutPresetField(selection: ComposeLayoutPresetSelection, key: string): ComposeLayoutPresetSelection {
    return Object.fromEntries(Object.entries(selection).map(([fieldKey, value]) => [fieldKey, fieldKey === key ? '' : value]));
}

function uniquePresetFacetValues(presets: ComposeLayoutPreset[], field: ComposeLayoutPresetField): string[] {
    const values = new Set(presets.flatMap((preset) => preset.facets[field.key] ?? []));
    // catalog-provided values define display order; contributed values still
    // appear, but after known values so existing UX stays stable.
    return [
        ...field.values.filter((value) => values.has(value)),
        ...Array.from(values).filter((value) => !field.values.includes(value)),
    ];
}

function presetFieldIsComplete(
    preset: ComposeLayoutPreset,
    selection: ComposeLayoutPresetSelection,
    key: string,
): boolean {
    const values = preset.facets[key] ?? [];
    const selected = selection[key] ?? '';
    return values.length === 0 ? selected === '' : values.includes(selected);
}

function canonicalLayoutSpecsText(specsText: string): string {
    try {
        return formatSpecsText(parseLayoutSpecs(specsText));
    } catch {
        return specsText.replace(/\r\n/g, '\n').trim();
    }
}
