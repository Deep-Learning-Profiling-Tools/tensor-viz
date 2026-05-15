/** raw fields accepted for older in-tree preset definitions.
 *
 * New preset families should prefer `facets`: the selector UI only needs to
 * know which facet values identify a preset, so keeping that data explicit
 * prevents widget code from growing instruction-specific branches.
 */
export type ComposeLayoutPresetFields = {
    gpuArch?: string;
    instruction?: string;
    matrixSize?: string;
    dtype?: string;
    operand?: string;
    trans?: string;
    major?: string;
    comments?: string[];
    inputName?: string;
    title?: string;
    facets?: ComposeLayoutPresetFacets;
};

export type ComposeLayoutPresetFacetValue = string | readonly string[];

/** selector values that must match before a preset can be chosen. */
export type ComposeLayoutPresetFacets = Record<string, ComposeLayoutPresetFacetValue>;

/** one text field/dropdown shown by the preset widget. */
export type ComposeLayoutPresetFieldDefinition = {
    key: string;
    label: string;
    placeholder: string;
    order: number;
    required?: boolean;
    /** field-level visibility dependency; value-specific hiding comes from filtered facet options. */
    dependsOn?: string[];
    values?: readonly string[];
};

/** one independently reviewable preset family, usually one GPU instruction family. */
export type ComposeLayoutPresetFamily = {
    fields?: readonly ComposeLayoutPresetFieldDefinition[];
    presets: readonly ComposeLayoutPresetDefinition[];
};

export type ComposeLayoutPresetTextDefinition = ComposeLayoutPresetFields & {
    specsText: string;
};

export type ComposeLayoutPresetNamedDefinition = ComposeLayoutPresetFields & {
    name: string;
    signature: string;
    comments?: string[];
    rows: Array<[label: string, bases: string]>;
};

export type ComposeLayoutPresetDefinition = ComposeLayoutPresetTextDefinition | ComposeLayoutPresetNamedDefinition;

export type MatrixTransferLayoutDefinition = {
    name: string;
    gpuArch: string;
    instruction: 'ldmatrix' | 'stmatrix';
    matrixSize: string;
    dtype: string;
    operand: string;
    inputName: string;
    rowsByTrans: Partial<Record<'no' | 'yes', Array<[label: string, bases: string]>>>;
};
