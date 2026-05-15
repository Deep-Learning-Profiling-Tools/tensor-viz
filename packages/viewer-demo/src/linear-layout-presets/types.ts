/** raw fields accepted for older in-tree preset definitions.
 *
 * new preset families should prefer `facets`: the selector UI only needs to
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

/** selector values that must match before a preset can be chosen.
 *
 * arrays mean one preset is valid for several selector values, such as a layout
 * that applies to multiple GPU archs.  The invariant tests expand these arrays
 * to make sure every concrete selection still resolves to exactly one preset.
 */
export type ComposeLayoutPresetFacets = Record<string, ComposeLayoutPresetFacetValue>;

/** one text field/dropdown shown by the preset widget. */
export type ComposeLayoutPresetFieldDefinition = {
    key: string;
    label: string;
    placeholder: string;
    /** lower values render earlier; keep shared hardware concepts before family-specific fields. */
    order: number;
    /** required fields are always visible and must be filled before a preset resolves. */
    required?: boolean;
    /** field-level visibility dependency; value-specific hiding comes from filtered facet options. */
    dependsOn?: string[];
    /** preferred display order for known values; unknown contributed values append after this list. */
    values?: readonly string[];
};

/** one independently reviewable preset family, usually one GPU instruction family. */
export type ComposeLayoutPresetFamily = {
    fields?: readonly ComposeLayoutPresetFieldDefinition[];
    presets: readonly ComposeLayoutPresetDefinition[];
};

export type ComposeLayoutPresetTextDefinition = ComposeLayoutPresetFields & {
    /** complete specs text for layouts that are clearer as hand-written notation. */
    specsText: string;
};

export type ComposeLayoutPresetNamedDefinition = ComposeLayoutPresetFields & {
    /** layout name used both in the generated signature and as the operation text. */
    name: string;
    signature: string;
    comments?: string[];
    /** basis rows kept as strings so preset files stay compact and close to ISA tables. */
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
