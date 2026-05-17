/**
 * raw fields accepted for older in-tree preset definitions.
 *
 * new preset families should prefer `facets`: the selector UI only needs to
 * know which facet values identify a preset, so keeping that data explicit
 * prevents widget code from growing instruction-specific branches.
 *
 * @example
 * const value: ComposeLayoutPresetFields = {} as ComposeLayoutPresetFields;
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

/**
 * shape of compose layout preset facet value data used by the viewer.
 *
 * @example
 * const value: ComposeLayoutPresetFacetValue = {} as ComposeLayoutPresetFacetValue;
 */
export type ComposeLayoutPresetFacetValue = string | readonly string[];

/**
 * selector values that must match before a preset can be chosen.
 *
 * arrays mean one preset is valid for several selector values, such as a layout
 * that applies to multiple GPU archs.  The invariant tests expand these arrays
 * to make sure every concrete selection still resolves to exactly one preset.
 *
 * @example
 * const value: ComposeLayoutPresetFacets = {} as ComposeLayoutPresetFacets;
 */
export type ComposeLayoutPresetFacets = Record<string, ComposeLayoutPresetFacetValue>;

/**
 * one text field/dropdown shown by the preset widget.
 *
 * @example
 * const value: ComposeLayoutPresetFieldDefinition = {} as ComposeLayoutPresetFieldDefinition;
 */
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

/**
 * one independently reviewable preset family, usually one GPU instruction family.
 *
 * @example
 * const value: ComposeLayoutPresetFamily = {} as ComposeLayoutPresetFamily;
 */
export type ComposeLayoutPresetFamily = {
    fields?: readonly ComposeLayoutPresetFieldDefinition[];
    presets: readonly ComposeLayoutPresetDefinition[];
};

/**
 * shape of compose layout preset text definition data used by the viewer.
 *
 * @example
 * const value: ComposeLayoutPresetTextDefinition = {} as ComposeLayoutPresetTextDefinition;
 */
export type ComposeLayoutPresetTextDefinition = ComposeLayoutPresetFields & {
    /** complete specs text for layouts that are clearer as hand-written notation. */
    specsText: string;
};

/**
 * shape of compose layout preset named definition data used by the viewer.
 *
 * @example
 * const value: ComposeLayoutPresetNamedDefinition = {} as ComposeLayoutPresetNamedDefinition;
 */
export type ComposeLayoutPresetNamedDefinition = ComposeLayoutPresetFields & {
    /** layout name used both in the generated signature and as the operation text. */
    name: string;
    signature: string;
    comments?: string[];
    /** basis rows kept as strings so preset files stay compact and close to ISA tables. */
    rows: Array<[label: string, bases: string]>;
};

/**
 * shape of compose layout preset definition data used by the viewer.
 *
 * @example
 * const value: ComposeLayoutPresetDefinition = {} as ComposeLayoutPresetDefinition;
 */
export type ComposeLayoutPresetDefinition = ComposeLayoutPresetTextDefinition | ComposeLayoutPresetNamedDefinition;

/**
 * shape of matrix transfer layout definition data used by the viewer.
 *
 * @example
 * const value: MatrixTransferLayoutDefinition = {} as MatrixTransferLayoutDefinition;
 */
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
