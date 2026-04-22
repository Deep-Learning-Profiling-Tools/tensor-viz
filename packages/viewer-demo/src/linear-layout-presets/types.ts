export type ComposeLayoutPresetFields = {
    gpuArch: string;
    instruction: string;
    matrixSize: string;
    dtype: string;
    operand: string;
    trans?: string;
    major?: string;
    comments?: string[];
    inputName?: string;
    title?: string;
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
