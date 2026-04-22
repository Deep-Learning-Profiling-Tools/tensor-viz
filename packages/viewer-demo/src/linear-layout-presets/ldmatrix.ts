import type { ComposeLayoutPresetDefinition, MatrixTransferLayoutDefinition } from './types.js';

export const MATRIX_TRANSFER_LAYOUTS = [
    {
        name: 'ldmatrix_m8n8_x1_b16',
        gpuArch: 'sm_75',
        instruction: 'ldmatrix',
        matrixSize: 'm8n8.x1',
        dtype: 'b16',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0]]'], ['C', '[[0,0],[1,0],[2,0]]']],
            yes: [['R', '[[0,0],[1,0],[2,0]]'], ['C', '[[4,0],[8,0],[16,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n8_x2_b16',
        gpuArch: 'sm_75',
        instruction: 'ldmatrix',
        matrixSize: 'm8n8.x2',
        dtype: 'b16',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1]]'], ['C', '[[0,0],[1,0],[2,0]]']],
            yes: [['R', '[[0,0],[1,0],[2,0],[0,1]]'], ['C', '[[4,0],[8,0],[16,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n8_x4_b16',
        gpuArch: 'sm_75',
        instruction: 'ldmatrix',
        matrixSize: 'm8n8.x4',
        dtype: 'b16',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1],[0,2]]'], ['C', '[[0,0],[1,0],[2,0]]']],
            yes: [['R', '[[0,0],[1,0],[2,0],[0,1],[0,2]]'], ['C', '[[4,0],[8,0],[16,0]]']],
        },
    },
    {
        name: 'ldmatrix_m16n16_x1_b8',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm16n16.x1',
        dtype: 'b8',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            yes: [['R', '[[0,0],[0,0],[1,0],[2,0]]'], ['C', '[[4,0],[8,0],[16,0],[0,1]]']],
        },
    },
    {
        name: 'ldmatrix_m16n16_x2_b8',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm16n16.x2',
        dtype: 'b8',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            yes: [['R', '[[0,0],[0,0],[1,0],[2,0]]'], ['C', '[[4,0],[8,0],[16,0],[0,1],[0,2]]']],
        },
    },
    {
        name: 'ldmatrix_m8n16_x1_b4',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm8n16.x1',
        dtype: 'b4',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0]]'], ['C', '[[0,0],[0,0],[1,0],[2,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n16_x2_b4',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm8n16.x2',
        dtype: 'b4',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1]]'], ['C', '[[0,0],[0,0],[1,0],[2,0]]']],
        },
    },
    {
        name: 'ldmatrix_m8n16_x4_b4',
        gpuArch: 'sm_100',
        instruction: 'ldmatrix',
        matrixSize: 'm8n16.x4',
        dtype: 'b4',
        operand: '',
        inputName: 'Shared Memory',
        rowsByTrans: {
            no: [['R', '[[4,0],[8,0],[16,0],[0,1],[0,2]]'], ['C', '[[0,0],[0,0],[1,0],[2,0]]']],
        },
    },
] satisfies MatrixTransferLayoutDefinition[];

export const LDMATRIX_PRESET_DEFINITIONS: ComposeLayoutPresetDefinition[] = MATRIX_TRANSFER_LAYOUTS.flatMap((layout) => (
    ['no', 'yes']
        .filter((trans): trans is 'no' | 'yes' => Boolean(layout.rowsByTrans[trans]))
        .map((trans) => ({
            gpuArch: layout.gpuArch,
            instruction: 'ldmatrix',
            matrixSize: layout.matrixSize,
            dtype: layout.dtype,
            operand: '',
            trans,
            inputName: layout.inputName,
            signature: '[R, C] -> [T, R32]',
            name: trans === 'yes' ? layout.name.replace(/_([^_]+)$/, '_trans_$1') : layout.name,
            rows: layout.rowsByTrans[trans]!,
            comments: trans === 'yes'
                ? [
                    'Consecutive columns need not be contiguous in memory; each row address points to the start of a matrix column.',
                    'trans = yes means the matrix is loaded in column-major format.',
                ]
                : ['Consecutive rows need not be contiguous in memory; each row address points to the start of a matrix row.'],
        }))
));
