import { LDMATRIX_PRESET_DEFINITIONS } from './ldmatrix.js';
import { MMA_PRESET_DEFINITIONS } from './mma.js';
import { STMATRIX_PRESET_DEFINITIONS } from './stmatrix.js';
import { SWIZZLE_PRESET_DEFINITIONS } from './swizzle.js';
import type { ComposeLayoutPresetFamily, ComposeLayoutPresetFieldDefinition } from './types.js';
import { WGMMA_PRESET_DEFINITIONS } from './wgmma.js';

export const PRESET_GPU_ARCHS = [
    'sm_70',
    'sm_75',
    'sm_80',
    'sm_90',
    'sm_90a',
    'sm_100',
    'sm_100a',
    'sm_100f',
    'sm_110',
    'sm_110a',
    'sm_110f',
    'sm_120',
    'sm_120a',
    'sm_120f',
] as const;

const DEFAULT_PRESET_FIELDS = [
    {
        key: 'gpuArch',
        label: 'GPU Arch',
        placeholder: 'Type GPU arch',
        order: 10,
        required: true,
        values: PRESET_GPU_ARCHS,
    },
    {
        key: 'instruction',
        label: 'Instruction',
        placeholder: 'Type instruction',
        order: 20,
        required: true,
    },
    {
        key: 'matrixSize',
        label: 'Matrix Size',
        placeholder: 'Type matrix size',
        order: 30,
        required: true,
    },
    {
        key: 'dtype',
        label: 'DType',
        placeholder: 'Type dtype',
        order: 40,
        required: true,
    },
    {
        key: 'operand',
        label: 'Operand',
        placeholder: 'Type operand',
        order: 50,
        dependsOn: ['instruction'],
    },
    {
        key: 'trans',
        label: 'Transpose',
        placeholder: 'Type transpose',
        order: 60,
        dependsOn: ['instruction'],
    },
    {
        key: 'major',
        label: 'Major',
        placeholder: 'Type major',
        order: 70,
        dependsOn: ['instruction'],
    },
] satisfies ComposeLayoutPresetFieldDefinition[];

const NVIDIA_PRESET_DEFINITIONS = [
    ...MMA_PRESET_DEFINITIONS,
    ...SWIZZLE_PRESET_DEFINITIONS,
    ...LDMATRIX_PRESET_DEFINITIONS,
    ...STMATRIX_PRESET_DEFINITIONS,
    ...WGMMA_PRESET_DEFINITIONS,
];

export const COMPOSE_LAYOUT_PRESET_FAMILIES = [
    { fields: DEFAULT_PRESET_FIELDS, presets: NVIDIA_PRESET_DEFINITIONS },
] satisfies ComposeLayoutPresetFamily[];
