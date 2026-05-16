import { GPU_ARCHS_SWIZZLE } from './gpu-archs.js';
import type { ComposeLayoutPresetDefinition } from './types.js';

function swizzleBases(leadingVectors: number, major: 'MN-major' | 'K-major'): string {
    const contiguousBases = Array.from({ length: leadingVectors }, (_, index) => major === 'MN-major'
        ? [2 ** index, 0]
        : [0, 2 ** index]);
    const crossBases = Array.from({ length: 3 }, (_, index) => major === 'MN-major'
        ? [2 ** (leadingVectors - 3 + index), 2 ** index]
        : [2 ** index, 2 ** (leadingVectors - 3 + index)]);
    return JSON.stringify([...contiguousBases, ...crossBases]);
}

const SWIZZLE_DTYPES = [
    ['b8', 7],
    ['b16', 6],
    ['b32', 5],
    ['b64', 4],
    ['b128', 3],
] as const;

export const SWIZZLE_PRESET_DEFINITIONS: ComposeLayoutPresetDefinition[] = SWIZZLE_DTYPES.flatMap(([dtype, leadingVectors]) => ([
    {
        name: `swizzle_128B_MN_major_${dtype}`,
        facets: {
            gpuArch: GPU_ARCHS_SWIZZLE,
            instruction: 'swizzle',
            matrixSize: '128B',
            dtype,
            major: 'MN-major',
        },
        comments: ['bX means each element is X bits wide.'],
        signature: '[O] -> [M, K]',
        rows: [['O', swizzleBases(leadingVectors, 'MN-major')]],
        inputName: 'Logical Offsets',
    },
    {
        name: `swizzle_128B_K_major_${dtype}`,
        facets: {
            gpuArch: GPU_ARCHS_SWIZZLE,
            instruction: 'swizzle',
            matrixSize: '128B',
            dtype,
            major: 'K-major',
        },
        comments: ['bX means each element is X bits wide.'],
        signature: '[O] -> [M, K]',
        rows: [['O', swizzleBases(leadingVectors, 'K-major')]],
        inputName: 'Logical Offsets',
    },
])) satisfies ComposeLayoutPresetDefinition[];
