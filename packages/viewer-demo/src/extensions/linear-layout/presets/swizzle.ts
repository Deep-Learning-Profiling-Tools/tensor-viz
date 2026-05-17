import { GPU_ARCHS_SWIZZLE } from './gpu-archs.js';
import type { ComposeLayoutPresetDefinition } from './types.js';

// swizzle presets are generated because the major variants differ only by which
// output axis receives the contiguous and cross terms.
// the generated rows are intentionally kept in the same notation contributors
// use in hand-written preset files.
// b8 through b128 reuse the same 128-byte tile rule with fewer leading vectors
// as element width increases.

/**
 * return swizzle bases for the current viewer state.
 *
 * @param leadingVectors - leading vectors input used by this operation (number).
 * @param major - major input used by this operation ('MN-major' | 'K-major').
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * swizzleBases(leadingVectors, major);
 */
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
