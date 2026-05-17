import type { DType, NumericArray } from './types.js';
import { isDType } from './validation.js';

const DTYPE_TO_ARRAY = {
    float64: Float64Array,
    float32: Float32Array,
    int32: Int32Array,
    uint8: Uint8Array,
} satisfies Record<DType, new (buffer: ArrayBuffer) => NumericArray>;

// this table is the only place where persisted dtype strings become platform
// typed-array constructors; validation keeps unsupported strings out earlier.

/**
 * Construct the viewer's typed-array wrapper for one dtype and raw tensor buffer.
 *
 * @param dtype - Tensor dtype string from a manifest, such as `float32`, used to choose the matching JavaScript typed-array constructor.
 * @param buffer - Raw binary payload bytes for one tensor, already fetched or decoded into an ArrayBuffer.
 * @returns A numeric typed-array view over `buffer` whose element type matches `dtype`.
 * @throws Error when `dtype` is not one of the viewer-supported dtype strings.
 * @example
 * const buffer = new ArrayBuffer(8);
 * new Float32Array(buffer).set([1.5, 2.5]);
 *
 * const values = createTypedArray('float32', buffer);
 *
 * expect(values).toBeInstanceOf(Float32Array);
 * expect(Array.from(values)).toEqual([1.5, 2.5]);
 *
 * @example
 * expect(() => createTypedArray('complex64' as DType, new ArrayBuffer(8)))
 *   .toThrow('Unsupported dtype complex64.');
 */
export function createTypedArray(dtype: DType, buffer: ArrayBuffer): NumericArray {
    if (!isDType(dtype)) throw new Error(`Unsupported dtype ${String(dtype)}.`);
    const ctor = DTYPE_TO_ARRAY[dtype];
    return new ctor(buffer);
}
