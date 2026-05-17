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
 * Construct the viewer's typed-array wrapper for one dtype and raw buffer.
 *
 * @param dtype - dtype input used by this operation (DType).
 * @param buffer - buffer input used by this operation (ArrayBuffer).
 * @returns Computed NumericArray value for the caller.
 * @throws Error when the requested input or state is invalid.
 * @example
 * createTypedArray(dtype, buffer);
 */
export function createTypedArray(dtype: DType, buffer: ArrayBuffer): NumericArray {
    if (!isDType(dtype)) throw new Error(`Unsupported dtype ${String(dtype)}.`);
    const ctor = DTYPE_TO_ARRAY[dtype];
    return new ctor(buffer);
}
