import type { DType, NumericArray } from './types.js';

const DTYPE_TO_ARRAY = {
    float64: Float64Array,
    float32: Float32Array,
    int32: Int32Array,
    uint8: Uint8Array,
} satisfies Record<DType, new (buffer: ArrayBuffer) => NumericArray>;

/** Construct the viewer's typed-array wrapper for one dtype and raw buffer. */
export function createTypedArray(dtype: DType, buffer: ArrayBuffer): NumericArray {
    const ctor = DTYPE_TO_ARRAY[dtype];
    return new ctor(buffer);
}
