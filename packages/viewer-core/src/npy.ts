import { product } from './view.js';
import type { DType, NumericArray } from './types.js';

const MAGIC = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]);

const DTYPE_TO_ARRAY = {
    float64: Float64Array,
    float32: Float32Array,
    int32: Int32Array,
    uint8: Uint8Array,
} satisfies Record<DType, new (buffer: ArrayBuffer) => NumericArray>;

const DTYPE_TO_DESCR = {
    float64: '<f8',
    float32: '<f4',
    int32: '<i4',
    uint8: '|u1',
} satisfies Record<DType, string>;

const DESCR_TO_DTYPE = {
    '<f8': 'float64',
    '<f4': 'float32',
    '<i4': 'int32',
    '|u1': 'uint8',
    '<u1': 'uint8',
} satisfies Record<string, DType>;

const DTYPE_BYTES = {
    float64: 8,
    float32: 4,
    int32: 4,
    uint8: 1,
} satisfies Record<DType, number>;

function linearIndex(coord: number[], shape: number[]): number {
    let index = 0;
    coord.forEach((value, axis) => {
        index = (index * shape[axis]) + value;
    });
    return index;
}

function unravelFortran(index: number, shape: number[]): number[] {
    const coord = new Array(shape.length).fill(0);
    for (let axis = 0; axis < shape.length; axis += 1) {
        const dim = shape[axis] ?? 1;
        coord[axis] = index % dim;
        index = Math.floor(index / dim);
    }
    return coord;
}

function parseHeader(header: string): { dtype: DType; fortranOrder: boolean; shape: number[] } {
    const descrMatch = /'descr':\s*'([^']+)'/.exec(header);
    const orderMatch = /'fortran_order':\s*(True|False)/.exec(header);
    const shapeMatch = /'shape':\s*\(([^)]*)\)/.exec(header);
    if (!descrMatch || !orderMatch || !shapeMatch) throw new Error('Invalid .npy header.');
    const dtype = DESCR_TO_DTYPE[descrMatch[1] as keyof typeof DESCR_TO_DTYPE];
    if (!dtype) throw new Error(`Unsupported .npy dtype ${descrMatch[1]}.`);
    const shape = (shapeMatch[1] ?? '')
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean)
        .map((value) => Number.parseInt(value, 10));
    return {
        dtype,
        fortranOrder: orderMatch[1] === 'True',
        shape,
    };
}

function normalizeFortran(dtype: DType, shape: number[], data: NumericArray): NumericArray {
    if (shape.length <= 1) return data;
    const ctor = DTYPE_TO_ARRAY[dtype];
    const normalized = new ctor(new ArrayBuffer(data.byteLength));
    for (let index = 0; index < data.length; index += 1) {
        const coord = unravelFortran(index, shape);
        normalized[linearIndex(coord, shape)] = data[index] as never;
    }
    return normalized;
}

/** Construct the viewer's typed-array wrapper for one dtype and raw buffer. */
export function createTypedArray(dtype: DType, buffer: ArrayBuffer): NumericArray {
    const ctor = DTYPE_TO_ARRAY[dtype];
    return new ctor(buffer);
}

/** Check whether a blob starts with the NumPy `.npy` file signature. */
export async function isNpyFile(file: Blob): Promise<boolean> {
    const prefix = new Uint8Array(await file.slice(0, MAGIC.length).arrayBuffer());
    return MAGIC.every((byte, index) => prefix[index] === byte);
}

/** Load one `.npy` file into dtype, shape, and typed-array payload metadata. */
export async function loadNpy(file: Blob): Promise<{ dtype: DType; shape: number[]; data: NumericArray }> {
    const buffer = await file.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    if (!MAGIC.every((byte, index) => bytes[index] === byte)) throw new Error('File is not a .npy array.');
    const versionMajor = bytes[6] ?? 0;
    const headerLengthOffset = 8;
    const headerLength = versionMajor <= 1
        ? new DataView(buffer).getUint16(headerLengthOffset, true)
        : new DataView(buffer).getUint32(headerLengthOffset, true);
    const dataOffset = headerLengthOffset + (versionMajor <= 1 ? 2 : 4) + headerLength;
    const header = new TextDecoder('latin1').decode(bytes.subarray(headerLengthOffset + (versionMajor <= 1 ? 2 : 4), dataOffset));
    const { dtype, fortranOrder, shape } = parseHeader(header);
    const expectedBytes = product(shape) * DTYPE_BYTES[dtype];
    const raw = createTypedArray(dtype, buffer.slice(dataOffset, dataOffset + expectedBytes));
    return {
        dtype,
        shape,
        data: fortranOrder ? normalizeFortran(dtype, shape, raw) : raw,
    };
}

/** Serialize one tensor payload into a little-endian `.npy` blob. */
export function saveNpy(shape: number[], data: NumericArray, dtype: DType): Blob {
    const descr = DTYPE_TO_DESCR[dtype];
    const shapeText = shape.length === 0
        ? ''
        : shape.length === 1
            ? `${shape[0]},`
            : shape.join(', ');
    const headerBase = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shapeText}), }`;
    const padding = (16 - ((10 + headerBase.length + 1) % 16)) % 16;
    const header = `${headerBase}${' '.repeat(padding)}\n`;
    const headerBytes = new TextEncoder().encode(header);
    const output = new Uint8Array(10 + headerBytes.length + data.byteLength);
    output.set(MAGIC, 0);
    output[6] = 1;
    output[7] = 0;
    new DataView(output.buffer).setUint16(8, headerBytes.length, true);
    output.set(headerBytes, 10);
    output.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 10 + headerBytes.length);
    return new Blob([output], { type: 'application/octet-stream' });
}
