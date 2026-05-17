import type {
    BundleManifest,
    ColorInstruction,
    DType,
    SessionBundleManifest,
    TensorViewEditor,
    Vec3,
    ViewerSnapshot,
} from './types.js';

export const VIEWER_LIMITS = {
    maxTabs: 64,
    maxTensors: 256,
    maxRank: 12,
    maxDimension: 1_000_000,
    maxTensorElements: 1_000_000,
    maxPayloadBytes: 64 * 1024 * 1024,
    maxTextLength: 256,
    maxEditorInputLength: 20_000,
    maxEditorEntries: 512,
    maxColorInstructions: 256,
    maxCustomColorEntries: 1_000_000,
    maxAbsWorldCoordinate: 1_000_000,
    minCameraZoom: 1e-9,
    maxCameraZoom: 1_000_000,
    minDimensionBlockGapMultiple: 1,
    maxDimensionBlockGapMultiple: 100,
} as const;

const DTYPE_BYTES = {
    float64: 8,
    float32: 4,
    int32: 4,
    uint8: 1,
} satisfies Record<DType, number>;

const DISPLAY_MODES = new Set(['2d', '3d']);
const INTERACTION_MODES = new Set(['pan', 'select', 'rotate']);
const MAPPING_SCHEMES = new Set(['z-order', 'contiguous']);

/**
 * Checks whether an unknown manifest or API value is one of the dtype strings supported by viewer-core tensor storage.
 *
 * @param value - Candidate dtype field read from a manifest, snapshot, or public API input.
 * @returns `true` when `value` is a supported `DType` key; otherwise `false`, allowing callers to narrow the value before indexing dtype tables.
 * @noThrows The check only performs a string type test and membership lookup against the dtype byte-size table, so non-string, null, and undefined inputs are reported as `false` instead of throwing.
 * @example
 * isDType('float32'); // true
 * isDType('complex64'); // false
 */
export function isDType(value: unknown): value is DType {
    return typeof value === 'string' && value in DTYPE_BYTES;
}

/**
 * Looks up the number of bytes used by one tensor element of a supported viewer dtype.
 *
 * @param dtype - Candidate dtype string from tensor metadata or caller input.
 * @returns The byte width for one tensor element, such as `4` for `float32`.
 * @throws Error when `dtype` is not a supported dtype string.
 * @example
 * dtypeByteLength('float32'); // 4
 *
 * @example
 * expect(() => dtypeByteLength('complex64')).toThrow('Unsupported dtype complex64.');
 */
export function dtypeByteLength(dtype: unknown): number {
    if (!isDType(dtype)) throw new Error(`Unsupported dtype ${String(dtype)}.`);
    return DTYPE_BYTES[dtype];
}

/**
 * Validates that a manifest or snapshot field is a non-null, non-array object before reading its named properties.
 *
 * @param value - Candidate field value being normalized, such as `manifest`, `viewer.camera`, or a tensor entry.
 * @param label - Human-readable path included in validation errors for the candidate field.
 * @returns The original value narrowed to a string-keyed record so callers can validate its properties.
 * @throws Error when `value` is `null`, an array, or a primitive value; the message names the supplied `label`.
 * @example
 * const tensor = assertObject({ id: 'weights', dtype: 'float32' }, 'manifest.tensors[0]');
 * tensor.id; // 'weights'
 *
 * @example
 * expect(() => assertObject([], 'viewer.tensors')).toThrow('viewer.tensors must be an object.');
 */
function assertObject(value: unknown, label: string): Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new Error(`${label} must be an object.`);
    }
    return value as Record<string, unknown>;
}

/**
 * Validates that a manifest or viewer snapshot field is an array and is small enough for core normalization.
 *
 * @param value - Raw field value read from untrusted bundle, tensor, editor, or viewer snapshot data.
 * @param label - Path-like field name, such as `manifest.tensors` or `viewer.tensors`, used in error messages.
 * @param maxLength - Inclusive maximum number of array entries allowed before the field is rejected.
 * @returns The original array value, narrowed to `unknown[]`, so callers can map each entry through field-specific validators.
 * @throws Error when `value` is not an array or when its length is greater than `maxLength`.
 * @example
 * const tensors = assertArray([{ id: 'weights' }], 'manifest.tensors', 4);
 * console.assert(tensors.length === 1);
 *
 * @example
 * try {
 *   assertArray('weights', 'manifest.tensors', 4);
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'manifest.tensors must be an array.');
 * }
 */
function assertArray(value: unknown, label: string, maxLength: number = VIEWER_LIMITS.maxEditorEntries): unknown[] {
    if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
    if (value.length > maxLength) throw new Error(`${label} has too many entries.`);
    return value;
}

/**
 * Coerces a raw manifest or viewer field to a JavaScript number and rejects non-finite numeric values.
 *
 * @param value - Raw scalar value from untrusted viewer data, such as a coordinate, zoom, color channel, or spacing setting.
 * @param label - Path-like field name, such as `viewer.camera.zoom`, used in error messages.
 * @returns The `Number(value)` result after confirming it is finite, for use by more specific validators.
 * @throws Error when `value` converts to `NaN`, `Infinity`, or `-Infinity`.
 * @example
 * const zoom = finiteNumber('2.5', 'viewer.camera.zoom');
 * console.assert(zoom === 2.5);
 *
 * @example
 * try {
 *   finiteNumber(Number.POSITIVE_INFINITY, 'viewer.camera.zoom');
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'viewer.camera.zoom must be finite.');
 * }
 */
function finiteNumber(value: unknown, label: string): number {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`${label} must be finite.`);
    return number;
}

/**
 * Coerces a raw manifest or viewer field to a finite integer for tensor dimensions, coordinates, and editor indices.
 *
 * @param value - Raw scalar value expected to represent an integer index, dimension size, or editor position.
 * @param label - Path-like field name, such as `shape[0]` or `viewer.tensors[0].view.hiddenIndices[1]`, used in error messages.
 * @returns The `Number(value)` result after confirming it is finite and has no fractional part.
 * @throws Error when `value` is non-finite or when it converts to a fractional number.
 * @example
 * const axisSize = finiteInteger('16', 'shape[0]');
 * console.assert(axisSize === 16);
 *
 * @example
 * try {
 *   finiteInteger(3.5, 'shape[0]');
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'shape[0] must be an integer.');
 * }
 */
function finiteInteger(value: unknown, label: string): number {
    const number = finiteNumber(value, label);
    if (!Number.isInteger(number)) throw new Error(`${label} must be an integer.`);
    return number;
}

/**
 * Coerces a raw numeric viewer field and verifies that it falls within the inclusive range accepted by the viewer engine.
 *
 * @param value - Raw scalar value from viewer data, such as a camera zoom, world coordinate, or layout spacing setting.
 * @param label - Path-like field name, such as `viewer.camera.zoom`, used in error messages.
 * @param min - Inclusive lower bound accepted for the field.
 * @param max - Inclusive upper bound accepted for the field.
 * @returns The finite `Number(value)` result when it is greater than or equal to `min` and less than or equal to `max`.
 * @throws Error when `value` is non-finite or when the converted number is outside the inclusive `[min, max]` range.
 * @example
 * const zoom = boundedFiniteNumber('1.25', 'viewer.camera.zoom', 0.01, 100);
 * console.assert(zoom === 1.25);
 *
 * @example
 * try {
 *   boundedFiniteNumber(0, 'viewer.camera.zoom', 0.01, 100);
 * } catch (error) {
 *   console.assert(error instanceof Error);
 *   console.assert(error.message === 'viewer.camera.zoom is out of range.');
 * }
 */
function boundedFiniteNumber(value: unknown, label: string, min: number, max: number): number {
    const number = finiteNumber(value, label);
    if (number < min || number > max) throw new Error(`${label} is out of range.`);
    return number;
}

/**
 * Validates a string field from a viewer manifest, session snapshot, or editor snapshot and enforces the field's character limit.
 *
 * @param value - Unknown field value read from caller-provided serialized viewer data.
 * @param label - Dot-path label, such as `tensor.name`, used in validation error messages.
 * @param maxLength - Maximum allowed string length for this field; defaults to the viewer-wide text limit.
 * @returns The original string once it has been confirmed to be a string within the configured length limit.
 * @throws Error when `value` is not a string or when its length is greater than `maxLength`.
 * @example
 * boundedString('activation', 'tensor.name', 20);
 * // Returns: 'activation'
 *
 * @example
 * boundedString(42, 'tensor.name');
 * // Throws: Error('tensor.name must be a string.')
 *
 * @example
 * boundedString('abcdef', 'tensor.name', 3);
 * // Throws: Error('tensor.name is too long.')
 */
function boundedString(value: unknown, label: string, maxLength: number = VIEWER_LIMITS.maxTextLength): string {
    if (typeof value !== 'string') throw new Error(`${label} must be a string.`);
    if (value.length > maxLength) throw new Error(`${label} is too long.`);
    return value;
}

/**
 * Multiplies a tensor shape's dimensions to determine how many cells the tensor contains, stopping when the viewer element limit would be exceeded.
 *
 * @param shape - Already-normalized tensor dimensions, one positive integer per axis.
 * @param label - Name or path for the shape being counted, used in overflow and size-limit error messages.
 * @returns The product of all dimensions, which callers use as the tensor cell count for payload sizing and per-cell validation.
 * @throws Error when an intermediate product is not a safe integer or exceeds `VIEWER_LIMITS.maxTensorElements`; the message includes the axis where the limit was crossed.
 * @example
 * tensorElementCount([2, 3, 4], 'tensor.shape');
 * // Returns: 24
 *
 * @example
 * tensorElementCount([VIEWER_LIMITS.maxTensorElements + 1], 'tensor.shape');
 * // Throws: Error('tensor.shape has too many elements at axis 0.')
 */
export function tensorElementCount(shape: readonly number[], label = 'shape'): number {
    return shape.reduce((total, dim, axis) => {
        const next = total * dim;
        if (!Number.isSafeInteger(next) || next > VIEWER_LIMITS.maxTensorElements) {
            throw new Error(`${label} has too many elements at axis ${axis}.`);
        }
        return next;
    }, 1);
}

/**
 * Normalizes an unknown tensor shape value into the positive integer dimensions accepted by the viewer engine.
 *
 * @param value - Caller-provided shape value from a tensor spec, manifest, or viewer API call.
 * @param label - Name or dot-path for the shape, used to identify invalid dimensions in error messages.
 * @returns A new array of positive integer dimensions suitable for tensor-view parsing, layout math, and payload validation.
 * @throws Error when `value` is not an array, has too many axes, contains a non-finite or non-integer dimension, contains a dimension less than 1, contains a dimension above `VIEWER_LIMITS.maxDimension`, or has too many total elements.
 * @example
 * validateTensorShape([2, 3, 4], 'tensor.shape');
 * // Returns: [2, 3, 4]
 *
 * @example
 * validateTensorShape([2, 0], 'tensor.shape');
 * // Throws: Error('tensor.shape[1] must be positive.')
 */
export function validateTensorShape(value: unknown, label = 'shape'): number[] {
    const shape = assertArray(value, label, VIEWER_LIMITS.maxRank).map((dim, axis) => {
        const number = finiteInteger(dim, `${label}[${axis}]`);
        if (number < 1) throw new Error(`${label}[${axis}] must be positive.`);
        if (number > VIEWER_LIMITS.maxDimension) throw new Error(`${label}[${axis}] is too large.`);
        return number;
    });
    tensorElementCount(shape, label);
    return shape;
}

/**
 * Computes the exact byte length required to store a tensor payload with the given dtype and shape.
 *
 * @param dtype - Supported tensor data type whose element width is used for the payload calculation.
 * @param shape - Tensor dimensions whose product gives the number of payload elements.
 * @returns The number of bytes that a matching tensor payload file or buffer must contain.
 * @throws Error when the shape has too many elements, the element count is not a safe integer, or the resulting payload size exceeds `VIEWER_LIMITS.maxPayloadBytes`.
 * @example
 * expectedTensorByteLength('float32', [2, 3]);
 * // Returns: 24
 *
 * @example
 * expectedTensorByteLength('float32', [VIEWER_LIMITS.maxTensorElements + 1]);
 * // Throws: Error('shape has too many elements at axis 0.')
 */
export function expectedTensorByteLength(dtype: DType, shape: readonly number[]): number {
    const bytes = tensorElementCount(shape) * dtypeByteLength(dtype);
    if (!Number.isSafeInteger(bytes) || bytes > VIEWER_LIMITS.maxPayloadBytes) {
        throw new Error('Tensor payload is too large.');
    }
    return bytes;
}

/**
 * Verifies that a tensor data buffer contains exactly the number of bytes implied by its dtype and shape.
 *
 * @param dtype - Tensor element type used to determine bytes per element.
 * @param shape - Tensor dimensions whose product determines the element count.
 * @param byteLength - Actual byte length of the payload or typed-array buffer being attached to the tensor.
 * @returns Nothing; returns normally when the payload size matches the tensor metadata.
 * @throws Error when byteLength does not equal the expected byte count for dtype and shape.
 * @example
 * validateTensorPayload('float32', [2, 3], 24); // OK: 6 float32 values at 4 bytes each.
 *
 * expect(() => validateTensorPayload('float32', [1], 8)).toThrow(
 *   'Tensor payload byte length 8 does not match expected 4.',
 * );
 */
export function validateTensorPayload(dtype: DType, shape: readonly number[], byteLength: number): void {
    const expectedBytes = expectedTensorByteLength(dtype, shape);
    if (byteLength !== expectedBytes) {
        throw new Error(`Tensor payload byte length ${byteLength} does not match expected ${expectedBytes}.`);
    }
}

/**
 * Validates a three-component viewer vector such as a tensor offset or camera position, target, or rotation.
 *
 * @param value - Unknown snapshot field that must be an array with exactly three numeric components.
 * @param label - Dot-path label for the field, used to identify the failing component in validation errors.
 * @returns A Vec3 containing the three finite components after bounds checks against viewer coordinate limits.
 * @throws Error when value is not a three-item array, a component is not finite, or a component exceeds the allowed world-coordinate range.
 * @example
 * const offset = validateVec3([10, 0, -5], 'viewer.tensors[0].offset');
 * // offset is [10, 0, -5]
 *
 * expect(() => validateVec3([1, 2], 'viewer.camera.position')).toThrow(
 *   'viewer.camera.position must have three values.',
 * );
 */
function validateVec3(value: unknown, label: string): Vec3 {
    const tuple = assertArray(value, label, 3);
    if (tuple.length !== 3) throw new Error(`${label} must have three values.`);
    return [
        boundedFiniteNumber(
            tuple[0],
            `${label}[0]`,
            -VIEWER_LIMITS.maxAbsWorldCoordinate,
            VIEWER_LIMITS.maxAbsWorldCoordinate,
        ),
        boundedFiniteNumber(
            tuple[1],
            `${label}[1]`,
            -VIEWER_LIMITS.maxAbsWorldCoordinate,
            VIEWER_LIMITS.maxAbsWorldCoordinate,
        ),
        boundedFiniteNumber(
            tuple[2],
            `${label}[2]`,
            -VIEWER_LIMITS.maxAbsWorldCoordinate,
            VIEWER_LIMITS.maxAbsWorldCoordinate,
        ),
    ];
}

/**
 * Ensures that a validated manifest, session, or viewer entry list does not contain repeated string ids.
 *
 * @param entries - Array of already validated objects that each expose an id property.
 * @param label - Dot-path label for the entry list, used in the duplicate-id error message.
 * @returns The same entries array when every id appears only once.
 * @throws Error when two or more entries in the array have the same id.
 * @example
 * const tensors = assertUniqueIds([{ id: 'activation' }, { id: 'weights' }], 'manifest.tensors');
 * // tensors is the original array because both ids are unique.
 *
 * expect(() => assertUniqueIds([{ id: 'layer-0' }, { id: 'layer-0' }], 'viewer.tensors')).toThrow(
 *   'viewer.tensors contains duplicate id layer-0.',
 * );
 */
function assertUniqueIds<T extends { id: string }>(entries: T[], label: string): T[] {
    const seen = new Set<string>();
    entries.forEach((entry) => {
        if (seen.has(entry.id)) throw new Error(`${label} contains duplicate id ${entry.id}.`);
        seen.add(entry.id);
    });
    return entries;
}

/**
 * Validates the numeric color tuple attached to coordinate or region color instructions.
 *
 * @param value - Unknown color field that must be an array of two or three numeric channel values.
 * @param label - Dot-path label for the color field, used to identify invalid channels in errors.
 * @returns The validated color channels as finite numbers, preserving the caller-provided channel order.
 * @throws Error when value is not an array, a channel is not finite, or the tuple has any length other than two or three.
 * @example
 * const color = validateColorTuple([0.25, 0.5, 1], 'manifest.tensors[0].colorInstructions[0].color');
 * // color is [0.25, 0.5, 1]
 *
 * expect(() => validateColorTuple([1], 'manifest.tensors[0].colorInstructions[0].color')).toThrow(
 *   'manifest.tensors[0].colorInstructions[0].color must have two or three channels.',
 * );
 */
function validateColorTuple(value: unknown, label: string): number[] {
    const tuple = assertArray(value, label, 3).map((entry, index) => finiteNumber(entry, `${label}[${index}]`));
    if (tuple.length !== 2 && tuple.length !== 3) throw new Error(`${label} must have two or three channels.`);
    return tuple;
}

/**
 * Validates a tensor coordinate and returns one in-bounds integer index for each tensor axis.
 *
 * @param value - Candidate coordinate array whose length must match the tensor rank.
 * @param shape - Tensor dimensions used to check each coordinate entry against its axis bounds.
 * @param label - Prefix used in validation error messages, such as `tensor.markerCoords[0]`.
 * @returns The coordinate entries as finite integers that can be used to index the tensor.
 * @throws Error when `value` is not an array, has the wrong rank, contains a non-integer entry, or contains an index below `0` or greater than or equal to the corresponding axis size.
 * @example
 * validateCoord([1, 2], [3, 4], 'markerCoords[0]');
 * // => [1, 2]
 *
 * @example
 * validateCoord([3, 0], [3, 4], 'markerCoords[0]');
 * // throws Error: markerCoords[0][0] is out of bounds.
 */
function validateCoord(value: unknown, shape: readonly number[], label: string): number[] {
    const coord = assertArray(value, label, shape.length).map((entry, axis) => {
        const index = finiteInteger(entry, `${label}[${axis}]`);
        if (index < 0 || index >= shape[axis]!) throw new Error(`${label}[${axis}] is out of bounds.`);
        return index;
    });
    if (coord.length !== shape.length) throw new Error(`${label} rank does not match tensor rank.`);
    return coord;
}

/**
 * Validates the per-axis extent of a tensor color region.
 *
 * @param value - Candidate region shape array containing one positive integer extent per tensor axis.
 * @param tensorShape - Tensor dimensions whose rank the region shape must match.
 * @param label - Prefix used in validation error messages, such as `colorInstructions[0].shape`.
 * @returns The validated region extents as positive integers, preserving axis order.
 * @throws Error when `value` is not an array, has the wrong rank, contains a non-integer entry, or contains an extent less than `1`.
 * @example
 * validateRegionShape([2, 3], [10, 20], 'colorInstructions[0].shape');
 * // => [2, 3]
 *
 * @example
 * validateRegionShape([2, 0], [10, 20], 'colorInstructions[0].shape');
 * // throws Error: colorInstructions[0].shape[1] must be positive.
 */
function validateRegionShape(value: unknown, tensorShape: readonly number[], label: string): number[] {
    const shape = assertArray(value, label, tensorShape.length).map((entry, axis) => {
        const dim = finiteInteger(entry, `${label}[${axis}]`);
        if (dim < 1) throw new Error(`${label}[${axis}] must be positive.`);
        return dim;
    });
    if (shape.length !== tensorShape.length) throw new Error(`${label} rank does not match tensor rank.`);
    return shape;
}

/**
 * Validates one custom tensor-color instruction and counts the tensor cells it affects.
 *
 * @param instruction - Candidate instruction object with `mode` set to `rgb` or `hs` and `kind` set to `dense`, `coords`, or `region`.
 * @param shape - Tensor dimensions used to validate coordinate lists, region bases, and region bounds.
 * @param label - Prefix used in validation error messages, such as `colorInstructions[0]`.
 * @returns The normalized color instruction plus the number of tensor cells represented by that instruction.
 * @throws Error when the instruction is not an object, has an unsupported `mode` or `kind`, provides a dense value count that is not two or three channels per tensor cell, contains invalid coordinates or color tuples, has region jumps with the wrong rank, or defines a region that extends outside the tensor bounds.
 * @example
 * validateColorInstruction(
 *   { mode: 'rgb', kind: 'coords', coords: [[0, 1], [2, 3]], color: [1, 0, 0] },
 *   [3, 4],
 *   'colorInstructions[0]',
 * );
 * // => {
 * //   instruction: { mode: 'rgb', kind: 'coords', coords: [[0, 1], [2, 3]], color: [1, 0, 0] },
 * //   entries: 2,
 * // }
 *
 * @example
 * validateColorInstruction(
 *   { mode: 'rgb', kind: 'dense', values: [1, 0, 0] },
 *   [2, 2],
 *   'colorInstructions[0]',
 * );
 * // throws Error: colorInstructions[0].values must have two or three channels per tensor cell.
 */
function validateColorInstruction(instruction: unknown, shape: readonly number[], label: string): {
    instruction: ColorInstruction;
    entries: number;
} {
    const value = assertObject(instruction, label);
    const mode = value.mode;
    const kind = value.kind;
    if (mode !== 'rgb' && mode !== 'hs') throw new Error(`${label}.mode is invalid.`);
    if (kind === 'dense') {
        const values = assertArray(value.values, `${label}.values`, VIEWER_LIMITS.maxTensorElements * 3)
            .map((entry, index) => finiteNumber(entry, `${label}.values[${index}]`));
        const cells = tensorElementCount(shape);
        if (values.length !== cells * 2 && values.length !== cells * 3) {
            throw new Error(`${label}.values must have two or three channels per tensor cell.`);
        }
        return { instruction: { mode, kind, values }, entries: cells };
    }
    if (kind === 'coords') {
        const color = validateColorTuple(value.color, `${label}.color`);
        const coords = assertArray(value.coords, `${label}.coords`, VIEWER_LIMITS.maxCustomColorEntries)
            .map((coord, index) => validateCoord(coord, shape, `${label}.coords[${index}]`));
        return { instruction: { mode, kind, coords, color }, entries: coords.length };
    }
    if (kind === 'region') {
        const base = validateCoord(value.base, shape, `${label}.base`);
        const regionShape = validateRegionShape(value.shape, shape, `${label}.shape`);
        const jumps = assertArray(value.jumps, `${label}.jumps`, shape.length)
            .map((entry, axis) => finiteInteger(entry, `${label}.jumps[${axis}]`));
        if (jumps.length !== shape.length) throw new Error(`${label}.jumps rank does not match tensor rank.`);
        const entries = tensorElementCount(regionShape, `${label}.shape`);
        regionShape.forEach((dim, axis) => {
            const last = base[axis]! + (dim - 1) * jumps[axis]!;
            if (last < 0 || last >= shape[axis]!) throw new Error(`${label}.shape exceeds tensor bounds.`);
        });
        return {
            instruction: {
                mode,
                kind,
                base,
                shape: regionShape,
                jumps,
                color: validateColorTuple(value.color, `${label}.color`),
            },
            entries,
        };
    }
    throw new Error(`${label}.kind is invalid.`);
}

/**
 * Validates an optional list of custom color instructions for a tensor.
 *
 * @param value - Optional candidate color-instruction array from a tensor manifest or viewer API call; `undefined` means no custom colors were supplied.
 * @param shape - Tensor dimensions used to validate each instruction's coordinates, dense channel count, and region bounds.
 * @param label - Prefix used in validation error messages; defaults to `colorInstructions`.
 * @returns `undefined` when `value` is `undefined`, otherwise the normalized color instructions in their original order.
 * @throws Error when `value` is not a valid instruction array, any nested instruction is invalid, or the combined coordinate and region instructions touch more than `VIEWER_LIMITS.maxCustomColorEntries` tensor cells.
 * @example
 * validateColorInstructions(undefined, [2, 2]);
 * // => undefined
 *
 * @example
 * validateColorInstructions(
 *   [{ mode: 'rgb', kind: 'coords', coords: [[0, 0]], color: [0, 1, 0] }],
 *   [2, 2],
 * );
 * // => [{ mode: 'rgb', kind: 'coords', coords: [[0, 0]], color: [0, 1, 0] }]
 *
 * @example
 * validateColorInstructions(
 *   [{ mode: 'rgb', kind: 'region', base: [1, 1], shape: [2, 1], jumps: [1, 1], color: [1, 1, 0] }],
 *   [2, 2],
 * );
 * // throws Error: colorInstructions[0].shape exceeds tensor bounds.
 */
export function validateColorInstructions(value: unknown, shape: readonly number[], label = 'colorInstructions'): ColorInstruction[] | undefined {
    if (value === undefined) return undefined;
    const instructions = assertArray(value, label, VIEWER_LIMITS.maxColorInstructions);
    let totalEntries = 0;
    return instructions.map((instruction, index) => {
        const validated = validateColorInstruction(instruction, shape, `${label}[${index}]`);
        totalEntries += validated.entries;
        if (totalEntries > VIEWER_LIMITS.maxCustomColorEntries) {
            throw new Error(`${label} touches too many tensor cells.`);
        }
        return validated.instruction;
    });
}

/**
 * Validates a serialized version-2 tensor-view editor snapshot and returns the canonical editor state used by view parsing.
 *
 * @param value - Unknown snapshot object with version, view input strings, base dimension descriptors, permutation ids, slice keys, singleton axes, and slice value entries.
 * @param label - Error-path prefix that identifies the editor field being validated, such as `manifest.tensors[0].view.editor`.
 * @returns A `TensorViewEditor` with bounded strings, finite integer dimensions and slice values, boolean flatten separators, and `version` fixed to `2`.
 * @throws Error when `value` is not an object, `version` is not `2`, a dimension size is outside the viewer limits, an editor string is too long, `sliceValues` has too many entries, or any nested field has the wrong type.
 * @example
 * const editor = normalizeTensorViewEditor({
 *   version: 2,
 *   viewTensorInput: 'height width',
 *   finalViewInput: undefined,
 *   baseDims: [{ id: 'height', label: 'Height', size: 4 }],
 *   permutedDimIds: ['height'],
 *   flattenSeparators: [false],
 *   singletons: [],
 *   slicedTokenKeys: ['depth'],
 *   sliceValues: { depth: 0 },
 * });
 *
 * expect(editor).toEqual({
 *   version: 2,
 *   viewTensorInput: 'height width',
 *   finalViewInput: undefined,
 *   baseDims: [{ id: 'height', label: 'Height', size: 4 }],
 *   permutedDimIds: ['height'],
 *   flattenSeparators: [false],
 *   singletons: [],
 *   slicedTokenKeys: ['depth'],
 *   sliceValues: { depth: 0 },
 * });
 *
 * @example
 * expect(() => normalizeTensorViewEditor({ version: 1 }, 'view.editor'))
 *   .toThrow('view.editor.version is unsupported.');
 */
export function normalizeTensorViewEditor(value: unknown, label = 'view.editor'): TensorViewEditor {
    const editor = assertObject(value, label);
    if (editor.version !== 2) throw new Error(`${label}.version is unsupported.`);
    const viewTensorInput = boundedString(editor.viewTensorInput, `${label}.viewTensorInput`, VIEWER_LIMITS.maxEditorInputLength);
    const finalViewInput = editor.finalViewInput === undefined
        ? undefined
        : boundedString(editor.finalViewInput, `${label}.finalViewInput`, VIEWER_LIMITS.maxEditorInputLength);
    const baseDims = assertArray(editor.baseDims, `${label}.baseDims`).map((dim, index) => {
        const entry = assertObject(dim, `${label}.baseDims[${index}]`);
        const size = finiteInteger(entry.size, `${label}.baseDims[${index}].size`);
        if (size < 1 || size > VIEWER_LIMITS.maxDimension) {
            throw new Error(`${label}.baseDims[${index}].size is out of range.`);
        }
        return {
            id: boundedString(entry.id, `${label}.baseDims[${index}].id`),
            label: boundedString(entry.label, `${label}.baseDims[${index}].label`),
            size,
        };
    });
    const permutedDimIds = assertArray(editor.permutedDimIds, `${label}.permutedDimIds`)
        .map((entry, index) => boundedString(entry, `${label}.permutedDimIds[${index}]`));
    const flattenSeparators = assertArray(editor.flattenSeparators, `${label}.flattenSeparators`)
        .map((entry) => Boolean(entry));
    const singletons = assertArray(editor.singletons, `${label}.singletons`).map((singleton, index) => {
        const entry = assertObject(singleton, `${label}.singletons[${index}]`);
        return {
            id: boundedString(entry.id, `${label}.singletons[${index}].id`),
            position: finiteInteger(entry.position, `${label}.singletons[${index}].position`),
        };
    });
    const slicedTokenKeys = assertArray(editor.slicedTokenKeys, `${label}.slicedTokenKeys`)
        .map((entry, index) => boundedString(entry, `${label}.slicedTokenKeys[${index}]`));
    const sliceValues = assertObject(editor.sliceValues, `${label}.sliceValues`);
    if (Object.keys(sliceValues).length > VIEWER_LIMITS.maxEditorEntries) {
        throw new Error(`${label}.sliceValues has too many entries.`);
    }
    return {
        version: 2,
        viewTensorInput,
        finalViewInput,
        baseDims,
        permutedDimIds,
        flattenSeparators,
        singletons,
        slicedTokenKeys,
        sliceValues: Object.fromEntries(Object.entries(sliceValues).map(([key, entry]) => [
            boundedString(key, `${label}.sliceValues key`),
            finiteInteger(entry, `${label}.sliceValues[${key}]`),
        ])),
    };
}

/**
 * Validates persisted tensor-view state against the tensor shape that the view will index into.
 *
 * @param value - Unknown snapshot object containing an `editor` snapshot and one hidden index for each tensor axis.
 * @param shape - Tensor axis lengths used to require `hiddenIndices.length === shape.length` and to bound each hidden index to its axis.
 * @param label - Error-path prefix that identifies the snapshot location, such as `manifest.tensors[0].view`.
 * @returns A normalized view snapshot with a canonical editor and finite hidden indices that are valid for the supplied shape.
 * @throws Error when the snapshot is not an object, hidden indices are missing or have the wrong length, a hidden index is not an integer or is outside its axis bounds, or the nested editor snapshot is invalid.
 * @example
 * const snapshot = validateTensorViewSnapshot({
 *   editor: {
 *     version: 2,
 *     viewTensorInput: 'row col',
 *     finalViewInput: undefined,
 *     baseDims: [{ id: 'row', label: 'Row', size: 2 }, { id: 'col', label: 'Column', size: 3 }],
 *     permutedDimIds: ['row', 'col'],
 *     flattenSeparators: [false, false],
 *     singletons: [],
 *     slicedTokenKeys: [],
 *     sliceValues: {},
 *   },
 *   hiddenIndices: [0, 2],
 * }, [2, 3], 'tensor.view');
 *
 * expect(snapshot.hiddenIndices).toEqual([0, 2]);
 * expect(snapshot.editor.version).toBe(2);
 *
 * @example
 * expect(() => validateTensorViewSnapshot({ editor: validEditor, hiddenIndices: [2] }, [2], 'tensor.view'))
 *   .toThrow('tensor.view.hiddenIndices[0] is out of bounds.');
 */
function validateTensorViewSnapshot(value: unknown, shape: readonly number[], label: string) {
    const snapshot = assertObject(value, label);
    const hiddenIndices = assertArray(snapshot.hiddenIndices, `${label}.hiddenIndices`, shape.length)
        .map((entry, axis) => {
            const index = finiteInteger(entry, `${label}.hiddenIndices[${axis}]`);
            if (index < 0 || index >= shape[axis]!) throw new Error(`${label}.hiddenIndices[${axis}] is out of bounds.`);
            return index;
        });
    return {
        editor: normalizeTensorViewEditor(snapshot.editor, `${label}.editor`),
        hiddenIndices,
    };
}

/**
 * Validates one tensor entry from a bundle manifest before the viewer loads its bytes, view state, colors, and markers.
 *
 * @param value - Unknown manifest tensor object with id, name, dtype, shape, little-endian byte order, view snapshot, color instructions, and optional data-file, offset, axis-label, placeholder, and marker fields.
 * @param label - Error-path prefix for this tensor entry, such as `manifest.tensors[0]`.
 * @returns A manifest tensor record with validated dtype, shape, axis labels, byte order, view snapshot, color instructions, and marker coordinates safe for viewer-core consumers.
 * @throws Error when the tensor is not an object, `dtype` is unsupported, `byteOrder` is not `little`, the shape or axis labels are invalid, or nested view, color instruction, offset, data-file, or marker-coordinate validation fails.
 * @example
 * const tensor = validateTensorManifest({
 *   id: 'activation',
 *   name: 'Activation',
 *   dtype: 'float32',
 *   shape: [2, 3],
 *   axisLabels: ['row', 'column'],
 *   byteOrder: 'little',
 *   view: validViewSnapshot,
 *   colorInstructions: [],
 * }, 'manifest.tensors[0]');
 *
 * expect(tensor).toMatchObject({
 *   id: 'activation',
 *   name: 'Activation',
 *   dtype: 'float32',
 *   shape: [2, 3],
 *   byteOrder: 'little',
 * });
 *
 * @example
 * expect(() => validateTensorManifest({
 *   id: 'activation',
 *   name: 'Activation',
 *   dtype: 'float32',
 *   shape: [2, 3],
 *   byteOrder: 'big',
 *   view: validViewSnapshot,
 *   colorInstructions: [],
 * }, 'manifest.tensors[0]')).toThrow('manifest.tensors[0].byteOrder must be little.');
 */
function validateTensorManifest(value: unknown, label: string): BundleManifest['tensors'][number] {
    const tensor = assertObject(value, label);
    const dtype = tensor.dtype;
    if (!isDType(dtype)) throw new Error(`${label}.dtype is unsupported.`);
    const shape = validateTensorShape(tensor.shape, `${label}.shape`);
    if (tensor.byteOrder !== 'little') throw new Error(`${label}.byteOrder must be little.`);
    return {
        id: boundedString(tensor.id, `${label}.id`),
        name: boundedString(tensor.name, `${label}.name`),
        dtype,
        shape,
        axisLabels: tensor.axisLabels === undefined
            ? undefined
            : assertArray(tensor.axisLabels, `${label}.axisLabels`, shape.length)
                .map((entry, index) => boundedString(entry, `${label}.axisLabels[${index}]`)),
        byteOrder: 'little',
        dataFile: tensor.dataFile === undefined ? undefined : boundedString(tensor.dataFile, `${label}.dataFile`, VIEWER_LIMITS.maxTextLength),
        placeholderData: tensor.placeholderData === undefined ? undefined : Boolean(tensor.placeholderData),
        offset: tensor.offset === undefined ? undefined : validateVec3(tensor.offset, `${label}.offset`),
        view: validateTensorViewSnapshot(tensor.view, shape, `${label}.view`),
        colorInstructions: validateColorInstructions(tensor.colorInstructions, shape, `${label}.colorInstructions`),
        markerCoords: tensor.markerCoords === undefined
            ? undefined
            : assertArray(tensor.markerCoords, `${label}.markerCoords`, VIEWER_LIMITS.maxTensorElements)
                .map((coord, index) => validateCoord(coord, shape, `${label}.markerCoords[${index}]`)),
    };
}

/**
 * Validates one viewer snapshot tensor entry and resolves its id against the loaded manifest tensors.
 *
 * @param value - Unknown viewer tensor snapshot object containing a manifest tensor `id`, display `name`, optional 3D offset, and persisted view snapshot.
 * @param tensorById - Map from manifest tensor ids to validated manifest tensor records; the matched tensor supplies the shape used to validate the view snapshot.
 * @param label - Error-path prefix for this viewer tensor entry, such as `viewer.tensors[0]`.
 * @returns A viewer tensor snapshot whose id exists in the manifest map and whose name, optional offset, and shape-dependent view state have been validated.
 * @throws Error when the snapshot is not an object, the id does not match a manifest tensor, the name or offset is invalid, or the nested view snapshot is invalid for the referenced tensor shape.
 * @example
 * const tensorById = new Map([['activation', { id: 'activation', name: 'Activation', dtype: 'float32', shape: [2, 3], byteOrder: 'little', view: validViewSnapshot, colorInstructions: [] }]]);
 * const snapshot = validateViewerTensorSnapshot({
 *   id: 'activation',
 *   name: 'Activation panel',
 *   offset: [1, 0, 0],
 *   view: validViewSnapshot,
 * }, tensorById, 'viewer.tensors[0]');
 *
 * expect(snapshot.id).toBe('activation');
 * expect(snapshot.offset).toEqual([1, 0, 0]);
 *
 * @example
 * expect(() => validateViewerTensorSnapshot({ id: 'missing', name: 'Missing', view: validViewSnapshot }, new Map(), 'viewer.tensors[0]'))
 *   .toThrow('viewer.tensors[0].id does not match a tensor manifest.');
 */
function validateViewerTensorSnapshot(
    value: unknown,
    tensorById: Map<string, BundleManifest['tensors'][number]>,
    label: string,
): ViewerSnapshot['tensors'][number] {
    const entry = assertObject(value, label);
    const id = boundedString(entry.id, `${label}.id`);
    const tensor = tensorById.get(id);
    if (!tensor) throw new Error(`${label}.id does not match a tensor manifest.`);
    return {
        id,
        name: boundedString(entry.name, `${label}.name`),
        offset: entry.offset === undefined ? undefined : validateVec3(entry.offset, `${label}.offset`),
        view: validateTensorViewSnapshot(entry.view, tensor.shape, `${label}.view`),
    };
}

/**
 * Normalizes a serialized viewer snapshot and verifies that its tensor references match the bundle manifest.
 *
 * @param value - Serialized `viewer` object from a bundle or session tab, including display mode, camera, tensor views, and active tensor id.
 * @param manifestTensors - Tensor manifest entries that the snapshot is allowed to reference from `viewer.tensors` and `viewer.activeTensorId`.
 * @returns A `ViewerSnapshot` with booleans and bounded numeric fields coerced into the shape used to restore viewer state.
 * @throws Error when the viewer object is malformed, uses an unsupported display/interation/mapping mode, has out-of-range camera or gap values, contains invalid tensor-view entries, or references an active tensor id that is not present in `manifestTensors`.
 * @example
 * const tensors = [{ id: 'weights', name: 'weights', dtype: 'float32', shape: [2, 2], byteOrder: 'little' }];
 * const snapshot = validateViewerSnapshot({
 *   displayMode: '2d',
 *   heatmap: true,
 *   showDimensionLines: false,
 *   showInspectorPanel: true,
 *   showHoverDetailsPanel: false,
 *   camera: { position: [0, 0, 5], target: [0, 0, 0], rotation: [0, 0, 0], zoom: 1 },
 *   tensors: [],
 *   activeTensorId: 'weights',
 * }, tensors);
 *
 * snapshot.activeTensorId;
 * // => 'weights'
 * snapshot.camera.zoom;
 * // => 1
 *
 * @example
 * expect(() => validateViewerSnapshot({
 *   displayMode: '2d',
 *   heatmap: false,
 *   showDimensionLines: false,
 *   showInspectorPanel: false,
 *   showHoverDetailsPanel: false,
 *   camera: { position: [0, 0, 5], target: [0, 0, 0], rotation: [0, 0, 0], zoom: 1 },
 *   tensors: [],
 *   activeTensorId: 'missing',
 * }, tensors)).toThrow('viewer.activeTensorId does not match a tensor manifest.');
 */
function validateViewerSnapshot(value: unknown, manifestTensors: BundleManifest['tensors']): ViewerSnapshot {
    const snapshot = assertObject(value, 'viewer');
    const displayMode = snapshot.displayMode;
    if (!DISPLAY_MODES.has(String(displayMode))) throw new Error('viewer.displayMode is invalid.');
    const interactionMode = snapshot.interactionMode;
    if (interactionMode !== undefined && !INTERACTION_MODES.has(String(interactionMode))) {
        throw new Error('viewer.interactionMode is invalid.');
    }
    const dimensionMappingScheme = snapshot.dimensionMappingScheme;
    if (dimensionMappingScheme !== undefined && !MAPPING_SCHEMES.has(String(dimensionMappingScheme))) {
        throw new Error('viewer.dimensionMappingScheme is invalid.');
    }
    const camera = assertObject(snapshot.camera, 'viewer.camera');
    const tensorById = new Map(manifestTensors.map((tensor) => [tensor.id, tensor]));
    const viewerTensors = assertUniqueIds(
        assertArray(snapshot.tensors, 'viewer.tensors', VIEWER_LIMITS.maxTensors)
            .map((entry, index) => validateViewerTensorSnapshot(entry, tensorById, `viewer.tensors[${index}]`)),
        'viewer.tensors',
    );
    const activeTensorId = snapshot.activeTensorId === null
        ? null
        : boundedString(snapshot.activeTensorId, 'viewer.activeTensorId');
    if (activeTensorId !== null && !tensorById.has(activeTensorId)) {
        throw new Error('viewer.activeTensorId does not match a tensor manifest.');
    }
    return {
        version: 1,
        displayMode: displayMode as '2d' | '3d',
        interactionMode: interactionMode as ViewerSnapshot['interactionMode'],
        heatmap: Boolean(snapshot.heatmap),
        dimensionBlockGapMultiple: snapshot.dimensionBlockGapMultiple === undefined
            ? undefined
            : boundedFiniteNumber(
                snapshot.dimensionBlockGapMultiple,
                'viewer.dimensionBlockGapMultiple',
                VIEWER_LIMITS.minDimensionBlockGapMultiple,
                VIEWER_LIMITS.maxDimensionBlockGapMultiple,
            ),
        displayGaps: snapshot.displayGaps === undefined ? undefined : Boolean(snapshot.displayGaps),
        logScale: snapshot.logScale === undefined ? undefined : Boolean(snapshot.logScale),
        collapseHiddenAxes: snapshot.collapseHiddenAxes === undefined ? undefined : Boolean(snapshot.collapseHiddenAxes),
        showSlicesInSamePlace: snapshot.showSlicesInSamePlace === undefined ? undefined : Boolean(snapshot.showSlicesInSamePlace),
        dimensionMappingScheme: dimensionMappingScheme as ViewerSnapshot['dimensionMappingScheme'],
        showDimensionLines: Boolean(snapshot.showDimensionLines),
        showTensorNames: snapshot.showTensorNames === undefined ? undefined : Boolean(snapshot.showTensorNames),
        showInspectorPanel: Boolean(snapshot.showInspectorPanel),
        showSelectionPanel: snapshot.showSelectionPanel === undefined ? undefined : Boolean(snapshot.showSelectionPanel),
        showHoverDetailsPanel: Boolean(snapshot.showHoverDetailsPanel),
        camera: {
            position: validateVec3(camera.position, 'viewer.camera.position'),
            target: validateVec3(camera.target, 'viewer.camera.target'),
            rotation: validateVec3(camera.rotation, 'viewer.camera.rotation'),
            zoom: boundedFiniteNumber(
                camera.zoom,
                'viewer.camera.zoom',
                VIEWER_LIMITS.minCameraZoom,
                VIEWER_LIMITS.maxCameraZoom,
            ),
        },
        tensors: viewerTensors,
        activeTensorId,
    };
}

/**
 * Validates the single-bundle manifest loaded by the viewer and normalizes its tensors and saved viewer snapshot.
 *
 * @param value - Candidate bundle manifest object with `version: 1`, a `tensors` array, and a nested `viewer` snapshot.
 * @returns A `BundleManifest` whose tensor metadata, viewer tensor views, camera values, and active tensor reference are safe to load.
 * @throws Error when the manifest is not an object, has an unsupported version, contains invalid or duplicate tensor metadata, exceeds manifest limits, or has an invalid nested viewer snapshot.
 * @example
 * const manifest = validateBundleManifest({
 *   version: 1,
 *   tensors: [{ id: 'weights', name: 'weights', dtype: 'float32', shape: [2, 2], byteOrder: 'little' }],
 *   viewer: {
 *     displayMode: '2d',
 *     heatmap: false,
 *     showDimensionLines: false,
 *     showInspectorPanel: true,
 *     showHoverDetailsPanel: false,
 *     camera: { position: [0, 0, 5], target: [0, 0, 0], rotation: [0, 0, 0], zoom: 1 },
 *     tensors: [],
 *     activeTensorId: 'weights',
 *   },
 * });
 *
 * manifest.version;
 * // => 1
 * manifest.tensors[0].id;
 * // => 'weights'
 *
 * @example
 * expect(() => validateBundleManifest({ version: 2, tensors: [], viewer: {} }))
 *   .toThrow('Unsupported bundle version 2.');
 */
export function validateBundleManifest(value: unknown): BundleManifest {
    const manifest = assertObject(value, 'manifest');
    if (manifest.version !== 1) throw new Error(`Unsupported bundle version ${String(manifest.version)}.`);
    const tensors = assertUniqueIds(
        assertArray(manifest.tensors, 'manifest.tensors', VIEWER_LIMITS.maxTensors)
            .map((tensor, index) => validateTensorManifest(tensor, `manifest.tensors[${index}]`)),
        'manifest.tensors',
    );
    return {
        version: 1,
        viewer: validateViewerSnapshot(manifest.viewer, tensors),
        tensors,
    };
}

/**
 * Validates a saved multi-tab viewer session and normalizes each tab's tensor manifests and viewer snapshot.
 *
 * @param value - Candidate session object with `version: 1` and a `tabs` array containing tab ids, titles, tensor manifests, and nested viewer snapshots.
 * @returns A `SessionBundleManifest` whose tab ids, titles, tensors, and per-tab viewer states are safe to restore.
 * @throws Error when the session is not an object, has an unsupported version, has malformed or duplicate tabs, contains invalid or duplicate tensor metadata within a tab, exceeds configured limits, or has an invalid nested viewer snapshot.
 * @example
 * const session = validateSessionBundleManifest({
 *   version: 1,
 *   tabs: [{
 *     id: 'tab-main',
 *     title: 'Weights',
 *     tensors: [{ id: 'weights', name: 'weights', dtype: 'float32', shape: [2, 2], byteOrder: 'little' }],
 *     viewer: {
 *       displayMode: '2d',
 *       heatmap: true,
 *       showDimensionLines: false,
 *       showInspectorPanel: true,
 *       showHoverDetailsPanel: false,
 *       camera: { position: [0, 0, 5], target: [0, 0, 0], rotation: [0, 0, 0], zoom: 1 },
 *       tensors: [],
 *       activeTensorId: 'weights',
 *     },
 *   }],
 * });
 *
 * session.tabs[0].title;
 * // => 'Weights'
 * session.tabs[0].viewer.activeTensorId;
 * // => 'weights'
 *
 * @example
 * expect(() => validateSessionBundleManifest({ version: 2, tabs: [] }))
 *   .toThrow('Unsupported session version 2.');
 */
export function validateSessionBundleManifest(value: unknown): SessionBundleManifest {
    const manifest = assertObject(value, 'session');
    if (manifest.version !== 1) throw new Error(`Unsupported session version ${String(manifest.version)}.`);
    const tabs = assertUniqueIds(
        assertArray(manifest.tabs, 'session.tabs', VIEWER_LIMITS.maxTabs).map((tab, index) => {
            const entry = assertObject(tab, `session.tabs[${index}]`);
            const tensors = assertUniqueIds(
                assertArray(entry.tensors, `session.tabs[${index}].tensors`, VIEWER_LIMITS.maxTensors)
                    .map((tensor, tensorIndex) => validateTensorManifest(tensor, `session.tabs[${index}].tensors[${tensorIndex}]`)),
                `session.tabs[${index}].tensors`,
            );
            return {
                id: boundedString(entry.id, `session.tabs[${index}].id`),
                title: boundedString(entry.title, `session.tabs[${index}].title`),
                viewer: validateViewerSnapshot(entry.viewer, tensors),
                tensors,
            };
        }),
        'session.tabs',
    );
    return {
        version: 1,
        tabs,
    };
}
