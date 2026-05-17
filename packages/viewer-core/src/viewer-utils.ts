import { Color, Vector3 } from 'three';
import type { CustomColor, DType, HueSaturation, NumericArray, RGB, Vec3 } from './types.js';

/**
 * Applies the viewer's sign-preserving logarithmic scale transform to a numeric value.
 *
 * Positive values become `log1p(value)`, negative values become `-log1p(abs(value))`, and zero remains zero.
 *
 * @param value - Tensor cell value or heatmap bound to transform for log-scale color normalization.
 * @returns Sign-preserving `log1p` magnitude used to compare positive and negative values on the same heatmap scale.
 * @noThrows `Math.sign`, `Math.abs`, and `Math.log1p` do not throw for JavaScript number inputs; non-finite inputs propagate as normal numeric results.
 * @example
 * expect(signedLog1p(3)).toBeCloseTo(Math.log1p(3));
 * expect(signedLog1p(-3)).toBeCloseTo(-Math.log1p(3));
 * expect(signedLog1p(0)).toBe(0);
 */
export function signedLog1p(value: number): number {
    return Math.sign(value) * Math.log1p(Math.abs(value));
}

/**
 * Scans tensor sample storage to derive the numeric value range used for viewer scaling and heatmap metadata.
 *
 * @param data - NumericArray containing tensor values; each element is coerced with `Number`, and nullish reads are treated as `0`.
 * @returns The smallest and largest finite values found in `data`; returns `{ min: 0, max: 1 }` when the scan cannot produce a finite range, such as for an empty array.
 * @noThrows The function only reads `data.length`, indexes the array, and performs numeric comparisons; it has no validation branch or explicit error path for supported NumericArray inputs.
 * @example
 * const range = computeMinMax(new Float32Array([3.5, -2, 8]));
 * // range is { min: -2, max: 8 }
 *
 * @example
 * const fallback = computeMinMax(new Float32Array([]));
 * // fallback is { min: 0, max: 1 }
 */
export function computeMinMax(data: NumericArray): { min: number; max: number } {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < data.length; index += 1) {
        const value = Number(data[index] ?? 0);
        if (value < min) min = value;
        if (value > max) max = value;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
    return { min, max };
}

/**
 * Converts a serialized viewer coordinate tuple into the Three.js vector type used by camera, mesh, and layout math.
 *
 * @param tuple - Three-number `[x, y, z]` Vec3 from viewer state, tensor offsets, or camera snapshots.
 * @returns A new `Vector3` whose `x`, `y`, and `z` components match the tuple entries in order.
 * @noThrows The function only passes the three tuple entries to the `Vector3` constructor and performs no parsing or validation.
 * @example
 * const position = vectorFromTuple([12, -4, 0.5]);
 * // position.x === 12
 * // position.y === -4
 * // position.z === 0.5
 */
export function vectorFromTuple(tuple: Vec3): Vector3 {
    return new Vector3(tuple[0], tuple[1], tuple[2]);
}

/**
 * Serializes a Three.js vector into the viewer's plain Vec3 tuple format for snapshots and persisted state.
 *
 * @param vector - Three.js `Vector3` whose `x`, `y`, and `z` components should be captured.
 * @returns A `[x, y, z]` tuple containing the vector components in coordinate order.
 * @noThrows The function only reads the numeric `x`, `y`, and `z` fields from an existing `Vector3` instance.
 * @example
 * const tuple = tupleFromVector(new Vector3(1.25, 2, -3));
 * // tuple is [1.25, 2, -3]
 */
export function tupleFromVector(vector: Vector3): Vec3 {
    return [vector.x, vector.y, vector.z];
}

/**
 * Infers the viewer dtype label that corresponds to a tensor's JavaScript numeric array storage.
 *
 * @param data - NumericArray instance supplied for tensor values, such as `Float32Array`, `Int32Array`, `Uint8Array`, or a float64-compatible fallback array.
 * @returns The manifest dtype label for the array constructor: `'float32'`, `'int32'`, `'uint8'`, or `'float64'` for any other supported numeric array.
 * @noThrows The function uses only `instanceof` checks and a default return value, so unrecognized NumericArray variants fall back to `'float64'` instead of throwing.
 * @example
 * const dtype = dtypeFromArray(new Uint8Array([0, 128, 255]));
 * // dtype is 'uint8'
 *
 * @example
 * const fallback = dtypeFromArray(new Float64Array([0.1, 0.2]));
 * // fallback is 'float64'
 */
export function dtypeFromArray(data: NumericArray): DType {
    if (data instanceof Float32Array) return 'float32';
    if (data instanceof Int32Array) return 'int32';
    if (data instanceof Uint8Array) return 'uint8';
    return 'float64';
}

/**
 * Reads a tensor data entry as a JavaScript number, using `0` when the tensor has no data array or the offset is missing.
 *
 * @param data - Numeric tensor backing storage, or `null` for tensors without loaded cell values.
 * @param index - Linear offset into the tensor data array, usually produced from a tensor coordinate and shape.
 * @returns The cell value coerced with `Number(...)`; returns `0` for `null` data or an out-of-range/undefined slot.
 * @noThrows Optional indexing and `Number` coercion handle `null` and missing entries without an expected exception path.
 * @example
 * numericValue(new Float32Array([1.5, 2.25, 3]), 1);
 * // => 2.25
 *
 * numericValue(null, 4);
 * // => 0
 */
export function numericValue(data: NumericArray | null, index: number): number {
    return Number(data?.[index] ?? 0);
}

/**
 * Serializes a tensor coordinate vector into the comma-delimited key used by viewer Sets and Maps.
 *
 * @param coord - Ordered tensor coordinate components, such as `[row, column]` or an empty scalar coordinate `[]`.
 * @returns A comma-separated coordinate key; for example, `[2, 5, 1]` becomes `"2,5,1"` and `[]` becomes `""`.
 * @noThrows The helper only delegates to `Array.prototype.join` on the provided coordinate array and performs no validation.
 * @example
 * coordKey([3, 0, 2]);
 * // => "3,0,2"
 *
 * coordKey([]);
 * // => ""
 */
export function coordKey(coord: number[]): string {
    return coord.join(',');
}

/**
 * Parses a serialized viewer coordinate key back into numeric tensor coordinate components.
 *
 * @param key - Comma-delimited coordinate key produced by `coordKey`, or `""` for a scalar/empty coordinate.
 * @returns The numeric coordinate vector represented by the key; each comma-delimited segment is converted with `Number`.
 * @noThrows String splitting and `Number` conversion do not throw for normal key strings; malformed numeric text becomes `NaN` rather than an exception.
 * @example
 * coordFromKey("3,0,2");
 * // => [3, 0, 2]
 *
 * coordFromKey("");
 * // => []
 */
export function coordFromKey(key: string): number[] {
    return key === '' ? [] : key.split(',').map((value) => Number(value));
}

/**
 * Calculates an interpolated percentile value from an already sorted numeric sample list.
 *
 * @param sortedValues - Values sorted in ascending order before calling; interpolation assumes this order is already correct.
 * @param percentile - Fractional percentile position, typically from `0` for the minimum through `1` for the maximum.
 * @returns The value at the requested percentile, linearly interpolated between adjacent sorted entries when the position is fractional.
 * @noThrows The helper performs arithmetic and guarded array indexing with `0` fallbacks, and it does not validate empty arrays or percentile bounds.
 * @example
 * quantile([10, 20, 30, 40], 0.5);
 * // => 25
 *
 * quantile([7], 0.95);
 * // => 7
 */
export function quantile(sortedValues: number[], percentile: number): number {
    if (sortedValues.length === 1) return sortedValues[0] ?? 0;
    const position = (sortedValues.length - 1) * percentile;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    const lowerValue = sortedValues[lower] ?? 0;
    const upperValue = sortedValues[upper] ?? lowerValue;
    if (lower === upper) return lowerValue;
    return lowerValue + (upperValue - lowerValue) * (position - lower);
}

/**
 * Tests whether two axis-aligned rectangles overlap or touch in viewer coordinate space.
 *
 * @param left - First rectangle bounds, with horizontal edges in `left`/`right` and vertical edges in `top`/`bottom`.
 * @param right - Second rectangle bounds to compare against the first rectangle.
 * @returns `true` when the rectangles share any area or boundary edge; `false` when one is completely outside the other.
 * @noThrows Performs only numeric comparisons on the supplied bounds object properties, so valid bounds objects have no expected throw path.
 * @example
 * const marquee = { left: 10, right: 30, top: 10, bottom: 30 };
 * const cell = { left: 25, right: 40, top: 20, bottom: 35 };
 * boxesIntersect(marquee, cell); // true
 *
 * const offscreenCell = { left: 31, right: 40, top: 20, bottom: 35 };
 * boxesIntersect(marquee, offscreenCell); // false
 */
export function boxesIntersect(
    left: { left: number; right: number; top: number; bottom: number },
    right: { left: number; right: number; top: number; bottom: number },
): boolean {
    return left.left <= right.right
        && left.right >= right.left
        && left.top <= right.bottom
        && left.bottom >= right.top;
}

/**
 * Converts a custom RGB color tuple from 8-bit channel values into a Three.js `Color`.
 *
 * @param rgb - Three-element `[red, green, blue]` tuple whose channel values use the 0-255 RGB range stored in viewer metadata.
 * @returns A `Color` with each channel normalized to Three.js's 0-1 component range for canvas, SVG, and mesh rendering.
 * @noThrows Reads three numeric tuple entries and constructs a `Color`; no validation or branching introduces an expected throw path for a well-formed RGB tuple.
 * @example
 * const color = colorFromRgb([255, 128, 0]);
 * color.r; // 1
 * color.g; // 0.5019607843137255
 * color.b; // 0
 */
export function colorFromRgb(rgb: RGB): Color {
    return new Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
}

/**
 * Converts a hue expressed as turns or degrees into the wrapped unit interval used by hue-saturation color rendering.
 *
 * @param hue - Hue value from a custom hue-saturation color; values with absolute magnitude greater than 1 are treated as degrees, otherwise as turns.
 * @returns The hue wrapped into the range `[0, 1)`, where `0` and `1` represent the same point on the color wheel.
 * @noThrows Uses arithmetic modulo and absolute value on the numeric argument, so there is no expected throw path for numeric hue input.
 * @example
 * normalizeHue(180); // 0.5
 * normalizeHue(-0.25); // 0.75
 * normalizeHue(450); // 0.25
 */
function normalizeHue(hue: number): number {
    const unit = Math.abs(hue) > 1 ? hue / 360 : hue;
    return ((unit % 1) + 1) % 1;
}

/**
 * Converts a saturation expressed as a unit value or percentage into the clamped unit range used for color rendering.
 *
 * @param saturation - Saturation component from a custom hue-saturation color; values with absolute magnitude greater than 1 are treated as percentages.
 * @returns The saturation clamped to `[0, 1]`, suitable for interpolating between grayscale and fully saturated color.
 * @noThrows Uses numeric division and min/max clamping only, so numeric saturation input has no expected throw path.
 * @example
 * normalizeSaturation(75); // 0.75
 * normalizeSaturation(1.5); // 0.015
 * normalizeSaturation(-0.2); // 0
 */
function normalizeSaturation(saturation: number): number {
    const unit = Math.abs(saturation) > 1 ? saturation / 100 : saturation;
    return Math.max(0, Math.min(1, unit));
}

/**
 * Converts a hue/saturation custom color and scalar brightness into the RGB color used for tensor cell rendering.
 *
 * @param color - Two-number hue/saturation tuple where the first entry is the hue position around the color wheel and the second entry is saturation; both entries are normalized before conversion.
 * @param brightness - Tensor intensity for the HSV value channel; values below 0 render black and values above 1 render at full brightness.
 * @returns Three.js `Color` whose `r`, `g`, and `b` channels are normalized RGB components for the rendered cell.
 * @noThrows The conversion only normalizes numbers, clamps brightness, selects an HSV sector, and constructs a `Color`; it performs no validation or external I/O.
 * @example
 * const color = colorFromHueSaturation([0, 1], 0.5);
 *
 * console.assert(color.r === 0.5);
 * console.assert(color.g === 0);
 * console.assert(color.b === 0);
 */
export function colorFromHueSaturation(color: HueSaturation, brightness: number): Color {
    const hue = normalizeHue(color[0]);
    const saturation = normalizeSaturation(color[1]);
    const value = Math.max(0, Math.min(1, brightness));
    const sector = hue * 6;
    const index = Math.floor(sector);
    const fraction = sector - index;
    const p = value * (1 - saturation);
    const q = value * (1 - saturation * fraction);
    const t = value * (1 - saturation * (1 - fraction));
    const components = [
        [value, t, p],
        [q, value, p],
        [p, value, t],
        [p, q, value],
        [t, p, value],
        [value, p, q],
    ][index % 6] ?? [value, value, value];
    return new Color(components[0], components[1], components[2]);
}

/**
 * Classifies a custom color tuple as hue/saturation or RGB metadata for tensor cell overrides.
 *
 * @param value - Numeric color tuple from custom-color tensor metadata: `[hue, saturation]` for hue/saturation colors or `[red, green, blue]` for RGB colors.
 * @returns Discriminated custom color object that downstream rendering uses to choose `colorFromHueSaturation` or `colorFromRgb`.
 * @throws Error when `value` has any length other than 2 or 3.
 * @example
 * parseCustomColor([0.25, 0.8]);
 * // => { kind: 'hs', value: [0.25, 0.8] }
 *
 * parseCustomColor([255, 128, 0]);
 * // => { kind: 'rgb', value: [255, 128, 0] }
 *
 * try {
 *     parseCustomColor([1]);
 * } catch (error) {
 *     console.assert(error instanceof Error);
 *     console.assert(error.message === 'Expected color tuple of length 2 or 3, received 1.');
 * }
 */
export function parseCustomColor(value: number[]): CustomColor {
    if (value.length === 2) return { kind: 'hs', value: [value[0], value[1]] };
    if (value.length === 3) return { kind: 'rgb', value: [value[0], value[1], value[2]] };
    throw new Error(`Expected color tuple of length 2 or 3, received ${value.length}.`);
}
