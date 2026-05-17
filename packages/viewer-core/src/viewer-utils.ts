import { Color, Vector3 } from 'three';
import type { CustomColor, DType, HueSaturation, NumericArray, RGB, Vec3 } from './types.js';

/**
 * return signed log1p for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @returns Numeric result computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * signedLog1p(value);
 */
export function signedLog1p(value: number): number {
    return Math.sign(value) * Math.log1p(Math.abs(value));
}

/**
 * return compute min max for the current viewer state.
 *
 * @param data - data input used by this operation (NumericArray).
 * @returns Object containing computed state for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * computeMinMax(data);
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
 * return vector from tuple for the current viewer state.
 *
 * @param tuple - tuple input used by this operation (Vec3).
 * @returns Computed Vector3 value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * vectorFromTuple(tuple);
 */
export function vectorFromTuple(tuple: Vec3): Vector3 {
    return new Vector3(tuple[0], tuple[1], tuple[2]);
}

/**
 * return tuple from vector for the current viewer state.
 *
 * @param vector - vector input used by this operation (Vector3).
 * @returns Computed Vec3 value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * tupleFromVector(vector);
 */
export function tupleFromVector(vector: Vector3): Vec3 {
    return [vector.x, vector.y, vector.z];
}

/**
 * return dtype from array for the current viewer state.
 *
 * @param data - data input used by this operation (NumericArray).
 * @returns Computed DType value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * dtypeFromArray(data);
 */
export function dtypeFromArray(data: NumericArray): DType {
    if (data instanceof Float32Array) return 'float32';
    if (data instanceof Int32Array) return 'int32';
    if (data instanceof Uint8Array) return 'uint8';
    return 'float64';
}

/**
 * return numeric value for the current viewer state.
 *
 * @param data - data input used by this operation (NumericArray | null).
 * @param index - Index used by this operation.
 * @returns Numeric result computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * numericValue(data, index);
 */
export function numericValue(data: NumericArray | null, index: number): number {
    return Number(data?.[index] ?? 0);
}

/**
 * return coord key for the current viewer state.
 *
 * @param coord - Coordinate used by this operation.
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * coordKey(coord);
 */
export function coordKey(coord: number[]): string {
    return coord.join(',');
}

/**
 * return coord from key for the current viewer state.
 *
 * @param key - key input used by this operation (string).
 * @returns Array of computed entries for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * coordFromKey(key);
 */
export function coordFromKey(key: string): number[] {
    return key === '' ? [] : key.split(',').map((value) => Number(value));
}

/**
 * return quantile for the current viewer state.
 *
 * @param sortedValues - sorted values input used by this operation (number[]).
 * @param percentile - percentile input used by this operation (number).
 * @returns Numeric result computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * quantile(sortedValues, percentile);
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
 * return boxes intersect for the current viewer state.
 *
 * @param left - left input used by this operation ({ left: number; right: number; top: number; bottom: number }).
 * @param right - right input used by this operation ({ left: number; right: number; top: number; bottom: number }).
 * @returns Whether the requested condition holds.
 * @noThrows This function has no direct throw path.
 * @example
 * boxesIntersect(left, right);
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
 * color from rgb for the current viewer state.
 *
 * @param rgb - rgb input used by this operation (RGB).
 * @returns Computed Color value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * colorFromRgb(rgb);
 */
export function colorFromRgb(rgb: RGB): Color {
    return new Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
}

/**
 * normalize hue for the current viewer state.
 *
 * @param hue - hue input used by this operation (number).
 * @returns Numeric result computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizeHue(hue);
 */
function normalizeHue(hue: number): number {
    const unit = Math.abs(hue) > 1 ? hue / 360 : hue;
    return ((unit % 1) + 1) % 1;
}

/**
 * normalize saturation for the current viewer state.
 *
 * @param saturation - saturation input used by this operation (number).
 * @returns Numeric result computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizeSaturation(saturation);
 */
function normalizeSaturation(saturation: number): number {
    const unit = Math.abs(saturation) > 1 ? saturation / 100 : saturation;
    return Math.max(0, Math.min(1, unit));
}

/**
 * color from hue saturation for the current viewer state.
 *
 * @param color - color input used by this operation (HueSaturation).
 * @param brightness - brightness input used by this operation (number).
 * @returns Computed Color value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * colorFromHueSaturation(color, brightness);
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
 * parse custom color for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @returns Computed CustomColor value for the caller.
 * @throws Error when the requested input or state is invalid.
 * @example
 * parseCustomColor(value);
 */
export function parseCustomColor(value: number[]): CustomColor {
    if (value.length === 2) return { kind: 'hs', value: [value[0], value[1]] };
    if (value.length === 3) return { kind: 'rgb', value: [value[0], value[1], value[2]] };
    throw new Error(`Expected color tuple of length 2 or 3, received ${value.length}.`);
}
