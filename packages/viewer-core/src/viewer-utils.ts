import { Color, Vector3 } from 'three';
import type { CustomColor, DType, HueSaturation, NumericArray, RGB, Vec3 } from './types.js';

export function signedLog1p(value: number): number {
    return Math.sign(value) * Math.log1p(Math.abs(value));
}

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

export function vectorFromTuple(tuple: Vec3): Vector3 {
    return new Vector3(tuple[0], tuple[1], tuple[2]);
}

export function tupleFromVector(vector: Vector3): Vec3 {
    return [vector.x, vector.y, vector.z];
}

export function dtypeFromArray(data: NumericArray): DType {
    if (data instanceof Float32Array) return 'float32';
    if (data instanceof Int32Array) return 'int32';
    if (data instanceof Uint8Array) return 'uint8';
    return 'float64';
}

export function numericValue(data: NumericArray | null, index: number): number {
    return Number(data?.[index] ?? 0);
}

export function coordKey(coord: number[]): string {
    return coord.join(',');
}

export function coordFromKey(key: string): number[] {
    return key === '' ? [] : key.split(',').map((value) => Number(value));
}

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

export function boxesIntersect(
    left: { left: number; right: number; top: number; bottom: number },
    right: { left: number; right: number; top: number; bottom: number },
): boolean {
    return left.left <= right.right
        && left.right >= right.left
        && left.top <= right.bottom
        && left.bottom >= right.top;
}

export function colorFromRgb(rgb: RGB): Color {
    return new Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255);
}

function normalizeHue(hue: number): number {
    const unit = Math.abs(hue) > 1 ? hue / 360 : hue;
    return ((unit % 1) + 1) % 1;
}

function normalizeSaturation(saturation: number): number {
    const unit = Math.abs(saturation) > 1 ? saturation / 100 : saturation;
    return Math.max(0, Math.min(1, unit));
}

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

export function parseCustomColor(value: number[]): CustomColor {
    if (value.length === 2) return { kind: 'hs', value: [value[0], value[1]] };
    if (value.length === 3) return { kind: 'rgb', value: [value[0], value[1], value[2]] };
    throw new Error(`Expected color tuple of length 2 or 3, received ${value.length}.`);
}
