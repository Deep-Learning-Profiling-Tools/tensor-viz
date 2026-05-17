import { axisWorldKeyForMode, type DimensionMappingScheme, type ViewerSnapshot } from '@tensor-viz/viewer-core';

/**
 * format range value for the current viewer state.
 */
export function formatRangeValue(value: number): string {
    return Number.isInteger(value) ? String(value) : value.toPrecision(6);
}

/**
 * return selection enabled for the current viewer state.
 */
export function selectionEnabled(snapshot: ViewerSnapshot): boolean {
    return snapshot.displayMode === '2d' && (snapshot.dimensionMappingScheme ?? 'z-order') === 'contiguous';
}

/**
 * escape html for the current viewer state.
 */
export function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

export const escapeInfo = escapeHtml;

/**
 * return info button for the current viewer state.
 */
export function infoButton(text: string): string {
    const escaped = escapeHtml(text);
    return `<button class="info-button" type="button" tabindex="-1" aria-label="${escaped}" data-info="${escaped}">i</button>`;
}

/**
 * return title with info for the current viewer state.
 */
export function titleWithInfo(title: string, info: string): string {
    return `<div class="title-row"><h2>${title}</h2>${infoButton(info)}</div>`;
}

/**
 * return label with info for the current viewer state.
 */
export function labelWithInfo(label: string, info: string, htmlFor?: string): string {
    const target = htmlFor ? ` for="${htmlFor}"` : '';
    return `<label${target} class="label-row"><span class="meta-label">${label}</span>${infoButton(info)}</label>`;
}

/**
 * return axis color for the current viewer state.
 */
function axisColor(displayMode: '2d' | '3d', rank: number, axis: number, scheme: DimensionMappingScheme): string {
    const worldKey = axisWorldKeyForMode(displayMode, rank, axis, scheme);
    const family = Array.from({ length: rank }, (_entry, index) => index)
        .filter((index) => axisWorldKeyForMode(displayMode, rank, index, scheme) === worldKey);
    const familyIndex = Math.max(0, family.indexOf(axis));
    const intensity = Math.max(1, familyIndex + 1) / Math.max(1, family.length);
    const channel = Math.round(intensity * 255);
    if (worldKey === 1) return `rgb(0 ${channel} 0)`;
    if (worldKey === 2) return `rgb(0 0 ${channel})`;
    return `rgb(${channel} 0 0)`;
}

/**
 * return axis span for the current viewer state.
 */
function axisSpan(content: string, color: string): string {
    return `<span class="axis-value-segment" style="--axis-color: ${color};">${escapeHtml(content)}</span>`;
}

/**
 * format axis values for the current viewer state.
 */
export function formatAxisValues(values: readonly (number | string)[], displayMode: '2d' | '3d', scheme: DimensionMappingScheme): string {
    if (values.length === 0) return '[]';
    const segments = values.map((value, axis) => axisSpan(String(value), axisColor(displayMode, values.length, axis, scheme)));
    return `[${segments.join('<span class="axis-value-punct">, </span>')}]`;
}

/**
 * format axis tokens for the current viewer state.
 */
export function formatAxisTokens(tokens: readonly string[], displayMode: '2d' | '3d', scheme: DimensionMappingScheme): string {
    if (tokens.length === 0) return '';
    return tokens.map((token, axis) => axisSpan(token, axisColor(displayMode, tokens.length, axis, scheme))).join('<span class="axis-value-punct"> </span>');
}

/**
 * format named axis values for the current viewer state.
 */
export function formatNamedAxisValues(
    labels: readonly string[],
    values: readonly (number | string)[],
    displayMode: '2d' | '3d',
    scheme: DimensionMappingScheme,
): string {
    if (values.length === 0) return '[]';
    return values
        .map((value, axis) => axisSpan(`${labels[axis] ?? axis}:${value}`, axisColor(displayMode, values.length, axis, scheme)))
        .join('<span class="axis-value-punct"> </span>');
}
