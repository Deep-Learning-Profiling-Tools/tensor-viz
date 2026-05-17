import { axisWorldKeyForMode, type DimensionMappingScheme, type ViewerSnapshot } from '@tensor-viz/viewer-core';

/**
 * Formats a numeric range or selection statistic for compact display in the
 * demo UI, preserving exact integer text and rounding fractional values to six
 * significant digits.
 *
 * @param value - Numeric statistic or range endpoint from the viewer summary.
 * @returns The label text to place in the UI, such as `"4"` for an integer or a six-significant-digit string for a fractional value.
 * @noThrows The formatter only checks whether the numeric argument is an integer and then uses built-in number-to-string formatting.
 * @example
 * formatRangeValue(12);
 * // '12'
 *
 * formatRangeValue(1 / 3);
 * // '0.333333'
 */
export function formatRangeValue(value: number): string {
    return Number.isInteger(value) ? String(value) : value.toPrecision(6);
}

/**
 * Reports whether the viewer snapshot supports box selection in the demo UI.
 * Selection is available only in 2D display mode with contiguous dimension
 * mapping; an omitted mapping scheme is treated as the default z-order mapping
 * and therefore disables selection.
 *
 * @param snapshot - Viewer state snapshot containing at least `displayMode` and the optional `dimensionMappingScheme` used by the control dock.
 * @returns `true` when selection controls and selection-dependent panels should be enabled; otherwise `false`.
 * @noThrows The predicate only reads fields from the supplied snapshot and compares them with known display and mapping mode strings.
 * @example
 * selectionEnabled({ displayMode: '2d', dimensionMappingScheme: 'contiguous' } as ViewerSnapshot);
 * // true
 *
 * selectionEnabled({ displayMode: '3d', dimensionMappingScheme: 'contiguous' } as ViewerSnapshot);
 * // false
 */
export function selectionEnabled(snapshot: ViewerSnapshot): boolean {
    return snapshot.displayMode === '2d' && (snapshot.dimensionMappingScheme ?? 'z-order') === 'contiguous';
}

/**
 * Escapes the HTML-significant characters used in demo markup so user- or model-facing labels can be inserted into attributes and text nodes without being interpreted as markup.
 *
 * @param text - Raw label, tooltip, tensor id, error message, or other string that may contain `&`, `"`, `<`, or `>` characters.
 * @returns The input string with ampersands, double quotes, less-than signs, and greater-than signs replaced by their HTML entity forms.
 * @noThrows The helper only performs built-in string replacements on a TypeScript `string` value and does not parse HTML, touch the DOM, or validate external state.
 * @example
 * escapeHtml('A&B <axis "x">');
 * // returns 'A&amp;B &lt;axis &quot;x&quot;&gt;'
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
 * Builds the small demo info-button HTML fragment used beside widget titles and field labels, with the help text copied into both accessibility and tooltip data attributes.
 *
 * @param text - Help or tooltip text to expose through the button's `aria-label` and `data-info` attributes.
 * @returns A `<button class="info-button">` HTML string whose label and `data-info` value contain the escaped help text.
 * @noThrows The function only escapes the provided string and interpolates it into a fixed button template; it does not query the DOM or depend on viewer state.
 * @example
 * infoButton('Choose tensor <main>');
 * // returns '<button class="info-button" type="button" tabindex="-1" aria-label="Choose tensor &lt;main&gt;" data-info="Choose tensor &lt;main&gt;">i</button>'
 */
export function infoButton(text: string): string {
    const escaped = escapeHtml(text);
    return `<button class="info-button" type="button" tabindex="-1" aria-label="${escaped}" data-info="${escaped}">i</button>`;
}

/**
 * Creates a title-row HTML fragment for demo panels by pairing an `h2` heading with the standard escaped info button.
 *
 * @param title - Heading text to place inside the row's `<h2>` element.
 * @param info - Help text for the adjacent info button; this value is escaped before it is placed in button attributes.
 * @returns A `<div class="title-row">` HTML string containing the heading and info button for insertion into panel markup.
 * @noThrows The helper only concatenates string fragments and delegates info-text escaping to `infoButton`; it does not read DOM state or perform validation.
 * @example
 * titleWithInfo('Tensor View', 'Edit axes <rank>.');
 * // returns '<div class="title-row"><h2>Tensor View</h2><button class="info-button" type="button" tabindex="-1" aria-label="Edit axes &lt;rank&gt;." data-info="Edit axes &lt;rank&gt;.">i</button></div>'
 */
export function titleWithInfo(title: string, info: string): string {
    return `<div class="title-row"><h2>${title}</h2>${infoButton(info)}</div>`;
}

/**
 * Creates the demo's label-row HTML fragment with visible label text, an optional target control id, and the standard escaped info button.
 *
 * @param label - Visible field label text to place inside the `meta-label` span.
 * @param info - Help text for the adjacent info button; this value is escaped before it is placed in button attributes.
 * @param htmlFor - Optional id of the form control that the returned `<label>` should target with a `for` attribute.
 * @returns A `<label class="label-row">` HTML string, including `for="..."` when `htmlFor` is provided, ready to insert into widget field markup.
 * @noThrows The helper only checks whether the optional id is present, composes strings, and uses `infoButton` for help-text escaping; it does not access the DOM.
 * @example
 * labelWithInfo('Tensor', 'Choose tensor <active>.', 'tensor-select');
 * // returns '<label for="tensor-select" class="label-row"><span class="meta-label">Tensor</span><button class="info-button" type="button" tabindex="-1" aria-label="Choose tensor &lt;active&gt;." data-info="Choose tensor &lt;active&gt;.">i</button></label>'
 *
 * labelWithInfo('Rank', 'Number of dimensions.');
 * // returns '<label class="label-row"><span class="meta-label">Rank</span><button class="info-button" type="button" tabindex="-1" aria-label="Number of dimensions." data-info="Number of dimensions.">i</button></label>'
 */
export function labelWithInfo(label: string, info: string, htmlFor?: string): string {
    const target = htmlFor ? ` for="${htmlFor}"` : '';
    return `<label${target} class="label-row"><span class="meta-label">${label}</span>${infoButton(info)}</label>`;
}

/**
 * Chooses the CSS RGB color used to mark one tensor axis in axis-value HTML.
 * Axes that map to the same world dimension share the same color channel, with
 * later duplicate axes rendered at a lighter channel intensity.
 *
 * @param displayMode - Viewer projection used to interpret the mapping scheme: `'2d'` for planar axes or `'3d'` for x/y/z-style world axes.
 * @param rank - Total number of axes in the tensor coordinate or shape being rendered.
 * @param axis - Zero-based tensor axis whose display color is being requested.
 * @param scheme - Dimension mapping scheme that `axisWorldKeyForMode` uses to group tensor axes into displayed world dimensions.
 * @returns A CSS `rgb(...)` color string for the axis segment's `--axis-color` style.
 * @noThrows Performs bounded array construction, numeric clamping, and string formatting only; unsupported mappings fall back to the red channel rather than throwing here.
 * @example
 * const color = axisColor('3d', 3, 1, scheme);
 * // color is a CSS RGB string such as 'rgb(0 255 0)' for an axis mapped to the green/Y world channel.
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
 * Wraps one axis label or value in the HTML span used by the demo inspector and
 * assigns the supplied axis color through the `--axis-color` CSS custom property.
 *
 * @param content - Axis token or value text to display inside the segment; it is HTML-escaped before insertion.
 * @param color - CSS color string, usually from `axisColor`, assigned to `--axis-color` on the span.
 * @returns An HTML fragment containing one `axis-value-segment` span with escaped text content.
 * @noThrows Builds a string and delegates escaping to `escapeHtml`; it does not query the DOM or validate CSS color syntax.
 * @example
 * axisSpan('x<1', 'rgb(255 0 0)');
 * // '<span class="axis-value-segment" style="--axis-color: rgb(255 0 0);">x&lt;1</span>'
 */
function axisSpan(content: string, color: string): string {
    return `<span class="axis-value-segment" style="--axis-color: ${color};">${escapeHtml(content)}</span>`;
}

/**
 * Formats tensor coordinate or shape values as the bracketed, color-coded HTML
 * list shown in the demo UI.
 *
 * @param values - Ordered tensor-axis values, such as a shape `[64, 128]` or a hovered coordinate `[3, 7]`.
 * @param displayMode - Viewer projection used when assigning each axis to a display color.
 * @param scheme - Dimension mapping scheme used to color each value according to its displayed world axis.
 * @returns A bracketed HTML fragment with comma punctuation between colored axis segments, or the literal string `'[]'` when `values` is empty.
 * @noThrows Converts each value with `String(...)` and builds markup from the provided array; empty input is handled explicitly.
 * @example
 * formatAxisValues([], '2d', scheme);
 * // '[]'
 *
 * @example
 * const html = formatAxisValues([16, 32], '2d', scheme);
 * // html starts with '[' and contains two 'axis-value-segment' spans separated by an 'axis-value-punct' comma.
 */
export function formatAxisValues(values: readonly (number | string)[], displayMode: '2d' | '3d', scheme: DimensionMappingScheme): string {
    if (values.length === 0) return '[]';
    const segments = values.map((value, axis) => axisSpan(String(value), axisColor(displayMode, values.length, axis, scheme)));
    return `[${segments.join('<span class="axis-value-punct">, </span>')}]`;
}

/**
 * Formats already-rendered axis tokens as color-coded HTML segments separated by
 * the punctuation span used in the demo's coordinate displays.
 *
 * @param tokens - Ordered token strings for tensor axes, such as binary coordinate labels produced by the inspector.
 * @param displayMode - Viewer projection used when choosing the color for each token's axis.
 * @param scheme - Dimension mapping scheme used to group token positions by displayed world axis.
 * @returns An HTML fragment with one colored span per token separated by space punctuation, or an empty string when `tokens` is empty.
 * @noThrows Iterates over the provided string array and concatenates markup only; empty input is returned before color calculation.
 * @example
 * formatAxisTokens([], '3d', scheme);
 * // ''
 *
 * @example
 * const html = formatAxisTokens(['x:0011', 'y:0101'], '3d', scheme);
 * // html contains two 'axis-value-segment' spans separated by an 'axis-value-punct' space.
 */
export function formatAxisTokens(tokens: readonly string[], displayMode: '2d' | '3d', scheme: DimensionMappingScheme): string {
    if (tokens.length === 0) return '';
    return tokens.map((token, axis) => axisSpan(token, axisColor(displayMode, tokens.length, axis, scheme))).join('<span class="axis-value-punct"> </span>');
}

/**
 * Builds the inspector HTML for one coordinate by pairing each axis value with its display label and dimension color.
 *
 * @param labels - Axis labels from the tensor or extension inspector row; missing labels fall back to the zero-based axis index.
 * @param values - Coordinate components to render in axis order, such as `[2, 7]` for a two-dimensional tensor cell.
 * @param displayMode - Viewer projection mode used to choose the axis color palette for 2D or 3D rendering.
 * @param scheme - Dimension mapping scheme that determines which color is assigned to each coordinate axis.
 * @returns HTML markup for the inspector value column; callers can assign it to an element's `innerHTML` to show colored `label:value` pairs, or receive `'[]'` for an empty coordinate.
 * @noThrows The formatter only reads the provided arrays, substitutes numeric fallback labels, and concatenates markup; it performs no validation or DOM access.
 * @example
 * formatNamedAxisValues([], [], '2d', dimensionMappingScheme);
 * // '[]'
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
