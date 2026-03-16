import { describe, expect, it } from 'vitest';
import {
    defaultTensorView,
    mapDisplayCoordToFullCoord,
    mapDisplayCoordToOutlineCoord,
    mapDisplayCoordToViewedCoord,
    mapOutlineCoordToDisplayCoord,
    mapViewedCoordToDisplayCoord,
    outlineCoordIsVisible,
    parseTensorView,
    viewedCoordIsVisible,
    viewedShape,
    viewedTokenLabels,
} from './view.js';

describe('parseTensorView', () => {
    it('parses a simple default view', () => {
        const result = parseTensorView([2, 3, 4], 'A B C');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.canonical).toBe('A B C');
        expect(result.spec.displayShape).toEqual([2, 3, 4]);
    });

    it('treats empty string as reset to default', () => {
        const result = parseTensorView([2, 3, 4], '');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.canonical).toBe(defaultTensorView([2, 3, 4]));
    });

    it('rejects mixed case grouped tokens', () => {
        const result = parseTensorView([2, 3, 4], 'A bC');
        expect(result.ok).toBe(false);
    });

    it('supports custom axis labels', () => {
        expect(defaultTensorView([2, 3, 4, 5], ['B', 'C', 'H', 'W'])).toBe('B C H W');
        const result = parseTensorView([2, 3, 4, 5], 'b C H W', [1, 0, 0, 0], ['B', 'C', 'H', 'W']);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.canonical).toBe('b C H W');
        expect(result.spec.hiddenTokens[0]?.token).toBe('b');
        expect(result.spec.axisLabels).toEqual(['B', 'C', 'H', 'W']);
    });

    it('supports labels with trailing non-letters', () => {
        expect(defaultTensorView([2, 3], ['T0', 'T11'])).toBe('T0 T11');
        const result = parseTensorView([2, 3], 't0 T11', [1, 0], ['T0', 'T11']);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.canonical).toBe('t0 T11');
        expect(result.spec.hiddenTokens[0]?.token).toBe('t0');
        const grouped = parseTensorView([2, 3], 'T0T11', undefined, ['T0', 'T11']);
        expect(grouped.ok).toBe(true);
        if (!grouped.ok) return;
        expect(grouped.spec.canonical).toBe('T0T11');
        expect(grouped.spec.displayShape).toEqual([6]);
    });

    it('rejects invalid custom axis labels', () => {
        const result = parseTensorView([2, 3], 'A B', undefined, ['A', 'A']);
        expect(result.ok).toBe(false);
    });

    it('rejects labels with extra letters after the prefix', () => {
        const result = parseTensorView([2], 'T0', undefined, ['TH']);
        expect(result.ok).toBe(false);
    });

    it('supports hidden grouped tokens', () => {
        const result = parseTensorView([2, 3, 4, 5], 'a BC D');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.hiddenTokens).toHaveLength(1);
        expect(result.spec.hiddenTokens[0].token).toBe('a');
    });

    it('keeps outline shape for hidden tokens', () => {
        const hidden = parseTensorView([4, 8], 'A b');
        const visible = parseTensorView([4, 8], 'A B');
        expect(hidden.ok).toBe(true);
        expect(visible.ok).toBe(true);
        if (!hidden.ok || !visible.ok) return;
        expect(hidden.spec.outlineShape).toEqual(visible.spec.outlineShape);
        expect(hidden.spec.displayShape).toEqual([4]);
    });

    it('keeps singleton display axes in coordinate mapping', () => {
        const result = parseTensorView([2, 3], 'A 1 B');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.displayShape).toEqual([2, 1, 3]);
        expect(mapDisplayCoordToFullCoord([1, 0, 2], result.spec)).toEqual([1, 2]);
        expect(mapDisplayCoordToOutlineCoord([1, 0, 2], result.spec)).toEqual([1, 0, 2]);
    });

    it('maps outline coordinates back to visible display coordinates', () => {
        const result = parseTensorView([2, 3, 4, 5], 'ab 1 C D');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(mapOutlineCoordToDisplayCoord([5, 0, 2, 4], result.spec)).toEqual([0, 2, 4]);
    });

    it('rejects outline coordinates from invisible hidden slices', () => {
        const result = parseTensorView([3, 4, 5], 'a B C', [1, 0, 0]);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(outlineCoordIsVisible([1, 2, 3], result.spec)).toBe(true);
        expect(outlineCoordIsVisible([0, 2, 3], result.spec)).toBe(false);
    });

    it('can show hidden slices in the same visual place', () => {
        const result = parseTensorView([2, 3, 4, 5], 'A b C d', [0, 1, 0, 2]);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(viewedShape(result.spec, true)).toEqual([2, 4]);
        expect(viewedTokenLabels(result.spec, true)).toEqual(['A', 'C']);
        expect(mapDisplayCoordToViewedCoord([1, 2], result.spec, true)).toEqual([1, 2]);
        expect(mapViewedCoordToDisplayCoord([1, 2], result.spec, true)).toEqual([1, 2]);
        expect(viewedCoordIsVisible([1, 2], result.spec, true)).toBe(true);
    });
});
