import { describe, expect, it } from 'vitest';
import {
    buildPreviewExpression,
    defaultTensorView,
    layoutAxisLabels,
    layoutCoordIsVisible,
    layoutCoordMatchesSlice,
    layoutShape,
    mapLayoutCoordToViewCoord,
    mapViewCoordToLayoutCoord,
    mapViewCoordToTensorCoord,
    parseTensorView,
    supportsContiguousSelectionFastPath2D,
} from './view.js';

describe('parseTensorView', () => {
    it('parses a simple default view', () => {
        const result = parseTensorView([2, 3, 4], 'A B C');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.canonical).toBe('A B C');
        expect(result.spec.viewShape).toEqual([2, 3, 4]);
    });

    it('treats empty string as reset to default', () => {
        const result = parseTensorView([2, 3, 4], '');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.canonical).toBe(defaultTensorView([2, 3, 4]));
    });

    it('uses unambiguous default labels beyond Z', () => {
        expect(defaultTensorView(new Array(28).fill(1))).toBe(
            'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z A0 B0',
        );
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
        expect(result.spec.sliceTokens[0]?.token).toBe('b');
        expect(result.spec.axisLabels).toEqual(['B', 'C', 'H', 'W']);
    });

    it('supports labels with trailing non-letters', () => {
        expect(defaultTensorView([2, 3], ['T0', 'T11'])).toBe('T0 T11');
        const result = parseTensorView([2, 3], 't0 T11', [1, 0], ['T0', 'T11']);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.canonical).toBe('t0 T11');
        expect(result.spec.sliceTokens[0]?.token).toBe('t0');
        const grouped = parseTensorView([2, 3], 'T0T11', undefined, ['T0', 'T11']);
        expect(grouped.ok).toBe(true);
        if (!grouped.ok) return;
        expect(grouped.spec.canonical).toBe('T0T11');
        expect(grouped.spec.viewShape).toEqual([6]);
    });

    it('rejects invalid custom axis labels', () => {
        const result = parseTensorView([2, 3], 'A B', undefined, ['A', 'A']);
        expect(result.ok).toBe(false);
    });

    it('rejects labels with extra letters after the prefix', () => {
        const result = parseTensorView([2], 'T0', undefined, ['TH']);
        expect(result.ok).toBe(false);
    });

    it('supports sliced grouped tokens', () => {
        const result = parseTensorView([2, 3, 4, 5], 'a BC D');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.sliceTokens).toHaveLength(1);
        expect(result.spec.sliceTokens[0].token).toBe('a');
    });

    it('keeps layout shape for sliced tokens', () => {
        const sliced = parseTensorView([4, 8], 'A b');
        const visible = parseTensorView([4, 8], 'A B');
        expect(sliced.ok).toBe(true);
        expect(visible.ok).toBe(true);
        if (!sliced.ok || !visible.ok) return;
        expect(sliced.spec.layoutShape).toEqual(visible.spec.layoutShape);
        expect(sliced.spec.viewShape).toEqual([4]);
    });

    it('keeps singleton view axes in coordinate mapping', () => {
        const result = parseTensorView([2, 3], 'A 1 B');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.viewShape).toEqual([2, 1, 3]);
        expect(mapViewCoordToTensorCoord([1, 0, 2], result.spec)).toEqual([1, 2]);
        expect(mapViewCoordToLayoutCoord([1, 0, 2], result.spec)).toEqual([1, 0, 2]);
    });

    it('maps layout coordinates back to visible view coordinates', () => {
        const result = parseTensorView([2, 3, 4, 5], 'ab 1 C D');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(mapLayoutCoordToViewCoord([5, 0, 2, 4], result.spec)).toEqual([0, 2, 4]);
    });

    it('rejects layout coordinates from invisible sliced views', () => {
        const result = parseTensorView([3, 4, 5], 'a B C', [1, 0, 0]);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(layoutCoordMatchesSlice([1, 2, 3], result.spec)).toBe(true);
        expect(layoutCoordMatchesSlice([0, 2, 3], result.spec)).toBe(false);
    });

    it('can collapse hidden axes into the same visual place', () => {
        const result = parseTensorView([2, 3, 4, 5], 'A b C d', [0, 1, 0, 2]);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(layoutShape(result.spec, true)).toEqual([2, 4]);
        expect(layoutAxisLabels(result.spec, true)).toEqual(['A', 'C']);
        expect(mapViewCoordToLayoutCoord([1, 2], result.spec, true)).toEqual([1, 2]);
        expect(mapLayoutCoordToViewCoord([1, 2], result.spec, true)).toEqual([1, 2]);
        expect(layoutCoordIsVisible([1, 2], result.spec, true)).toBe(true);
    });

    it('detects 2d contiguous row-major fast-path views', () => {
        const fast = parseTensorView([2, 3, 4, 5], 'A b C D', [0, 1, 0, 0]);
        const slow = parseTensorView([2, 3, 4, 5], 'A C b D', [0, 0, 1, 0]);
        expect(fast.ok).toBe(true);
        expect(slow.ok).toBe(true);
        if (!fast.ok || !slow.ok) return;
        expect(supportsContiguousSelectionFastPath2D(fast.spec)).toBe(true);
        expect(supportsContiguousSelectionFastPath2D(slow.spec)).toBe(false);
        expect(supportsContiguousSelectionFastPath2D(fast.spec, true)).toBe(false);
    });

    it('shows hidden axes in preview slices using original tensor order', () => {
        const allHidden = parseTensorView([2, 3, 4], 'a b c', [1, 2, 3]);
        const middleHidden = parseTensorView([2, 3, 4], 'A b C', [0, 2, 0]);
        expect(allHidden.ok).toBe(true);
        expect(middleHidden.ok).toBe(true);
        if (!allHidden.ok || !middleHidden.ok) return;
        expect(buildPreviewExpression(allHidden.spec)).toBe('tensor[1, 2, 3]');
        expect(buildPreviewExpression(middleHidden.spec)).toBe('tensor[:, 2, :]');
    });

    it('uses one preview index per hidden grouped token', () => {
        const hiddenGroup = parseTensorView([2, 3, 4], 'A bc', [0, 1, 2]);
        const permuted = parseTensorView([2, 3, 4], 'a B c', [1, 0, 3]);
        expect(hiddenGroup.ok).toBe(true);
        expect(permuted.ok).toBe(true);
        if (!hiddenGroup.ok || !permuted.ok) return;
        expect(buildPreviewExpression(hiddenGroup.spec)).toBe('tensor.reshape(2, 3*4)[:, 6]');
        expect(buildPreviewExpression(permuted.spec)).toBe('tensor[1, :, 3]');
    });
});
