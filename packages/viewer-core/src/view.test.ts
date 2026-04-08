import { describe, expect, it } from 'vitest';
import {
    buildTensorViewExpression,
    layoutAxisLabels,
    layoutCoordIsVisible,
    layoutCoordMatchesSlice,
    layoutShape,
    mapLayoutCoordToViewCoord,
    mapViewCoordToLayoutCoord,
    mapViewCoordToTensorCoord,
    parseTensorView,
    serializeTensorViewEditor,
    supportsContiguousSelectionFastPath2D,
} from './view.js';

describe('parseTensorView', () => {
    it('treats empty string as reset to the default editor state', () => {
        const result = parseTensorView([2, 3, 4], '');
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.editor.viewTensorInput).toBe('[A=2, B=3, C=4]');
        expect(result.spec.viewShape).toEqual([2, 3, 4]);
    });

    it('rejects legacy grammar strings', () => {
        const result = parseTensorView([2, 3, 4], 'A B C');
        expect(result.ok).toBe(false);
        if (result.ok) return;
        expect(result.errors).toEqual(['Tensor view state must be serialized editor data.']);
    });

    it('parses structured editor views with permutation, singleton, and slice state', () => {
        const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
            version: 2,
            viewTensorInput: '[X=2, Y=3, Z=4]',
            baseDims: [],
            permutedDimIds: ['dim-2', 'dim-0', 'dim-1'],
            flattenSeparators: [false, true],
            singletons: [{ id: 'singleton-0', position: 1 }],
            slicedTokenKeys: ['group:dim-2+dim-0'],
            sliceValues: { 'group:dim-2+dim-0': 5 },
        }));
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.baseShape).toEqual([2, 3, 4]);
        expect(result.spec.layoutShape).toEqual([8, 1, 3]);
        expect(result.spec.viewShape).toEqual([1, 3]);
        expect(result.spec.hiddenIndices).toEqual([1, 0, 2]);
        expect(buildTensorViewExpression(result.spec)).toBe('tensor.view(X=2, Y=3, Z=4).permute(2, 0, 1)[5, :, :]');
    });

    it('canonicalizes anonymous editor dimensions as star-a labels', () => {
        const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
            version: 2,
            viewTensorInput: '[2, B, 4]',
            baseDims: [],
            permutedDimIds: [],
            flattenSeparators: [],
            singletons: [],
            slicedTokenKeys: [],
            sliceValues: {},
        }), undefined, ['A', 'B', 'C']);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.editor.viewTensorInput).toBe('[*A0=2, B=3, *A1=4]');
        expect(result.spec.baseDims.map((dim) => dim.label)).toEqual(['*A0', 'B', '*A1']);
    });

    it('maps structured editor coordinates back to the original tensor', () => {
        const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
            version: 2,
            viewTensorInput: '[A, B, C]',
            baseDims: [],
            permutedDimIds: ['dim-1', 'dim-2', 'dim-0'],
            flattenSeparators: [true, true],
            singletons: [],
            slicedTokenKeys: ['group:dim-0'],
            sliceValues: { 'group:dim-0': 1 },
        }), undefined, ['A', 'B', 'C']);
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(mapViewCoordToTensorCoord([2, 3], result.spec)).toEqual([1, 2, 3]);
    });

    it('preserves the first permutation edit after normalization', () => {
        const initial = parseTensorView([2, 3, 4], '');
        expect(initial.ok).toBe(true);
        if (!initial.ok) return;
        const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
            ...initial.spec.editor,
            permutedDimIds: ['axis-2', 'axis-0', 'axis-1'],
        }));
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.editor.permutedDimIds).toEqual(['axis-2', 'axis-0', 'axis-1']);
        expect(buildTensorViewExpression(result.spec)).toBe('tensor.view(A=2, B=3, C=4).permute(2, 0, 1)');
    });

    it('keeps existing labels across non-view edits after numeric canonicalization', () => {
        const initial = parseTensorView([2, 3, 4], '');
        expect(initial.ok).toBe(true);
        if (!initial.ok) return;
        const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
            ...initial.spec.editor,
            viewTensorInput: '[2, 3, 4]',
            flattenSeparators: [false, true],
        }));
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.editor.viewTensorInput).toBe('[*A0=2, *A1=3, *A2=4]');
        expect(result.spec.baseDims.map((dim) => dim.id)).toEqual(['axis-0', 'axis-1', 'axis-2']);
        expect(result.spec.baseDims.map((dim) => dim.label)).toEqual(['*A0', '*A1', '*A2']);
    });

    it('supports explicit final view splits after permute', () => {
        const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
            version: 2,
            viewTensorInput: '[2, 3, 4]',
            finalViewInput: '[2, 2, 3, 2]',
            baseDims: [
                { id: 'axis-0', label: 'A', size: 2 },
                { id: 'axis-1', label: 'B', size: 3 },
                { id: 'axis-2', label: 'C', size: 4 },
            ],
            permutedDimIds: ['axis-2', 'axis-0', 'axis-1'],
            flattenSeparators: [true, true],
            singletons: [],
            slicedTokenKeys: [],
            sliceValues: {},
        }));
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.viewShape).toEqual([2, 2, 3, 2]);
        expect(buildTensorViewExpression(result.spec)).toBe('tensor.view(*A0=2, *A1=3, *A2=4).permute(2, 0, 1).view(2, 2, 3, 2)');
        expect(mapViewCoordToTensorCoord([1, 0, 2, 1], result.spec)).toEqual([1, 2, 2]);
    });

    it('preserves explicit labels with stars and mixed case in view inputs', () => {
        const result = parseTensorView([2, 3, 4], serializeTensorViewEditor({
            version: 2,
            viewTensorInput: '[batch*head=2, q-kv=3, Ab=4]',
            finalViewInput: '[token*idx=6, Value.range=4]',
            baseDims: [],
            permutedDimIds: [],
            flattenSeparators: [],
            singletons: [],
            slicedTokenKeys: ['view:token*idx'],
            sliceValues: { 'view:token*idx': 1 },
        }));
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(result.spec.baseDims.map((dim) => dim.label)).toEqual(['batch*head', 'q-kv', 'Ab']);
        expect(result.spec.sliceTokens[0]?.token).toBe('token*idx');
    });

    it('uses explicit final-view slice keys for starred labels', () => {
        const result = parseTensorView([2048], serializeTensorViewEditor({
            version: 2,
            viewTensorInput: '[A=2048]',
            finalViewInput: '[*A0=32, *A1=64]',
            baseDims: [{ id: 'dim-0', label: 'A', size: 2048 }],
            permutedDimIds: ['dim-0'],
            flattenSeparators: [],
            singletons: [],
            slicedTokenKeys: ['view:*A0'],
            sliceValues: { 'view:*A0': 1 },
        }));
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(buildTensorViewExpression(result.spec)).toBe('tensor.view(A=2048).view(*A0=32, *A1=64)[1, :]');
        expect(mapViewCoordToLayoutCoord([3], result.spec)).toEqual([1, 3]);
        expect(layoutAxisLabels(result.spec, true)).toEqual(['*A1']);
    });

    it('supports collapsed-axis utilities through structured editor state', () => {
        const result = parseTensorView([2, 3, 4, 5], serializeTensorViewEditor({
            version: 2,
            viewTensorInput: '[A=2, B=3, C=4, D=5]',
            baseDims: [],
            permutedDimIds: ['dim-0', 'dim-1', 'dim-2', 'dim-3'],
            flattenSeparators: [true, true, true],
            singletons: [],
            slicedTokenKeys: ['group:dim-1', 'group:dim-3'],
            sliceValues: { 'group:dim-1': 1, 'group:dim-3': 2 },
        }));
        expect(result.ok).toBe(true);
        if (!result.ok) return;
        expect(layoutShape(result.spec, true)).toEqual([2, 4]);
        expect(layoutAxisLabels(result.spec, true)).toEqual(['A', 'C']);
        expect(mapViewCoordToLayoutCoord([1, 2], result.spec, true)).toEqual([1, 2]);
        expect(mapLayoutCoordToViewCoord([1, 2], result.spec, true)).toEqual([1, 2]);
        expect(layoutCoordIsVisible([1, 2], result.spec, true)).toBe(true);
        expect(layoutCoordMatchesSlice([1, 1, 2, 2], result.spec)).toBe(true);
        expect(layoutCoordMatchesSlice([1, 0, 2, 2], result.spec)).toBe(false);
        expect(supportsContiguousSelectionFastPath2D(result.spec)).toBe(false);
        expect(supportsContiguousSelectionFastPath2D(result.spec, true)).toBe(false);
    });
});
