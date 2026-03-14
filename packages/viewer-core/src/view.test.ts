import { describe, expect, it } from 'vitest';
import {
    defaultTensorView,
    mapDisplayCoordToFullCoord,
    mapDisplayCoordToOutlineCoord,
    parseTensorView,
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
});
