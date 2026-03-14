import { describe, expect, it } from 'vitest';
import { loadNpy, saveNpy } from './npy.js';

describe('npy', () => {
    it('round-trips a float64 tensor', async () => {
        const data = new Float64Array([1.5, -2, 3.25, 4]);
        const blob = saveNpy([2, 2], data, 'float64');
        const result = await loadNpy(blob);
        expect(result.dtype).toBe('float64');
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.data)).toEqual(Array.from(data));
    });

    it('round-trips a uint8 tensor', async () => {
        const data = new Uint8Array([1, 2, 3, 255]);
        const blob = saveNpy([4], data, 'uint8');
        const result = await loadNpy(blob);
        expect(result.dtype).toBe('uint8');
        expect(result.shape).toEqual([4]);
        expect(Array.from(result.data)).toEqual(Array.from(data));
    });
});
