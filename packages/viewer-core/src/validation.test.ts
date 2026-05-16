import { describe, expect, it } from 'vitest';
import { createBundleManifest } from './session.js';
import { parseTensorView } from './view.js';
import {
    validateBundleManifest,
    validateTensorPayload,
    VIEWER_LIMITS,
} from './validation.js';

describe('runtime validation', () => {
    it('rejects shapes that would allocate too many cells', () => {
        const manifest = createBundleManifest({
            tensors: [{ name: 'small', dtype: 'float32', shape: [2, 2] }],
        });
        expect(() => validateBundleManifest({
            ...manifest,
            tensors: [{
                ...manifest.tensors[0],
                shape: [VIEWER_LIMITS.maxTensorElements + 1],
            }],
        })).toThrow(/too (large|many elements)/);
    });

    it('rejects payloads with the wrong byte length', () => {
        expect(() => validateTensorPayload('float32', [1], 8)).toThrow(/byte length/);
    });

    it('rejects color regions that expand past the tensor shape', () => {
        const manifest = createBundleManifest({
            tensors: [{
                name: 'colored',
                dtype: 'float32',
                shape: [4, 4],
                colorInstructions: [{
                    mode: 'rgb',
                    kind: 'region',
                    base: [3, 3],
                    shape: [2, 2],
                    jumps: [1, 1],
                    color: [255, 0, 0],
                }],
            }],
        });
        expect(() => validateBundleManifest(manifest)).toThrow(/exceeds tensor bounds/);
    });

    it('rejects oversized serialized editor state before normalization work', () => {
        const editor = {
            version: 2,
            viewTensorInput: '[A=2]',
            baseDims: [],
            permutedDimIds: [],
            flattenSeparators: [],
            singletons: Array.from({ length: VIEWER_LIMITS.maxEditorEntries + 1 }, (_entry, index) => ({
                id: `singleton-${index}`,
                position: 0,
            })),
            slicedTokenKeys: [],
            sliceValues: {},
        };
        const result = parseTensorView([2], `tv2:${encodeURIComponent(JSON.stringify(editor))}`);
        expect(result.ok).toBe(false);
    });
});
