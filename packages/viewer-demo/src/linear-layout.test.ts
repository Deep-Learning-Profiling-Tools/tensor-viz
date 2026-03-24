import { describe, expect, it } from 'vitest';
import {
    createLinearLayoutDocument,
    defaultLinearLayoutSpecText,
    parseLinearLayoutSpec,
} from './linear-layout.js';

describe('linear layout helpers', () => {
    it('builds the default blocked-layout example', () => {
        const spec = parseLinearLayoutSpec(defaultLinearLayoutSpecText());
        const document = createLinearLayoutDocument(spec);
        const hardware = document.tensors.get('tensor-1') as Float32Array;
        const logical = document.tensors.get('tensor-2') as Float32Array;

        expect(document.title).toBe('Blocked Layout');
        expect(document.manifest.viewer.dimensionMappingScheme).toBe('contiguous');
        expect(document.manifest.tensors.map((tensor) => tensor.shape)).toEqual([[32, 4, 4], [32, 16]]);
        expect(document.manifest.tensors[0]?.axisLabels).toEqual(['T', 'W', 'R']);
        expect(document.manifest.tensors[1]?.axisLabels).toEqual(['Y', 'X']);
        expect(hardware[1]).toBe(1);
        expect(logical[1]).toBe(1);
    });

    it('infers output sizes from xor-composed bases', () => {
        const spec = parseLinearLayoutSpec(JSON.stringify({
            name: 'xor',
            bases: [
                ['lane', [[1], [1]]],
            ],
            out_dims: ['x'],
            color_axes: { lane: 'L' },
            color_ranges: { L: [0, 1] },
        }));
        const document = createLinearLayoutDocument(spec);
        const logical = document.tensors.get('tensor-2') as Float32Array;

        expect(spec.output_dims).toEqual([{ name: 'x', size: 2 }]);
        expect(logical[0]).toBe(3);
        expect(logical[1]).toBe(2);
    });

    it('forces contiguous mapping when rebuilding a layout', () => {
        const spec = parseLinearLayoutSpec(defaultLinearLayoutSpecText());
        const document = createLinearLayoutDocument(spec, {
            dimensionMappingScheme: 'z-order',
            tensors: [{ id: 'tensor-1', name: 'Hardware tensor', view: { view: 'T W R', hiddenIndices: [] } }],
        });

        expect(document.manifest.viewer.dimensionMappingScheme).toBe('contiguous');
    });
});
