import { describe, expect, it } from 'vitest';
import { createBundleManifest, createSessionBundleManifest, createViewerSnapshot } from './session.js';

describe('session builders', () => {
    it('builds a bundle manifest with default tensor views', () => {
        const manifest = createBundleManifest({
            tensors: [
                { name: 'weights', dtype: 'float32', shape: [4, 8], axisLabels: ['O', 'I'] },
            ],
        });

        expect(manifest.tensors[0]?.id).toBe('tensor-1');
        expect(manifest.tensors[0]?.placeholderData).toBe(true);
        expect(manifest.tensors[0]?.view.editor?.viewTensorInput).toBe('[O=4, I=8]');
        expect(manifest.viewer.activeTensorId).toBe('tensor-1');
        expect(manifest.viewer.dimensionMappingScheme).toBe('z-order');
        expect(manifest.viewer.tensors[0]?.view.editor?.viewTensorInput).toBe('[O=4, I=8]');
    });

    it('builds a multi-tab session manifest', () => {
        const session = createSessionBundleManifest([
            {
                title: 'inputs',
                tensors: [
                    { name: 'image', dtype: 'float32', shape: [3, 32, 32], axisLabels: ['C', 'H', 'W'] },
                ],
            },
            {
                title: 'weights',
                tensors: [
                    { name: 'conv', dtype: 'float32', shape: [16, 3, 3, 3], axisLabels: ['O', 'I', 'K0', 'K1'] },
                ],
            },
        ]);

        expect(session.tabs.map((tab) => tab.id)).toEqual(['tab-1', 'tab-2']);
        expect(session.tabs.map((tab) => tab.title)).toEqual(['inputs', 'weights']);
        expect(session.tabs[1]?.viewer.tensors[0]?.view.editor?.viewTensorInput).toBe('[O=16, I=3, K0=3, K1=3]');
    });

    it('defaults tensor names on and preserves explicit overrides', () => {
        expect(createViewerSnapshot().showTensorNames).toBe(true);
        expect(createViewerSnapshot({ showTensorNames: false }).showTensorNames).toBe(false);
    });

    it('defaults selection panel on and preserves explicit overrides', () => {
        expect(createViewerSnapshot().showSelectionPanel).toBe(true);
        expect(createViewerSnapshot({ showSelectionPanel: false }).showSelectionPanel).toBe(false);
    });

});
