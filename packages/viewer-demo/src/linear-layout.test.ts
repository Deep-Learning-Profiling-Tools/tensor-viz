import { describe, expect, it } from 'vitest';
import {
    buildComposeRuntime,
    composeLayoutStateFromLegacySpec,
    createComposeLayoutDocument,
    defaultComposeLayoutState,
} from './linear-layout.js';

const DEFAULT_RANGES = {
    H: ['0', '0.8'],
    S: ['1', '1'],
    L: ['0', '1'],
} as const;

function composeState(specsText: string, operationText: string) {
    return {
        specsText,
        operationText,
        inputName: 'Input Space',
        visibleTensors: {},
        mapping: { H: 'none', S: 'none', L: 'none' } as const,
        ranges: {
            H: [...DEFAULT_RANGES.H],
            S: [...DEFAULT_RANGES.S],
            L: [...DEFAULT_RANGES.L],
        },
    };
}

describe('compose layout helpers', () => {
    it('builds the default blocked-layout example', () => {
        const state = defaultComposeLayoutState();
        const runtime = buildComposeRuntime(state);
        const document = createComposeLayoutDocument(state);

        expect(runtime.tensors.map((tensor) => ({ title: tensor.title, shape: tensor.shape }))).toEqual([
            { title: 'Hardware Layout', shape: [32, 4, 4] },
            { title: 'Blocked_Layout', shape: [16, 32] },
        ]);
        expect(document.title).toBe('Blocked_Layout');
        expect(document.manifest.viewer.dimensionMappingScheme).toBe('contiguous');
        expect(document.manifest.tensors.map((tensor) => ({
            name: tensor.name,
            shape: tensor.shape,
            axisLabels: tensor.axisLabels,
        }))).toEqual([
            { name: 'Hardware Layout', shape: [32, 4, 4], axisLabels: ['T', 'W', 'R'] },
            { name: 'Blocked_Layout', shape: [16, 32], axisLabels: ['A', 'B'] },
        ]);
        expect(state.mapping).toEqual({ H: 'T', S: 'W', L: 'R' });
    });

    it('renders composition chains left to right and emits compose code in the same order', () => {
        const state = composeState([
            'Layout1: [T] -> [A]',
            '[[1]]',
            '',
            'Layout2: [A] -> [B]',
            '[[1]]',
        ].join('\n'), 'Layout2(Layout1)');
        const runtime = buildComposeRuntime(state);

        expect(runtime.tensors.map((tensor) => tensor.title)).toEqual([
            'Input Space',
            'Layout1',
            'Layout2(Layout1)',
        ]);
        expect(runtime.tensors.map((tensor) => tensor.shape)).toEqual([[2], [2], [2]]);
        expect(runtime.matrixBlocks.map((block) => block.title)).toEqual([
            'Layout1',
            'Layout2',
            'layout_tmp1 = Layout2(Layout1)',
        ]);
        expect(runtime.pythonCode).toContain('layout_tmp1 = Layout1.compose(Layout2)');
    });

    it('normalizes legacy warp-thread-register ordering to thread-warp-register', () => {
        const state = composeLayoutStateFromLegacySpec({
            name: 'Legacy',
            input_dims: [
                ['warp', []],
                ['thread', []],
                ['register', []],
            ],
            out_dims: ['x'],
        });

        expect(state.specsText.startsWith('Legacy: [T,W,R] -> [X]')).toBe(true);
    });

    it('composes layouts with matching labels even when intermediate bit widths differ', () => {
        const state = composeState([
            'Layout1: [T] -> [A]',
            '[[1]]',
            '',
            'Layout2: [A] -> [B]',
            '[[1],[2]]',
        ].join('\n'), 'Layout2(Layout1)');
        const runtime = buildComposeRuntime(state);

        expect(runtime.tensors.map((tensor) => tensor.title)).toEqual([
            'Input Space',
            'Layout1',
            'Layout2(Layout1)',
        ]);
        expect(runtime.tensors.map((tensor) => tensor.shape)).toEqual([[2], [2], [2]]);
    });

    it('rejects compositions that become non-injective after width alignment', () => {
        const state = composeState([
            'Blocked_Layout: [T,W,R] -> [Y,X]',
            '[[4,0],[8,0],[0,1],[0,2],[0,4]]',
            '[[0,8],[0,16]]',
            '[[1,0],[2,0]]',
            '',
            'Layout2: [Y,X] -> [Y,X]',
            '[[4,0],[8,0],[0,1],[0,2],[0,4]]',
            '[[0,8],[0,16]]',
        ].join('\n'), 'Layout2(Blocked_Layout)');

        expect(() => buildComposeRuntime(state)).toThrow('Layout2(Blocked_Layout) is not injective over its full input domain.');
    });

    it('supports products as atomic render steps', () => {
        const state = composeState([
            'Left: [T] -> [A]',
            '[[1]]',
            '',
            'Right: [U] -> [B]',
            '[[1]]',
        ].join('\n'), 'Left * Right');
        const runtime = buildComposeRuntime(state);

        expect(runtime.inputLabels).toEqual(['T', 'U']);
        expect(runtime.tensors.map((tensor) => ({ title: tensor.title, shape: tensor.shape }))).toEqual([
            { title: 'Input Space', shape: [2, 2] },
            { title: 'Left * Right', shape: [2, 2] },
        ]);
        expect(runtime.pythonCode).toContain('layout_tmp1 = Left * Right');
    });

    it('rejects non-injective specs', () => {
        const state = composeState([
            'Bad: [T] -> [A]',
            '[[0],[0]]',
        ].join('\n'), 'Bad');

        expect(() => buildComposeRuntime(state)).toThrow('Bad is not injective over its full input domain.');
    });

    it('rejects inverse on injective-but-non-bijective layouts', () => {
        const state = composeState([
            'Stretch: [T] -> [A]',
            '[[2]]',
        ].join('\n'), 'inv(Stretch)');

        expect(() => buildComposeRuntime(state)).toThrow('Stretch is not bijective, so inv(Stretch) is invalid.');
    });
});
