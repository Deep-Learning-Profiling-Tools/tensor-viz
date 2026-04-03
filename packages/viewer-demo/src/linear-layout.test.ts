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
            'T: [[1]]',
            '',
            'Layout2: [A] -> [B]',
            'A: [[1]]',
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
            'T: [[1]]',
            '',
            'Layout2: [A] -> [B]',
            'A: [[1],[2]]',
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
            'T: [[4,0],[8,0],[0,1],[0,2],[0,4]]',
            'W: [[0,8],[0,16]]',
            'R: [[1,0],[2,0]]',
            '',
            'Layout2: [Y,X] -> [Y,X]',
            'Y: [[4,0],[8,0],[0,1],[0,2],[0,4]]',
            'X: [[0,8],[0,16]]',
        ].join('\n'), 'Layout2(Blocked_Layout)');

        expect(() => buildComposeRuntime(state)).toThrow('Layout2(Blocked_Layout) is not injective over its full input domain.');
    });

    it('supports products as atomic render steps', () => {
        const state = composeState([
            'Left: [T] -> [A]',
            'T: [[1]]',
            '',
            'Right: [U] -> [B]',
            'U: [[1]]',
        ].join('\n'), 'Left * Right');
        const runtime = buildComposeRuntime(state);

        expect(runtime.inputLabels).toEqual(['T', 'U']);
        expect(runtime.tensors.map((tensor) => ({ title: tensor.title, shape: tensor.shape }))).toEqual([
            { title: 'Input Space', shape: [2, 2] },
            { title: 'Left * Right', shape: [2, 2] },
        ]);
        expect(runtime.pythonCode).toContain('layout_tmp1 = Left * Right');
    });

    it('supports products with duplicate inputs and distinct outputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Left: [A,B] -> [C,D]',
            'A: [[1,0],[2,0]]',
            'B: [[0,1],[0,2]]',
            '',
            'Right: [A,B] -> [C2,D2]',
            'A: [[1,0],[2,0]]',
            'B: [[0,1],[0,2]]',
        ].join('\n'), 'Left * Right'));

        expect(runtime.inputLabels).toEqual(['A', 'B']);
        expect(runtime.inputShape).toEqual([16, 16]);
        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['C', 'D', 'C2', 'D2'],
            shape: [4, 4, 4, 4],
        });
        expect(runtime.matrixBlocks.at(-1)?.values).toEqual([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]);
    });

    it('supports products with distinct inputs and duplicate outputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Left: [A] -> [C]',
            'A: [[1]]',
            '',
            'Right: [B] -> [C]',
            'B: [[1]]',
        ].join('\n'), 'Left * Right'));

        expect(runtime.inputLabels).toEqual(['A', 'B']);
        expect(runtime.inputShape).toEqual([2, 2]);
        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['C'],
            shape: [4],
            rootToTensor: [[0], [2], [1], [3]],
        });
    });

    it('supports products with duplicate inputs and duplicate outputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Left: [A] -> [C]',
            'A: [[1]]',
            '',
            'Right: [A] -> [C]',
            'A: [[1]]',
        ].join('\n'), 'Left * Right'));

        expect(runtime.inputLabels).toEqual(['A']);
        expect(runtime.inputShape).toEqual([4]);
        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['C'],
            shape: [4],
            rootToTensor: [[0], [1], [2], [3]],
        });
    });

    it('rejects non-injective specs', () => {
        const state = composeState([
            'Bad: [T] -> [A]',
            'T: [[0],[0]]',
        ].join('\n'), 'Bad');

        expect(() => buildComposeRuntime(state)).toThrow('Bad is not injective over its full input domain.');
    });

    it('rejects inverse on injective-but-non-bijective layouts', () => {
        const state = composeState([
            'Stretch: [T] -> [A]',
            'T: [[2]]',
        ].join('\n'), 'inv(Stretch)');

        expect(() => buildComposeRuntime(state)).toThrow('Stretch is not bijective, so inv(Stretch) is invalid.');
    });

    it('accepts labeled basis rows with flexible whitespace and reordered inputs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Blocked_Layout: [T,W,R] -> [Y,X]',
            'R:[[1,0],[2,0]]',
            'W : [[0,8],[0,16]]',
            'T   :[[4,0],[8,0],[0,1],[0,2],[0,4]]',
        ].join('\n'), 'Blocked_Layout'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            title: 'Blocked_Layout',
            shape: [16, 32],
            axisLabels: ['Y', 'X'],
        });
    });

    it('rejects unlabeled basis rows', () => {
        const state = composeState([
            'Bad: [T] -> [A]',
            '[[1]]',
        ].join('\n'), 'Bad');

        expect(() => buildComposeRuntime(state)).toThrow('Layout Bad basis rows must use "<label>: <json>" syntax.');
    });
});
