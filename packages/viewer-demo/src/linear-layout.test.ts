import { describe, expect, it } from 'vitest';
import {
    autoColorLayoutState,
    buildComposeRuntime,
    composeLayoutStateFromLegacySpec,
    createComposeLayoutDocument,
    defaultComposeLayoutState,
} from './linear-layout.js';
import {
    coordsForRootIndexes,
    linearLayoutDisplayModel,
    linearLayoutMultiInputModel,
    linearLayoutSelectionMapForMeta,
    rootIndexesForCoords,
} from './linear-layout-multi-input.js';
import { linearLayoutHoverPopupEntries } from './linear-layout-viewer-sync.js';

const DEFAULT_RANGES = {
    H: ['0', '0.8'],
    S: ['1', '0.2'],
    L: ['1', '0.2'],
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
            { name: 'Blocked_Layout', shape: [16, 32], axisLabels: ['Y', 'X'] },
        ]);
        expect(state.mapping).toEqual({ H: 'T', S: 'W', L: 'R' });
        expect(state.ranges).toEqual({ H: ['0', '0.8'], S: ['1', '0.2'], L: ['1', '0.2'] });
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
        expect(runtime.tensors.map((tensor) => tensor.shape)).toEqual([[2], [2], [4]]);
    });

    it('composes swizzles after tiles without dropping reachable coordinates', () => {
        const runtime = buildComposeRuntime(composeState([
            'Tile: [T,W] -> [Y,X]',
            'T: [[0,1],[0,2]]',
            'W: [[1,0]]',
            '',
            'Swizzle: [Y,X] -> [S,B]',
            'Y: [[1,1],[2,2]]',
            'X: [[0,1],[0,2]]',
        ].join('\n'), 'Swizzle(Tile)'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            axisLabels: ['S', 'B'],
            shape: [4, 4],
            rootToTensor: [[0, 0], [1, 1], [0, 1], [1, 0], [0, 2], [1, 3], [0, 3], [1, 2]],
        });
    });

    it('supports compositions that become non-injective after width alignment', () => {
        const runtime = buildComposeRuntime(composeState([
            'L: [X] -> [X]',
            'X: [[0],[1]]',
        ].join('\n'), 'L(L)'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            title: 'L(L)',
            shape: [2],
            rootToTensor: [[0], [0], [0], [0]],
        });
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

    it('accepts non-injective specs', () => {
        const runtime = buildComposeRuntime(composeState([
            'Bad: [T] -> [A]',
            'T: [[0],[0]]',
        ].join('\n'), 'Bad'));

        expect(runtime.tensors.at(-1)).toMatchObject({
            title: 'Bad',
            shape: [1],
            rootToTensor: [[0], [0], [0], [0]],
        });
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

    it('maps largest inputs to H, then L, then S', () => {
        expect(autoColorLayoutState([
            'Blocked_Layout: [T,W,R] -> [Y,X]',
            'T: [[4,0],[8,0],[0,1],[0,2],[0,4]]',
            'W: [[0,8],[0,16]]',
            'R: [[1,0],[2,0]]',
        ].join('\n'), 'Blocked_Layout').mapping).toEqual({ H: 'T', S: 'W', L: 'R' });
    });

    it('leaves unused color channels unmapped when fewer than three inputs exist', () => {
        expect(autoColorLayoutState([
            'Pair: [A,B] -> [C]',
            'A: [[1],[2],[4]]',
            'B: [[8]]',
        ].join('\n'), 'Pair').mapping).toEqual({ H: 'A', S: 'none', L: 'B' });
    });

    it('prefers later dimensions when equal sizes compete for the next color axis', () => {
        expect(autoColorLayoutState([
            'Square: [Y,X] -> [A]',
            'Y: [[1]]',
            'X: [[2]]',
        ].join('\n'), 'Square').mapping).toEqual({ H: 'X', S: 'none', L: 'Y' });
    });

    it('renders the first member of a non-injective cell by default', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[0],[1]]',
        ].join('\n'), 'L(L)'));

        expect(Array.from(document.tensors.get('compose-step-2') ?? [])).toEqual([0, -1]);
    });

    it('derives multi-input display layers from the focused tensor membership', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[0],[1]]',
        ].join('\n'), 'L(L)'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const ctx = {
            viewer: {
                getState: () => ({ activeTensorId: 'compose-step-2' }),
                getTensorStatus: (tensorId: string) => {
                    const tensor = document.manifest.tensors.find((entry) => entry.id === tensorId)!;
                    return {
                        id: tensor.id,
                        name: tensor.name,
                        rank: tensor.shape.length,
                        shape: tensor.shape.slice(),
                        axisLabels: tensor.axisLabels ?? [],
                        dtype: tensor.dtype,
                        hasData: true,
                        valueRange: null,
                    };
                },
                getTensorView: (tensorId: string) => document.manifest.tensors.find((entry) => entry.id === tensorId)!.view,
            },
            state: {
                linearLayoutMultiInputState: { 'compose-step-2': 2 },
                linearLayoutSelectionMaps: new Map(),
            },
        } as never;

        expect(linearLayoutMultiInputModel(ctx, mapping)).toEqual({
            focusedTensorId: 'compose-step-2',
            size: 4,
            value: 2,
        });

        const display = linearLayoutDisplayModel(ctx, mapping);
        expect(Array.from(display.rootIndexes)).toEqual([2]);
        expect(display.displayedRootIndexByTensor.get('compose-root')).toEqual([null, null, 2, null]);
        expect(display.displayedRootIndexByTensor.get('compose-step-1')).toEqual([null, 2]);
        expect(display.displayedRootIndexByTensor.get('compose-step-2')).toEqual([2, null]);
    });

    it('keeps multi-input disabled by default until the slider leaves -1', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const ctx = {
            viewer: {
                getState: () => ({ activeTensorId: 'compose-step-1' }),
                getTensorStatus: (tensorId: string) => {
                    const tensor = document.manifest.tensors.find((entry) => entry.id === tensorId)!;
                    return {
                        id: tensor.id,
                        name: tensor.name,
                        rank: tensor.shape.length,
                        shape: tensor.shape.slice(),
                        axisLabels: tensor.axisLabels ?? [],
                        dtype: tensor.dtype,
                        hasData: true,
                        valueRange: null,
                    };
                },
                getTensorView: (tensorId: string) => document.manifest.tensors.find((entry) => entry.id === tensorId)!.view,
            },
            state: {
                linearLayoutMultiInputState: {},
                linearLayoutSelectionMaps: new Map(),
            },
        } as never;

        expect(linearLayoutMultiInputModel(ctx, mapping)).toEqual({
            focusedTensorId: 'compose-step-1',
            size: 2,
            value: -1,
        });
        expect(Array.from(linearLayoutDisplayModel(ctx, mapping).rootIndexes)).toEqual([0, 1, 2, 3]);
        expect(linearLayoutDisplayModel(ctx, mapping).ghostRootIndexesByTensor.get('compose-step-1')).toEqual([
            { coord: [0], rootIndex: 2, layer: 1 },
            { coord: [1], rootIndex: 3, layer: 1 },
        ]);
    });

    it('switches displayed members for a many-to-one tensor when multi-input changes', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const makeCtx = (value: number) => ({
            viewer: {
                getState: () => ({ activeTensorId: 'compose-step-1' }),
                getTensorStatus: (tensorId: string) => {
                    const tensor = document.manifest.tensors.find((entry) => entry.id === tensorId)!;
                    return {
                        id: tensor.id,
                        name: tensor.name,
                        rank: tensor.shape.length,
                        shape: tensor.shape.slice(),
                        axisLabels: tensor.axisLabels ?? [],
                        dtype: tensor.dtype,
                        hasData: true,
                        valueRange: null,
                    };
                },
                getTensorView: (tensorId: string) => document.manifest.tensors.find((entry) => entry.id === tensorId)!.view,
            },
            state: {
                linearLayoutMultiInputState: { 'compose-step-1': value },
                linearLayoutSelectionMaps: new Map(),
            },
        }) as never;

        expect(linearLayoutDisplayModel(makeCtx(0), mapping).displayedRootIndexByTensor.get('compose-step-1')).toEqual([0, 1]);
        expect(linearLayoutDisplayModel(makeCtx(1), mapping).displayedRootIndexByTensor.get('compose-step-1')).toEqual([2, 3]);
    });

    it('selects every input root in an output cell regardless of the displayed member', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const selectedRoots = rootIndexesForCoords(mapping, 'compose-step-1', [[0]]);

        expect(Array.from(selectedRoots)).toEqual([0, 2]);
        expect(coordsForRootIndexes(mapping, 'compose-root', selectedRoots, null)).toEqual([[0], [2]]);
    });

    it('lists every hovered input cell with text and color for many-to-one outputs', () => {
        const document = createComposeLayoutDocument(composeState([
            'L: [X] -> [X]',
            'X: [[1],[0]]',
        ].join('\n'), 'L'));
        const mapping = linearLayoutSelectionMapForMeta(document)!;
        const entries = linearLayoutHoverPopupEntries({
            state: {
                linearLayoutCellTextState: { X: true },
                linearLayoutState: composeState([
                    'L: [X] -> [X]',
                    'X: [[1],[0]]',
                ].join('\n'), 'L'),
            },
        } as never, {
            tensorId: 'compose-step-1',
            tensorName: 'L',
            viewCoord: [0],
            layoutCoord: [0],
            tensorCoord: [0],
            value: 0,
            colorSource: 'custom',
        }, mapping);

        expect(entries).toEqual([
            { text: 'X:0', color: 'rgb(255 0 0)' },
            { text: 'X:2', color: 'rgb(255 0 0)' },
        ]);
    });
});
