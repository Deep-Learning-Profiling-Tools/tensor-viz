import { describe, expect, it } from 'vitest';
import {
    BoxGeometry,
    BufferGeometry,
    Color,
    InstancedBufferAttribute,
    InstancedMesh,
    Matrix4,
    PlaneGeometry,
    Vector3,
} from 'three';
import type { TensorRecord, TensorViewSpec, ViewerState } from './types.js';
import { layoutShape, layoutAxisLabels, mapViewCoordToLayoutCoord, parseTensorView, product } from './view.js';
import { buildTensorGroup } from './viewer-mesh.js';

function tensorView(shape: number[]): TensorViewSpec {
    const parsed = parseTensorView(shape, '');
    if (!parsed.ok) throw new Error(parsed.errors.join('\n'));
    return parsed.spec;
}

function viewerState(): ViewerState {
    return {
        displayMode: '2d',
        interactionMode: 'pan',
        heatmap: false,
        dimensionBlockGapMultiple: 3,
        displayGaps: false,
        logScale: false,
        collapseHiddenAxes: false,
        dimensionMappingScheme: 'z-order',
        showDimensionLines: false,
        showTensorNames: false,
        showInspectorPanel: false,
        showSelectionPanel: true,
        showHoverDetailsPanel: false,
        activeTensorId: null,
        hover: null,
        lastHover: null,
    };
}

function tensorRecord(): TensorRecord {
    const shape = [2, 2];
    return {
        id: 'tensor-1',
        name: 'Tensor',
        shape,
        axisLabels: ['Y', 'X'],
        dtype: 'float32',
        data: new Float32Array([1, 2, 3, 4]),
        hasData: true,
        valueRange: { min: 1, max: 4 },
        offset: [0, 0, 0],
        view: tensorView(shape),
        customColors: new Map(),
        markerCoords: null,
        visibleCoords: null,
        cellLabels: null,
        ghostLayers: null,
        autoOffset: true,
    };
}

function meshContext(selectedKey = ''): Parameters<typeof buildTensorGroup>[0] {
    const state = viewerState();
    return {
        cubeGeometry: new BoxGeometry(1, 1, 1),
        planeGeometry: new PlaneGeometry(1, 1) as BufferGeometry,
        state,
        tensorMeshes: new Map(),
        instanceShape: (spec) => (spec.viewShape.length === 0 ? [1] : spec.viewShape),
        layoutShape: (spec) => layoutShape(spec, state.collapseHiddenAxes),
        layoutAxisLabels,
        layoutGapMultiple: () => 0,
        mapViewCoordToLayoutCoord,
        selectionStateAttribute: (mesh) => mesh.geometry.getAttribute('selectionState') as InstancedBufferAttribute | null,
        installSelectionPreviewShader: () => {},
        heatmapNormalizedValue: () => 0,
        baseCellColor: () => new Color(1, 1, 1),
        tensorCoordVisible: () => true,
        isSelectedCell: (_tensorId, coord) => coord.join(',') === selectedKey,
        selectedColor: (color) => color.clone().lerp(new Color(1, 1, 1), 0.35),
        linearIndex: (coord, shape) => coord.reduce((index, value, axis) => (index * shape[axis]!) + value, 0),
        clearHover: () => {},
        requestRender: () => {},
        emit: () => {},
    };
}

describe('buildTensorGroup', () => {
    it('builds stable 2d instance buffers for the default tensor view', () => {
        const group = buildTensorGroup(meshContext('0,1'), tensorRecord());
        const mesh = group.children.find((child): child is InstancedMesh => child instanceof InstancedMesh)!;
        const selection = mesh.geometry.getAttribute('selectionState') as InstancedBufferAttribute;
        const matrix = new Matrix4();

        mesh.getMatrixAt(1, matrix);
        const position = new Vector3().setFromMatrixPosition(matrix);

        expect(mesh.count).toBe(product([2, 2]));
        expect(group.children).toHaveLength(2);
        expect(Array.from(selection.array)).toEqual([0, 1, 0, 0]);
        expect(position.x).toBeCloseTo(0.5);
        expect(position.y).toBeCloseTo(0.5);
        expect(mesh.boundingBox?.min.x).toBeCloseTo(-1);
        expect(mesh.boundingBox?.max.x).toBeCloseTo(1);
    });
});
