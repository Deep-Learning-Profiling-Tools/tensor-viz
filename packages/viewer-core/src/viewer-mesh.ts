import {
    BufferGeometry,
    Box3,
    BoxGeometry,
    EdgesGeometry,
    Group,
    InstancedBufferAttribute,
    InstancedMesh,
    Line,
    LineBasicMaterial,
    LineSegments,
    Matrix4,
    Mesh,
    MeshBasicMaterial,
    Sphere,
    Vector3,
} from 'three';
import { axisWorldKeyForMode, displayExtent, displayExtent2D, displayPositionForCoord, displayPositionForCoord2D, unravelIndex } from './layout.js';
import { layoutShape, mapLayoutCoordToViewCoord, mapViewCoordToLayoutCoord, mapViewCoordToTensorCoord, product } from './view.js';
import { BASE_COLOR, type MeshMeta } from './viewer-config.js';
import { axisFamilyColor, createLine, createTextLabel } from './viewer-graphics.js';
import { coordKey, numericValue, vectorFromTuple } from './viewer-utils.js';
import type { TensorRecord, TensorViewSpec, ViewerState, Vec3 } from './types.js';

type MeshViewerContext = {
    cubeGeometry: BoxGeometry;
    planeGeometry: BufferGeometry;
    state: ViewerState;
    tensorMeshes: Map<string, Group>;
    instanceShape(spec: TensorViewSpec): number[];
    layoutShape(spec: TensorViewSpec): number[];
    layoutAxisLabels(spec: TensorViewSpec): string[];
    layoutGapMultiple(): number;
    mapViewCoordToLayoutCoord(viewCoord: number[], spec: TensorViewSpec): number[];
    selectionStateAttribute(mesh: InstancedMesh): InstancedBufferAttribute | null;
    installSelectionPreviewShader(mesh: InstancedMesh): void;
    heatmapNormalizedValue(value: number, min: number, max: number): number;
    baseCellColor(
        tensor: TensorRecord,
        tensorCoord: number[],
        value: number,
        heatmapRange: { min: number; max: number } | null,
    ): { r: number; g: number; b: number };
    isSelectedCell(tensorId: string, tensorCoord: number[]): boolean;
    selectedColor(color: { clone(): { lerp(target: { r: number; g: number; b: number }, alpha: number): { r: number; g: number; b: number } } }): {
        r: number;
        g: number;
        b: number;
    };
    linearIndex(coord: number[], shape: number[]): number;
    clearHover(): void;
    requestRender(): void;
    emit(): void;
};

function populateFastMesh2D(
    viewer: MeshViewerContext,
    tensor: TensorRecord,
    mesh: InstancedMesh,
    instanceShape: number[],
    heatmapRange: { min: number; max: number } | null,
): boolean {
    const colorArray = mesh.instanceColor?.array as Float32Array | undefined;
    const selectionAttribute = viewer.selectionStateAttribute(mesh);
    const selectionState = selectionAttribute?.array as Float32Array | undefined;
    const isIdentityView = viewer.state.displayMode === '2d'
        && tensor.shape.length <= 2
        && tensor.view.sliceTokens.length === 0
        && tensor.view.tokens.length === tensor.shape.length
        && tensor.view.tokens.every((token, index) => token.kind === 'axis_group'
            && token.visible
            && token.axes.length === 1
            && token.axes[0] === index);
    if (!isIdentityView || tensor.customColors.size !== 0 || !colorArray) return false;

    // default 1d/2d views are affine grids, so write instance buffers directly.
    const matrixArray = mesh.instanceMatrix.array as Float32Array;
    const extent = displayExtent2D(instanceShape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
    const rowCount = instanceShape.length > 1 ? instanceShape[0] : 1;
    const columnCount = instanceShape.length > 1 ? instanceShape[1] : (instanceShape[0] ?? 1);
    const startX = tensor.offset[0] - extent.x / 2 + 0.5;
    const startY = tensor.offset[1] + extent.y / 2 - 0.5;
    const stepX = columnCount > 1 ? (extent.x - 1) / (columnCount - 1) : 0;
    const stepY = rowCount > 1 ? (extent.y - 1) / (rowCount - 1) : 0;
    let cellIndex = 0;

    for (let row = 0; row < rowCount; row += 1) {
        const y = startY - row * stepY;
        for (let column = 0; column < columnCount; column += 1) {
            const x = startX + column * stepX;
            const matrixOffset = cellIndex * 16;
            matrixArray[matrixOffset] = 1;
            matrixArray[matrixOffset + 1] = 0;
            matrixArray[matrixOffset + 2] = 0;
            matrixArray[matrixOffset + 3] = 0;
            matrixArray[matrixOffset + 4] = 0;
            matrixArray[matrixOffset + 5] = 1;
            matrixArray[matrixOffset + 6] = 0;
            matrixArray[matrixOffset + 7] = 0;
            matrixArray[matrixOffset + 8] = 0;
            matrixArray[matrixOffset + 9] = 0;
            matrixArray[matrixOffset + 10] = 1;
            matrixArray[matrixOffset + 11] = 0;
            matrixArray[matrixOffset + 12] = x;
            matrixArray[matrixOffset + 13] = y;
            matrixArray[matrixOffset + 14] = 0;
            matrixArray[matrixOffset + 15] = 1;

            const colorOffset = cellIndex * 3;
            if (heatmapRange) {
                const gray = viewer.heatmapNormalizedValue(
                    numericValue(tensor.data, cellIndex),
                    heatmapRange.min,
                    heatmapRange.max,
                );
                colorArray[colorOffset] = gray;
                colorArray[colorOffset + 1] = gray;
                colorArray[colorOffset + 2] = gray;
            } else {
                colorArray[colorOffset] = BASE_COLOR.r;
                colorArray[colorOffset + 1] = BASE_COLOR.g;
                colorArray[colorOffset + 2] = BASE_COLOR.b;
            }
            if (selectionState) {
                const tensorCoord = instanceShape.length > 1 ? [row, column] : [column];
                selectionState[cellIndex] = viewer.isSelectedCell(tensor.id, tensorCoord) ? 1 : 0;
            }
            cellIndex += 1;
        }
    }

    if (selectionAttribute) selectionAttribute.needsUpdate = true;
    return true;
}

function buildOutline(extent: Vector3, offset: Vec3): LineSegments {
    const outline = new LineSegments(
        new EdgesGeometry(new BoxGeometry(extent.x + 0.2, extent.y + 0.2, extent.z + 0.2)),
        new LineBasicMaterial({ color: '#334155' }),
    );
    outline.position.copy(vectorFromTuple(offset));
    return outline;
}

function buildOutline2D(extent: { x: number; y: number }, offset: Vec3): Line {
    const halfX = extent.x / 2;
    const halfY = extent.y / 2;
    const outline = createLine([
        new Vector3(-halfX, halfY, 0.02),
        new Vector3(halfX, halfY, 0.02),
        new Vector3(halfX, -halfY, 0.02),
        new Vector3(-halfX, -halfY, 0.02),
        new Vector3(-halfX, halfY, 0.02),
    ], '#334155');
    outline.position.copy(vectorFromTuple(offset));
    return outline;
}

function buildDimensionGuides2D(
    viewer: MeshViewerContext,
    shape: number[],
    offset: Vec3,
    labels: string[],
    guideOffset: number,
    linearStep: number,
    labelOffset: number,
    labelScale: number,
): Group {
    const group = new Group();
    const rank = shape.length;
    const families = new Map<number, number[]>();
    for (let axis = 0; axis < rank; axis += 1) {
        const key = axisWorldKeyForMode('2d', rank, axis, viewer.state.dimensionMappingScheme) as 0 | 1;
        const family = families.get(key) ?? [];
        family.push(axis);
        families.set(key, family);
    }

    shape.forEach((size, axis) => {
        const familyKey = axisWorldKeyForMode('2d', rank, axis, viewer.state.dimensionMappingScheme) as 0 | 1;
        const family = families.get(familyKey) ?? [axis];
        const familyPos = Math.max(0, family.indexOf(axis));
        const start = new Array(rank).fill(0);
        const end = start.slice();
        family.forEach((familyAxis) => {
            if (familyAxis >= axis) end[familyAxis] = Math.max(0, shape[familyAxis] - 1);
        });
        const startPos = displayPositionForCoord2D(start, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        const endPos = displayPositionForCoord2D(end, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        const delta = { x: endPos.x - startPos.x, y: endPos.y - startPos.y };
        const length = Math.hypot(delta.x, delta.y) || 1;
        const axisDir = { x: delta.x / length, y: delta.y / length };
        const extentStart = new Vector3(offset[0] + startPos.x - axisDir.x * 0.5, offset[1] + startPos.y - axisDir.y * 0.5, 0.02);
        const extentEnd = new Vector3(offset[0] + endPos.x + axisDir.x * 0.5, offset[1] + endPos.y + axisDir.y * 0.5, 0.02);
        const color = axisFamilyColor(familyKey as 0 | 1 | 2, familyPos, family.length);
        const dir = familyKey === 0 ? new Vector3(0, 1, 0) : new Vector3(-1, 0, 0);
        const reverseIndex = family.length - 1 - familyPos;
        const worldOffset = guideOffset + reverseIndex * linearStep;
        const startGuide = extentStart.clone().add(dir.clone().multiplyScalar(worldOffset));
        const endGuide = extentEnd.clone().add(dir.clone().multiplyScalar(worldOffset));
        group.add(createLine([extentStart, startGuide], color));
        group.add(createLine([extentEnd, endGuide], color));
        group.add(createLine([startGuide, endGuide], color));
        const label = createTextLabel(`${labels[axis] ?? 'X'}: ${size}`, color);
        label.position.copy(startGuide.clone().add(endGuide).multiplyScalar(0.5).add(dir.clone().multiplyScalar(labelOffset)));
        label.scale.setScalar(labelScale);
        group.add(label);
    });
    return group;
}

function buildDimensionGuides(viewer: MeshViewerContext, extent: Vector3, shape: number[], offset: Vec3, labels: string[]): Group {
    const group = new Group();
    const halfX = extent.x / 2;
    const halfY = extent.y / 2;
    const rank = shape.length;
    const families = new Map<number, number[]>();
    for (let axis = 0; axis < rank; axis += 1) {
        const key = axisWorldKeyForMode('3d', rank, axis, viewer.state.dimensionMappingScheme);
        const family = families.get(key) ?? [];
        family.push(axis);
        families.set(key, family);
    }
    const entries = shape.map((size, axis) => {
        const worldKey = axisWorldKeyForMode('3d', rank, axis, viewer.state.dimensionMappingScheme);
        const family = families.get(worldKey) ?? [axis];
        const familyPos = Math.max(0, family.indexOf(axis));
        const start = new Array(rank).fill(0);
        const end = start.slice();
        if (worldKey === 2) {
            family.forEach((familyAxis, index) => {
                end[familyAxis] = index >= familyPos ? Math.max(0, shape[familyAxis] - 1) : 0;
            });
        } else {
            family.forEach((familyAxis) => {
                if (familyAxis >= axis) end[familyAxis] = Math.max(0, shape[familyAxis] - 1);
            });
        }
        const startPos = displayPositionForCoord(start, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        const endPos = displayPositionForCoord(end, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        const axisDir = endPos.clone().sub(startPos);
        const direction = axisDir.lengthSq() > 1e-9
            ? axisDir.normalize()
            : worldKey === 1
                ? new Vector3(0, 1, 0)
                : worldKey === 2
                    ? new Vector3(0, 0, 1)
                    : new Vector3(1, 0, 0);
        return {
            worldKey,
            color: axisFamilyColor(worldKey, familyPos, family.length),
            label: `${labels[axis] ?? 'X'}: ${size}`,
            start: startPos.add(direction.clone().multiplyScalar(-0.5)),
            end: endPos.add(direction.clone().multiplyScalar(0.5)),
        };
    });
    const offsetBase = 2.25;
    const linearStep = 1.85;
    ([0, 1, 2] as const).forEach((worldKey) => {
        const familyEntries = entries.filter((entry) => entry.worldKey === worldKey);
        familyEntries.forEach((entry, index) => {
            const reverseIndex = familyEntries.length - 1 - index;
            const midpoint = entry.start.clone().add(entry.end).multiplyScalar(0.5);
            const extensionDirection = worldKey === 0
                ? new Vector3(0, midpoint.y >= 0 ? 1 : -1, 0)
                : worldKey === 1
                    ? new Vector3(midpoint.x >= 0 ? 1 : -1, 0, 0)
                    : new Vector3(midpoint.x >= 0 ? 1 : -1, midpoint.y >= 0 ? 1 : -1, 0).normalize();
            const boundaryOffset = worldKey === 0
                ? Math.max(0, halfY - Math.abs(midpoint.y))
                : worldKey === 1
                    ? Math.max(0, halfX - Math.abs(midpoint.x))
                    : Math.max(0, Math.max(halfX - Math.abs(midpoint.x), halfY - Math.abs(midpoint.y)));
            const guideOffset = extensionDirection.clone().multiplyScalar(boundaryOffset + offsetBase + reverseIndex * linearStep);
            const startGuide = entry.start.clone().add(guideOffset);
            const endGuide = entry.end.clone().add(guideOffset);
            group.add(createLine([entry.start.clone(), startGuide], entry.color));
            group.add(createLine([entry.end.clone(), endGuide], entry.color));
            group.add(createLine([startGuide, endGuide], entry.color));
            const label = createTextLabel(entry.label, entry.color);
            label.position.copy(startGuide.clone().add(endGuide).multiplyScalar(0.5).add(extensionDirection.clone().multiplyScalar(0.75)));
            label.renderOrder = 10_000;
            group.add(label);
        });
    });

    group.position.copy(vectorFromTuple(offset));
    return group;
}

export function buildTensorGroup(viewer: MeshViewerContext, tensor: TensorRecord): Group {
    const group = new Group();
    const instanceShape = viewer.instanceShape(tensor.view);
    const shape = viewer.layoutShape(tensor.view);
    const labels = viewer.layoutAxisLabels(tensor.view);
    const count = product(instanceShape);
    const geometry = viewer.state.displayMode === '2d' ? viewer.planeGeometry.clone() : viewer.cubeGeometry;
    const mesh = new InstancedMesh(
        geometry,
        new MeshBasicMaterial({ color: 0xffffff, vertexColors: true, toneMapped: false }),
        count,
    );
    mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(count * 3), 3);
    if (viewer.state.displayMode === '2d') {
        mesh.geometry.setAttribute('selectionState', new InstancedBufferAttribute(new Float32Array(count), 1));
        viewer.installSelectionPreviewShader(mesh);
    }
    const outlineExtent2D = displayExtent2D(shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
    const outlineSpan2D = Math.max(outlineExtent2D.x, outlineExtent2D.y);
    const baseOutlineLabelScale2D = Math.max(1.25, Math.min(10, outlineSpan2D * 0.05));
    const guideLabelScale2D = baseOutlineLabelScale2D / 5;
    const guideStartOffset2D = Math.max(1.15, guideLabelScale2D * 2.5);
    const guideLevelStep2D = Math.max(0.75, guideLabelScale2D * 3.5);
    const guideLabelOffset2D = Math.max(0.3, guideLabelScale2D * 1.2);
    const tensorNameScale2D = (baseOutlineLabelScale2D * 1.25) / 2;
    const heatmapRange = viewer.state.heatmap ? tensor.valueRange : null;
    const selectionAttribute = viewer.selectionStateAttribute(mesh);
    const selectionState = selectionAttribute?.array as Float32Array | undefined;
    if (!populateFastMesh2D(viewer, tensor, mesh, instanceShape, heatmapRange)) {
        const matrix = new Matrix4();
        const offset = vectorFromTuple(tensor.offset);
        for (let index = 0; index < count; index += 1) {
            const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
            const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
            const layoutCoord = viewer.mapViewCoordToLayoutCoord(viewCoord, tensor.view);
            const tensorLinear = viewer.linearIndex(tensorCoord, tensor.shape);
            const value = numericValue(tensor.data, tensorLinear);
            const position = viewer.state.displayMode === '2d'
                ? (() => {
                    const flat = displayPositionForCoord2D(layoutCoord, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
                    return new Vector3(tensor.offset[0] + flat.x, tensor.offset[1] + flat.y, 0);
                })()
                : displayPositionForCoord(layoutCoord, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme).add(offset);
            matrix.makeTranslation(position.x, position.y, position.z);
            mesh.setMatrixAt(index, matrix);
            const baseColor = viewer.baseCellColor(tensor, tensorCoord, value, heatmapRange);
            mesh.setColorAt(index, selectionState ? baseColor as never : viewer.isSelectedCell(tensor.id, tensorCoord) ? viewer.selectedColor(baseColor as never) as never : baseColor as never);
            if (selectionState) selectionState[index] = viewer.isSelectedCell(tensor.id, tensorCoord) ? 1 : 0;
        }
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (selectionAttribute) selectionAttribute.needsUpdate = true;
    if (viewer.state.displayMode === '2d') {
        const halfX = outlineExtent2D.x / 2;
        const halfY = outlineExtent2D.y / 2;
        const center = new Vector3(tensor.offset[0], tensor.offset[1], 0);
        mesh.boundingBox = new Box3(
            new Vector3(center.x - halfX, center.y - halfY, 0),
            new Vector3(center.x + halfX, center.y + halfY, 0),
        );
        mesh.boundingSphere = new Sphere(center, Math.hypot(halfX, halfY));
    } else {
        const outlineExtent = displayExtent(shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        const halfExtent = outlineExtent.clone().multiplyScalar(0.5);
        const center = vectorFromTuple(tensor.offset);
        mesh.boundingBox = new Box3(center.clone().sub(halfExtent), center.clone().add(halfExtent));
        mesh.boundingSphere = new Sphere(center, halfExtent.length());
    }
    mesh.material.needsUpdate = true;
    mesh.userData.meta = { tensorId: tensor.id, instanceShape } satisfies MeshMeta;
    group.add(mesh);
    if (viewer.state.displayMode === '2d') {
        group.add(buildOutline2D(outlineExtent2D, tensor.offset));
        const showDimensionGuides = viewer.state.showDimensionLines && labels.length > 0;
        if (showDimensionGuides) {
            group.add(buildDimensionGuides2D(
                viewer,
                shape,
                tensor.offset,
                labels,
                guideStartOffset2D,
                guideLevelStep2D,
                guideLabelOffset2D,
                guideLabelScale2D,
            ));
        }
        if (viewer.state.showTensorNames) {
            const topGuideCount = shape.reduce((count, _size, axis) => (
                count + Number(axisWorldKeyForMode('2d', shape.length, axis, viewer.state.dimensionMappingScheme) === 0)
            ), 0);
            const nameLabel = createTextLabel(tensor.name || tensor.id, '#0f172a');
            const nameMesh = nameLabel.children[0];
            const nameGeometry = nameMesh instanceof Mesh ? nameMesh.geometry : null;
            nameGeometry?.computeBoundingBox();
            const nameWidth = nameGeometry?.boundingBox?.getSize(new Vector3()).x ?? 0;
            const fittedTensorNameScale2D = nameWidth > 0
                ? Math.min(tensorNameScale2D, (outlineExtent2D.x * 0.95) / nameWidth)
                : tensorNameScale2D;
            const guideClearance = showDimensionGuides
                ? guideStartOffset2D
                    + Math.max(0, topGuideCount - 1) * guideLevelStep2D
                    + guideLabelOffset2D
                    + fittedTensorNameScale2D * 1.5
                : fittedTensorNameScale2D * 1.75;
            nameLabel.position.set(tensor.offset[0], tensor.offset[1] + outlineExtent2D.y / 2 + guideClearance, 0.02);
            nameLabel.scale.setScalar(fittedTensorNameScale2D);
            group.add(nameLabel);
        }
    } else {
        const outlineExtent = displayExtent(shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        group.add(buildOutline(outlineExtent, tensor.offset));
        if (viewer.state.showDimensionLines && labels.length > 0) group.add(buildDimensionGuides(viewer, outlineExtent, shape, tensor.offset, labels));
    }
    return group;
}

export function updateSliceMesh(viewer: MeshViewerContext, tensor: TensorRecord, previousView: TensorViewSpec): boolean {
    if (previousView.canonical !== tensor.view.canonical) return false;
    if (!previousView.hiddenIndices.some((value, index) => value !== tensor.view.hiddenIndices[index])) return false;
    if (previousView.viewShape.length !== tensor.view.viewShape.length
        || previousView.viewShape.some((size, index) => size !== tensor.view.viewShape[index])) return false;
    const shape = viewer.layoutShape(tensor.view);
    const previousShape = layoutShape(previousView, viewer.state.collapseHiddenAxes);
    if (previousShape.length !== shape.length
        || previousShape.some((size, index) => size !== shape[index])) return false;

    const group = viewer.tensorMeshes.get(tensor.id);
    const mesh = group?.children[0];
    const colorArray = mesh instanceof InstancedMesh ? mesh.instanceColor?.array as Float32Array | undefined : undefined;
    const selectionAttribute = mesh instanceof InstancedMesh ? viewer.selectionStateAttribute(mesh) : null;
    const selectionState = selectionAttribute?.array as Float32Array | undefined;
    if (!(mesh instanceof InstancedMesh) || !colorArray) return false;

    const instanceShape = viewer.instanceShape(tensor.view);
    const anchorViewCoord = tensor.view.viewShape.length === 0 ? [] : new Array(tensor.view.viewShape.length).fill(0);
    const previousAnchor = mapViewCoordToLayoutCoord(anchorViewCoord, previousView, viewer.state.collapseHiddenAxes);
    const nextAnchor = viewer.mapViewCoordToLayoutCoord(anchorViewCoord, tensor.view);
    if (viewer.state.displayMode === '2d') {
        const previousPosition = displayPositionForCoord2D(previousAnchor, previousShape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        const nextPosition = displayPositionForCoord2D(nextAnchor, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        mesh.position.x += nextPosition.x - previousPosition.x;
        mesh.position.y += nextPosition.y - previousPosition.y;
        mesh.position.z = 0;
    } else {
        const previousPosition = displayPositionForCoord(previousAnchor, previousShape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        const nextPosition = displayPositionForCoord(nextAnchor, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
        mesh.position.add(nextPosition.sub(previousPosition));
    }

    const heatmapRange = viewer.state.heatmap ? tensor.valueRange : null;
    const count = product(instanceShape);
    for (let index = 0; index < count; index += 1) {
        const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
        const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
        const value = numericValue(tensor.data, viewer.linearIndex(tensorCoord, tensor.shape));
        const colorOffset = index * 3;
        const baseColor = viewer.baseCellColor(tensor, tensorCoord, value, heatmapRange);
        const color = selectionState
            ? baseColor
            : viewer.isSelectedCell(tensor.id, tensorCoord) ? viewer.selectedColor(baseColor as never) as never : baseColor;
        colorArray[colorOffset] = color.r;
        colorArray[colorOffset + 1] = color.g;
        colorArray[colorOffset + 2] = color.b;
        if (selectionState) selectionState[index] = viewer.isSelectedCell(tensor.id, tensorCoord) ? 1 : 0;
    }

    if (viewer.state.hover?.tensorId === tensor.id || viewer.state.lastHover?.tensorId === tensor.id) viewer.clearHover();
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    if (selectionAttribute) selectionAttribute.needsUpdate = true;
    mesh.updateMatrixWorld(true);
    viewer.requestRender();
    viewer.emit();
    return true;
}
