import {
    BufferGeometry,
    Box3,
    BoxGeometry,
    Color,
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

// viewer-mesh.ts owns renderable geometry, not viewer state.
// TensorViewer passes a narrow context so this file can build groups, instance
// buffers, labels, outlines, and dimension guides without reaching into private
// class fields.
// coordinate transforms still live in layout/view modules; mesh code should
// call those helpers instead of re-deriving tensor-view semantics.
// if a feature changes what cells exist, update the view/model layer first.
// if a feature changes how existing cells are drawn, this is usually the right
// file.

const MULTI_INPUT_Z_STEP = 1.15;

/**
 * Rendering boundary that `viewer-mesh` needs from `TensorViewer` to build tensor geometry.
 *
 * The context keeps mesh construction downstream of viewer state and coordinate
 * math while allowing tests to provide a small fake instead of constructing a
 * WebGL renderer. Implementations supply shared geometries, tensor visibility
 * and selection queries, color mapping, layout conversion, render invalidation,
 * and event emission hooks.
 *
 * @example
 * const context = {
 *   cubeGeometry: new BoxGeometry(1, 1, 1),
 *   planeGeometry: new BufferGeometry(),
 *   state: viewerState,
 *   tensorMeshes: new Map(),
 *   instanceShape: spec => spec.shape,
 *   layoutShape: spec => spec.shape,
 *   layoutAxisLabels: spec => spec.axes.map(String),
 *   layoutGapMultiple: () => 1,
 *   mapViewCoordToLayoutCoord: coord => coord,
 *   selectionStateAttribute: () => null,
 *   installSelectionPreviewShader: () => undefined,
 *   heatmapNormalizedValue: (value, min, max) => (value - min) / (max - min),
 *   baseCellColor: () => ({ r: 1, g: 1, b: 1 }),
 *   tensorCoordVisible: () => true,
 *   isSelectedCell: () => false,
 *   selectedColor: color => color.clone().lerp({ r: 1, g: 1, b: 0 }, 0.35),
 *   linearIndex: (coord, shape) => coord.reduce((index, value, axis) => index * shape[axis] + value, 0),
 *   clearHover: () => undefined,
 *   requestRender: () => undefined,
 *   emit: () => undefined,
 * } satisfies MeshViewerContext;
 */
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
    tensorCoordVisible(tensor: TensorRecord, tensorCoord: number[]): boolean;
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

/**
 * Writes instance transforms, base or heatmap colors, and optional selection flags for the conservative 2D identity-view mesh path.
 *
 * @param viewer - Mesh viewer context whose state is in 2D mode and whose helpers provide layout spacing, heatmap normalization, and selected-cell lookup.
 * @param tensor - Tensor record being rendered; the fast path only accepts unsliced, unpermuted identity axis-group views with no custom colors or hidden coordinates.
 * @param mesh - Instanced mesh whose instanceMatrix buffer is writable and whose instanceColor buffer must already exist for the fast path to run.
 * @param instanceShape - Rendered tensor shape for the identity 1D or 2D grid; vectors are written as a single row and matrices as row/column cells.
 * @param heatmapRange - Numeric min/max range used to convert tensor values to grayscale, or null to write the viewer base color into every instance.
 * @returns True when the mesh buffers were populated directly; false when the tensor/view/mesh requires the caller to fall back to the generic coordinate-mapping renderer.
 * @noThrows The helper only performs guard checks and writes numeric buffer entries; unsupported views return false instead of raising an error.
 * @example
 * const usedFastPath = populateFastMesh2D(viewer, tensor, mesh, [2, 3], { min: 0, max: 1 });
 * expect(usedFastPath).toBe(true);
 * expect(mesh.instanceMatrix.array[12]).toBeCloseTo(tensor.offset[0] - extent.x / 2 + 0.5);
 * expect(mesh.instanceColor?.array[0]).toBe(viewer.heatmapNormalizedValue(tensor.data[0], 0, 1));
 *
 * @example
 * tensor.view.sliceTokens = [{ axis: 0, index: 1 }];
 * expect(populateFastMesh2D(viewer, tensor, mesh, [3], null)).toBe(false);
 */
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
    // the fast path is intentionally conservative.  custom colors, hidden
    // coords, slices, and permutations need the generic coordinate mapping path.
    if (!isIdentityView || tensor.customColors.size !== 0 || tensor.visibleCoords || !colorArray) return false;

    // default 1d/2d views are affine grids, so write instance buffers directly.
    const matrixArray = mesh.instanceMatrix.array as Float32Array;
    const extent = displayExtent2D(instanceShape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme);
    // one-dimensional tensors render as a single row so the same direct buffer
    // writer can handle both vector and matrix defaults.
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
                // the fast path only handles identity views, so instance index
                // is already the canonical tensor coordinate.
                const tensorCoord = instanceShape.length > 1 ? [row, column] : [column];
                selectionState[cellIndex] = viewer.isSelectedCell(tensor.id, tensorCoord) ? 1 : 0;
            }
            cellIndex += 1;
        }
    }

    if (selectionAttribute) selectionAttribute.needsUpdate = true;
    return true;
}

/**
 * Creates the 3D box-edge outline that frames an instanced tensor block in the scene.
 *
 * @param extent - Width, height, and depth of the rendered tensor block before the outline padding is added.
 * @param offset - Tensor world-space origin used to position the outline around the block.
 * @returns A LineSegments object with padded box-edge geometry, the standard outline material color, and its position copied from the tensor offset.
 * @noThrows The helper constructs Three.js geometry from the supplied numeric extent and offset and does not branch into validation or I/O that would throw under normal viewer inputs.
 * @example
 * const outline = buildOutline(new Vector3(4, 2, 3), [10, 20, 30]);
 * expect(outline).toBeInstanceOf(LineSegments);
 * expect(outline.position.toArray()).toEqual([10, 20, 30]);
 */
function buildOutline(extent: Vector3, offset: Vec3): LineSegments {
    // 3d outlines use box edges so depth sorting and camera rotation stay
    // consistent with the instanced cube cells.
    const outline = new LineSegments(
        new EdgesGeometry(new BoxGeometry(extent.x + 0.2, extent.y + 0.2, extent.z + 0.2)),
        new LineBasicMaterial({ color: '#334155' }),
    );
    outline.position.copy(vectorFromTuple(offset));
    return outline;
}

/**
 * Creates the flat rectangular outline used to frame a 2D tensor grid and SVG-export-compatible geometry.
 *
 * @param extent - Rendered grid width and height in world units.
 * @param offset - Tensor world-space origin where the rectangle outline is placed.
 * @returns A closed Line tracing the grid perimeter at z=0.02, positioned at the supplied tensor offset.
 * @noThrows The helper derives five rectangle vertices from numeric dimensions and copies the provided offset without performing validation or external operations.
 * @example
 * const outline = buildOutline2D({ x: 6, y: 4 }, [1, 2, 0]);
 * expect(outline).toBeInstanceOf(Line);
 * expect(outline.position.toArray()).toEqual([1, 2, 0]);
 */
function buildOutline2D(extent: { x: number; y: number }, offset: Vec3): Line {
    // 2d outlines are lines instead of box edges because SVG export mirrors this
    // flat geometry path.
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

/**
 * Builds the 2D dimension-line overlay that shows each tensor axis extent and label beside the rendered grid.
 *
 * @param viewer - Mesh viewer context whose dimension-mapping scheme and layout gap determine each axis family and display coordinate.
 * @param shape - Tensor axis sizes used to compute guide extents and label text such as "Rows: 4".
 * @param offset - Tensor world-space origin added to every guide endpoint and label position.
 * @param labels - Axis labels from the tensor metadata; missing entries fall back to "X".
 * @param guideOffset - Distance from the tensor edge to the first guide line in world units.
 * @param linearStep - Extra spacing between stacked guides when multiple logical axes share the same rendered x or y family.
 * @param labelOffset - Distance from each guide segment to its text label.
 * @param labelScale - Scalar applied to each generated text label mesh.
 * @returns A Group containing, for each tensor axis, two connector lines, one extent line, and one scaled text label positioned according to the 2D mapping scheme.
 * @noThrows The helper computes guide geometry from in-memory shape, labels, and viewer layout state; absent labels are handled with a fallback instead of throwing.
 * @example
 * const guides = buildDimensionGuides2D(viewer, [2, 3], [0, 0, 0], ['Rows', 'Columns'], 0.8, 0.35, 0.2, 0.1);
 * expect(guides).toBeInstanceOf(Group);
 * expect(guides.children).toHaveLength(8);
 * expect(guides.children.some((child) => child.type.includes('Line'))).toBe(true);
 */
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
    // guide offsets are grouped by rendered world family so labels do not stack
    // on top of each other when several logical axes share x or y.
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
        // later axes in the same rendered family extend from the current axis so
        // stacked labels show nested block structure rather than full extents.
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

/**
 * Creates the 3D dimension-line overlay for a tensor block.
 *
 * The returned group contains extension segments, dimension spans, and text labels for each layout axis,
 * offset just outside the supplied tensor outline and then translated to the tensor's world offset.
 *
 * @param viewer - Mesh rendering context that supplies the current dimension mapping scheme and layout gap used to map tensor axes into world X/Y/Z families.
 * @param extent - World-space size of the tensor outline that the guides should sit outside.
 * @param shape - Layout-axis sizes for the tensor view; each entry produces one labeled guide.
 * @param offset - Tensor world offset applied to the completed guide group.
 * @param labels - Axis labels matching `shape`; missing entries fall back to `X` in the rendered label text.
 * @returns Three.js group containing the guide lines and text labels, positioned at `offset`, ready to add beside the tensor outline.
 * @noThrows Uses normalized viewer layout state and creates Three.js objects without explicit validation; unsupported or degenerate axis spans fall back to unit X/Y/Z guide directions instead of throwing.
 * @example
 * const guides = buildDimensionGuides(viewer, new Vector3(4, 3, 2), [2, 3, 4], [10, 0, 0], ['batch', 'row', 'col']);
 * expect(guides.position.toArray()).toEqual([10, 0, 0]);
 * expect(guides.children.length).toBeGreaterThan(0);
 * expect(guides.children.some((child) => child.type === 'Sprite')).toBe(true);
 */
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
        // z-family guides need a different endpoint rule because deeper axes
        // move away from the camera instead of across the screen.
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

/**
 * Builds all Three.js objects needed to render one tensor in the current viewer mode.
 *
 * The group starts with an instanced cell mesh and may also include selection attributes, ghost-layer cells,
 * tensor outlines, dimension guides, and tensor-name labels according to viewer state.
 *
 * @param viewer - Mesh rendering context that provides geometry templates, viewer state, layout helpers, coloring, selection lookup, and render hooks.
 * @param tensor - Normalized tensor record containing id, data, shape, view, offset, value range, and optional ghost-layer metadata for the tensor being rendered.
 * @returns Three.js group for the tensor; callers add it to the scene and store it by tensor id for picking, updates, and later removal.
 * @noThrows Expects the tensor record and viewer context to have already been normalized by the viewer model; incompatible visibility or fast-path cases are represented in mesh buffers rather than by explicit throws.
 * @example
 * const group = buildTensorGroup(meshContext('0,1'), tensorRecord({ id: 'weights', shape: [2, 2] }));
 * const mesh = group.children.find((child): child is InstancedMesh => child instanceof InstancedMesh);
 * expect(mesh?.count).toBe(4);
 * expect(mesh?.userData.meta).toMatchObject({ tensorId: 'weights', instanceShape: [2, 2] });
 */
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
            if (!viewer.tensorCoordVisible(tensor, tensorCoord)) {
                matrix.makeScale(0, 0, 0);
                mesh.setMatrixAt(index, matrix);
                mesh.setColorAt(index, BASE_COLOR as never);
                if (selectionState) selectionState[index] = 0;
                continue;
            }
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
    if (viewer.state.displayMode === '3d' && tensor.ghostLayers?.length) {
        // ghost layers show extra roots for many-to-one cells.  They are only
        // useful in 3d because 2d uses popup text instead of stacked geometry.
        const ghostMesh = new InstancedMesh(
            viewer.cubeGeometry,
            new MeshBasicMaterial({ color: 0xffffff, vertexColors: true, toneMapped: false }),
            tensor.ghostLayers.length,
        );
        ghostMesh.instanceColor = new InstancedBufferAttribute(new Float32Array(tensor.ghostLayers.length * 3), 3);
        const matrix = new Matrix4();
        const offset = vectorFromTuple(tensor.offset);
        tensor.ghostLayers.forEach((layer, index) => {
            const position = displayPositionForCoord(layer.coord, shape, viewer.layoutGapMultiple(), viewer.state.dimensionMappingScheme)
                .add(offset)
                .add(new Vector3(0, 0, -layer.layer * MULTI_INPUT_Z_STEP));
            matrix.makeTranslation(position.x, position.y, position.z);
            ghostMesh.setMatrixAt(index, matrix);
            ghostMesh.setColorAt(index, new Color(layer.color[0] / 255, layer.color[1] / 255, layer.color[2] / 255));
        });
        ghostMesh.instanceMatrix.needsUpdate = true;
        if (ghostMesh.instanceColor) ghostMesh.instanceColor.needsUpdate = true;
        ghostMesh.material.needsUpdate = true;
        group.add(ghostMesh);
    }
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
            // account for guide levels before placing tensor names or long labels
            // overlap dimension guides in dense high-rank views.
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
        if (viewer.state.showTensorNames) {
            const nameLabel = createTextLabel(tensor.name || tensor.id, '#0f172a');
            const nameMesh = nameLabel.children[0];
            const nameGeometry = nameMesh instanceof Mesh ? nameMesh.geometry : null;
            nameGeometry?.computeBoundingBox();
            const nameWidth = nameGeometry?.boundingBox?.getSize(new Vector3()).x ?? 0;
            const tensorNameScale3D = Math.max(0.45, Math.min(1.8, Math.max(outlineExtent.x, outlineExtent.y) * 0.08));
            const fittedTensorNameScale3D = nameWidth > 0
                ? Math.min(tensorNameScale3D, (outlineExtent.x * 0.95) / nameWidth)
                : tensorNameScale3D;
            nameLabel.position.set(
                tensor.offset[0],
                tensor.offset[1] + outlineExtent.y / 2 + fittedTensorNameScale3D * 1.75,
                tensor.offset[2],
            );
            nameLabel.scale.setScalar(fittedTensorNameScale3D);
            group.add(nameLabel);
        }
    }
    return group;
}

/**
 * Recolors and repositions an existing tensor instanced mesh after a slice-only view change.
 *
 * The fast path is used only when the canonical view and layout shape are unchanged and the hidden-index
 * values changed; otherwise callers should rebuild the tensor group.
 *
 * @param viewer - Mesh rendering context that owns the existing tensor mesh map, layout state, selection state, hover state, and render/event hooks.
 * @param tensor - Tensor record after the slice change; its id is used to find the existing mesh and its current view/data drive the new colors.
 * @param previousView - Tensor view before the slice change, used to verify fast-path eligibility and compute the anchor-position delta.
 * @returns `true` when the existing instanced mesh was updated in place and render/event notifications were sent; `false` when the view change or mesh state requires a full rebuild.
 * @noThrows Ineligible view changes, missing tensor groups, and non-instanced meshes are returned as `false`; the function performs no explicit error throwing for those states.
 * @example
 * const updated = updateSliceMesh(viewer, tensorWithHiddenIndexChanged, previousView);
 * expect(updated).toBe(true);
 * expect(viewer.requestRender).toHaveBeenCalled();
 * expect(viewer.emit).toHaveBeenCalled();
 *
 * const needsRebuild = updateSliceMesh(viewer, tensorWithDifferentViewShape, previousView);
 * expect(needsRebuild).toBe(false);
 */
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
    // slice-only edits keep the same instance count, so move the mesh by the
    // anchor delta instead of rebuilding every instance and label.
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
        if (!viewer.tensorCoordVisible(tensor, tensorCoord)) {
            const colorOffset = index * 3;
            colorArray[colorOffset] = BASE_COLOR.r;
            colorArray[colorOffset + 1] = BASE_COLOR.g;
            colorArray[colorOffset + 2] = BASE_COLOR.b;
            if (selectionState) selectionState[index] = 0;
            continue;
        }
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
