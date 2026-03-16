import { Vector3 } from 'three';

const CELL_SIZE = 1;
const GAP = 0.15;
export const DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE = 3;

type Extent3 = {
    x: number;
    y: number;
    z: number;
};

type Extent2 = {
    x: number;
    y: number;
};

type CoordHit2D = {
    coord: number[];
    position: {
        x: number;
        y: number;
    };
};

const CELL_HIT_EPSILON = 1e-6;

function normalizeDisplayShape(shape: number[]): number[] {
    return shape.length === 0 ? [1] : shape.map((value) => Math.max(1, value));
}

function normalizeDimensionBlockGapMultiple(value: number): number {
    return Number.isFinite(value) ? Math.max(0, value) : DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
}

function levelGap(level: number, dimensionBlockGapMultiple: number): number {
    if (dimensionBlockGapMultiple <= 0) return 0;
    return GAP * Math.pow(dimensionBlockGapMultiple, Math.max(0, level));
}

function depthHeightWidth(shape: number[]): { depth: number; height: number; width: number } {
    if (shape.length === 1) return { depth: 1, height: 1, width: shape[0] };
    if (shape.length === 2) return { depth: 1, height: shape[0], width: shape[1] };
    return {
        depth: shape[shape.length - 3],
        height: shape[shape.length - 2],
        width: shape[shape.length - 1],
    };
}

function baseGridExtent3D(shapeInput: number[], cellExtent: Extent3, level: number, dimensionBlockGapMultiple: number): Extent3 {
    const shape = normalizeDisplayShape(shapeInput);
    const { depth, height, width } = depthHeightWidth(shape);
    const gap = levelGap(level, dimensionBlockGapMultiple);
    return {
        x: (width - 1) * (cellExtent.x + gap) + cellExtent.x,
        y: (height - 1) * (cellExtent.y + gap) + cellExtent.y,
        z: (depth - 1) * (cellExtent.z + gap) + cellExtent.z,
    };
}

function recursiveExtent3D(shapeInput: number[], cellExtent: Extent3, level: number, dimensionBlockGapMultiple: number): Extent3 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 3) return baseGridExtent3D(shape, cellExtent, level, dimensionBlockGapMultiple);
    const split = shape.length - 3;
    const innerExtent = recursiveExtent3D(shape.slice(split), cellExtent, level, dimensionBlockGapMultiple);
    return recursiveExtent3D(shape.slice(0, split), innerExtent, level + 1, dimensionBlockGapMultiple);
}

function baseGridPosition3D(coord: number[], shapeInput: number[], cellExtent: Extent3, level: number, dimensionBlockGapMultiple: number): Vector3 {
    const shape = normalizeDisplayShape(shapeInput);
    const { depth, height, width } = depthHeightWidth(shape);
    const gap = levelGap(level, dimensionBlockGapMultiple);
    const stepX = cellExtent.x + gap;
    const stepY = cellExtent.y + gap;
    const stepZ = cellExtent.z + gap;
    const x = coord[shape.length - 1] ?? 0;
    const y = shape.length >= 2 ? (coord[shape.length - 2] ?? 0) : 0;
    const z = shape.length >= 3 ? (coord[shape.length - 3] ?? 0) : 0;
    return new Vector3(
        x * stepX - ((width - 1) * stepX) / 2,
        -y * stepY + ((height - 1) * stepY) / 2,
        -z * stepZ + ((depth - 1) * stepZ) / 2,
    );
}

function recursivePosition3D(coord: number[], shapeInput: number[], cellExtent: Extent3, level: number, dimensionBlockGapMultiple: number): Vector3 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 3) return baseGridPosition3D(coord, shape, cellExtent, level, dimensionBlockGapMultiple);
    const split = shape.length - 3;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent3D(innerShape, cellExtent, level, dimensionBlockGapMultiple);
    const outerPosition = recursivePosition3D(coord.slice(0, split), outerShape, innerExtent, level + 1, dimensionBlockGapMultiple);
    const innerPosition = recursivePosition3D(coord.slice(split), innerShape, cellExtent, level, dimensionBlockGapMultiple);
    return outerPosition.add(innerPosition);
}

function baseGridExtent2D(
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
    dimensionBlockGapMultiple: number,
): Extent2 {
    const shape = normalizeDisplayShape(shapeInput);
    const gap = levelGap(level, dimensionBlockGapMultiple);
    if (shape.length <= 1) {
        const length = shape[0] ?? 1;
        return verticalFor1D
            ? { x: cellExtent.x, y: (length - 1) * (cellExtent.y + gap) + cellExtent.y }
            : { x: (length - 1) * (cellExtent.x + gap) + cellExtent.x, y: cellExtent.y };
    }
    const width = shape[shape.length - 1] ?? 1;
    const height = shape[shape.length - 2] ?? 1;
    return {
        x: (width - 1) * (cellExtent.x + gap) + cellExtent.x,
        y: (height - 1) * (cellExtent.y + gap) + cellExtent.y,
    };
}

function axisWorldKey2D(rank: number, axis: number): 0 | 1 {
    return ((rank - 1 - axis) % 2) as 0 | 1;
}

function recursiveExtent2D(
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
    originalRank: number,
    axisOffset: number,
    dimensionBlockGapMultiple: number,
): Extent2 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 2) {
        const orient1D = shape.length === 1 ? axisWorldKey2D(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridExtent2D(shape, cellExtent, level, orient1D, dimensionBlockGapMultiple);
    }
    const split = shape.length - 2;
    const innerExtent = recursiveExtent2D(shape.slice(split), cellExtent, level, false, originalRank, axisOffset + split, dimensionBlockGapMultiple);
    const outerVertical = axisWorldKey2D(originalRank, axisOffset + split - 1) === 1;
    return recursiveExtent2D(shape.slice(0, split), innerExtent, level + 1, outerVertical, originalRank, axisOffset, dimensionBlockGapMultiple);
}

function baseGridPosition2D(
    coord: number[],
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
    dimensionBlockGapMultiple: number,
): { x: number; y: number } {
    const shape = normalizeDisplayShape(shapeInput);
    const gap = levelGap(level, dimensionBlockGapMultiple);
    const stepX = cellExtent.x + gap;
    const stepY = cellExtent.y + gap;
    if (shape.length <= 1) {
        const length = shape[0] ?? 1;
        return verticalFor1D
            ? { x: 0, y: -(coord[0] ?? 0) * stepY + ((length - 1) * stepY) / 2 }
            : { x: (coord[0] ?? 0) * stepX - ((length - 1) * stepX) / 2, y: 0 };
    }
    const width = shape[shape.length - 1] ?? 1;
    const height = shape[shape.length - 2] ?? 1;
    const x = coord[shape.length - 1] ?? 0;
    const y = coord[shape.length - 2] ?? 0;
    return {
        x: x * stepX - ((width - 1) * stepX) / 2,
        y: -y * stepY + ((height - 1) * stepY) / 2,
    };
}

function insideCell(point: number, center: number, extent: number): boolean {
    return Math.abs(point - center) <= extent / 2 + CELL_HIT_EPSILON;
}

function normalizeZero(value: number): number {
    return Object.is(value, -0) ? 0 : value;
}

function baseGridHit2D(
    point: { x: number; y: number },
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
    dimensionBlockGapMultiple: number,
): CoordHit2D | null {
    const shape = normalizeDisplayShape(shapeInput);
    const gap = levelGap(level, dimensionBlockGapMultiple);
    const stepX = cellExtent.x + gap;
    const stepY = cellExtent.y + gap;
    if (shape.length <= 1) {
        const length = shape[0] ?? 1;
        if (verticalFor1D) {
            if (!insideCell(point.x, 0, cellExtent.x)) return null;
            const index = Math.round((((length - 1) * stepY) / 2 - point.y) / stepY);
            if (index < 0 || index >= length) return null;
            const centerY = normalizeZero(-index * stepY + ((length - 1) * stepY) / 2);
            if (!insideCell(point.y, centerY, cellExtent.y)) return null;
            return { coord: [normalizeZero(index)], position: { x: 0, y: centerY } };
        }
        if (!insideCell(point.y, 0, cellExtent.y)) return null;
        const index = Math.round((point.x + ((length - 1) * stepX) / 2) / stepX);
        if (index < 0 || index >= length) return null;
        const centerX = normalizeZero(index * stepX - ((length - 1) * stepX) / 2);
        if (!insideCell(point.x, centerX, cellExtent.x)) return null;
        return { coord: [normalizeZero(index)], position: { x: centerX, y: 0 } };
    }

    const width = shape[shape.length - 1] ?? 1;
    const height = shape[shape.length - 2] ?? 1;
    const x = Math.round((point.x + ((width - 1) * stepX) / 2) / stepX);
    const y = Math.round((((height - 1) * stepY) / 2 - point.y) / stepY);
    if (x < 0 || x >= width || y < 0 || y >= height) return null;
    const centerX = normalizeZero(x * stepX - ((width - 1) * stepX) / 2);
    const centerY = normalizeZero(-y * stepY + ((height - 1) * stepY) / 2);
    if (!insideCell(point.x, centerX, cellExtent.x) || !insideCell(point.y, centerY, cellExtent.y)) return null;
    return {
        coord: [normalizeZero(y), normalizeZero(x)],
        position: { x: centerX, y: centerY },
    };
}

function recursivePosition2D(
    coord: number[],
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
    originalRank: number,
    axisOffset: number,
    dimensionBlockGapMultiple: number,
): { x: number; y: number } {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 2) {
        const orient1D = shape.length === 1 ? axisWorldKey2D(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridPosition2D(coord, shape, cellExtent, level, orient1D, dimensionBlockGapMultiple);
    }
    const split = shape.length - 2;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent2D(innerShape, cellExtent, level, false, originalRank, axisOffset + split, dimensionBlockGapMultiple);
    const outerVertical = axisWorldKey2D(originalRank, axisOffset + split - 1) === 1;
    const outerPosition = recursivePosition2D(coord.slice(0, split), outerShape, innerExtent, level + 1, outerVertical, originalRank, axisOffset, dimensionBlockGapMultiple);
    const innerPosition = recursivePosition2D(coord.slice(split), innerShape, cellExtent, level, false, originalRank, axisOffset + split, dimensionBlockGapMultiple);
    return {
        x: outerPosition.x + innerPosition.x,
        y: outerPosition.y + innerPosition.y,
    };
}

function recursiveHit2D(
    point: { x: number; y: number },
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
    originalRank: number,
    axisOffset: number,
    dimensionBlockGapMultiple: number,
): CoordHit2D | null {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 2) {
        const orient1D = shape.length === 1 ? axisWorldKey2D(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridHit2D(point, shape, cellExtent, level, orient1D, dimensionBlockGapMultiple);
    }
    const split = shape.length - 2;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent2D(innerShape, cellExtent, level, false, originalRank, axisOffset + split, dimensionBlockGapMultiple);
    const outerVertical = axisWorldKey2D(originalRank, axisOffset + split - 1) === 1;
    const outerHit = recursiveHit2D(point, outerShape, innerExtent, level + 1, outerVertical, originalRank, axisOffset, dimensionBlockGapMultiple);
    if (!outerHit) return null;
    const innerHit = recursiveHit2D(
        { x: point.x - outerHit.position.x, y: point.y - outerHit.position.y },
        innerShape,
        cellExtent,
        level,
        false,
        originalRank,
        axisOffset + split,
        dimensionBlockGapMultiple,
    );
    if (!innerHit) return null;
    return {
        coord: outerHit.coord.concat(innerHit.coord),
        position: {
            x: outerHit.position.x + innerHit.position.x,
            y: outerHit.position.y + innerHit.position.y,
        },
    };
}

export function unravelIndex(index: number, shapeInput: number[]): number[] {
    const shape = normalizeDisplayShape(shapeInput);
    const coord = new Array(shape.length).fill(0);
    let remaining = index;
    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
        coord[axis] = remaining % shape[axis];
        remaining = Math.floor(remaining / shape[axis]);
    }
    return coord;
}

export function displayPositionForCoord(
    coord: number[],
    shape: number[],
    dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
): Vector3 {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    return recursivePosition3D(coord, shape, { x: CELL_SIZE, y: CELL_SIZE, z: CELL_SIZE }, 0, normalizedGapMultiple);
}

export function displayExtent(shape: number[], dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE): Vector3 {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    const extent = recursiveExtent3D(shape, { x: CELL_SIZE, y: CELL_SIZE, z: CELL_SIZE }, 0, normalizedGapMultiple);
    return new Vector3(extent.x, extent.y, extent.z);
}

export function displayPositionForCoord2D(
    coord: number[],
    shape: number[],
    dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
): { x: number; y: number } {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    return recursivePosition2D(coord, shape, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalizeDisplayShape(shape).length, 0, normalizedGapMultiple);
}

export function displayExtent2D(shape: number[], dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE): { x: number; y: number } {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    const normalized = normalizeDisplayShape(shape);
    return recursiveExtent2D(normalized, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalized.length, 0, normalizedGapMultiple);
}

export function displayHitForPoint2D(
    x: number,
    y: number,
    shape: number[],
    dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
): CoordHit2D | null {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    const normalized = normalizeDisplayShape(shape);
    return recursiveHit2D({ x, y }, normalized, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalized.length, 0, normalizedGapMultiple);
}
