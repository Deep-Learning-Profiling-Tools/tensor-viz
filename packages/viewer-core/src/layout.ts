import { Vector3 } from 'three';

const CELL_SIZE = 1;
const GAP = 0.15;
const OUTER_GAP_SCALE_3D = 3;
const OUTER_GAP_SCALE_2D = 2;

type Extent3 = {
    x: number;
    y: number;
    z: number;
};

type Extent2 = {
    x: number;
    y: number;
};

function normalizeDisplayShape(shape: number[]): number[] {
    return shape.length === 0 ? [1] : shape.map((value) => Math.max(1, value));
}

function levelGap(level: number, scale: number): number {
    return GAP * Math.pow(scale, Math.max(0, level));
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

function baseGridExtent3D(shapeInput: number[], cellExtent: Extent3, level: number): Extent3 {
    const shape = normalizeDisplayShape(shapeInput);
    const { depth, height, width } = depthHeightWidth(shape);
    const gap = levelGap(level, OUTER_GAP_SCALE_3D);
    return {
        x: (width - 1) * (cellExtent.x + gap) + cellExtent.x,
        y: (height - 1) * (cellExtent.y + gap) + cellExtent.y,
        z: (depth - 1) * (cellExtent.z + gap) + cellExtent.z,
    };
}

function recursiveExtent3D(shapeInput: number[], cellExtent: Extent3, level: number): Extent3 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 3) return baseGridExtent3D(shape, cellExtent, level);
    const split = shape.length - 3;
    const innerExtent = recursiveExtent3D(shape.slice(split), cellExtent, level);
    return recursiveExtent3D(shape.slice(0, split), innerExtent, level + 1);
}

function baseGridPosition3D(coord: number[], shapeInput: number[], cellExtent: Extent3, level: number): Vector3 {
    const shape = normalizeDisplayShape(shapeInput);
    const { depth, height, width } = depthHeightWidth(shape);
    const gap = levelGap(level, OUTER_GAP_SCALE_3D);
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

function recursivePosition3D(coord: number[], shapeInput: number[], cellExtent: Extent3, level: number): Vector3 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 3) return baseGridPosition3D(coord, shape, cellExtent, level);
    const split = shape.length - 3;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent3D(innerShape, cellExtent, level);
    const outerPosition = recursivePosition3D(coord.slice(0, split), outerShape, innerExtent, level + 1);
    const innerPosition = recursivePosition3D(coord.slice(split), innerShape, cellExtent, level);
    return outerPosition.add(innerPosition);
}

function baseGridExtent2D(shapeInput: number[], cellExtent: Extent2, level: number, verticalFor1D: boolean): Extent2 {
    const shape = normalizeDisplayShape(shapeInput);
    const gap = levelGap(level, OUTER_GAP_SCALE_2D);
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
): Extent2 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 2) {
        const orient1D = shape.length === 1 ? axisWorldKey2D(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridExtent2D(shape, cellExtent, level, orient1D);
    }
    const split = shape.length - 2;
    const innerExtent = recursiveExtent2D(shape.slice(split), cellExtent, level, false, originalRank, axisOffset + split);
    const outerVertical = axisWorldKey2D(originalRank, axisOffset + split - 1) === 1;
    return recursiveExtent2D(shape.slice(0, split), innerExtent, level + 1, outerVertical, originalRank, axisOffset);
}

function baseGridPosition2D(
    coord: number[],
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
): { x: number; y: number } {
    const shape = normalizeDisplayShape(shapeInput);
    const gap = levelGap(level, OUTER_GAP_SCALE_2D);
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

function recursivePosition2D(
    coord: number[],
    shapeInput: number[],
    cellExtent: Extent2,
    level: number,
    verticalFor1D: boolean,
    originalRank: number,
    axisOffset: number,
): { x: number; y: number } {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 2) {
        const orient1D = shape.length === 1 ? axisWorldKey2D(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridPosition2D(coord, shape, cellExtent, level, orient1D);
    }
    const split = shape.length - 2;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent2D(innerShape, cellExtent, level, false, originalRank, axisOffset + split);
    const outerVertical = axisWorldKey2D(originalRank, axisOffset + split - 1) === 1;
    const outerPosition = recursivePosition2D(coord.slice(0, split), outerShape, innerExtent, level + 1, outerVertical, originalRank, axisOffset);
    const innerPosition = recursivePosition2D(coord.slice(split), innerShape, cellExtent, level, false, originalRank, axisOffset + split);
    return {
        x: outerPosition.x + innerPosition.x,
        y: outerPosition.y + innerPosition.y,
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

export function displayPositionForCoord(coord: number[], shape: number[]): Vector3 {
    return recursivePosition3D(coord, shape, { x: CELL_SIZE, y: CELL_SIZE, z: CELL_SIZE }, 0);
}

export function displayExtent(shape: number[]): Vector3 {
    const extent = recursiveExtent3D(shape, { x: CELL_SIZE, y: CELL_SIZE, z: CELL_SIZE }, 0);
    return new Vector3(extent.x, extent.y, extent.z);
}

export function displayPositionForCoord2D(coord: number[], shape: number[]): { x: number; y: number } {
    return recursivePosition2D(coord, shape, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalizeDisplayShape(shape).length, 0);
}

export function displayExtent2D(shape: number[]): { x: number; y: number } {
    const normalized = normalizeDisplayShape(shape);
    return recursiveExtent2D(normalized, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalized.length, 0);
}
