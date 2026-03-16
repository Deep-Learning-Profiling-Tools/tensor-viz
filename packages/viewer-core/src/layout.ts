import { Vector3 } from 'three';
import type { DimensionMappingScheme } from './types.js';

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

export function axisWorldKeyForMode(
    displayMode: '2d' | '3d',
    rank: number,
    axis: number,
    scheme: DimensionMappingScheme = 'z-order',
): 0 | 1 | 2 {
    if (scheme === 'z-order') {
        return (displayMode === '2d' ? (rank - 1 - axis) % 2 : (rank - 1 - axis) % 3) as 0 | 1 | 2;
    }
    if (displayMode === '2d') {
        return (axis < Math.floor(rank / 2) ? 1 : 0) as 0 | 1 | 2;
    }
    const base = Math.floor(rank / 3);
    const remainder = rank % 3;
    const xCount = base + (remainder >= 1 ? 1 : 0);
    const yCount = base + (remainder >= 2 ? 1 : 0);
    const zCount = rank - xCount - yCount;
    if (axis < zCount) return 2;
    if (axis < zCount + yCount) return 1;
    return 0;
}

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

function axisWorldKey2DZOrder(rank: number, axis: number): 0 | 1 {
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
        const orient1D = shape.length === 1 ? axisWorldKey2DZOrder(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridExtent2D(shape, cellExtent, level, orient1D, dimensionBlockGapMultiple);
    }
    const split = shape.length - 2;
    const innerExtent = recursiveExtent2D(shape.slice(split), cellExtent, level, false, originalRank, axisOffset + split, dimensionBlockGapMultiple);
    const outerVertical = axisWorldKey2DZOrder(originalRank, axisOffset + split - 1) === 1;
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

function familyAxes(rank: number, displayMode: '2d' | '3d', familyKey: 0 | 1 | 2, scheme: DimensionMappingScheme): number[] {
    return Array.from({ length: rank }, (_entry, axis) => axis)
        .filter((axis) => axisWorldKeyForMode(displayMode, rank, axis, scheme) === familyKey);
}

function familyExtent1D(shape: number[], axes: number[], dimensionBlockGapMultiple: number): number {
    let extent = CELL_SIZE;
    for (let index = axes.length - 1, level = 0; index >= 0; index -= 1, level += 1) {
        const axis = axes[index];
        const step = extent + levelGap(level, dimensionBlockGapMultiple);
        extent = (shape[axis] - 1) * step + extent;
    }
    return extent;
}

function familyPosition1D(
    coord: number[],
    shape: number[],
    axes: number[],
    sign: 1 | -1,
    dimensionBlockGapMultiple: number,
): number {
    let extent = CELL_SIZE;
    let position = 0;
    for (let index = axes.length - 1, level = 0; index >= 0; index -= 1, level += 1) {
        const axis = axes[index];
        const step = extent + levelGap(level, dimensionBlockGapMultiple);
        position += sign * ((coord[axis] ?? 0) * step - ((shape[axis] - 1) * step) / 2);
        extent = (shape[axis] - 1) * step + extent;
    }
    return normalizeZero(position);
}

function familyHit1D(
    point: number,
    shape: number[],
    axes: number[],
    sign: 1 | -1,
    dimensionBlockGapMultiple: number,
): { coord: number[]; position: number } | null {
    if (axes.length === 0) {
        return insideCell(point, 0, CELL_SIZE) ? { coord: new Array(shape.length).fill(0), position: 0 } : null;
    }
    const innerExtents = new Map<number, number>();
    const levels = new Map<number, number>();
    let extent = CELL_SIZE;
    for (let index = axes.length - 1, level = 0; index >= 0; index -= 1, level += 1) {
        const axis = axes[index];
        innerExtents.set(axis, extent);
        levels.set(axis, level);
        const step = extent + levelGap(level, dimensionBlockGapMultiple);
        extent = (shape[axis] - 1) * step + extent;
    }

    const coord = new Array(shape.length).fill(0);
    let remaining = point;
    let position = 0;
    for (const axis of axes) {
        const innerExtent = innerExtents.get(axis) ?? CELL_SIZE;
        const step = innerExtent + levelGap(levels.get(axis) ?? 0, dimensionBlockGapMultiple);
        const size = shape[axis];
        const centered = sign === 1
            ? (remaining + ((size - 1) * step) / 2) / step
            : ((((size - 1) * step) / 2) - remaining) / step;
        const value = Math.round(centered);
        if (value < 0 || value >= size) return null;
        const center = normalizeZero(sign === 1
            ? value * step - ((size - 1) * step) / 2
            : -value * step + ((size - 1) * step) / 2);
        if (!insideCell(remaining, center, innerExtent)) return null;
        coord[axis] = value;
        remaining -= center;
        position += center;
    }
    return { coord, position: normalizeZero(position) };
}

function contiguousPosition3D(coord: number[], shape: number[], dimensionBlockGapMultiple: number): Vector3 {
    const rank = shape.length;
    return new Vector3(
        familyPosition1D(coord, shape, familyAxes(rank, '3d', 0, 'contiguous'), 1, dimensionBlockGapMultiple),
        familyPosition1D(coord, shape, familyAxes(rank, '3d', 1, 'contiguous'), -1, dimensionBlockGapMultiple),
        familyPosition1D(coord, shape, familyAxes(rank, '3d', 2, 'contiguous'), -1, dimensionBlockGapMultiple),
    );
}

function contiguousExtent3D(shape: number[], dimensionBlockGapMultiple: number): Extent3 {
    const rank = shape.length;
    return {
        x: familyExtent1D(shape, familyAxes(rank, '3d', 0, 'contiguous'), dimensionBlockGapMultiple),
        y: familyExtent1D(shape, familyAxes(rank, '3d', 1, 'contiguous'), dimensionBlockGapMultiple),
        z: familyExtent1D(shape, familyAxes(rank, '3d', 2, 'contiguous'), dimensionBlockGapMultiple),
    };
}

function contiguousPosition2D(coord: number[], shape: number[], dimensionBlockGapMultiple: number): { x: number; y: number } {
    const rank = shape.length;
    return {
        x: familyPosition1D(coord, shape, familyAxes(rank, '2d', 0, 'contiguous'), 1, dimensionBlockGapMultiple),
        y: familyPosition1D(coord, shape, familyAxes(rank, '2d', 1, 'contiguous'), -1, dimensionBlockGapMultiple),
    };
}

function contiguousExtent2D(shape: number[], dimensionBlockGapMultiple: number): Extent2 {
    const rank = shape.length;
    return {
        x: familyExtent1D(shape, familyAxes(rank, '2d', 0, 'contiguous'), dimensionBlockGapMultiple),
        y: familyExtent1D(shape, familyAxes(rank, '2d', 1, 'contiguous'), dimensionBlockGapMultiple),
    };
}

function contiguousHit2D(point: { x: number; y: number }, shape: number[], dimensionBlockGapMultiple: number): CoordHit2D | null {
    const xHit = familyHit1D(point.x, shape, familyAxes(shape.length, '2d', 0, 'contiguous'), 1, dimensionBlockGapMultiple);
    if (!xHit) return null;
    const yHit = familyHit1D(point.y, shape, familyAxes(shape.length, '2d', 1, 'contiguous'), -1, dimensionBlockGapMultiple);
    if (!yHit) return null;
    const coord = new Array(shape.length).fill(0);
    for (let axis = 0; axis < shape.length; axis += 1) {
        coord[axis] = xHit.coord[axis] || yHit.coord[axis] || 0;
    }
    return {
        coord: coord.map((value) => normalizeZero(value)),
        position: {
            x: xHit.position,
            y: yHit.position,
        },
    };
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
        const orient1D = shape.length === 1 ? axisWorldKey2DZOrder(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridPosition2D(coord, shape, cellExtent, level, orient1D, dimensionBlockGapMultiple);
    }
    const split = shape.length - 2;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent2D(innerShape, cellExtent, level, false, originalRank, axisOffset + split, dimensionBlockGapMultiple);
    const outerVertical = axisWorldKey2DZOrder(originalRank, axisOffset + split - 1) === 1;
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
        const orient1D = shape.length === 1 ? axisWorldKey2DZOrder(originalRank, axisOffset) === 1 : verticalFor1D;
        return baseGridHit2D(point, shape, cellExtent, level, orient1D, dimensionBlockGapMultiple);
    }
    const split = shape.length - 2;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent2D(innerShape, cellExtent, level, false, originalRank, axisOffset + split, dimensionBlockGapMultiple);
    const outerVertical = axisWorldKey2DZOrder(originalRank, axisOffset + split - 1) === 1;
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
    scheme: DimensionMappingScheme = 'z-order',
): Vector3 {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    if (scheme === 'contiguous') return contiguousPosition3D(coord, normalizeDisplayShape(shape), normalizedGapMultiple);
    return recursivePosition3D(coord, shape, { x: CELL_SIZE, y: CELL_SIZE, z: CELL_SIZE }, 0, normalizedGapMultiple);
}

export function displayExtent(
    shape: number[],
    dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
    scheme: DimensionMappingScheme = 'z-order',
): Vector3 {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    const extent = scheme === 'contiguous'
        ? contiguousExtent3D(normalizeDisplayShape(shape), normalizedGapMultiple)
        : recursiveExtent3D(shape, { x: CELL_SIZE, y: CELL_SIZE, z: CELL_SIZE }, 0, normalizedGapMultiple);
    return new Vector3(extent.x, extent.y, extent.z);
}

export function displayPositionForCoord2D(
    coord: number[],
    shape: number[],
    dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
    scheme: DimensionMappingScheme = 'z-order',
): { x: number; y: number } {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    const normalized = normalizeDisplayShape(shape);
    if (scheme === 'contiguous') return contiguousPosition2D(coord, normalized, normalizedGapMultiple);
    return recursivePosition2D(coord, shape, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalized.length, 0, normalizedGapMultiple);
}

export function displayExtent2D(
    shape: number[],
    dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
    scheme: DimensionMappingScheme = 'z-order',
): { x: number; y: number } {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    const normalized = normalizeDisplayShape(shape);
    return scheme === 'contiguous'
        ? contiguousExtent2D(normalized, normalizedGapMultiple)
        : recursiveExtent2D(normalized, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalized.length, 0, normalizedGapMultiple);
}

export function displayHitForPoint2D(
    x: number,
    y: number,
    shape: number[],
    dimensionBlockGapMultiple = DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
    scheme: DimensionMappingScheme = 'z-order',
): CoordHit2D | null {
    const normalizedGapMultiple = normalizeDimensionBlockGapMultiple(dimensionBlockGapMultiple);
    const normalized = normalizeDisplayShape(shape);
    return scheme === 'contiguous'
        ? contiguousHit2D({ x, y }, normalized, normalizedGapMultiple)
        : recursiveHit2D({ x, y }, normalized, { x: CELL_SIZE, y: CELL_SIZE }, 0, false, normalized.length, 0, normalizedGapMultiple);
}
