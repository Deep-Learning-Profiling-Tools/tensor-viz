import { Vector3 } from 'three';
import type { DimensionMappingScheme } from './types.js';

const CELL_SIZE = 1;
const GAP = 0.15;

/** Default multiplier used when spacing nested dimension blocks apart. */
export const DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE = 3;

/**
 * Width, height, and depth occupied by a rendered 3D tensor block in layout-space units.
 *
 * The layout engine passes these extents between recursive block calculations so outer
 * tensor axes can reserve enough x, y, and z space for nested cells.
 *
 * @example
 * const cellBlock: Extent3 = { x: 1, y: 1, z: 1 };
 * const tensorBlock: Extent3 = { x: 4, y: 3, z: 2 };
 *
 * console.assert(tensorBlock.x > cellBlock.x);
 * console.assert(tensorBlock.z === 2);
 */
type Extent3 = {
    x: number;
    y: number;
    z: number;
};

/**
 * Width and height occupied by a rendered 2D tensor block in layout-space units.
 *
 * Recursive 2D layout calculations use this size to position rows, columns, and higher-rank
 * blocks without needing a z measurement.
 *
 * @example
 * const cellBlock: Extent2 = { x: 1, y: 1 };
 * const matrixBlock: Extent2 = { x: 5, y: 3 };
 *
 * console.assert(matrixBlock.x === 5);
 * console.assert(matrixBlock.y > cellBlock.y);
 */
type Extent2 = {
    x: number;
    y: number;
};

/**
 * One 2D hit-test result containing the tensor layout coordinate under the pointer and the
 * rendered world-space center for that cell.
 *
 * Callers can use `coord` to identify the tensor element and `position` to place overlays,
 * tooltips, or selection markers on the 2D view.
 *
 * @example
 * const hit: CoordHit2D = {
 *     coord: [2, 4],
 *     position: { x: 4.5, y: -2.5 },
 * };
 *
 * console.assert(hit.coord.join(',') === '2,4');
 * console.assert(hit.position.x === 4.5);
 */
export type CoordHit2D = {
    coord: number[];
    position: {
        x: number;
        y: number;
    };
};

const CELL_HIT_EPSILON = 1e-6;

/**
 * Assign a tensor axis to the layout family that controls horizontal, vertical, or depth
 * placement for the selected display mode.
 *
 * In z-order mode, high-rank axes alternate across the visible families so nested tensor
 * blocks separate visually. In contiguous mode, adjacent leading axes are grouped together
 * for monotonic row, column, and depth ranges.
 *
 * @param displayMode - Viewer projection to map into: `'2d'` uses x/y families, while `'3d'` uses x/y/z families.
 * @param rank - Number of axes in the tensor shape that is being laid out.
 * @param axis - Zero-based tensor axis within `rank` whose display family should be selected.
 * @param scheme - Axis grouping strategy: `'z-order'` alternates families, and contiguous mode groups neighboring axes.
 * @returns Family key for the axis: `0` for x placement, `1` for y placement, and `2` for z placement in 3D layouts.
 * @noThrows The function only branches on the typed display mode and scheme and performs numeric comparisons/modulo operations; it does not validate inputs or call code that throws.
 * @example
 * const xFamily = axisWorldKeyForMode('2d', 4, 3, 'z-order');
 * console.assert(xFamily === 0);
 *
 * const zFamily = axisWorldKeyForMode('3d', 5, 0, 'contiguous');
 * console.assert(zFamily === 2);
 */
export function axisWorldKeyForMode(
    displayMode: '2d' | '3d',
    rank: number,
    axis: number,
    scheme: DimensionMappingScheme = 'z-order',
): 0 | 1 | 2 {
    if (scheme === 'z-order') {
        // z-order keeps adjacent high-rank axes alternating across screen/world
        // directions, which makes nested tensor blocks visually separate.
        return (displayMode === '2d' ? (rank - 1 - axis) % 2 : (rank - 1 - axis) % 3) as 0 | 1 | 2;
    }
    // contiguous mode groups leading axes together so box selection can use
    // monotonic row/column ranges instead of scanning every rendered cell.
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

/**
 * Converts a tensor shape into dimensions that can be laid out on screen.
 * Empty scalar shapes become a single display cell, and each provided dimension is clamped to at least one cell so extent and hit-test math never uses zero-sized axes.
 *
 * @param shape - Tensor shape from the view/session data, ordered by tensor axis; entries may be zero or negative before display normalization.
 * @returns A display-shape array with at least one dimension and no entry below `1`, suitable for position, extent, and index-unraveling calculations.
 * @noThrows Uses only array length checks and numeric clamping on the provided entries; it does not validate rank or allocate from untrusted sizes beyond the returned mapped array.
 * @example
 * normalizeDisplayShape([]);
 * // Returns [1] for a scalar tensor display.
 *
 * normalizeDisplayShape([2, 0, -4]);
 * // Returns [2, 1, 1].
 */
function normalizeDisplayShape(shape: number[]): number[] {
    return shape.length === 0 ? [1] : shape.map((value) => Math.max(1, value));
}

/**
 * Sanitizes the caller-configured multiplier used to widen gaps between nested dimension blocks.
 * Non-finite values fall back to the default spacing multiplier, while finite negative values disable additional block spacing by becoming `0`.
 *
 * @param value - Dimension-block gap multiplier from layout options or viewer configuration.
 * @returns A finite, non-negative multiplier used by layout position, extent, and hit-test calculations.
 * @noThrows Uses `Number.isFinite` and `Math.max` only; invalid numeric inputs are converted to fallback values instead of being rejected.
 * @example
 * normalizeDimensionBlockGapMultiple(-2);
 * // Returns 0.
 *
 * normalizeDimensionBlockGapMultiple(Number.NaN);
 * // Returns DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE.
 */
function normalizeDimensionBlockGapMultiple(value: number): number {
    return Number.isFinite(value) ? Math.max(0, value) : DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
}

/**
 * Computes the spacing inserted between cells or nested blocks at a layout nesting level.
 * A non-positive gap multiplier removes the gap entirely; otherwise the base gap grows by the multiplier for each outer level.
 *
 * @param level - Zero-based nesting level for the dimension block; negative levels are treated like level `0`.
 * @param dimensionBlockGapMultiple - Sanitized non-negative multiplier controlling how much wider gaps become for outer dimension blocks.
 * @returns The world-space gap distance to add to the cell or inner-block extent at the requested level.
 * @noThrows Performs deterministic arithmetic on numeric arguments and treats non-positive multipliers as a valid request for no extra spacing.
 * @example
 * levelGap(3, 0);
 * // Returns 0 because a zero multiplier disables block gaps.
 *
 * levelGap(-1, 2);
 * // Returns GAP because negative levels are clamped to level 0.
 */
function levelGap(level: number, dimensionBlockGapMultiple: number): number {
    if (dimensionBlockGapMultiple <= 0) return 0;
    return GAP * Math.pow(dimensionBlockGapMultiple, Math.max(0, level));
}

/**
 * Projects a normalized tensor display shape onto the depth, height, and width axes used by 3D grid layout.
 * One-dimensional shapes render as a row, two-dimensional shapes render as a plane, and higher-rank shapes use the final three axes as depth, height, and width.
 *
 * @param shape - Normalized display shape whose trailing axes represent the visible 3D grid dimensions.
 * @returns The depth, height, and width counts used to size and position cells in the base 3D grid.
 * @noThrows Reads array length and trailing entries only; callers are expected to pass the normalized non-empty display shape produced by `normalizeDisplayShape`.
 * @example
 * depthHeightWidth([7]);
 * // Returns { depth: 1, height: 1, width: 7 }.
 *
 * depthHeightWidth([2, 3, 4, 5]);
 * // Returns { depth: 3, height: 4, width: 5 }.
 */
function depthHeightWidth(shape: number[]): { depth: number; height: number; width: number } {
    if (shape.length === 1) return { depth: 1, height: 1, width: shape[0] };
    if (shape.length === 2) return { depth: 1, height: shape[0], width: shape[1] };
    return {
        depth: shape[shape.length - 3],
        height: shape[shape.length - 2],
        width: shape[shape.length - 1],
    };
}

/**
 * Calculates the bounding size of one displayed grid block whose last three shape axes map to x, y, and z.
 *
 * @param shapeInput - Display shape for the block; the last axis is width, the previous axis is height, and the third-from-last axis is depth. Missing axes are treated as length 1 by shape normalization.
 * @param cellExtent - Rendered size of one cell or nested child block along each Three.js axis.
 * @param level - Nesting level used to scale the spacing inserted between adjacent cells in this grid.
 * @param dimensionBlockGapMultiple - Multiplier for the inter-cell gap at the supplied nesting level.
 * @returns Total x, y, and z extent occupied by the grid, including the cells and the gaps between neighboring cells.
 * @noThrows Shape normalization supplies default axes and the remaining work is deterministic numeric arithmetic over the provided extents and gap multiplier.
 * @example
 * const extent = baseGridExtent3D([2, 3], { x: 1, y: 1, z: 1 }, 0, 0);
 * // extent is { x: 3, y: 2, z: 1 }: three columns, two rows, and one normalized depth layer.
 */
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

/**
 * Calculates the bounding size for a display shape by nesting higher-dimensional axes into outer 3D blocks.
 *
 * @param shapeInput - Full display shape for the visible tensor axes; shapes with more than three axes are split into outer axes and an inner 3D block.
 * @param cellExtent - Rendered size of a single leaf cell before any outer block nesting is applied.
 * @param level - Current nesting level, used to increase block spacing as recursion moves outward.
 * @param dimensionBlockGapMultiple - Multiplier for the gap inserted between cells or child blocks at each nesting level.
 * @returns Full x, y, and z extent needed to contain every nested block for the display shape.
 * @noThrows The recursion only normalizes the shape, slices arrays, and combines numeric extents; it does not reject particular shape lengths or coordinate values.
 * @example
 * const extent = recursiveExtent3D([2, 2, 3, 4], { x: 1, y: 1, z: 1 }, 0, 0);
 * // extent is { x: 8, y: 3, z: 2 }: two outer blocks, each containing a 2-by-3-by-4 inner grid.
 */
function recursiveExtent3D(shapeInput: number[], cellExtent: Extent3, level: number, dimensionBlockGapMultiple: number): Extent3 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 3) return baseGridExtent3D(shape, cellExtent, level, dimensionBlockGapMultiple);
    const split = shape.length - 3;
    const innerExtent = recursiveExtent3D(shape.slice(split), cellExtent, level, dimensionBlockGapMultiple);
    // outer axes arrange whole inner blocks, so the inner block extent becomes
    // the next recursion's effective cell size.
    return recursiveExtent3D(shape.slice(0, split), innerExtent, level + 1, dimensionBlockGapMultiple);
}

/**
 * Converts a coordinate within one displayed grid block into a centered Three.js position.
 *
 * @param coord - Coordinate indexes for the display shape; the last entry selects the x column, the previous entry selects the y row, and the third-from-last entry selects the z layer. Missing entries default to 0.
 * @param shapeInput - Display shape for the block used to center the coordinate within the grid's width, height, and depth.
 * @param cellExtent - Rendered size of one cell or nested child block along each Three.js axis.
 * @param level - Nesting level used to scale the spacing between adjacent coordinates in this grid.
 * @param dimensionBlockGapMultiple - Multiplier for the inter-cell gap at the supplied nesting level.
 * @returns Centered render position for the coordinate, with positive x to the right and negative y/z moving down and deeper through the displayed grid.
 * @noThrows The function tolerates missing coordinate components with zero defaults and otherwise performs only shape normalization and arithmetic.
 * @example
 * const position = baseGridPosition3D([1, 2], [2, 3], { x: 1, y: 1, z: 1 }, 0, 0);
 * // position has components x=1, y=-0.5, z=0 for the last column of the second row.
 */
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

/**
 * Converts a coordinate in a higher-dimensional display shape into the final centered position of its nested cell.
 *
 * @param coord - Coordinate indexes for the full display shape; leading indexes select outer blocks and trailing indexes select the inner 3D cell position.
 * @param shapeInput - Full display shape whose axes are recursively split into outer block axes and inner 3D grid axes.
 * @param cellExtent - Rendered size of a single leaf cell before nesting is applied.
 * @param level - Current nesting level, used to scale spacing as the recursion places outer blocks.
 * @param dimensionBlockGapMultiple - Multiplier for the gap inserted between cells or child blocks at each nesting level.
 * @returns Three.js position formed by adding the outer block origin to the inner cell offset for the supplied coordinate.
 * @noThrows The recursion uses array slicing, extent calculation, vector addition, and zero-default coordinate lookup rather than throwing for short shapes or missing coordinate entries.
 * @example
 * const position = recursivePosition3D([1, 1, 2, 3], [2, 2, 3, 4], { x: 1, y: 1, z: 1 }, 0, 0);
 * // position has components x=3.5, y=-1, z=-0.5: the inner cell offset is added to the second outer block's origin.
 */
function recursivePosition3D(coord: number[], shapeInput: number[], cellExtent: Extent3, level: number, dimensionBlockGapMultiple: number): Vector3 {
    const shape = normalizeDisplayShape(shapeInput);
    if (shape.length <= 3) return baseGridPosition3D(coord, shape, cellExtent, level, dimensionBlockGapMultiple);
    const split = shape.length - 3;
    const innerShape = shape.slice(split);
    const outerShape = shape.slice(0, split);
    const innerExtent = recursiveExtent3D(innerShape, cellExtent, level, dimensionBlockGapMultiple);
    const outerPosition = recursivePosition3D(coord.slice(0, split), outerShape, innerExtent, level + 1, dimensionBlockGapMultiple);
    const innerPosition = recursivePosition3D(coord.slice(split), innerShape, cellExtent, level, dimensionBlockGapMultiple);
    // rendered coordinates are block origin plus in-block offset; keeping the two
    // parts separate makes extent, position, and hit testing share the same model.
    return outerPosition.add(innerPosition);
}

/**
 * Computes the total rendered width and height of a one- or two-dimensional cell grid.
 *
 * The last shape entry is treated as the grid width, the next-to-last entry as the
 * grid height, and a one-dimensional shape is laid out horizontally or vertically
 * according to `verticalFor1D`.
 *
 * @param shapeInput - Display shape for the base grid; empty shapes behave like a single cell, one-entry shapes are 1-D, and two-or-more-entry shapes use the last two dimensions.
 * @param cellExtent - Width and height of one rendered tensor cell before inter-cell gaps are added.
 * @param level - Nesting level whose gap size separates adjacent cells or blocks.
 * @param verticalFor1D - Whether a one-dimensional shape stacks cells along y instead of x.
 * @param dimensionBlockGapMultiple - Multiplier used by `levelGap` to add spacing between cells at this nesting level.
 * @returns The x/y extent occupied by the base grid, including level-dependent gaps between adjacent cells.
 * @noThrows Uses normalized shape values, default dimensions, and arithmetic only; missing shape entries are treated as length 1 instead of raising an error.
 * @example
 * const extent = baseGridExtent2D([2, 3], { x: 10, y: 10 }, 0, false, 0);
 * expect(extent).toEqual({ x: 30, y: 20 });
 *
 * const verticalLine = baseGridExtent2D([4], { x: 10, y: 5 }, 0, true, 0);
 * expect(verticalLine).toEqual({ x: 10, y: 20 });
 */
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

/**
 * Maps a world-axis index to the alternating two-dimensional nesting order used by recursive layout.
 *
 * Axes are counted from the trailing display dimension toward the leading dimension so adjacent
 * world axes alternate between the two z-order/orientation buckets.
 *
 * @param rank - Number of axes in the normalized display shape being laid out.
 * @param axis - Zero-based world-axis index whose recursive 2-D order is needed.
 * @returns `0` or `1`, indicating which alternating layout bucket the axis belongs to.
 * @noThrows Performs only numeric subtraction and modulo arithmetic and does not inspect external state.
 * @example
 * expect(axisWorldKey2DZOrder(3, 0)).toBe(0);
 * expect(axisWorldKey2DZOrder(3, 1)).toBe(1);
 */
function axisWorldKey2DZOrder(rank: number, axis: number): 0 | 1 {
    return ((rank - 1 - axis) % 2) as 0 | 1;
}

/**
 * Computes the rendered footprint of a normalized tensor display shape by recursively nesting 2-D blocks.
 *
 * Higher-rank shapes are split into an inner trailing 2-D grid and an outer leading block grid;
 * the outer block size is based on the extent of the inner block.
 *
 * @param shapeInput - Remaining display shape to measure, with leading dimensions becoming outer blocks and trailing dimensions becoming the innermost grid.
 * @param cellExtent - Width and height of one rendered tensor cell at the innermost level.
 * @param level - Current recursive nesting depth used to scale gaps between blocks.
 * @param verticalFor1D - Orientation to use when the current recursive slice is one-dimensional and not overridden by axis order.
 * @param originalRank - Rank of the full display shape, used to keep axis orientation stable during recursive slices.
 * @param axisOffset - Number of leading axes already skipped before this recursive slice.
 * @param dimensionBlockGapMultiple - Multiplier used by `levelGap` for spacing between cells and nested blocks.
 * @returns The total x/y extent needed to render the full nested block structure for the provided shape slice.
 * @noThrows Recurses over array slices and uses normalized/defaulted dimensions; empty or short slices collapse to base-grid extents rather than throwing.
 * @example
 * const extent = recursiveExtent2D([2, 3, 4], { x: 10, y: 10 }, 0, false, 3, 0, 0);
 * expect(extent).toEqual({ x: 80, y: 30 });
 */
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

/**
 * Converts a coordinate in a one- or two-dimensional base grid into a centered render offset.
 *
 * The returned point is relative to the grid center: x increases to the right and y increases
 * upward, so larger row coordinates move downward in screen-space layout math.
 *
 * @param coord - Coordinate within the base grid; the last entry is the column, the next-to-last entry is the row, and missing entries default to 0.
 * @param shapeInput - Display shape for the base grid; one-entry shapes use `verticalFor1D`, and two-or-more-entry shapes use the last two dimensions.
 * @param cellExtent - Width and height of one rendered tensor cell before inter-cell gaps are added.
 * @param level - Nesting level whose gap size separates adjacent cells or blocks.
 * @param verticalFor1D - Whether a one-dimensional coordinate advances along y instead of x.
 * @param dimensionBlockGapMultiple - Multiplier used by `levelGap` to add spacing between adjacent cells at this level.
 * @returns Center-relative `{ x, y }` position for the cell coordinate, suitable for render placement and hit-test calculations.
 * @noThrows Missing coordinate or shape entries are defaulted with nullish coalescing, and the function performs only normalization and arithmetic.
 * @example
 * const position = baseGridPosition2D([1, 2], [2, 3], { x: 10, y: 10 }, 0, false, 0);
 * expect(position).toEqual({ x: 10, y: -5 });
 *
 * const verticalPosition = baseGridPosition2D([2], [4], { x: 10, y: 5 }, 0, true, 0);
 * expect(verticalPosition).toEqual({ x: 0, y: -2.5 });
 */
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

/**
 * Tests whether a 1D pointer coordinate still falls within a rendered cell's hit area.
 *
 * @param point - World-space coordinate along one layout axis, such as the pointer x or y value being hit-tested.
 * @param center - World-space center coordinate of the candidate cell on the same axis.
 * @param extent - Rendered width of the candidate cell on that axis.
 * @returns `true` when `point` is no farther than half the cell extent from `center`, including the small hit-test epsilon used to avoid rejecting boundary clicks.
 * @noThrows Uses only numeric subtraction, absolute value, and comparison; it performs no parsing, allocation, or validation that would introduce an expected throw path.
 * @example
 * insideCell(12, 10, 4);
 * // => true, because 12 is on the right edge of a cell spanning 8 through 12.
 *
 * insideCell(13, 10, 4);
 * // => false, because 13 is outside that cell's hit area.
 */
function insideCell(point: number, center: number, extent: number): boolean {
    return Math.abs(point - center) <= extent / 2 + CELL_HIT_EPSILON;
}

/**
 * Converts JavaScript's signed negative zero to the ordinary zero used in displayed coordinates.
 *
 * @param value - Numeric coordinate or index produced by layout arithmetic.
 * @returns The same number unless it is `-0`; callers can compare or display the result without exposing a negative-zero artifact.
 * @noThrows Uses `Object.is` and a ternary expression on a primitive number, so there is no expected throw path.
 * @example
 * Object.is(normalizeZero(-0), 0);
 * // => true
 *
 * normalizeZero(-3);
 * // => -3
 */
function normalizeZero(value: number): number {
    return Object.is(value, -0) ? 0 : value;
}

/**
 * Selects the tensor axes that are assigned to one world-axis family in the 2D or 3D layout mapping.
 *
 * @param rank - Number of dimensions in the tensor shape; candidate axis indices are generated from `0` through `rank - 1`.
 * @param displayMode - Layout projection being built: `'2d'` for x/y families or `'3d'` for x/y/z families.
 * @param familyKey - World-axis family to collect: `0` for x, `1` for y, and `2` for z when the display mode supports it.
 * @param scheme - Dimension-to-world-axis mapping strategy used by the layout engine, such as `'contiguous'`.
 * @returns Tensor axis indices whose mapped world-axis family equals `familyKey`; callers pass this list to position, extent, and hit-test helpers for that family.
 * @noThrows Builds an in-memory range from `rank` and filters it with deterministic mapping logic; it does not perform I/O or throw its own validation errors.
 * @example
 * familyAxes(0, '3d', 0, 'contiguous');
 * // => [] for a scalar tensor, because there are no tensor axes to assign to the x family.
 */
function familyAxes(rank: number, displayMode: '2d' | '3d', familyKey: 0 | 1 | 2, scheme: DimensionMappingScheme): number[] {
    return Array.from({ length: rank }, (_entry, axis) => axis)
        .filter((axis) => axisWorldKeyForMode(displayMode, rank, axis, scheme) === familyKey);
}

/**
 * Computes the rendered 1D span occupied by a group of tensor axes in one layout family.
 *
 * @param shape - Tensor dimension lengths indexed by tensor axis.
 * @param axes - Tensor axis indices, ordered from outermost to innermost for the selected world-axis family.
 * @param dimensionBlockGapMultiple - Multiplier applied by `levelGap` to separate nested dimension blocks.
 * @returns World-space extent needed to draw every cell along the selected family, including nested block gaps; callers use it for layout bounds and hit testing.
 * @noThrows Iterates over the supplied axis list and performs numeric indexing and arithmetic only; valid layout callers provide axes that refer to entries in `shape`.
 * @example
 * familyExtent1D([4, 5], [], 0.25);
 * // => CELL_SIZE, because an empty axis family still occupies one cell span.
 */
function familyExtent1D(shape: number[], axes: number[], dimensionBlockGapMultiple: number): number {
    let extent = CELL_SIZE;
    for (let index = axes.length - 1, level = 0; index >= 0; index -= 1, level += 1) {
        const axis = axes[index];
        const step = extent + levelGap(level, dimensionBlockGapMultiple);
        // each outer contiguous axis repeats the complete inner extent.
        extent = (shape[axis] - 1) * step + extent;
    }
    return extent;
}

/**
 * Projects a tensor coordinate onto one display axis family and returns the cell center offset from that family's origin.
 *
 * The last axis in `axes` is packed innermost; earlier axes wrap around the accumulated inner extent with a level gap between blocks.
 *
 * @param coord - Tensor coordinate indexed by tensor axis; missing entries are treated as zero for axes in this family.
 * @param shape - Tensor extents by axis, used to center each family level around zero.
 * @param axes - Axis indices that belong to this rendered family, ordered outermost to innermost.
 * @param sign - Direction of increasing tensor index on the display axis: `1` for positive display coordinates, `-1` for reversed coordinates.
 * @param dimensionBlockGapMultiple - Multiplier applied to the per-level gap inserted between nested dimension blocks.
 * @returns Centered display-space offset for `coord` along the requested one-dimensional family.
 * @noThrows Uses numeric array lookups and arithmetic only; out-of-range or missing coordinate entries collapse to zero instead of throwing.
 * @example
 * const shape = [3];
 * const coord = [2];
 *
 * // A single three-cell axis is centered at zero, so the last cell is one cell size to the right.
 * assert.equal(familyPosition1D(coord, shape, [0], 1, 0), CELL_SIZE);
 */
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

/**
 * Converts a display-space point on one axis family back into the tensor coordinate cell that contains it.
 *
 * The hit test mirrors `familyPosition1D`: it walks the family axes from outer to inner, verifies that the rounded candidate index is within the tensor shape, and rejects points that fall in inter-block gaps.
 *
 * @param point - Display-space coordinate to test along this one-dimensional family.
 * @param shape - Tensor extents by axis; the returned coordinate array has this length.
 * @param axes - Axis indices assigned to this display family, ordered outermost to innermost.
 * @param sign - Direction used by the family layout: `1` for positive increasing indices, `-1` for reversed display coordinates.
 * @param dimensionBlockGapMultiple - Multiplier for gaps between nested dimension blocks, matching the value used to place cells.
 * @returns The tensor coordinate and resolved family-center position for the hit cell, or `null` when `point` lands outside all cells or inside a gap.
 * @noThrows Misses and out-of-range rounded indices are reported as `null`; the function only allocates arrays/maps and performs bounded numeric arithmetic.
 * @example
 * const shape = [3];
 *
 * assert.deepEqual(familyHit1D(CELL_SIZE, shape, [0], 1, 0), {
 *   coord: [2],
 *   position: CELL_SIZE,
 * });
 * assert.equal(familyHit1D(2 * CELL_SIZE, shape, [0], 1, 0), null);
 */
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
    // hit testing walks outer-to-inner, but it first needs the inner extent for
    // each axis so rounded candidate indices can be verified against cell bounds.
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

/**
 * Computes the 3D display-space center for a tensor coordinate in the contiguous layout scheme.
 *
 * Contiguous layout assigns tensor axes to x, y, and z axis families, then uses the one-dimensional family projection for each component.
 *
 * @param coord - Tensor coordinate to place, indexed by axis in the same order as `shape`.
 * @param shape - Display-normalized tensor extents used to choose the contiguous axis families and center the layout.
 * @param dimensionBlockGapMultiple - Multiplier for gaps between nested dimension blocks in each axis family.
 * @returns Three.js vector whose x, y, and z components are the rendered cell center for `coord`.
 * @noThrows Delegates to deterministic axis-family selection and numeric position calculations; invalid hits are not evaluated here.
 * @example
 * const position = contiguousPosition3D([2], [3], 0);
 *
 * assert.equal(position.x, CELL_SIZE);
 * assert.equal(position.y, 0);
 * assert.equal(position.z, 0);
 */
function contiguousPosition3D(coord: number[], shape: number[], dimensionBlockGapMultiple: number): Vector3 {
    const rank = shape.length;
    return new Vector3(
        familyPosition1D(coord, shape, familyAxes(rank, '3d', 0, 'contiguous'), 1, dimensionBlockGapMultiple),
        familyPosition1D(coord, shape, familyAxes(rank, '3d', 1, 'contiguous'), -1, dimensionBlockGapMultiple),
        familyPosition1D(coord, shape, familyAxes(rank, '3d', 2, 'contiguous'), -1, dimensionBlockGapMultiple),
    );
}

/**
 * Measures the x, y, and z bounding extents occupied by a tensor in the contiguous 3D layout scheme.
 *
 * Each component is computed from the contiguous axis family assigned to that display direction, including nested dimension-block gaps.
 *
 * @param shape - Display-normalized tensor extents whose rank determines the contiguous axis-family assignment.
 * @param dimensionBlockGapMultiple - Multiplier for gaps between nested dimension blocks in each display direction.
 * @returns Object containing the total rendered width (`x`), height (`y`), and depth (`z`) required for the contiguous tensor layout.
 * @noThrows Performs axis-family lookup and arithmetic extent accumulation only; empty axis families resolve to a single cell extent.
 * @example
 * const extent = contiguousExtent3D([3], 0);
 *
 * assert.deepEqual(extent, {
 *   x: 3 * CELL_SIZE,
 *   y: CELL_SIZE,
 *   z: CELL_SIZE,
 * });
 */
function contiguousExtent3D(shape: number[], dimensionBlockGapMultiple: number): Extent3 {
    const rank = shape.length;
    return {
        x: familyExtent1D(shape, familyAxes(rank, '3d', 0, 'contiguous'), dimensionBlockGapMultiple),
        y: familyExtent1D(shape, familyAxes(rank, '3d', 1, 'contiguous'), dimensionBlockGapMultiple),
        z: familyExtent1D(shape, familyAxes(rank, '3d', 2, 'contiguous'), dimensionBlockGapMultiple),
    };
}

/**
 * Projects a tensor coordinate into the 2-D contiguous layout used by the viewer canvas.
 *
 * @param coord - Per-axis tensor indices in display-shape order; entries are read by the x and y contiguous axis families.
 * @param shape - Normalized display shape whose length determines the rank and axis-family mapping.
 * @param dimensionBlockGapMultiple - Scale factor applied to gaps between higher-dimensional blocks in the contiguous layout.
 * @returns Display-space cell center for the coordinate, with `x` from the horizontal axis family and `y` from the vertical axis family.
 * @noThrows Performs deterministic axis selection and numeric layout arithmetic for normalized inputs; it does not validate by throwing.
 * @example
 * const shape = normalizeDisplayShape([2, 3]);
 * const gap = normalizeDimensionBlockGapMultiple(0);
 * const coord = [1, 2];
 * const position = contiguousPosition2D(coord, shape, gap);
 *
 * // The projected center can be fed back into contiguous hit testing to recover the same tensor cell.
 * expect(contiguousHit2D(position, shape, gap)?.coord).toEqual(coord);
 */
function contiguousPosition2D(coord: number[], shape: number[], dimensionBlockGapMultiple: number): { x: number; y: number } {
    const rank = shape.length;
    return {
        x: familyPosition1D(coord, shape, familyAxes(rank, '2d', 0, 'contiguous'), 1, dimensionBlockGapMultiple),
        y: familyPosition1D(coord, shape, familyAxes(rank, '2d', 1, 'contiguous'), -1, dimensionBlockGapMultiple),
    };
}

/**
 * Computes the display-space width and height needed to contain every cell in a contiguous tensor layout.
 *
 * @param shape - Normalized display shape whose rank determines which axes contribute to the horizontal and vertical extents.
 * @param dimensionBlockGapMultiple - Scale factor applied to spacing between contiguous blocks for dimensions beyond the base axes.
 * @returns The `{ x, y }` span of the contiguous layout, used by callers to size or center the rendered tensor view.
 * @noThrows Uses only the supplied shape, axis-family mapping, and numeric extent arithmetic; unsupported hit locations are not involved.
 * @example
 * const shape = normalizeDisplayShape([2, 3]);
 * const gap = normalizeDimensionBlockGapMultiple(0);
 * const extent = contiguousExtent2D(shape, gap);
 *
 * expect(extent.x).toBeGreaterThan(0);
 * expect(extent.y).toBeGreaterThan(0);
 */
function contiguousExtent2D(shape: number[], dimensionBlockGapMultiple: number): Extent2 {
    const rank = shape.length;
    return {
        x: familyExtent1D(shape, familyAxes(rank, '2d', 0, 'contiguous'), dimensionBlockGapMultiple),
        y: familyExtent1D(shape, familyAxes(rank, '2d', 1, 'contiguous'), dimensionBlockGapMultiple),
    };
}

/**
 * Converts a display-space point into the tensor cell under that point in the contiguous 2-D layout.
 *
 * @param point - Canvas-layout coordinates to test, in the same coordinate space returned by `contiguousPosition2D`.
 * @param shape - Normalized display shape whose rank determines the horizontal and vertical contiguous axis families.
 * @param dimensionBlockGapMultiple - Scale factor for gaps between higher-dimensional blocks; must match the value used to position cells.
 * @returns The tensor coordinate and snapped cell-center position for a hit, or `null` when either axis-family test lands outside a cell.
 * @noThrows Misses are represented as `null`; the function only combines the results of numeric one-dimensional hit tests.
 * @example
 * const shape = normalizeDisplayShape([2, 3]);
 * const gap = normalizeDimensionBlockGapMultiple(0);
 * const center = contiguousPosition2D([1, 2], shape, gap);
 *
 * expect(contiguousHit2D(center, shape, gap)).toEqual({
 *   coord: [1, 2],
 *   position: center,
 * });
 * expect(contiguousHit2D({ x: Number.POSITIVE_INFINITY, y: 0 }, shape, gap)).toBeNull();
 */
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

/**
 * Hit-tests the base grid for a one- or two-dimensional display shape and returns the cell nearest the point.
 *
 * @param point - Display-space coordinates to test against the grid cell rectangles.
 * @param shapeInput - Display shape before local normalization; empty or missing dimensions are treated as a single visible cell.
 * @param cellExtent - Width and height of each rendered cell in display units.
 * @param level - Nesting level used to derive the gap between neighboring cells or blocks.
 * @param verticalFor1D - Whether a one-dimensional shape is arranged along the y axis instead of the x axis.
 * @param dimensionBlockGapMultiple - Scale factor used by `levelGap` to space cells at the requested nesting level.
 * @returns A `CoordHit2D` containing the tensor coordinate and snapped cell center, or `null` when the point is outside the candidate cell or grid bounds.
 * @noThrows Out-of-bounds and between-cell points return `null`; shape defaults and numeric bounds checks avoid throwing for ordinary layout misses.
 * @example
 * const cellExtent = { x: 10, y: 10 };
 * const gapMultiple = 0;
 *
 * expect(baseGridHit2D({ x: 0, y: 0 }, [1, 1], cellExtent, 0, false, gapMultiple)).toEqual({
 *   coord: [0, 0],
 *   position: { x: 0, y: 0 },
 * });
 * expect(baseGridHit2D({ x: 100, y: 100 }, [1, 1], cellExtent, 0, false, gapMultiple)).toBeNull();
 */
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

/**
 * Computes the 2D layout offset for one tensor cell by recursively nesting higher-rank axes around an inner two-axis grid.
 *
 * @param coord - Layout coordinate in axis order; each entry is the selected index for the corresponding entry in `shapeInput`.
 * @param shapeInput - Display shape for the visible layout axes before normalization; dimensions beyond the last two are split into outer blocks.
 * @param cellExtent - Width and height of a single innermost cell or nested block at this recursion level.
 * @param level - Current recursion depth, used to scale the gap inserted between dimension blocks.
 * @param verticalFor1D - Orientation to use when a two-dimensional split leaves a one-axis grid at this level.
 * @param originalRank - Rank of the original normalized display shape, used to keep z-order axis orientation stable across recursive splits.
 * @param axisOffset - Index of this recursive slice within the original axis list.
 * @param dimensionBlockGapMultiple - Multiplier applied to cell size when spacing nested dimension blocks.
 * @returns The `{ x, y }` center offset, relative to this recursion origin, where the requested cell should be drawn or used as a hit-test anchor.
 * @noThrows For finite numeric coordinates, extents, and gap values, the routine only normalizes the shape, slices arrays, recurses, and adds offsets; invalid or non-finite inputs are expected to be rejected by the public layout validation layer before this helper is called.
 * @example
 * const position = recursivePosition2D([1, 0, 2], [2, 2, 3], { x: 1, y: 1 }, 0, false, 3, 0, 0.25);
 * // The returned offset is the center used for rendering the cell at layout coordinate [1, 0, 2].
 * expect(Number.isFinite(position.x)).toBe(true);
 * expect(Number.isFinite(position.y)).toBe(true);
 */
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

/**
 * Maps a 2D pointer location back to the nested tensor coordinate whose drawn cell contains that point.
 *
 * @param point - Pointer position in the same local 2D layout coordinates produced by `recursivePosition2D`.
 * @param shapeInput - Display shape for the visible layout axes before normalization; higher-rank shapes are tested as outer blocks containing inner grids.
 * @param cellExtent - Width and height of a single innermost cell or nested block at this recursion level.
 * @param level - Current recursion depth, used to match the gap scale used when positions were generated.
 * @param verticalFor1D - Orientation to use when a two-dimensional split leaves a one-axis grid at this level.
 * @param originalRank - Rank of the original normalized display shape, used to keep z-order axis orientation stable across recursive splits.
 * @param axisOffset - Index of this recursive slice within the original axis list.
 * @param dimensionBlockGapMultiple - Multiplier applied to cell size when spacing nested dimension blocks.
 * @returns A hit containing the full layout coordinate and the cell center position, or `null` when the point falls outside every cell at this nested level.
 * @noThrows Pointer misses and gaps between nested blocks are represented as `null`; for validated numeric layout inputs the helper performs only shape normalization, arithmetic, slicing, and recursion.
 * @example
 * const center = recursivePosition2D([1, 0, 2], [2, 2, 3], { x: 1, y: 1 }, 0, false, 3, 0, 0.25);
 * const hit = recursiveHit2D(center, [2, 2, 3], { x: 1, y: 1 }, 0, false, 3, 0, 0.25);
 * expect(hit?.coord).toEqual([1, 0, 2]);
 * expect(hit?.position).toEqual(center);
 *
 * const miss = recursiveHit2D({ x: 10_000, y: 10_000 }, [2, 2, 3], { x: 1, y: 1 }, 0, false, 3, 0, 0.25);
 * expect(miss).toBeNull();
 */
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
    // subtract the outer block center before recursing into the inner grid; using
    // absolute coordinates here would make nested blocks impossible to pick.
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

/**
 * Converts a row-major flat storage index into the coordinate used by tensor view and layout code.
 *
 * @param index - Zero-based flat index into an array stored with the last shape axis varying fastest.
 * @param shapeInput - Per-axis extents for the tensor or view; the shape is normalized before modulo/division is applied.
 * @returns Per-axis coordinate in the same order as `shapeInput`, suitable for mapping view coordinates to tensor coordinates or filling mesh/color buffers.
 * @noThrows For numeric indices and normalized positive extents, conversion uses only modulo, division, and array writes; callers are responsible for passing an index within the tensor's element count.
 * @example
 * expect(unravelIndex(17, [2, 3, 4])).toEqual([1, 1, 1]);
 * expect(unravelIndex(5, [2, 3])).toEqual([1, 2]);
 */
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

/**
 * Returns the Three.js world-space position for the cube representing one visible tensor layout coordinate.
 *
 * @param coord - Layout coordinate after view slicing/grouping has been mapped into display-axis order.
 * @param shape - Display shape for the visible layout axes; it determines block nesting and axis orientation.
 * @param dimensionBlockGapMultiple - Gap size, as a multiple of the base cell size, inserted between higher-dimensional blocks.
 * @param scheme - Dimension mapping strategy: `contiguous` packs axes directly, while `z-order` recursively groups axes into spatial blocks.
 * @returns A `Vector3` center position that mesh builders, ghost layers, and axis guides can use to place objects in the 3D scene.
 * @noThrows The function chooses a layout strategy, normalizes the gap/shape values, and performs deterministic coordinate arithmetic; unsupported schemes are excluded by the `DimensionMappingScheme` type.
 * @example
 * const origin = displayPositionForCoord([0, 0, 0], [2, 3, 4], 0.25, 'z-order');
 * const neighbor = displayPositionForCoord([0, 0, 1], [2, 3, 4], 0.25, 'z-order');
 * expect(origin).toBeInstanceOf(Vector3);
 * expect(neighbor.equals(origin)).toBe(false);
 */
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

/**
 * Compute the world-space bounding span for the rendered cells of a tensor layout in 3D mode.
 *
 * @param shape - Visible layout dimensions after view slicing, grouping, and hidden-axis handling; each entry is the cell count along that layout axis.
 * @param dimensionBlockGapMultiple - Multiplier applied to the standard gap inserted between higher-dimensional blocks; omitted values use the viewer default gap.
 * @param scheme - Dimension-to-space mapping strategy: `'z-order'` recursively tiles tensor blocks through 3D space, while `'contiguous'` lays normalized dimensions in a contiguous grid.
 * @returns A `Vector3` whose `x`, `y`, and `z` components are the total world-space span occupied by the rendered layout, suitable for outlines, bounding volumes, and camera framing.
 * @noThrows For numeric layout shapes and gap values, this wrapper only normalizes the inputs, chooses the mapping algorithm, and returns a new `Vector3`; it performs no explicit validation that throws.
 * @example
 * const extent = displayExtent([2, 3, 4], 1, 'z-order');
 *
 * expect(extent).toBeInstanceOf(Vector3);
 * expect(extent.x).toBeGreaterThan(0);
 * expect(extent.y).toBeGreaterThan(0);
 * expect(extent.z).toBeGreaterThan(0);
 */
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

/**
 * Map one tensor layout coordinate to the center of its rendered cell in 2D world space.
 *
 * @param coord - Zero-based layout coordinate for a visible tensor cell; its entries correspond to the axes described by `shape`.
 * @param shape - Visible layout dimensions after view slicing, grouping, and hidden-axis handling; each entry is the cell count along that layout axis.
 * @param dimensionBlockGapMultiple - Multiplier applied to the standard gap between tiled dimension blocks before the coordinate is placed.
 * @param scheme - Dimension-to-space mapping strategy used for the same tensor mesh, either recursive `'z-order'` tiling or `'contiguous'` placement.
 * @returns The `{ x, y }` world-space center of the rendered cell, used by labels, selection overlays, hit testing, and canvas projection.
 * @noThrows For numeric coordinates, shapes, and gap values, this wrapper only normalizes layout inputs and delegates to pure coordinate mappers with no explicit throwing branch.
 * @example
 * const position = displayPositionForCoord2D([1, 2], [2, 3], 1, 'contiguous');
 * const hit = displayHitForPoint2D(position.x, position.y, [2, 3], 1, 'contiguous');
 *
 * expect(hit?.coord).toEqual([1, 2]);
 */
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

/**
 * Compute the world-space bounding span for the rendered cells of a tensor layout in 2D mode.
 *
 * @param shape - Visible layout dimensions after view slicing, grouping, and hidden-axis handling; each entry is the cell count along that layout axis.
 * @param dimensionBlockGapMultiple - Multiplier applied to the standard gap inserted between tiled dimension blocks; omitted values use the viewer default gap.
 * @param scheme - Dimension-to-space mapping strategy: `'z-order'` recursively tiles higher dimensions into a plane, while `'contiguous'` places normalized dimensions in a contiguous 2D grid.
 * @returns An object whose `x` and `y` components are the total world-space width and height occupied by the rendered layout, suitable for 2D outlines, pick bounds, and label scaling.
 * @noThrows For numeric layout shapes and gap values, this wrapper only normalizes the inputs and dispatches to deterministic extent calculations with no explicit throwing branch.
 * @example
 * const extent = displayExtent2D([2, 3], 1, 'contiguous');
 *
 * expect(extent.x).toBeGreaterThan(0);
 * expect(extent.y).toBeGreaterThan(0);
 */
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

/**
 * Hit-test a 2D world-space point against the rendered cells of one tensor layout.
 *
 * @param x - Layout-local world-space x coordinate, usually after subtracting the tensor's rendered x offset from a pointer or canvas point.
 * @param y - Layout-local world-space y coordinate, usually after subtracting the tensor's rendered y offset from a pointer or canvas point.
 * @param shape - Visible layout dimensions after view slicing, grouping, and hidden-axis handling; each entry is the cell count along that layout axis.
 * @param dimensionBlockGapMultiple - Multiplier applied to the standard gap between tiled dimension blocks; the same value must be used for rendering and hit testing.
 * @param scheme - Dimension-to-space mapping strategy used to render the tensor, either recursive `'z-order'` tiling or `'contiguous'` placement.
 * @returns The hit record for the cell under the point, including its layout coordinate, or `null` when the point falls outside every rendered cell or inside a layout gap.
 * @noThrows For finite numeric points, layout shapes, and gap values, hit testing only normalizes the shape and delegates to deterministic hit-test helpers; this wrapper has no explicit throwing branch.
 * @example
 * const shape = [2, 3];
 * const cellCenter = displayPositionForCoord2D([1, 2], shape, 1, 'contiguous');
 *
 * expect(displayHitForPoint2D(cellCenter.x, cellCenter.y, shape, 1, 'contiguous')?.coord).toEqual([1, 2]);
 * expect(displayHitForPoint2D(10_000, 10_000, shape, 1, 'contiguous')).toBeNull();
 */
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
