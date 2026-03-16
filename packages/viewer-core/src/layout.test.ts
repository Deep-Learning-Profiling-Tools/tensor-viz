import { describe, expect, it } from 'vitest';
import { axisWorldKeyForMode, displayExtent2D, displayHitForPoint2D, displayPositionForCoord, displayPositionForCoord2D } from './layout.js';

describe('displayHitForPoint2D', () => {
    it('inverts recursive 2d positions for higher-rank shapes', () => {
        const shape = [2, 3, 2, 2];
        for (let a = 0; a < shape[0]; a += 1) {
            for (let b = 0; b < shape[1]; b += 1) {
                for (let c = 0; c < shape[2]; c += 1) {
                    for (let d = 0; d < shape[3]; d += 1) {
                        const coord = [a, b, c, d];
                        const position = displayPositionForCoord2D(coord, shape);
                        const hit = displayHitForPoint2D(position.x, position.y, shape);
                        expect(hit?.coord).toEqual(coord);
                        expect(hit?.position).toEqual(position);
                    }
                }
            }
        }
    });

    it('returns null for points in layout gaps', () => {
        const left = displayPositionForCoord2D([0, 0, 0], [2, 2, 2]);
        const right = displayPositionForCoord2D([0, 0, 1], [2, 2, 2]);
        expect(displayHitForPoint2D((left.x + right.x) / 2, left.y, [2, 2, 2])).toBeNull();
    });

    it('uses the configured dimension block gap multiple', () => {
        const shape = [2, 2, 2];
        const compactExtent = displayExtent2D(shape, 1);
        const wideExtent = displayExtent2D(shape, 5);
        expect(wideExtent.x).toBeGreaterThan(compactExtent.x);
        const position = displayPositionForCoord2D([1, 1, 1], shape, 5);
        expect(displayHitForPoint2D(position.x, position.y, shape, 5)?.coord).toEqual([1, 1, 1]);
    });

    it('treats touching cells as pickable when gaps are disabled', () => {
        const left = displayPositionForCoord2D([0, 0, 0], [2, 2, 2], 0);
        const right = displayPositionForCoord2D([0, 0, 1], [2, 2, 2], 0);
        expect(displayHitForPoint2D((left.x + right.x) / 2, left.y, [2, 2, 2], 0)?.coord).toEqual([0, 0, 1]);
    });

    it('supports the contiguous dimension mapping scheme', () => {
        expect(Array.from({ length: 5 }, (_entry, axis) => axisWorldKeyForMode('2d', 5, axis, 'contiguous'))).toEqual([1, 1, 0, 0, 0]);
        expect(Array.from({ length: 5 }, (_entry, axis) => axisWorldKeyForMode('3d', 5, axis, 'contiguous'))).toEqual([2, 1, 1, 0, 0]);
        const shape = [2, 3, 2, 2];
        for (let a = 0; a < shape[0]; a += 1) {
            for (let b = 0; b < shape[1]; b += 1) {
                for (let c = 0; c < shape[2]; c += 1) {
                    for (let d = 0; d < shape[3]; d += 1) {
                        const coord = [a, b, c, d];
                        const position = displayPositionForCoord2D(coord, shape, 3, 'contiguous');
                        const hit = displayHitForPoint2D(position.x, position.y, shape, 3, 'contiguous');
                        expect(hit?.coord).toEqual(coord);
                    }
                }
            }
        }
    });

    it('keeps contiguous 3d families on their assigned world axes', () => {
        const shape = [2, 2, 2, 2, 2];
        const origin = displayPositionForCoord([0, 0, 0, 0, 0], shape, 3, 'contiguous');
        const xAxis = displayPositionForCoord([0, 0, 0, 0, 1], shape, 3, 'contiguous').sub(origin.clone());
        const yAxis = displayPositionForCoord([0, 0, 1, 0, 0], shape, 3, 'contiguous').sub(origin.clone());
        const zAxis = displayPositionForCoord([1, 0, 0, 0, 0], shape, 3, 'contiguous').sub(origin.clone());
        expect(xAxis.x).toBeGreaterThan(0);
        expect(xAxis.y).toBe(0);
        expect(xAxis.z).toBe(0);
        expect(yAxis.x).toBe(0);
        expect(yAxis.y).toBeLessThan(0);
        expect(yAxis.z).toBe(0);
        expect(zAxis.x).toBe(0);
        expect(zAxis.y).toBe(0);
        expect(zAxis.z).toBeLessThan(0);
    });

    it('scales contiguous gaps by nesting depth within each family', () => {
        const shape = [2, 2, 2, 2, 2, 2];
        const origin = displayPositionForCoord2D([0, 0, 0, 0, 0, 0], shape, 3, 'contiguous');
        const yMiddle = displayPositionForCoord2D([0, 1, 0, 0, 0, 0], shape, 3, 'contiguous');
        const xMiddle = displayPositionForCoord2D([0, 0, 0, 0, 1, 0], shape, 3, 'contiguous');
        const yOuter = displayPositionForCoord2D([1, 0, 0, 0, 0, 0], shape, 3, 'contiguous');
        const xOuter = displayPositionForCoord2D([0, 0, 0, 1, 0, 0], shape, 3, 'contiguous');
        const inner = displayPositionForCoord2D([0, 0, 1, 0, 0, 0], shape, 3, 'contiguous');
        expect(Math.abs(yMiddle.y - origin.y)).toBeCloseTo(Math.abs(xMiddle.x - origin.x));
        expect(Math.abs(yOuter.y - origin.y)).toBeCloseTo(Math.abs(xOuter.x - origin.x));
        expect(Math.abs(yMiddle.y - origin.y)).toBeGreaterThan(Math.abs(inner.y - origin.y));
    });
});
