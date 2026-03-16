import { describe, expect, it } from 'vitest';
import { displayExtent2D, displayHitForPoint2D, displayPositionForCoord2D } from './layout.js';

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
});
