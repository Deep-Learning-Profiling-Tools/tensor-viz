import { Box3, Color, InstancedMesh, Vector4 } from 'three';
import type { NumericArray, TensorDataRequestReason, TensorStatus } from './types.js';

export const BASE_COLOR = new Color('#90a4ae');
export const ACTIVE_COLOR = new Color('#1976d2');
export const HOVER_COLOR = new Color('#f59e0b');
export const CANVAS_WORLD_SCALE = 4;
export const MIN_CANVAS_ZOOM = 1e-9;
export const MIN_CANVAS_FIT_INSET = 16;
export const MAX_CANVAS_FIT_INSET = 48;
export const AUTO_FIT_2D_SCALE = 0.75;
export const AUTO_FIT_3D_DISTANCE_SCALE = 1.25;
export const DEFAULT_TENSOR_SPACING = 4;

const LOG_PREFIX = '[tensor-viz]';
const LOG_ENABLED = (() => {
    try {
        return typeof window !== 'undefined'
            && ((window as { __TENSOR_VIZ_DEBUG__?: boolean }).__TENSOR_VIZ_DEBUG__ === true
                || window.localStorage.getItem('tensor-viz-debug') === '1');
    } catch {
        return false;
    }
})();

/**
 * Construction-time viewer options that do not need full state persistence.
 *
 * @example
 * const value: ViewerOptions = {} as ViewerOptions;
 */
export type ViewerOptions = {
    background?: string;
    requestTensorData?: (tensor: TensorStatus, reason: TensorDataRequestReason) => Promise<NumericArray | null | undefined> | NumericArray | null | undefined;
};

/**
 * shape of mesh meta data used by the viewer.
 *
 * @example
 * const value: MeshMeta = {} as MeshMeta;
 */
export type MeshMeta = {
    tensorId: string;
    instanceShape: number[];
};

/**
 * shape of pick mesh data used by the viewer.
 *
 * @example
 * const value: PickMesh = {} as PickMesh;
 */
export type PickMesh = {
    tensorId: string;
    mesh: InstancedMesh;
    bounds: Box3;
    rect2D: {
        minX: number;
        maxX: number;
        minY: number;
        maxY: number;
    };
};

/**
 * shape of selection drag state data used by the viewer.
 *
 * @example
 * const value: SelectionDragState = {} as SelectionDragState;
 */
export type SelectionDragState = {
    source: '2d' | '3d';
    mode: 'replace' | 'add' | 'remove';
    tensorId: string | null;
    startClient: { x: number; y: number };
    startWorld: { x: number; y: number } | null;
    currentClient: { x: number; y: number };
    baseSelections: Map<string, Set<string>>;
    previewSelections: Map<string, Set<string>>;
};

/**
 * shape of selection preview uniforms data used by the viewer.
 *
 * @example
 * const value: SelectionPreviewUniforms = {} as SelectionPreviewUniforms;
 */
export type SelectionPreviewUniforms = {
    selectionPreviewActive: { value: number };
    selectionPreviewBounds: { value: Vector4 };
    selectionPreviewMode: { value: number };
    selectionColor: { value: Color };
};

/**
 * return log event for the current viewer state.
 *
 * @param event - Browser event that triggered this handler.
 * @param details - details input used by this operation (unknown).
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * logEvent(event, details);
 */
export function logEvent(event: string, details?: unknown): void {
    if (!LOG_ENABLED) return;
    if (details === undefined) console.log(LOG_PREFIX, event);
    else console.log(LOG_PREFIX, event, details);
}

/**
 * normalize canvas zoom for the current viewer state.
 *
 * @param value - Value supplied by the caller.
 * @returns Numeric result computed from the inputs.
 * @noThrows This function has no direct throw path.
 * @example
 * normalizeCanvasZoom(value);
 */
export function normalizeCanvasZoom(value: number): number {
    if (Number.isNaN(value) || value <= 0) return MIN_CANVAS_ZOOM;
    return Math.min(Number.MAX_VALUE, value);
}
