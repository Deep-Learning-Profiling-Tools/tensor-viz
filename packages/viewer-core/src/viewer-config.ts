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
export const DEFAULT_TENSOR_SPACING = 2;

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

/** Construction-time viewer options that do not need full state persistence. */
export type ViewerOptions = {
    background?: string;
    requestTensorData?: (tensor: TensorStatus, reason: TensorDataRequestReason) => Promise<NumericArray | null | undefined> | NumericArray | null | undefined;
};

export type MeshMeta = {
    tensorId: string;
    instanceShape: number[];
};

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

export type SelectionDragState = {
    source: '2d' | '3d';
    mode: 'replace' | 'add' | 'remove';
    tensorId: string | null;
    startClient: { x: number; y: number };
    currentClient: { x: number; y: number };
    baseSelections: Map<string, Set<string>>;
    previewSelections: Map<string, Set<string>>;
};

export type SelectionPreviewUniforms = {
    selectionPreviewActive: { value: number };
    selectionPreviewBounds: { value: Vector4 };
    selectionPreviewMode: { value: number };
    selectionColor: { value: Color };
};

export function logEvent(event: string, details?: unknown): void {
    if (!LOG_ENABLED) return;
    if (details === undefined) console.log(LOG_PREFIX, event);
    else console.log(LOG_PREFIX, event, details);
}

export function normalizeCanvasZoom(value: number): number {
    if (Number.isNaN(value) || value <= 0) return MIN_CANVAS_ZOOM;
    return Math.min(Number.MAX_VALUE, value);
}
