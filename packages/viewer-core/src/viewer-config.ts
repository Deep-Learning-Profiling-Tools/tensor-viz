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
 * Optional constructor settings that customize the viewer before any session is
 * loaded, including canvas background color and lazy tensor-byte retrieval.
 *
 * `requestTensorData` is called when the viewer needs numeric values for a
 * tensor that was loaded from metadata only. Returning `null` or `undefined`
 * leaves that tensor without client-side numeric data for the requested reason.
 *
 * @example
 * const options: ViewerOptions = {
 *   background: '#111827',
 *   requestTensorData: (tensor, reason) => {
 *     if (tensor.id === 'logits' && reason === 'render') {
 *       return new Float32Array([0.1, 0.7, 0.2]);
 *     }
 *     return null;
 *   },
 * };
 *
 * const loaded = options.requestTensorData?.(
 *   { id: 'logits' } as TensorStatus,
 *   'render' as TensorDataRequestReason,
 * );
 * // loaded is the Float32Array used to populate the logits tensor.
 */
export type ViewerOptions = {
    background?: string;
    requestTensorData?: (tensor: TensorStatus, reason: TensorDataRequestReason) => Promise<NumericArray | null | undefined> | NumericArray | null | undefined;
};

/**
 * Metadata attached to each rendered tensor `InstancedMesh` so pointer hits can
 * map a Three.js instance id back to the owning tensor and its view-coordinate
 * shape.
 *
 * `instanceShape` is the multidimensional shape used to unravel an instance id;
 * it matches the rendered tensor view rather than the original persisted tensor
 * shape when axes have been sliced or grouped.
 *
 * @example
 * const meta: MeshMeta = {
 *   tensorId: 'activation-layer-3',
 *   instanceShape: [2, 4, 8],
 * };
 *
 * // A hit on instance 13 belongs to tensor "activation-layer-3" and can be
 * // unraveled against [2, 4, 8] to recover the displayed cell coordinate.
 * console.assert(meta.tensorId === 'activation-layer-3');
 * console.assert(meta.instanceShape.length === 3);
 */
export type MeshMeta = {
    tensorId: string;
    instanceShape: number[];
};

/**
 * Pickable tensor mesh plus its cached 3D and 2D extents used by hover,
 * click, and box-selection hit testing.
 *
 * The viewer rebuilds these records alongside the rendered tensor meshes so hit
 * tests use bounds that match the current view, slice, gap, and display mode.
 *
 * @example
 * const mesh = new InstancedMesh(geometry, material, 6);
 * const pickMesh: PickMesh = {
 *   tensorId: 'weights',
 *   mesh,
 *   bounds: new Box3().setFromCenterAndSize(
 *     new Vector3(0, 0, 0),
 *     new Vector3(4, 2, 1),
 *   ),
 *   rect2D: { minX: -2, maxX: 2, minY: -1, maxY: 1 },
 * };
 *
 * const pointerInside2D =
 *   0 >= pickMesh.rect2D.minX &&
 *   0 <= pickMesh.rect2D.maxX &&
 *   0 >= pickMesh.rect2D.minY &&
 *   0 <= pickMesh.rect2D.maxY;
 * console.assert(pointerInside2D);
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
 * Mutable state captured while the user drags a selection rectangle across the
 * canvas, including the drag origin, current pointer position, selection mode,
 * and the base and preview cell sets being merged for feedback.
 *
 * `startWorld` is present for 2D canvas-space drags that need projection back to
 * client coordinates; 3D drags can keep it `null` and use client coordinates for
 * the selection box.
 *
 * @example
 * const drag: SelectionDragState = {
 *   source: '2d',
 *   mode: 'add',
 *   tensorId: 'activations',
 *   startClient: { x: 100, y: 120 },
 *   startWorld: { x: -1.5, y: 0.25 },
 *   currentClient: { x: 220, y: 260 },
 *   baseSelections: new Map([['activations', new Set(['0,0'])]]),
 *   previewSelections: new Map([['activations', new Set(['0,0', '0,1'])]]),
 * };
 *
 * console.assert(drag.mode === 'add');
 * console.assert(drag.previewSelections.get('activations')?.has('0,1'));
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
 * Uniform bundle stored on selection-preview materials so the viewer can toggle the overlay, pass the dragged rectangle, choose the preview mode, and tint selected cells without rebuilding the mesh.
 *
 * @example
 * const uniforms: SelectionPreviewUniforms = {
 *     selectionPreviewActive: { value: 1 },
 *     selectionPreviewBounds: { value: new Vector4(12, 24, 96, 144) },
 *     selectionPreviewMode: { value: 0 },
 *     selectionColor: { value: new Color('#f97316') },
 * };
 *
 * uniforms.selectionPreviewActive.value; // 1, so the shader should draw the preview overlay.
 */
export type SelectionPreviewUniforms = {
    selectionPreviewActive: { value: number };
    selectionPreviewBounds: { value: Vector4 };
    selectionPreviewMode: { value: number };
    selectionColor: { value: Color };
};

/**
 * Writes a viewer diagnostic event to the console with the viewer log prefix when core logging is enabled.
 *
 * @param event - Colon-delimited viewer event name such as `viewer:init`, `2d:zoom`, or `selection:update`.
 * @param details - Optional payload to print after the event name, commonly a zoom value, hover record, tensor id, or selection metadata object.
 * @returns Nothing. The only observable effect is a `console.log` call when logging is enabled; disabled logging returns before writing anything.
 * @noThrows The helper performs no parsing or validation and has no explicit throw branch; it only checks the logging flag and forwards the provided values to `console.log`.
 * @example
 * logEvent('2d:zoom', { zoom: 1.25 });
 * // With viewer logging enabled, the console receives the viewer log prefix,
 * // "2d:zoom", and the payload object { zoom: 1.25 }.
 */
export function logEvent(event: string, details?: unknown): void {
    if (!LOG_ENABLED) return;
    if (details === undefined) console.log(LOG_PREFIX, event);
    else console.log(LOG_PREFIX, event, details);
}

/**
 * Converts a proposed 2D canvas zoom factor into the safe range used by camera sync, wheel zooming, snapshots, and fit-to-view calculations.
 *
 * @param value - Proposed canvas zoom multiplier from viewer state, wheel input, snapshot data, or fit-to-view math.
 * @returns The minimum canvas zoom for `NaN`, zero, or negative inputs; otherwise the requested positive zoom capped at `Number.MAX_VALUE`.
 * @noThrows Invalid numeric zooms are normalized to a fallback value instead of being rejected, and the implementation uses only `Number.isNaN` and `Math.min`.
 * @example
 * normalizeCanvasZoom(2); // 2
 * normalizeCanvasZoom(0); // MIN_CANVAS_ZOOM
 * normalizeCanvasZoom(Number.NaN); // MIN_CANVAS_ZOOM
 */
export function normalizeCanvasZoom(value: number): number {
    if (Number.isNaN(value) || value <= 0) return MIN_CANVAS_ZOOM;
    return Math.min(Number.MAX_VALUE, value);
}
