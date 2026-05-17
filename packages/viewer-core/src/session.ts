import { DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE } from './layout.js';
import { defaultTensorViewEditor } from './view.js';
import type {
    BundleManifest,
    ColorInstruction,
    DType,
    SessionBundleManifest,
    TensorViewSnapshot,
    Vec3,
    ViewerSnapshot,
} from './types.js';
import { validateTensorShape } from './validation.js';

/**
 * Describes one tensor entry that a bundle or session manifest can load into
 * the viewer, including its logical name, element type, shape, optional byte
 * source, and optional view/rendering metadata.
 *
 * @example
 * const tensor: SessionTensorSpec = {
 *     name: 'activations.layer1',
 *     dtype: 'float32',
 *     shape: [2, 3, 4],
 *     axisLabels: ['batch', 'row', 'column'],
 *     dataFile: 'activations.layer1.f32',
 *     view: { expression: 'batch,row,column' },
 * };
 *
 * // The session builder can derive an id when one is not supplied.
 * expect(tensor.name).toBe('activations.layer1');
 */
export type SessionTensorSpec = {
    id?: string;
    name: string;
    dtype: DType;
    shape: number[];
    axisLabels?: string[];
    dataFile?: string;
    placeholderData?: boolean;
    offset?: Vec3;
    view?: TensorViewSnapshot;
    colorInstructions?: ColorInstruction[];
    markerCoords?: number[][];
};

/**
 * Describes the single-document manifest input accepted by bundle builders:
 * optional viewer settings plus the tensor declarations that should appear in
 * one viewer load unit.
 *
 * @example
 * const document: BundleDocumentSpec = {
 *     viewer: { displayMode: '2d', showTensorNames: false },
 *     tensors: [
 *         {
 *             name: 'weights',
 *             dtype: 'float32',
 *             shape: [4, 4],
 *             dataFile: 'weights.f32',
 *         },
 *     ],
 * };
 *
 * expect(document.tensors).toHaveLength(1);
 * expect(document.viewer?.showTensorNames).toBe(false);
 */
export type BundleDocumentSpec = {
    viewer?: Partial<ViewerSnapshot>;
    tensors: SessionTensorSpec[];
};

/**
 * Describes one tab in a multi-tab session manifest, combining a tab title and
 * optional stable tab id with the bundle document fields for that tab's tensors
 * and viewer overrides.
 *
 * @example
 * const tab: SessionTabSpec = {
 *     id: 'encoder-tab',
 *     title: 'Encoder activations',
 *     viewer: { activeTensorId: 'tensor-1' },
 *     tensors: [
 *         {
 *             id: 'tensor-1',
 *             name: 'encoder.block0.output',
 *             dtype: 'float32',
 *             shape: [1, 12, 64],
 *         },
 *     ],
 * };
 *
 * expect(tab.title).toBe('Encoder activations');
 * expect(tab.tensors[0]?.id).toBe('tensor-1');
 */
export type SessionTabSpec = BundleDocumentSpec & {
    id?: string;
    title: string;
};

/**
 * Builds the viewer-state portion of a bundle or session manifest, filling in
 * default display, panel, camera, and tensor-selection settings while preserving
 * any explicit viewer overrides supplied by the caller.
 *
 * Tensor metadata and payload references are not created here; they belong in
 * bundle or session manifests.
 *
 * @param overrides - Partial viewer snapshot fields to write over the defaults, such as display mode, panel visibility, camera position, tensor summaries, or active tensor id.
 * @returns A complete viewer snapshot suitable for embedding in a manifest or passing to viewer initialization code.
 * @noThrows The builder only reads optional properties and applies nullish-coalescing defaults; it performs no validation, parsing, I/O, or tensor loading.
 * @example
 * const snapshot = createViewerSnapshot({
 *     displayMode: '2d',
 *     showTensorNames: false,
 *     camera: { position: [1, 2, 10] },
 * });
 *
 * expect(snapshot.version).toBe(1);
 * expect(snapshot.displayMode).toBe('2d');
 * expect(snapshot.showTensorNames).toBe(false);
 * expect(snapshot.camera.position).toEqual([1, 2, 10]);
 * expect(snapshot.camera.target).toEqual([0, 0, 0]);
 */
export function createViewerSnapshot(overrides: Partial<ViewerSnapshot> = {}): ViewerSnapshot {
    return {
        version: 1,
        displayMode: overrides.displayMode ?? '2d',
        heatmap: overrides.heatmap ?? true,
        dimensionBlockGapMultiple: overrides.dimensionBlockGapMultiple ?? DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
        displayGaps: overrides.displayGaps ?? false,
        logScale: overrides.logScale ?? false,
        collapseHiddenAxes: overrides.collapseHiddenAxes ?? overrides.showSlicesInSamePlace ?? false,
        dimensionMappingScheme: overrides.dimensionMappingScheme ?? 'z-order',
        showDimensionLines: overrides.showDimensionLines ?? true,
        showTensorNames: overrides.showTensorNames ?? true,
        showInspectorPanel: overrides.showInspectorPanel ?? true,
        showSelectionPanel: overrides.showSelectionPanel ?? true,
        showHoverDetailsPanel: overrides.showHoverDetailsPanel ?? true,
        camera: {
            position: overrides.camera?.position ?? [0, 0, 30],
            target: overrides.camera?.target ?? [0, 0, 0],
            rotation: overrides.camera?.rotation ?? [0, 0, 0],
            zoom: overrides.camera?.zoom ?? 1,
        },
        tensors: overrides.tensors ?? [],
        activeTensorId: overrides.activeTensorId ?? null,
    };
}

/**
 * Normalizes the per-tensor view snapshot stored in a manifest entry.
 *
 * When a saved view is present, its editor and hidden indices are preserved.
 * Otherwise the helper creates the default tensor-view editor for the tensor's
 * shape and optional axis labels, and starts with no hidden dimensions.
 *
 * @param shape - Validated tensor dimensions used to build the default editor when no saved editor is provided.
 * @param axisLabels - Optional label for each tensor axis, used by the default editor expression.
 * @param view - Previously saved tensor view snapshot whose editor or hidden indices should be reused.
 * @returns A manifest-ready tensor view snapshot containing an editor and a hidden-index list.
 * @noThrows This helper only reads optional fields and calls the default editor builder with an already-validated shape.
 * @example
 * const view = tensorView([4, 8], ['O', 'I']);
 *
 * expect(view.hiddenIndices).toEqual([]);
 * expect(view.editor.version).toBe(2);
 */
function tensorView(shape: number[], axisLabels: string[] | undefined, view?: TensorViewSnapshot): TensorViewSnapshot {
    return {
        editor: view?.editor ?? defaultTensorViewEditor(shape, axisLabels),
        hiddenIndices: view?.hiddenIndices ?? [],
    };
}

/**
 * Build one bundle manifest from lightweight tensor specs plus optional viewer overrides.
 *
 * A bundle manifest is one viewer document: one viewer snapshot plus one or
 * more tensor declarations for a single tab/load unit. Missing tensor ids are
 * assigned as `tensor-1`, `tensor-2`, and so on; tensors without a data file are
 * marked as placeholder-backed.
 *
 * @param spec - Bundle document spec containing tensor metadata, optional binary payload references, and optional viewer state overrides.
 * @returns A versioned bundle manifest whose viewer tensor list mirrors the normalized tensor declarations.
 * @throws Error when any tensor shape fails runtime validation, such as a dimension list that would exceed the viewer's supported cell count.
 * @example
 * const manifest = createBundleManifest({
 *   tensors: [{ name: 'weights', dtype: 'float32', shape: [4, 8], axisLabels: ['O', 'I'] }],
 * });
 *
 * expect(manifest.version).toBe(1);
 * expect(manifest.tensors[0].id).toBe('tensor-1');
 * expect(manifest.viewer.activeTensorId).toBe('tensor-1');
 * expect(manifest.tensors[0].placeholderData).toBe(true);
 *
 * @example
 * expect(() => createBundleManifest({
 *   tensors: [{ name: 'too-large', dtype: 'float32', shape: [Number.MAX_SAFE_INTEGER] }],
 * })).toThrow(Error);
 */
export function createBundleManifest(spec: BundleDocumentSpec): BundleManifest {
    const tensors = spec.tensors.map((tensor, index) => {
        const id = tensor.id ?? `tensor-${index + 1}`;
        const shape = validateTensorShape(tensor.shape);
        const view = tensorView(shape, tensor.axisLabels, tensor.view);
        return {
            id,
            name: tensor.name,
            dtype: tensor.dtype,
            shape,
            axisLabels: tensor.axisLabels,
            byteOrder: 'little' as const,
            dataFile: tensor.dataFile,
            placeholderData: tensor.placeholderData ?? !tensor.dataFile,
            offset: tensor.offset,
            view,
            colorInstructions: tensor.colorInstructions,
            markerCoords: tensor.markerCoords,
        };
    });
    const viewer = createViewerSnapshot(spec.viewer);
    viewer.tensors = tensors.map((tensor) => ({
        id: tensor.id,
        name: tensor.name,
        offset: tensor.offset,
        view: tensor.view,
    }));
    viewer.activeTensorId ??= viewer.tensors[0]?.id ?? null;
    return {
        version: 1,
        viewer,
        tensors,
    };
}

/**
 * Build a multi-tab session manifest from lightweight tab specs.
 *
 * Each tab wraps the bundle-level `viewer` plus `tensors` pair with a tab id
 * and title. Missing tab ids are assigned as `tab-1`, `tab-2`, and so on.
 *
 * @param tabs - Ordered tab specs, each containing a title plus the tensor and viewer data accepted by {@link createBundleManifest}.
 * @returns A versioned session manifest whose tabs contain normalized bundle manifests for viewer loading.
 * @throws Error when any tensor shape in any tab fails bundle-manifest validation.
 * @example
 * const session = createSessionBundleManifest([
 *   { title: 'inputs', tensors: [{ name: 'x', dtype: 'float32', shape: [2, 2] }] },
 *   { id: 'outputs', title: 'outputs', tensors: [{ name: 'y', dtype: 'float32', shape: [1] }] },
 * ]);
 *
 * expect(session.version).toBe(1);
 * expect(session.tabs.map((tab) => tab.id)).toEqual(['tab-1', 'outputs']);
 * expect(session.tabs[0].viewer.activeTensorId).toBe('tensor-1');
 *
 * @example
 * expect(() => createSessionBundleManifest([
 *   { title: 'bad', tensors: [{ name: 'too-large', dtype: 'float32', shape: [Number.MAX_SAFE_INTEGER] }] },
 * ])).toThrow(Error);
 */
export function createSessionBundleManifest(tabs: SessionTabSpec[]): SessionBundleManifest {
    return {
        version: 1,
        tabs: tabs.map((tab, index) => {
            const manifest = createBundleManifest(tab);
            return {
                id: tab.id ?? `tab-${index + 1}`,
                title: tab.title,
                viewer: manifest.viewer,
                tensors: manifest.tensors,
            };
        }),
    };
}
