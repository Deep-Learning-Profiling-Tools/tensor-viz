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

/** Input spec for one tensor entry in a bundle/session manifest. */
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

/** Input spec for one viewer document containing multiple tensors. */
export type BundleDocumentSpec = {
    viewer?: Partial<ViewerSnapshot>;
    tensors: SessionTensorSpec[];
};

/** Input spec for one tab in a multi-tab session manifest. */
export type SessionTabSpec = BundleDocumentSpec & {
    id?: string;
    title: string;
};

/** Build the default viewer snapshot used by bundle/session builders.
 *
 * This returns viewer state only. Tensor metadata and payload references live
 * in bundle or session manifests.
 */
export function createViewerSnapshot(overrides: Partial<ViewerSnapshot> = {}): ViewerSnapshot {
    return {
        version: 1,
        displayMode: overrides.displayMode ?? '2d',
        heatmap: overrides.heatmap ?? true,
        dimensionBlockGapMultiple: overrides.dimensionBlockGapMultiple ?? DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
        displayGaps: overrides.displayGaps ?? true,
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

function tensorView(shape: number[], axisLabels: string[] | undefined, view?: TensorViewSnapshot): TensorViewSnapshot {
    return {
        editor: view?.editor ?? defaultTensorViewEditor(shape, axisLabels),
        hiddenIndices: view?.hiddenIndices ?? [],
    };
}

/** Build one bundle manifest from lightweight tensor specs plus optional viewer overrides.
 *
 * A bundle manifest is one viewer document: one viewer snapshot plus one or
 * more tensor declarations for a single tab/load unit.
 */
export function createBundleManifest(spec: BundleDocumentSpec): BundleManifest {
    const tensors = spec.tensors.map((tensor, index) => {
        const id = tensor.id ?? `tensor-${index + 1}`;
        const view = tensorView(tensor.shape, tensor.axisLabels, tensor.view);
        return {
            id,
            name: tensor.name,
            dtype: tensor.dtype,
            shape: tensor.shape,
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

/** Build a multi-tab session manifest from lightweight tab specs.
 *
 * Each tab wraps the bundle-level `viewer` plus `tensors` pair with a tab id
 * and title.
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
