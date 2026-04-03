/** Supported dense tensor dtypes for viewer storage and manifests. */
export type DType = 'float64' | 'float32' | 'int32' | 'uint8';

/** Strategy for assigning tensor axes to the x, y, and z layout families. */
export type DimensionMappingScheme = 'z-order' | 'contiguous';

/** Backing typed-array payload accepted by the viewer and bundle loaders. */
export type NumericArray = Float64Array | Float32Array | Int32Array | Uint8Array;

/** Fixed-length xyz tuple used for tensor offsets and camera values. */
export type Vec3 = readonly [number, number, number];

/** RGB color tuple using 0-255 channels. */
export type RGB = readonly [number, number, number];

/** Hue-saturation tuple used by the viewer's brightness-preserving color mode. */
export type HueSaturation = readonly [number, number];

/** One normalized custom color entry stored on a tensor cell. */
export type CustomColor =
    | { kind: 'rgb'; value: RGB }
    | { kind: 'hs'; value: HueSaturation };

/** Serializable custom-color instructions persisted in bundle manifests. */
export type ColorInstruction =
    | { mode: 'rgb' | 'hs'; kind: 'dense'; values: number[] }
    | { mode: 'rgb' | 'hs'; kind: 'coords'; coords: number[][]; color: number[] }
    | { mode: 'rgb' | 'hs'; kind: 'region'; base: number[]; shape: number[]; jumps: number[]; color: number[] };

/** Lightweight metadata returned when a tensor is added to a viewer. */
export type TensorHandle = {
    id: string;
    name: string;
    rank: number;
    shape: readonly number[];
    axisLabels: readonly string[];
    dtype: DType;
    hasData: boolean;
};

/** Live tensor metadata together with current dense-data availability and value range. */
export type TensorStatus = TensorHandle & {
    valueRange: { min: number; max: number } | null;
};

/** Reason the viewer is asking the host to hydrate a metadata-only tensor. */
export type TensorDataRequestReason = 'explicit' | 'heatmap' | 'save';

/** Persisted tensor-view string plus sliced indices for one tensor. */
export type TensorViewSnapshot = {
    view: string;
    hiddenIndices: number[];
    visible?: string;
};

/** Hover payload emitted for the currently pointed tensor cell. */
export type HoverInfo = {
    tensorId: string;
    tensorName: string;
    viewCoord: number[];
    layoutCoord: number[];
    tensorCoord: number[];
    value: number | null;
    colorSource: 'base' | 'heatmap' | 'custom';
};

/** Selected tensor coordinates grouped by tensor id. */
export type SelectionCoords = Map<string, number[][]>;

/** Primary left-drag interaction used by the viewer. */
export type InteractionMode = 'pan' | 'select' | 'rotate';

/** Serializable viewer state for one loaded document.
 *
 * This captures how the viewer should look after tensors are present:
 * display mode, camera, panel toggles, the active tensor, and each tensor's
 * current offset and tensor-view state.
 */
export type ViewerSnapshot = {
    version: 1;
    displayMode: '2d' | '3d';
    interactionMode?: InteractionMode;
    heatmap: boolean;
    dimensionBlockGapMultiple?: number;
    displayGaps?: boolean;
    logScale?: boolean;
    collapseHiddenAxes?: boolean;
    showSlicesInSamePlace?: boolean;
    dimensionMappingScheme?: DimensionMappingScheme;
    showDimensionLines: boolean;
    showTensorNames?: boolean;
    showInspectorPanel: boolean;
    showSelectionPanel?: boolean;
    showHoverDetailsPanel: boolean;
    camera: {
        position: Vec3;
        target: Vec3;
        rotation: Vec3;
        zoom: number;
    };
    tensors: Array<{
        id: string;
        name: string;
        offset?: Vec3;
        view: TensorViewSnapshot;
    }>;
    activeTensorId: string | null;
};

/** Minimal inspector entry used to populate the active-tensor selector. */
export type InspectorTensorOption = {
    id: string;
    name: string;
};

/** Parsed hidden-axis token together with its current slice value. */
export type SliceToken = {
    token: string;
    axes: number[];
    size: number;
    value: number;
};

/** One parsed token from a tensor-view string, either visible or sliced away. */
export type ViewToken = {
    kind: 'axis_group' | 'singleton';
    visible: boolean;
    label: string;
    axes: number[];
    size: number;
};

/** Fully parsed tensor-view specification derived from one view string. */
export type TensorViewSpec = {
    input: string;
    canonical: string;
    axisLabels: string[];
    tensorShape: number[];
    tokens: ViewToken[];
    viewAxes: number[];
    sliceAxes: number[];
    hiddenIndices: number[];
    sliceTokens: SliceToken[];
    viewShape: number[];
    layoutShape: number[];
};

/** Result of parsing a tensor-view string against one tensor shape. */
export type ViewParseResult =
    | {
        ok: true;
        spec: TensorViewSpec;
    }
    | {
        ok: false;
        errors: string[];
    };

/** One complete viewer document, typically one tab.
 *
 * A bundle manifest combines one viewer snapshot with one or more tensor
 * declarations. Each tensor entry describes the tensor's metadata plus where
 * its bytes live, whether it is metadata-only, and any manifest-driven colors.
 */
export type BundleManifest = {
    version: 1;
    viewer: ViewerSnapshot;
    tensors: Array<{
        id: string;
        name: string;
        dtype: DType;
        shape: number[];
        axisLabels?: string[];
        byteOrder: 'little';
        dataFile?: string;
        placeholderData?: boolean;
        offset?: Vec3;
        view: TensorViewSnapshot;
        colorInstructions?: ColorInstruction[];
        markerCoords?: number[][];
    }>;
};

/** Multi-tab session document loaded by the demo app or Python server.
 *
 * Each tab embeds the same `viewer` plus `tensors` pair used by a bundle
 * manifest, with an added tab id and title.
 */
export type SessionBundleManifest = {
    version: 1;
    tabs: Array<{
        id: string;
        title: string;
        viewer: ViewerSnapshot;
        tensors: BundleManifest['tensors'];
    }>;
};

/** One loaded tab together with decoded tensor buffers. */
export type LoadedBundleDocument = {
    id: string;
    title: string;
    manifest: BundleManifest;
    tensors: Map<string, NumericArray>;
};

/** Internal tensor record tracked by the live viewer scene. */
export type TensorRecord = {
    id: string;
    name: string;
    shape: number[];
    dtype: DType;
    data: NumericArray | null;
    hasData: boolean;
    valueRange: { min: number; max: number } | null;
    offset: Vec3;
    view: TensorViewSpec;
    customColors: Map<string, CustomColor>;
    markerCoords: Set<string> | null;
    visibleCoords: Set<string> | null;
    cellLabels: Map<string, string> | null;
    autoOffset: boolean;
};

/** Internal mutable viewer state mirrored into public snapshots. */
export type ViewerState = {
    displayMode: '2d' | '3d';
    interactionMode: InteractionMode;
    heatmap: boolean;
    dimensionBlockGapMultiple: number;
    displayGaps: boolean;
    logScale: boolean;
    collapseHiddenAxes: boolean;
    dimensionMappingScheme: DimensionMappingScheme;
    showDimensionLines: boolean;
    showTensorNames: boolean;
    showInspectorPanel: boolean;
    showSelectionPanel: boolean;
    showHoverDetailsPanel: boolean;
    activeTensorId: string | null;
    hover: HoverInfo | null;
    lastHover: HoverInfo | null;
};
