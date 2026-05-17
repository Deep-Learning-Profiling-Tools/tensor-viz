/**
 * Supported dense tensor element types recorded in bundle manifests and used to choose the matching typed-array storage.
 *
 * @example
 * const dtype: DType = 'float32';
 * const bytesPerElement = dtype === 'float32' ? 4 : 8;
 * console.assert(bytesPerElement === 4);
 */
export type DType = 'float64' | 'float32' | 'int32' | 'uint8';

/**
 * Axis-family assignment mode used when mapping tensor dimensions onto the viewer's x, y, and z layout directions.
 *
 * @example
 * const scheme: DimensionMappingScheme = 'z-order';
 * console.assert(scheme === 'z-order');
 */
export type DimensionMappingScheme = 'z-order' | 'contiguous';

/**
 * Typed-array tensor payload that can be loaded from a bundle and rendered by the viewer.
 *
 * @example
 * const values: NumericArray = new Float32Array([0, 0.5, 1]);
 * console.assert(values.length === 3);
 * console.assert(values[1] === 0.5);
 */
export type NumericArray = Float64Array | Float32Array | Int32Array | Uint8Array;

/**
 * Immutable three-number tuple for xyz positions such as tensor offsets and camera coordinates.
 *
 * @example
 * const offset: Vec3 = [10, 0, -2];
 * const [x, y, z] = offset;
 * console.assert(`${x},${y},${z}` === '10,0,-2');
 */
export type Vec3 = readonly [number, number, number];

/**
 * Red, green, and blue color channels for an explicit tensor-cell color.
 * Each channel is stored as a 0-255 intensity in tuple order `[red, green, blue]`.
 *
 * @example
 * const magenta: RGB = [255, 0, 255];
 */
export type RGB = readonly [number, number, number];

/**
 * Hue and saturation pair used when a custom color should preserve the cell's heatmap brightness.
 * The tuple is stored as `[hue, saturation]`; rendering combines it with the tensor value's normalized brightness.
 *
 * @example
 * const cyanTint: HueSaturation = [180, 0.75];
 */
export type HueSaturation = readonly [number, number];

/**
 * Normalized custom color stored for one tensor coordinate after manifest instructions are parsed.
 * RGB entries replace the displayed color directly, while hue-saturation entries keep brightness tied to the tensor value.
 *
 * @example
 * const fixedRed: CustomColor = { kind: 'rgb', value: [255, 0, 0] };
 * const valueShadedBlue: CustomColor = { kind: 'hs', value: [240, 1] };
 */
export type CustomColor =
    | { kind: 'rgb'; value: RGB }
    | { kind: 'hs'; value: HueSaturation };

/**
 * Serializable custom-color instruction loaded from a bundle manifest for a tensor.
 * Dense instructions provide one color tuple per tensor cell, coordinate instructions target explicit cells,
 * and region instructions color a strided rectangular block.
 *
 * @example
 * const highlightTwoCells: ColorInstruction = {
 *     mode: 'rgb',
 *     kind: 'coords',
 *     coords: [[0, 0], [1, 2]],
 *     color: [255, 128, 0],
 * };
 *
 * const tintRegion: ColorInstruction = {
 *     mode: 'hs',
 *     kind: 'region',
 *     base: [0, 0],
 *     shape: [2, 3],
 *     jumps: [1, 1],
 *     color: [210, 0.8],
 * };
 */
export type ColorInstruction =
    | { mode: 'rgb' | 'hs'; kind: 'dense'; values: number[] }
    | { mode: 'rgb' | 'hs'; kind: 'coords'; coords: number[][]; color: number[] }
    | { mode: 'rgb' | 'hs'; kind: 'region'; base: number[]; shape: number[]; jumps: number[]; color: number[] };

/**
 * Lightweight metadata returned when a tensor is added to a viewer.
 *
 * @example
 * const handle: TensorHandle = {
 *   id: 'tensor-0',
 *   name: 'attention_scores',
 *   rank: 2,
 *   shape: [4, 8],
 *   axisLabels: ['head', 'token'],
 *   dtype: 'float32',
 *   hasData: true,
 * };
 *
 * console.assert(handle.id === 'tensor-0');
 * console.assert(handle.shape.join('x') === '4x8');
 * console.assert(handle.hasData === true);
 */
export type TensorHandle = {
    id: string;
    name: string;
    rank: number;
    shape: readonly number[];
    axisLabels: readonly string[];
    dtype: DType;
    hasData: boolean;
};

/**
 * Live tensor metadata together with current dense-data availability and value range.
 *
 * @example
 * const status: TensorStatus = {
 *   id: 'tensor-0',
 *   name: 'attention_scores',
 *   rank: 2,
 *   shape: [4, 8],
 *   axisLabels: ['head', 'token'],
 *   dtype: 'float32',
 *   hasData: true,
 *   valueRange: { min: -1.25, max: 3.5 },
 * };
 *
 * console.assert(status.valueRange?.max === 3.5);
 * console.assert(status.hasData === true);
 */
export type TensorStatus = TensorHandle & {
    valueRange: { min: number; max: number } | null;
};

/**
 * Reason the viewer is asking the host to hydrate a metadata-only tensor.
 *
 * @example
 * const reason: TensorDataRequestReason = 'heatmap';
 * const priority = reason === 'explicit' ? 'user-requested' : 'viewer-generated';
 *
 * console.assert(priority === 'viewer-generated');
 */
export type TensorDataRequestReason = 'explicit' | 'heatmap' | 'save';

/**
 * Persisted tensor-view state for one tensor.
 *
 * @example
 * declare const editor: TensorViewEditor;
 *
 * const snapshot: TensorViewSnapshot = {
 *   editor,
 *   hiddenIndices: [2],
 * };
 *
 * console.assert(snapshot.hiddenIndices.includes(2));
 */
export type TensorViewSnapshot = {
    editor: TensorViewEditor;
    hiddenIndices: number[];
};

/**
 * Describes one base axis that the tensor-view editor can label, permute, flatten, or slice.
 * The `id` is the stable key used by editor arrays, `label` is the text shown in view expressions,
 * and `size` is the axis length from the tensor shape or a grouped view dimension.
 *
 * @example
 * const batchDim: TensorViewEditorDim = {
 *     id: 'axis-0',
 *     label: 'Batch',
 *     size: 32,
 * };
 *
 * console.assert(batchDim.id === 'axis-0');
 * console.assert(`${batchDim.label}=${batchDim.size}` === 'Batch=32');
 */
export type TensorViewEditorDim = {
    id: string;
    label: string;
    size: number;
};

/**
 * Records an editor-created size-one axis that is inserted into the permuted view order.
 * The `position` is the zero-based slot in the rendered dimension sequence where the singleton
 * axis should appear.
 *
 * @example
 * const channelSingleton: TensorViewEditorSingleton = {
 *     id: 'singleton-channel',
 *     position: 1,
 * };
 *
 * console.assert(channelSingleton.position === 1);
 */
export type TensorViewEditorSingleton = {
    id: string;
    position: number;
};

/**
 * Captures the structured state behind the tensor-view editor so the viewer can serialize,
 * restore, and parse the same staged operations: base view text, dimension permutation,
 * flatten separators, inserted singleton axes, and selected slice values.
 *
 * @example
 * const editor: TensorViewEditor = {
 *     version: 2,
 *     viewTensorInput: '[Batch=2, Row=3, Col=4]',
 *     finalViewInput: '[Row=3, Batch=2]',
 *     baseDims: [
 *         { id: 'axis-0', label: 'Batch', size: 2 },
 *         { id: 'axis-1', label: 'Row', size: 3 },
 *         { id: 'axis-2', label: 'Col', size: 4 },
 *     ],
 *     permutedDimIds: ['axis-1', 'axis-0', 'axis-2'],
 *     flattenSeparators: [false, true],
 *     singletons: [{ id: 'singleton-channel', position: 2 }],
 *     slicedTokenKeys: ['axis-2'],
 *     sliceValues: { 'axis-2': 1 },
 * };
 *
 * console.assert(editor.version === 2);
 * console.assert(editor.permutedDimIds[0] === 'axis-1');
 * console.assert(editor.sliceValues['axis-2'] === 1);
 */
export type TensorViewEditor = {
    version: 2;
    viewTensorInput: string;
    finalViewInput?: string;
    baseDims: TensorViewEditorDim[];
    permutedDimIds: string[];
    flattenSeparators: boolean[];
    singletons: TensorViewEditorSingleton[];
    slicedTokenKeys: string[];
    sliceValues: Record<string, number>;
};

/**
 * Payload published when hit testing identifies the tensor cell under the pointer.
 * It links the hovered screen/layout position back to the source tensor coordinates, numeric cell
 * value, and color pipeline so inspectors and extensions can render matching hover details.
 *
 * @example
 * const hover: HoverInfo = {
 *     tensorId: 'logits',
 *     tensorName: 'Decoder logits',
 *     viewCoord: [0, 5],
 *     layoutCoord: [0, 1, 5],
 *     tensorCoord: [0, 12, 5],
 *     value: 0.875,
 *     colorSource: 'heatmap',
 * };
 *
 * console.assert(hover.tensorName === 'Decoder logits');
 * console.assert(hover.tensorCoord.join(',') === '0,12,5');
 * console.assert(hover.value === 0.875);
 */
export type HoverInfo = {
    tensorId: string;
    tensorName: string;
    viewCoord: number[];
    layoutCoord: number[];
    tensorCoord: number[];
    value: number | null;
    colorSource: 'base' | 'heatmap' | 'custom';
};

/**
 * Selected tensor coordinates grouped by tensor id.
 *
 * Each map key is a tensor id from the loaded manifest, and each value is the
 * list of tensor-space index tuples selected for that tensor.
 *
 * @example
 * const selection: SelectionCoords = new Map([
 *   ['weights', [[0, 1], [0, 2]]],
 *   ['bias', [[1]]],
 * ]);
 *
 * selection.get('weights')?.[0]; // [0, 1]
 */
export type SelectionCoords = Map<string, number[][]>;

/**
 * Primary left-drag interaction used by the viewer.
 *
 * The mode determines whether a drag moves the camera, marks tensor cells, or
 * rotates the 3D view.
 *
 * @example
 * const mode: InteractionMode = 'select';
 * const dragCreatesSelection = mode === 'select'; // true
 */
export type InteractionMode = 'pan' | 'select' | 'rotate';

/**
 * Serializable viewer state for one loaded document.
 *
 * This captures how the viewer should look after tensors are present:
 * display mode, camera, panel toggles, the active tensor, and each tensor's
 * current offset and tensor-view state.
 *
 * @example
 * const snapshot: ViewerSnapshot = {
 *   version: 1,
 *   displayMode: '2d',
 *   interactionMode: 'select',
 *   heatmap: true,
 *   showDimensionLines: true,
 *   showInspectorPanel: true,
 *   showHoverDetailsPanel: false,
 *   camera: {
 *     position: [0, 0, 10],
 *     target: [0, 0, 0],
 *     rotation: [0, 0, 0],
 *     zoom: 1,
 *   },
 *   tensors: [
 *     {
 *       id: 'weights',
 *       name: 'Weights',
 *       offset: [0, 0, 0],
 *       view: { expression: '[:, :]' },
 *     },
 *   ],
 *   activeTensorId: 'weights',
 * };
 *
 * snapshot.tensors[0].id; // 'weights'
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

/**
 * Minimal inspector entry used to populate the active-tensor selector.
 *
 * The id is the stable tensor identifier stored in viewer state, while the name
 * is the label displayed in the inspector UI.
 *
 * @example
 * const option: InspectorTensorOption = {
 *   id: 'attention_qk',
 *   name: 'Attention QK Scores',
 * };
 *
 * option.name; // 'Attention QK Scores'
 */
export type InspectorTensorOption = {
    id: string;
    name: string;
};

/**
 * Parsed hidden-axis token together with the selected slice index used when a view hides that axis.
 *
 * @example
 * const channelSlice: SliceToken = {
 *     token: "C",
 *     key: "axis:C",
 *     axes: [2],
 *     size: 3,
 *     value: 1,
 * };
 *
 * console.assert(channelSlice.axes[0] === 2);
 * console.assert(channelSlice.value === 1);
 */
export type SliceToken = {
    token: string;
    key: string;
    axes: number[];
    size: number;
    value: number;
};

/**
 * One parsed token from a tensor-view string, either a visible layout axis or an axis that is sliced away.
 *
 * @example
 * const rowColumnGroup: ViewToken = {
 *     kind: "axis_group",
 *     key: "axis:H,W",
 *     visible: true,
 *     label: "H×W",
 *     axes: [1, 2],
 *     size: 28 * 28,
 * };
 *
 * console.assert(rowColumnGroup.visible === true);
 * console.assert(rowColumnGroup.size === 784);
 */
export type ViewToken = {
    kind: 'axis_group' | 'singleton';
    key: string;
    visible: boolean;
    label: string;
    axes: number[];
    size: number;
};

/**
 * Fully parsed tensor-view specification derived from one view string, including visible layout axes and hidden slice state.
 *
 * @example
 * const spec = parseTensorView("[N, H, W, C]", [2, 28, 28, 3], ["N", "H", "W", "C"]);
 *
 * if (spec.ok) {
 *     const view: TensorViewSpec = spec.spec;
 *     console.assert(view.input === "[N, H, W, C]");
 *     console.assert(view.tensorShape.join(",") === "2,28,28,3");
 *     console.assert(view.viewShape.join(",") === "2,28,28,3");
 * }
 */
export type TensorViewSpec = {
    input: string;
    canonical: string;
    axisLabels: string[];
    tensorShape: number[];
    baseDims: TensorViewEditorDim[];
    baseShape: number[];
    permutedBaseShape: number[];
    permutedBaseIndices: number[];
    baseIsTensorAxes: boolean;
    tokens: ViewToken[];
    viewAxes: number[];
    sliceAxes: number[];
    hiddenIndices: number[];
    sliceTokens: SliceToken[];
    viewShape: number[];
    layoutShape: number[];
    editor: TensorViewEditor;
};

/**
 * Discriminated result of parsing a tensor-view string against a tensor shape.
 *
 * @example
 * const parsed: ViewParseResult = parseTensorView("[N, H, W, C]", [2, 28, 28, 3], ["N", "H", "W", "C"]);
 *
 * if (parsed.ok) {
 *     console.assert(parsed.spec.viewShape.join(",") === "2,28,28,3");
 * } else {
 *     console.error(parsed.errors.join("\n"));
 * }
 *
 * @example
 * const invalid: ViewParseResult = parseTensorView("[N, MissingAxis]", [2, 28], ["N", "H"]);
 *
 * if (!invalid.ok) {
 *     console.assert(invalid.errors.length > 0);
 * }
 */
export type ViewParseResult =
    | {
        ok: true;
        spec: TensorViewSpec;
    }
    | {
        ok: false;
        errors: string[];
    };

/**
 * Serialized document for one viewer tab.
 *
 * A bundle manifest stores the `ViewerSnapshot` needed to restore camera,
 * display, and per-tensor view state, plus the tensor declarations that the
 * loader uses to find tensor bytes or represent metadata-only tensors. Tensor
 * entries carry stable ids, dtype and shape metadata, optional external data
 * file locations, offsets, marker coordinates, and manifest-driven coloring.
 *
 * @example
 * const manifest: BundleManifest = createBundleManifest(snapshot, tensorEntries);
 * console.assert(manifest.version === 1);
 * console.assert(manifest.tensors.every((tensor) => tensor.byteOrder === 'little'));
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

/**
 * Serialized multi-tab viewer session consumed by the demo app and Python server.
 *
 * Each tab contains its tab id and title together with the same viewer snapshot
 * and tensor manifest entries used by a single-tab `BundleManifest`, allowing a
 * saved session to restore several viewer tabs from one JSON document.
 *
 * @example
 * const session: SessionBundleManifest = createSessionBundleManifest([
 *     { id: 'main', title: 'Main', viewer: snapshot, tensors: manifest.tensors },
 * ]);
 * console.assert(session.tabs[0].title === 'Main');
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

/**
 * Runtime representation of one loaded viewer tab.
 *
 * The document keeps the tab id and title shown by the host UI, the original
 * bundle manifest used to restore viewer state, and a map from tensor id to the
 * decoded numeric array loaded from each tensor's manifest entry.
 *
 * @example
 * const document: LoadedBundleDocument = { id: 'main', title: 'Main', manifest, tensors };
 * console.assert(document.tensors.has(manifest.tensors[0].id));
 */
export type LoadedBundleDocument = {
    id: string;
    title: string;
    manifest: BundleManifest;
    tensors: Map<string, NumericArray>;
};

/**
 * Normalized tensor state tracked by the live viewer scene.
 *
 * A record combines manifest metadata, decoded tensor data when available,
 * computed value range, parsed view specification, color and marker overlays,
 * visibility filters, ghost-layer annotations, and the offset used when laying
 * the tensor out in the rendered scene.
 *
 * @example
 * const record: TensorRecord = viewer.getTensorRecord('weights');
 * console.assert(record.axisLabels.length === record.shape.length);
 * console.assert(record.hasData === (record.data !== null));
 */
export type TensorRecord = {
    id: string;
    name: string;
    shape: number[];
    axisLabels: string[];
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
    ghostLayers: Array<{
        coord: number[];
        color: RGB;
        bias: readonly [number, number];
        layer: number;
        text: string | null;
    }> | null;
    autoOffset: boolean;
};

/**
 * Mutable state object owned by the viewer runtime and consumed by mesh/rendering code when building public snapshots and renderable geometry.
 *
 * @example
 * const state: ViewerState = {
 *   displayMode: '2d',
 *   interactionMode: 'pan',
 *   heatmap: false,
 *   dimensionBlockGapMultiple: 1,
 *   displayGaps: true,
 *   logScale: false,
 *   collapseHiddenAxes: false,
 *   dimensionMappingScheme: 'auto',
 *   showDimensionLines: true,
 *   showTensorNames: true,
 *   showInspectorPanel: false,
 *   showSelectionPanel: true,
 *   showHoverDetailsPanel: true,
 *   activeTensorId: 'weights',
 *   hover: null,
 *   lastHover: null,
 * };
 *
 * if (state.displayMode === '2d' && state.activeTensorId === 'weights') {
 *   // Render the selected tensor with the 2D mesh path.
 * }
 */
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
