export type DType = 'float64' | 'float32' | 'int32' | 'uint8';
export type NumericArray = Float64Array | Float32Array | Int32Array | Uint8Array;
export type Vec3 = readonly [number, number, number];
export type RGBA = readonly [number, number, number, number];
export type HueSaturation = readonly [number, number];
export type CustomColor =
    | { kind: 'rgba'; value: RGBA }
    | { kind: 'hs'; value: HueSaturation };

export type ColorInstruction =
    | { mode: 'rgba' | 'hs'; kind: 'dense'; values: number[] }
    | { mode: 'rgba' | 'hs'; kind: 'coords'; coords: number[][]; color: number[] }
    | { mode: 'rgba' | 'hs'; kind: 'region'; base: number[]; shape: number[]; jumps: number[]; color: number[] };

export type TensorHandle = {
    id: string;
    name: string;
    rank: number;
    shape: readonly number[];
    dtype: DType;
};

export type TensorViewSnapshot = {
    visible: string;
    hiddenIndices: number[];
};

export type HoverInfo = {
    tensorId: string;
    tensorName: string;
    displayCoord: number[];
    fullCoord: number[];
    value: number;
    colorSource: 'base' | 'heatmap' | 'custom';
};

export type ViewerSnapshot = {
    version: 1;
    displayMode: '2d' | '3d';
    heatmap: boolean;
    showDimensionLines: boolean;
    showInspectorPanel: boolean;
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
        offset: Vec3;
        view: TensorViewSnapshot;
    }>;
    activeTensorId: string | null;
};

export type HiddenToken = {
    token: string;
    axes: number[];
    size: number;
    value: number;
};

export type ViewToken = {
    kind: 'axis_group' | 'singleton';
    visible: boolean;
    label: string;
    axes: number[];
    size: number;
};

export type TensorViewSpec = {
    input: string;
    canonical: string;
    axisLabels: string[];
    axisShape: number[];
    tokens: ViewToken[];
    visibleAxes: number[];
    hiddenAxes: number[];
    hiddenIndices: number[];
    hiddenTokens: HiddenToken[];
    displayShape: number[];
    outlineShape: number[];
};

export type ViewParseResult =
    | {
        ok: true;
        spec: TensorViewSpec;
    }
    | {
        ok: false;
        errors: string[];
    };

export type BundleManifest = {
    version: 1;
    viewer: ViewerSnapshot;
    tensors: Array<{
        id: string;
        name: string;
        dtype: DType;
        shape: number[];
        byteOrder: 'little';
        dataFile: string;
        offset: Vec3;
        view: TensorViewSnapshot;
        colorInstructions?: ColorInstruction[];
    }>;
};

export type TensorRecord = {
    id: string;
    name: string;
    shape: number[];
    dtype: DType;
    data: NumericArray;
    offset: Vec3;
    view: TensorViewSpec;
    customColors: Map<string, CustomColor>;
};

export type ViewerState = {
    displayMode: '2d' | '3d';
    heatmap: boolean;
    showDimensionLines: boolean;
    showInspectorPanel: boolean;
    showHoverDetailsPanel: boolean;
    activeTensorId: string | null;
    hover: HoverInfo | null;
    lastHover: HoverInfo | null;
};
