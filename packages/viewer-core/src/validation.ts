import type {
    BundleManifest,
    ColorInstruction,
    DType,
    SessionBundleManifest,
    TensorViewEditor,
    Vec3,
    ViewerSnapshot,
} from './types.js';

export const VIEWER_LIMITS = {
    maxTabs: 64,
    maxTensors: 256,
    maxRank: 12,
    maxDimension: 1_000_000,
    maxTensorElements: 1_000_000,
    maxPayloadBytes: 64 * 1024 * 1024,
    maxTextLength: 256,
    maxEditorInputLength: 20_000,
    maxEditorEntries: 512,
    maxColorInstructions: 256,
    maxCustomColorEntries: 1_000_000,
} as const;

const DTYPE_BYTES = {
    float64: 8,
    float32: 4,
    int32: 4,
    uint8: 1,
} satisfies Record<DType, number>;

const DISPLAY_MODES = new Set(['2d', '3d']);
const INTERACTION_MODES = new Set(['pan', 'select', 'rotate']);
const MAPPING_SCHEMES = new Set(['z-order', 'contiguous']);

export function isDType(value: unknown): value is DType {
    return typeof value === 'string' && value in DTYPE_BYTES;
}

export function dtypeByteLength(dtype: unknown): number {
    if (!isDType(dtype)) throw new Error(`Unsupported dtype ${String(dtype)}.`);
    return DTYPE_BYTES[dtype];
}

function assertObject(value: unknown, label: string): Record<string, unknown> {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        throw new Error(`${label} must be an object.`);
    }
    return value as Record<string, unknown>;
}

function assertArray(value: unknown, label: string, maxLength: number = VIEWER_LIMITS.maxEditorEntries): unknown[] {
    if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
    if (value.length > maxLength) throw new Error(`${label} has too many entries.`);
    return value;
}

function finiteNumber(value: unknown, label: string): number {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`${label} must be finite.`);
    return number;
}

function finiteInteger(value: unknown, label: string): number {
    const number = finiteNumber(value, label);
    if (!Number.isInteger(number)) throw new Error(`${label} must be an integer.`);
    return number;
}

function boundedString(value: unknown, label: string, maxLength: number = VIEWER_LIMITS.maxTextLength): string {
    if (typeof value !== 'string') throw new Error(`${label} must be a string.`);
    if (value.length > maxLength) throw new Error(`${label} is too long.`);
    return value;
}

export function tensorElementCount(shape: readonly number[], label = 'shape'): number {
    return shape.reduce((total, dim, axis) => {
        const next = total * dim;
        if (!Number.isSafeInteger(next) || next > VIEWER_LIMITS.maxTensorElements) {
            throw new Error(`${label} has too many elements at axis ${axis}.`);
        }
        return next;
    }, 1);
}

export function validateTensorShape(value: unknown, label = 'shape'): number[] {
    const shape = assertArray(value, label, VIEWER_LIMITS.maxRank).map((dim, axis) => {
        const number = finiteInteger(dim, `${label}[${axis}]`);
        if (number < 1) throw new Error(`${label}[${axis}] must be positive.`);
        if (number > VIEWER_LIMITS.maxDimension) throw new Error(`${label}[${axis}] is too large.`);
        return number;
    });
    tensorElementCount(shape, label);
    return shape;
}

export function expectedTensorByteLength(dtype: DType, shape: readonly number[]): number {
    const bytes = tensorElementCount(shape) * dtypeByteLength(dtype);
    if (!Number.isSafeInteger(bytes) || bytes > VIEWER_LIMITS.maxPayloadBytes) {
        throw new Error('Tensor payload is too large.');
    }
    return bytes;
}

export function validateTensorPayload(dtype: DType, shape: readonly number[], byteLength: number): void {
    const expectedBytes = expectedTensorByteLength(dtype, shape);
    if (byteLength !== expectedBytes) {
        throw new Error(`Tensor payload byte length ${byteLength} does not match expected ${expectedBytes}.`);
    }
}

function validateVec3(value: unknown, label: string): Vec3 {
    const tuple = assertArray(value, label, 3);
    if (tuple.length !== 3) throw new Error(`${label} must have three values.`);
    return [
        finiteNumber(tuple[0], `${label}[0]`),
        finiteNumber(tuple[1], `${label}[1]`),
        finiteNumber(tuple[2], `${label}[2]`),
    ];
}

function validateColorTuple(value: unknown, label: string): number[] {
    const tuple = assertArray(value, label, 3).map((entry, index) => finiteNumber(entry, `${label}[${index}]`));
    if (tuple.length !== 2 && tuple.length !== 3) throw new Error(`${label} must have two or three channels.`);
    return tuple;
}

function validateCoord(value: unknown, shape: readonly number[], label: string): number[] {
    const coord = assertArray(value, label, shape.length).map((entry, axis) => {
        const index = finiteInteger(entry, `${label}[${axis}]`);
        if (index < 0 || index >= shape[axis]!) throw new Error(`${label}[${axis}] is out of bounds.`);
        return index;
    });
    if (coord.length !== shape.length) throw new Error(`${label} rank does not match tensor rank.`);
    return coord;
}

function validateRegionShape(value: unknown, tensorShape: readonly number[], label: string): number[] {
    const shape = assertArray(value, label, tensorShape.length).map((entry, axis) => {
        const dim = finiteInteger(entry, `${label}[${axis}]`);
        if (dim < 1) throw new Error(`${label}[${axis}] must be positive.`);
        return dim;
    });
    if (shape.length !== tensorShape.length) throw new Error(`${label} rank does not match tensor rank.`);
    return shape;
}

function validateColorInstruction(instruction: unknown, shape: readonly number[], label: string): {
    instruction: ColorInstruction;
    entries: number;
} {
    const value = assertObject(instruction, label);
    const mode = value.mode;
    const kind = value.kind;
    if (mode !== 'rgb' && mode !== 'hs') throw new Error(`${label}.mode is invalid.`);
    if (kind === 'dense') {
        const values = assertArray(value.values, `${label}.values`, VIEWER_LIMITS.maxTensorElements * 3)
            .map((entry, index) => finiteNumber(entry, `${label}.values[${index}]`));
        const cells = tensorElementCount(shape);
        if (values.length !== cells * 2 && values.length !== cells * 3) {
            throw new Error(`${label}.values must have two or three channels per tensor cell.`);
        }
        return { instruction: { mode, kind, values }, entries: cells };
    }
    if (kind === 'coords') {
        const color = validateColorTuple(value.color, `${label}.color`);
        const coords = assertArray(value.coords, `${label}.coords`, VIEWER_LIMITS.maxCustomColorEntries)
            .map((coord, index) => validateCoord(coord, shape, `${label}.coords[${index}]`));
        return { instruction: { mode, kind, coords, color }, entries: coords.length };
    }
    if (kind === 'region') {
        const base = validateCoord(value.base, shape, `${label}.base`);
        const regionShape = validateRegionShape(value.shape, shape, `${label}.shape`);
        const jumps = assertArray(value.jumps, `${label}.jumps`, shape.length)
            .map((entry, axis) => finiteInteger(entry, `${label}.jumps[${axis}]`));
        if (jumps.length !== shape.length) throw new Error(`${label}.jumps rank does not match tensor rank.`);
        const entries = tensorElementCount(regionShape, `${label}.shape`);
        regionShape.forEach((dim, axis) => {
            const last = base[axis]! + (dim - 1) * jumps[axis]!;
            if (last < 0 || last >= shape[axis]!) throw new Error(`${label}.shape exceeds tensor bounds.`);
        });
        return {
            instruction: {
                mode,
                kind,
                base,
                shape: regionShape,
                jumps,
                color: validateColorTuple(value.color, `${label}.color`),
            },
            entries,
        };
    }
    throw new Error(`${label}.kind is invalid.`);
}

export function validateColorInstructions(value: unknown, shape: readonly number[], label = 'colorInstructions'): ColorInstruction[] | undefined {
    if (value === undefined) return undefined;
    const instructions = assertArray(value, label, VIEWER_LIMITS.maxColorInstructions);
    let totalEntries = 0;
    return instructions.map((instruction, index) => {
        const validated = validateColorInstruction(instruction, shape, `${label}[${index}]`);
        totalEntries += validated.entries;
        if (totalEntries > VIEWER_LIMITS.maxCustomColorEntries) {
            throw new Error(`${label} touches too many tensor cells.`);
        }
        return validated.instruction;
    });
}

export function normalizeTensorViewEditor(value: unknown, label = 'view.editor'): TensorViewEditor {
    const editor = assertObject(value, label);
    const viewTensorInput = boundedString(editor.viewTensorInput, `${label}.viewTensorInput`, VIEWER_LIMITS.maxEditorInputLength);
    const finalViewInput = editor.finalViewInput === undefined
        ? undefined
        : boundedString(editor.finalViewInput, `${label}.finalViewInput`, VIEWER_LIMITS.maxEditorInputLength);
    const baseDims = assertArray(editor.baseDims, `${label}.baseDims`).map((dim, index) => {
        const entry = assertObject(dim, `${label}.baseDims[${index}]`);
        const size = finiteInteger(entry.size, `${label}.baseDims[${index}].size`);
        if (size < 1 || size > VIEWER_LIMITS.maxDimension) {
            throw new Error(`${label}.baseDims[${index}].size is out of range.`);
        }
        return {
            id: boundedString(entry.id, `${label}.baseDims[${index}].id`),
            label: boundedString(entry.label, `${label}.baseDims[${index}].label`),
            size,
        };
    });
    const permutedDimIds = assertArray(editor.permutedDimIds, `${label}.permutedDimIds`)
        .map((entry, index) => boundedString(entry, `${label}.permutedDimIds[${index}]`));
    const flattenSeparators = assertArray(editor.flattenSeparators, `${label}.flattenSeparators`)
        .map((entry) => Boolean(entry));
    const singletons = assertArray(editor.singletons, `${label}.singletons`).map((singleton, index) => {
        const entry = assertObject(singleton, `${label}.singletons[${index}]`);
        return {
            id: boundedString(entry.id, `${label}.singletons[${index}].id`),
            position: finiteInteger(entry.position, `${label}.singletons[${index}].position`),
        };
    });
    const slicedTokenKeys = assertArray(editor.slicedTokenKeys, `${label}.slicedTokenKeys`)
        .map((entry, index) => boundedString(entry, `${label}.slicedTokenKeys[${index}]`));
    const sliceValues = assertObject(editor.sliceValues, `${label}.sliceValues`);
    if (Object.keys(sliceValues).length > VIEWER_LIMITS.maxEditorEntries) {
        throw new Error(`${label}.sliceValues has too many entries.`);
    }
    return {
        version: 2,
        viewTensorInput,
        finalViewInput,
        baseDims,
        permutedDimIds,
        flattenSeparators,
        singletons,
        slicedTokenKeys,
        sliceValues: Object.fromEntries(Object.entries(sliceValues).map(([key, entry]) => [
            boundedString(key, `${label}.sliceValues key`),
            finiteInteger(entry, `${label}.sliceValues[${key}]`),
        ])),
    };
}

function validateTensorViewSnapshot(value: unknown, shape: readonly number[], label: string) {
    const snapshot = assertObject(value, label);
    const hiddenIndices = assertArray(snapshot.hiddenIndices, `${label}.hiddenIndices`, shape.length)
        .map((entry, axis) => {
            const index = finiteInteger(entry, `${label}.hiddenIndices[${axis}]`);
            if (index < 0 || index >= shape[axis]!) throw new Error(`${label}.hiddenIndices[${axis}] is out of bounds.`);
            return index;
        });
    return {
        editor: normalizeTensorViewEditor(snapshot.editor, `${label}.editor`),
        hiddenIndices,
    };
}

function validateTensorManifest(value: unknown, label: string): BundleManifest['tensors'][number] {
    const tensor = assertObject(value, label);
    const dtype = tensor.dtype;
    if (!isDType(dtype)) throw new Error(`${label}.dtype is unsupported.`);
    const shape = validateTensorShape(tensor.shape, `${label}.shape`);
    if (tensor.byteOrder !== 'little') throw new Error(`${label}.byteOrder must be little.`);
    return {
        id: boundedString(tensor.id, `${label}.id`),
        name: boundedString(tensor.name, `${label}.name`),
        dtype,
        shape,
        axisLabels: tensor.axisLabels === undefined
            ? undefined
            : assertArray(tensor.axisLabels, `${label}.axisLabels`, shape.length)
                .map((entry, index) => boundedString(entry, `${label}.axisLabels[${index}]`)),
        byteOrder: 'little',
        dataFile: tensor.dataFile === undefined ? undefined : boundedString(tensor.dataFile, `${label}.dataFile`, VIEWER_LIMITS.maxTextLength),
        placeholderData: tensor.placeholderData === undefined ? undefined : Boolean(tensor.placeholderData),
        offset: tensor.offset === undefined ? undefined : validateVec3(tensor.offset, `${label}.offset`),
        view: validateTensorViewSnapshot(tensor.view, shape, `${label}.view`),
        colorInstructions: validateColorInstructions(tensor.colorInstructions, shape, `${label}.colorInstructions`),
        markerCoords: tensor.markerCoords === undefined
            ? undefined
            : assertArray(tensor.markerCoords, `${label}.markerCoords`, VIEWER_LIMITS.maxTensorElements)
                .map((coord, index) => validateCoord(coord, shape, `${label}.markerCoords[${index}]`)),
    };
}

function validateViewerTensorSnapshot(
    value: unknown,
    tensorById: Map<string, BundleManifest['tensors'][number]>,
    label: string,
): ViewerSnapshot['tensors'][number] {
    const entry = assertObject(value, label);
    const id = boundedString(entry.id, `${label}.id`);
    const tensor = tensorById.get(id);
    if (!tensor) throw new Error(`${label}.id does not match a tensor manifest.`);
    return {
        id,
        name: boundedString(entry.name, `${label}.name`),
        offset: entry.offset === undefined ? undefined : validateVec3(entry.offset, `${label}.offset`),
        view: validateTensorViewSnapshot(entry.view, tensor.shape, `${label}.view`),
    };
}

function validateViewerSnapshot(value: unknown, tensors: BundleManifest['tensors']): ViewerSnapshot {
    const snapshot = assertObject(value, 'viewer');
    const displayMode = snapshot.displayMode;
    if (!DISPLAY_MODES.has(String(displayMode))) throw new Error('viewer.displayMode is invalid.');
    const interactionMode = snapshot.interactionMode;
    if (interactionMode !== undefined && !INTERACTION_MODES.has(String(interactionMode))) {
        throw new Error('viewer.interactionMode is invalid.');
    }
    const dimensionMappingScheme = snapshot.dimensionMappingScheme;
    if (dimensionMappingScheme !== undefined && !MAPPING_SCHEMES.has(String(dimensionMappingScheme))) {
        throw new Error('viewer.dimensionMappingScheme is invalid.');
    }
    const camera = assertObject(snapshot.camera, 'viewer.camera');
    const tensorById = new Map(tensors.map((tensor) => [tensor.id, tensor]));
    return {
        version: 1,
        displayMode: displayMode as '2d' | '3d',
        interactionMode: interactionMode as ViewerSnapshot['interactionMode'],
        heatmap: Boolean(snapshot.heatmap),
        dimensionBlockGapMultiple: snapshot.dimensionBlockGapMultiple === undefined
            ? undefined
            : finiteNumber(snapshot.dimensionBlockGapMultiple, 'viewer.dimensionBlockGapMultiple'),
        displayGaps: snapshot.displayGaps === undefined ? undefined : Boolean(snapshot.displayGaps),
        logScale: snapshot.logScale === undefined ? undefined : Boolean(snapshot.logScale),
        collapseHiddenAxes: snapshot.collapseHiddenAxes === undefined ? undefined : Boolean(snapshot.collapseHiddenAxes),
        showSlicesInSamePlace: snapshot.showSlicesInSamePlace === undefined ? undefined : Boolean(snapshot.showSlicesInSamePlace),
        dimensionMappingScheme: dimensionMappingScheme as ViewerSnapshot['dimensionMappingScheme'],
        showDimensionLines: Boolean(snapshot.showDimensionLines),
        showTensorNames: snapshot.showTensorNames === undefined ? undefined : Boolean(snapshot.showTensorNames),
        showInspectorPanel: Boolean(snapshot.showInspectorPanel),
        showSelectionPanel: snapshot.showSelectionPanel === undefined ? undefined : Boolean(snapshot.showSelectionPanel),
        showHoverDetailsPanel: Boolean(snapshot.showHoverDetailsPanel),
        camera: {
            position: validateVec3(camera.position, 'viewer.camera.position'),
            target: validateVec3(camera.target, 'viewer.camera.target'),
            rotation: validateVec3(camera.rotation, 'viewer.camera.rotation'),
            zoom: finiteNumber(camera.zoom, 'viewer.camera.zoom'),
        },
        tensors: assertArray(snapshot.tensors, 'viewer.tensors', VIEWER_LIMITS.maxTensors)
            .map((entry, index) => validateViewerTensorSnapshot(entry, tensorById, `viewer.tensors[${index}]`)),
        activeTensorId: snapshot.activeTensorId === null
            ? null
            : boundedString(snapshot.activeTensorId, 'viewer.activeTensorId'),
    };
}

export function validateBundleManifest(value: unknown): BundleManifest {
    const manifest = assertObject(value, 'manifest');
    if (manifest.version !== 1) throw new Error(`Unsupported bundle version ${String(manifest.version)}.`);
    const tensors = assertArray(manifest.tensors, 'manifest.tensors', VIEWER_LIMITS.maxTensors)
        .map((tensor, index) => validateTensorManifest(tensor, `manifest.tensors[${index}]`));
    return {
        version: 1,
        viewer: validateViewerSnapshot(manifest.viewer, tensors),
        tensors,
    };
}

export function validateSessionBundleManifest(value: unknown): SessionBundleManifest {
    const manifest = assertObject(value, 'session');
    if (manifest.version !== 1) throw new Error(`Unsupported session version ${String(manifest.version)}.`);
    return {
        version: 1,
        tabs: assertArray(manifest.tabs, 'session.tabs', VIEWER_LIMITS.maxTabs).map((tab, index) => {
            const entry = assertObject(tab, `session.tabs[${index}]`);
            const tensors = assertArray(entry.tensors, `session.tabs[${index}].tensors`, VIEWER_LIMITS.maxTensors)
                .map((tensor, tensorIndex) => validateTensorManifest(tensor, `session.tabs[${index}].tensors[${tensorIndex}]`));
            return {
                id: boundedString(entry.id, `session.tabs[${index}].id`),
                title: boundedString(entry.title, `session.tabs[${index}].title`),
                viewer: validateViewerSnapshot(entry.viewer, tensors),
                tensors,
            };
        }),
    };
}
