// public package surface for embedding tensor-viz.
// keep exports grouped by subsystem so downstream imports stay predictable:
// typed-array owns binary dtype wrappers.
// layout/view own coordinate and tensor-view math.
// session/validation own persisted bundle shapes.
// viewer-utils owns small coordinate serialization helpers.
// TensorViewer is the imperative runtime class.
// types stay type-only so bundlers do not pull runtime code unnecessarily.
export { createTypedArray } from './typed-array.js';
export { axisWorldKeyForMode, displayExtent, displayHitForPoint2D, displayPositionForCoord, displayPositionForCoord2D, unravelIndex } from './layout.js';
export type { CoordHit2D } from './layout.js';
export { createBundleManifest, createSessionBundleManifest, createViewerSnapshot } from './session.js';
export type { BundleDocumentSpec, SessionTabSpec, SessionTensorSpec } from './session.js';
export {
    dtypeByteLength,
    expectedTensorByteLength,
    isDType,
    tensorElementCount,
    validateBundleManifest,
    validateSessionBundleManifest,
    validateTensorPayload,
    validateTensorShape,
    VIEWER_LIMITS,
} from './validation.js';
export {
    buildTensorViewExpression,
    defaultTensorViewEditor,
    expandGroupedIndex,
    layoutAxisLabels,
    layoutCoordIsVisible,
    layoutCoordMatchesSlice,
    layoutShape,
    mapLayoutCoordToViewCoord,
    mapViewCoordToLayoutCoord,
    mapViewCoordToTensorCoord,
    parseTensorView,
    product,
    serializeTensorViewEditor,
    visibleTensorCoords,
} from './view.js';
export { coordFromKey, coordKey } from './viewer-utils.js';
export { TensorViewer } from './viewer.js';
export type { ViewerOptions } from './viewer.js';
export type {
    BundleManifest,
    ColorInstruction,
    CustomColor,
    DType,
    DimensionMappingScheme,
    HoverInfo,
    InspectorTensorOption,
    InteractionMode,
    HueSaturation,
    LoadedBundleDocument,
    NumericArray,
    RGB,
    SelectionCoords,
    SessionBundleManifest,
    SliceToken,
    TensorDataRequestReason,
    TensorHandle,
    TensorStatus,
    TensorViewEditor,
    TensorViewEditorDim,
    TensorViewEditorSingleton,
    TensorViewSnapshot,
    TensorViewSpec,
    Vec3,
    ViewParseResult,
    ViewToken,
    ViewerSnapshot,
} from './types.js';
