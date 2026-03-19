# API Layers

`tensor-viz` now exposes four distinct API layers:

## Python: session builder + local-server convenience

Use the Python package when Python owns the workflow and you want a local viewer
session with minimal setup.

- `tensor_viz.viz(...)`
- `tensor_viz.Tab(...)`
- `tensor_viz.TensorMeta(...)`
- `tensor_viz.create_session_data(...)`
- `tensor_viz.ViewerSession`

This layer builds a session document and serves the full browser app for local
inspection.

## TypeScript shared: session builder + types

Use `@tensor-viz/viewer-core` session helpers when another TypeScript app needs
to build the same manifest structure as the Python API without directly driving
the viewer instance yet.

- `createViewerSnapshot(...)`
- `createBundleManifest(...)`
- `createSessionBundleManifest(...)`
- shared manifest, tensor, and view types

This layer is useful for host apps that want to prepare tabs, tensors, custom
colors, labels, and initial viewer state ahead of time.

## viewer-demo: mountable app API

Use `@tensor-viz/viewer-demo` when you want the full demo shell: renderer,
widgets, menus, tabs, and sidebar UI.

- `mountDemoApp(container, options)`

The current mount API is iframe-backed. That makes it easy to embed the full
demo page into another web app without coupling the host to the demo's DOM
internals.

## viewer-core: imperative viewer engine

Use `TensorViewer` when your app wants direct control over the rendered viewer
instance.

- `new TensorViewer(container, options)`
- `addTensor(...)`
- `addMetadataTensor(...)`
- `setTensorData(...)`
- `clearTensorData(...)`
- `setTensorView(...)`
- `toggleHeatmap(...)`
- subscriptions and snapshot APIs

This is the lowest-level browser API and the right layer for applications that
need to drive tensor-viz directly.
