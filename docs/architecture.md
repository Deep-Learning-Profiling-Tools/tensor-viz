# API Layers

`tensor-viz` now exposes four distinct API layers:

## Which layer should you use?

Choose the layer based on what you want to own in your app:

- Use the Python layer when Python owns the session lifecycle and you just want
  a local browser viewer.
- Use the TypeScript shared layer when you want to build or exchange
  `BundleManifest` or `SessionBundleManifest` data without rendering yet.
- Use `viewer-demo` when you want the stock tensor-viz app with its built-in
  UI.
- Use `viewer-core` when you want to build your own UI around the rendering
  engine.

The important distinction is that `viewer-demo` is an app, while
`viewer-core` is a library.

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

The main object here is `ViewerSession`. It is a small handle for the local HTTP
server, not the browser renderer itself. Use it to inspect the local URL, wait
for the session to end, or stop the server.

This surface is intentionally ergonomic rather than manifest-shaped. Python
callers usually start with arrays, mappings, metadata-only tensors, and tabs, so
the API accepts those inputs directly and normalizes them into session
documents.

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

The main objects here are data objects:

- `BundleManifest`: one viewer document with one viewer state plus one or more
  tensors
- `SessionBundleManifest`: a multi-tab wrapper around bundle manifests
- `ViewerSnapshot`: serialized viewer state such as display mode, camera, and
  per-tensor views

This layer does not render anything by itself.

This surface is intentionally explicit rather than ergonomic. TypeScript callers
usually already manage app state and serialization boundaries, so the API names
the manifest objects directly instead of hiding them behind a single convenience
entrypoint.

## viewer-demo: mountable app API

Use `@tensor-viz/viewer-demo` when you want the full demo shell: renderer,
widgets, menus, tabs, and sidebar UI.

- `mountDemoApp(container, options)`

The current mount API is iframe-backed. That makes it easy to embed the full
demo page into another web app without coupling the host to the demo's DOM
internals.

`viewer-demo` includes the stock application chrome:

- ribbon menus
- sidebar widgets
- command palette
- tab strip
- file open/save flows
- session loading from the demo app's HTTP endpoints

The main object here is the return value from `mountDemoApp(...)`. It is a small
mount handle with the iframe element and a `destroy()` method. It does not
expose low-level tensor mutation methods because `viewer-demo` is meant to be a
complete app shell.

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

`TensorViewer` is the central object in this layer. It is an imperative browser
class that:

- mounts into one host DOM element
- creates and owns the renderer, cameras, scene, and input handling
- stores the live tensor set and viewer state
- exposes methods for tensor data, tensor views, colors, hover, and snapshots

`TensorViewer` is not a full app shell. It does not create ribbon menus, tabs,
or side panels for your app. If you use `viewer-core`, those higher-level UI
pieces belong to your host application.

## Quick mental model

You can think about the layers like this:

- Python: "start a local tensor-viz session for me"
- shared manifests/types: "describe what should be shown"
- `viewer-demo`: "show it with the stock tensor-viz app UI"
- `viewer-core`: "render it, and I will build the surrounding UI myself"

The Python and TypeScript shared-builder layers target the same document model,
but they are not mirror-image APIs. Python optimizes for local workflow
ergonomics, while TypeScript optimizes for explicit manifest construction.
