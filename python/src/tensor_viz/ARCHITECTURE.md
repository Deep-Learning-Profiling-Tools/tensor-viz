The `tensor_viz/` Python package is a small transport layer between Python tensors and the browser viewer. It should stay boring: normalize inputs, create manifests, serve bytes, and let the TypeScript viewer own interaction behavior.

`bundle.py` owns the public session data model. `TensorMeta` represents metadata-only tensors, `Tab` groups tensors into viewer tabs, and `create_session_data` converts NumPy arrays, mappings, sequences, and tabs into the same manifest and byte payload shape consumed by the TypeScript viewer. If a change affects accepted Python inputs, axis labels, dtypes, tab behavior, or manifest generation, start here.

`server.py` owns local serving. It should know how to serve the built frontend, session manifest, and tensor payloads, but it should not duplicate layout or view logic from the TypeScript packages.

`__init__.py` is the public API surface. Keep exports intentional; adding a helper to an internal module does not mean it should become public.

When changing the Python package, add tests under `tensor-viz/python/tests/`. Run `PYTHONPATH=python/src python -m unittest discover -s python/tests -p 'test_*.py'` from `tensor-viz/`, then run the project build so the packaged frontend assets stay current.
