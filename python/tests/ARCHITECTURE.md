The `python/tests/` directory protects the Python-facing examples and server behavior.

These tests treat the documentation examples as executable contracts. If the README says a Python user can create tabs, metadata-only tensors, color instructions, or a local viewer session, the tests build that path and inspect the emitted manifest or HTTP response.

Server tests focus on security-sensitive behavior such as token-gated session APIs, loopback-only defaults, no-store cache headers, and content security policy. Rendering and widget behavior stay in the TypeScript tests because Python only transports data to the browser.
