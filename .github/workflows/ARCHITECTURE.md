The `.github/workflows/` directory is the standalone CI entry point for `tensor-viz`.

The CI job installs Python and Node because both halves of the package are part of the public contract. Python tests verify the server and manifest examples. TypeScript checks verify the core and demo packages. Playwright installs Chromium before `npm run test` because the test script includes a real browser smoke test of the demo app.

The final `npm run build` is intentionally separate from tests. A passing test suite does not prove that the demo assets copied into the Python package are current; the build step is the check that refreshes those packaged files.
