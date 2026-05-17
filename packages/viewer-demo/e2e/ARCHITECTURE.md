The `e2e/` directory contains browser tests that exercise the demo as a user would load it.

Unit tests can prove parsers and render models are correct, but they cannot prove the app actually boots in a browser. The smoke test starts Vite through Playwright, opens the demo in Chromium, checks that core and extension widgets appear, verifies the command palette, and samples a viewport screenshot so blank-canvas regressions fail automatically.

The test deliberately allows the static `/api/session.json` miss because the demo probes for an optional Python session before falling back to baked tabs. It also ignores a known headless Chromium Three.js shader-validation diagnostic; the screenshot assertion is the stronger user-facing render check.
