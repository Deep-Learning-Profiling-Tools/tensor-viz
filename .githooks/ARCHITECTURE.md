The `.githooks/` directory contains local git hooks that keep generated source in sync before a commit is created.

The current pre-commit hook runs `npm run sync:linear-layout-examples`. That command rewrites the baked linear-layout examples in the demo extension from the Python `demo_linear_layout.py` source when that source is present. This matters in the LL-viz checkout because the static demo must show the same examples as the Python script without asking contributors to edit generated TypeScript by hand.

Standalone `tensor-viz` checkouts may not include the LL-viz Python demo source. In that case the sync tool leaves the baked examples unchanged, so normal tensor-viz commits are not blocked by a file that only exists in the parent project.
