The `widgets/` directory exists because the linear-layout sidebar stopped being one thing.

At first, a single file was enough: all of the sidebar controls manipulated the same layout state, and seeing every control in one place made the coupling obvious. That stopped being helpful once the sidebar grew into several independent interfaces. A preset chooser has very different failure modes from a drag-and-drop color mapper, and both are easier to reason about when their event flow is local instead of buried in a thousand-line file.

The split here follows the shape of the UI the user sees:

- `linear-layout-preset-widget.ts` owns preset selection and loading.
- `linear-layout-specs-widget.ts` owns the editable layout spec itself and the apply/load flows.
- `linear-layout-visible-tensors-widget.ts` owns visibility toggles for rendered tensors.
- `linear-layout-color-widget.ts` owns propagated labels, ranges, and color-channel mapping.
- `linear-layout-cell-text-widget.ts` keeps the placeholder cell-text widget isolated, even though it is currently hidden.

`linear-layout-widget-shared.ts` is deliberately small. It only contains helpers that are genuinely shared across widgets, such as active-tab lookup, textarea sizing, clipboard copying, and the propagation-label fallback logic. The goal is not to invent a framework for widgets. The goal is to keep each widget file focused on one chunk of DOM and one chunk of state so interaction bugs stay local.
