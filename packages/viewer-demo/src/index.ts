// public demo package surface for embedders.
// app-extension types stay exported here so extensions do not import internals.
// app startup stays explicit: standalone tensor-viz passes no factories, while
// downstream apps pass their workflow-specific extension factories.
export { startDemoApp } from './app-entry.js';
export type { DemoAppRuntimeOptions } from './app-entry.js';
// iframe embedding is intentionally separate from startDemoApp because hosts
// often want isolation instead of sharing their DOM with the demo shell.
export { mountDemoApp } from './embed.js';
export type { DemoAppOptions, MountedDemoApp } from './embed.js';
// extension helpers also live under ./extension-api for tests/model code that
// should not import app-entry through this root startup surface.
export { escapeHtml, escapeInfo, infoButton, labelWithInfo } from './extension-api.js';
export { controlIcons, renderControlDockControls } from './extension-api.js';
export type {
    AppShellWidgetSlot,
    CommandAction,
    ControlSpec,
    DemoAppExtension,
    DemoExtensionContext,
    DemoExtensionFactory,
    DemoInspectorCoordEntry,
    DemoTensorViewContribution,
    DemoTensorViewSliderSpec,
    DemoWidgetSpec,
    LoadedSessionTab,
} from './extension-api.js';
