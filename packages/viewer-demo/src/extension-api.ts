// side-effect-light package surface for external demo extensions.
// unlike index.ts, this module never imports app-entry, so unit tests and model
// code can use extension contracts without booting the viewer or renderer.
export { escapeHtml, escapeInfo, infoButton, labelWithInfo } from './app-format.js';
export { controlIcons, renderControlDockControls } from './control-dock.js';
export type { AppShellWidgetSlot } from './app-shell.js';
export type { ControlSpec } from './control-dock.js';
export type {
    CommandAction,
    DemoAppExtension,
    DemoExtensionContext,
    DemoExtensionFactory,
    DemoInspectorCoordEntry,
    DemoTensorViewContribution,
    DemoTensorViewSliderSpec,
    DemoWidgetSpec,
    LoadedSessionTab,
} from './app-extension.js';
