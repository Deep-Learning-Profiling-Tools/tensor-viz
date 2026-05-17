// public demo package surface for embedders.
// app-extension types stay exported here so extensions do not import internals.
export { mountDemoApp } from './embed.js';
export type { DemoAppOptions, MountedDemoApp } from './embed.js';
export type {
    CommandAction,
    DemoAppExtension,
    DemoExtensionContext,
    DemoExtensionFactory,
    DemoWidgetSpec,
    LoadedSessionTab,
} from './app-extension.js';
