import type { DemoExtensionFactory } from './app-extension.js';
import { linearLayoutExtensionFactory } from './extensions/linear-layout/extension.js';

export const DEMO_EXTENSION_FACTORIES = [
    linearLayoutExtensionFactory,
] satisfies DemoExtensionFactory[];
