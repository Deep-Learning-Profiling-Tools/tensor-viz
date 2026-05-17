import type { DemoExtensionFactory } from './app-extension.js';
import { linearLayoutExtensionFactory } from './extensions/linear-layout/extension.js';

// extensions stay explicit so production builds do not depend on runtime module discovery
export const DEMO_EXTENSION_FACTORIES = [
    linearLayoutExtensionFactory,
] satisfies DemoExtensionFactory[];
