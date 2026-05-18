import type { DemoExtensionFactory } from './app-extension.js';

// extensions stay explicit so production builds do not depend on runtime module discovery
export const DEMO_EXTENSION_FACTORIES = [
] satisfies DemoExtensionFactory[];
