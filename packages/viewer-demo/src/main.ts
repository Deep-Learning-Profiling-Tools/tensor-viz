// vite entrypoint; app-entry owns the actual demo boot sequence.
import { startDemoApp } from './app-entry.js';
import { DEMO_EXTENSION_FACTORIES } from './registered-extensions.js';

startDemoApp({ extensionFactories: DEMO_EXTENSION_FACTORIES });
