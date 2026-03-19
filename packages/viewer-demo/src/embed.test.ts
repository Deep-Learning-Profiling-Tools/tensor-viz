import { afterEach, describe, expect, it } from 'vitest';
import { mountDemoApp } from './embed.js';

type FakeElement = {
    className: string;
    parentElement: FakeContainer | null;
    style: Record<string, string>;
    title?: string;
    src?: string;
};

type FakeContainer = {
    children: FakeElement[];
    replaceChildren: (...children: FakeElement[]) => void;
    removeChild: (child: FakeElement) => void;
};

function createContainer(): FakeContainer {
    return {
        children: [],
        replaceChildren(...children: FakeElement[]): void {
            this.children.forEach((child) => {
                child.parentElement = null;
            });
            this.children = children;
            children.forEach((child) => {
                child.parentElement = this;
            });
        },
        removeChild(child: FakeElement): void {
            this.children = this.children.filter((entry) => entry !== child);
            child.parentElement = null;
        },
    };
}

function installDocumentStub(): void {
    Object.assign(globalThis, {
        document: {
            createElement(tagName: string): FakeElement {
                if (tagName !== 'iframe') throw new Error(`Unexpected tag ${tagName}.`);
                return {
                    className: '',
                    parentElement: null,
                    style: {},
                };
            },
        },
    });
}

afterEach(() => {
    Reflect.deleteProperty(globalThis, 'document');
});

describe('mountDemoApp', () => {
    it('mounts and destroys an iframe-backed demo app', () => {
        installDocumentStub();
        const container = createContainer();

        const mounted = mountDemoApp(container as unknown as HTMLElement, {
            src: '/viewer',
            title: 'tensor-viz demo',
            className: 'demo-frame',
        });

        expect(container.children).toHaveLength(1);
        expect(mounted.iframe.src).toBe('/viewer');
        expect(mounted.iframe.title).toBe('tensor-viz demo');
        expect(mounted.iframe.className).toBe('demo-frame');
        expect(mounted.iframe.style.width).toBe('100%');
        expect(mounted.iframe.style.height).toBe('100%');
        expect(mounted.iframe.style.border).toBe('0');
        expect(mounted.iframe.style.display).toBe('block');

        mounted.destroy();

        expect(container.children).toHaveLength(0);
    });
});
