import { afterEach, describe, expect, it } from 'vitest';
import { mountDemoApp } from './embed.js';

type FakeElement = {
    attributes: Record<string, string>;
    className: string;
    parentElement: FakeContainer | null;
    referrerPolicy?: string;
    style: Record<string, string>;
    title?: string;
    src?: string;
    setAttribute: (name: string, value: string) => void;
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
                    attributes: {},
                    className: '',
                    parentElement: null,
                    setAttribute(name: string, value: string): void {
                        this.attributes[name] = value;
                    },
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

        const iframe = mounted.iframe as unknown as FakeElement;
        expect(container.children).toHaveLength(1);
        expect(iframe.src).toBe('/viewer');
        expect(iframe.title).toBe('tensor-viz demo');
        expect(iframe.className).toBe('demo-frame');
        expect(iframe.referrerPolicy).toBe('no-referrer');
        expect(iframe.attributes.sandbox).toBe('allow-downloads allow-same-origin allow-scripts');
        expect(iframe.style.width).toBe('100%');
        expect(iframe.style.height).toBe('100%');
        expect(iframe.style.border).toBe('0');
        expect(iframe.style.display).toBe('block');

        mounted.destroy();

        expect(container.children).toHaveLength(0);
    });

    it('rejects script-like iframe sources', () => {
        installDocumentStub();
        expect(() => mountDemoApp(createContainer() as unknown as HTMLElement, {
            src: 'javascript:alert(1)',
        })).toThrow(/Unsafe iframe src/);
    });
});
