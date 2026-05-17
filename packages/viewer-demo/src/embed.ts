/**
 * Construction-time options for mounting the full demo app in an iframe.
 *
 * @example
 * const options: DemoAppOptions = {
 *     src: '/demo/?manifest=/fixtures/linear-layout.json',
 *     title: 'Tensor Viz linear-layout demo',
 *     className: 'embedded-demo-frame',
 * };
 */
export type DemoAppOptions = {
    src?: string;
    title?: string;
    className?: string;
};

/**
 * Lifecycle handle for an iframe-mounted tensor-viz demo embedded into a host element.
 *
 * The handle exposes the created iframe so embedders can inspect or style it, and a
 * destroy callback that removes that iframe from the original container when the
 * embed is no longer needed.
 *
 * @example
 * const container = document.createElement('div');
 * const mounted = mountDemoApp(container, { src: '/viewer', title: 'tensor-viz demo' });
 *
 * console.assert(mounted.iframe.tagName === 'IFRAME');
 * console.assert(container.firstElementChild === mounted.iframe);
 * mounted.destroy();
 * console.assert(container.childElementCount === 0);
 */
export type MountedDemoApp = {
    iframe: HTMLIFrameElement;
    destroy: () => void;
};

/**
 * Normalizes an iframe source for the embedded demo and rejects protocols that could
 * execute inline script or load non-web content.
 *
 * @param src - Absolute or document-relative iframe URL supplied by the embedder.
 * @returns The trimmed source string to assign to HTMLIFrameElement.src after it resolves to an http or https URL.
 * @throws Error when src resolves against the current document base to a protocol other than http: or https:, such as javascript: or data:.
 * @example
 * safeIframeSrc('  /viewer  ');
 * // '/viewer'
 *
 * @example
 * expect(() => safeIframeSrc('javascript:alert(1)')).toThrow(/Unsafe iframe src javascript:alert\(1\)\./);
 */
function safeIframeSrc(src: string): string {
    const value = src.trim();
    const base = document.baseURI || globalThis.location?.href || 'http://localhost/';
    const protocol = new URL(value, base).protocol;
    if (protocol !== 'http:' && protocol !== 'https:') throw new Error(`Unsafe iframe src ${src}.`);
    return value;
}

/**
 * Mounts the tensor-viz demo shell into a host element by replacing the host's
 * children with a sandboxed iframe.
 *
 * @param container - Host DOM element whose existing children are replaced by the demo iframe.
 * @param options - Optional iframe configuration, including src, title, and className values for the embedded demo.
 * @returns A lifecycle handle containing the created iframe and a destroy callback that removes it from the original container.
 * @throws Error when options.src resolves to a non-http(s) URL such as javascript: or data:.
 * @example
 * const container = document.createElement('section');
 * const mounted = mountDemoApp(container, {
 *   src: '/viewer',
 *   title: 'tensor-viz demo',
 *   className: 'demo-frame',
 * });
 *
 * console.assert(container.firstElementChild === mounted.iframe);
 * console.assert(mounted.iframe.getAttribute('sandbox') === 'allow-downloads allow-same-origin allow-scripts');
 * console.assert(mounted.iframe.title === 'tensor-viz demo');
 * console.assert(mounted.iframe.className === 'demo-frame');
 *
 * mounted.destroy();
 * console.assert(container.childElementCount === 0);
 *
 * @example
 * const container = document.createElement('div');
 * expect(() => mountDemoApp(container, { src: 'data:text/html,<script></script>' })).toThrow(/Unsafe iframe src/);
 */
export function mountDemoApp(container: HTMLElement, options: DemoAppOptions = {}): MountedDemoApp {
    const iframe = document.createElement('iframe');
    iframe.src = safeIframeSrc(options.src ?? '/');
    iframe.title = options.title ?? 'tensor-viz';
    iframe.className = options.className ?? '';
    iframe.referrerPolicy = 'no-referrer';
    iframe.setAttribute('sandbox', 'allow-downloads allow-same-origin allow-scripts');
    iframe.style.width = '100%';
    iframe.style.height = '100%';
    iframe.style.border = '0';
    iframe.style.display = 'block';
    container.replaceChildren(iframe);
    return {
        iframe,
        destroy: () => {
            if (iframe.parentElement === container) container.removeChild(iframe);
        },
    };
}
