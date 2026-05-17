/**
 * Construction-time options for mounting the full demo app in an iframe.
 *
 * @example
 * const value: DemoAppOptions = {} as DemoAppOptions;
 */
export type DemoAppOptions = {
    src?: string;
    title?: string;
    className?: string;
};

/**
 * Handle returned by {@link mountDemoApp} for lifecycle control.
 *
 * @example
 * const value: MountedDemoApp = {} as MountedDemoApp;
 */
export type MountedDemoApp = {
    iframe: HTMLIFrameElement;
    destroy: () => void;
};

/**
 * return safe iframe src for the current viewer state.
 *
 * @param src - src input used by this operation (string).
 * @returns Text formatted for the caller.
 * @throws Error when the requested input or state is invalid.
 * @example
 * safeIframeSrc(src);
 */
function safeIframeSrc(src: string): string {
    const value = src.trim();
    const base = document.baseURI || globalThis.location?.href || 'http://localhost/';
    const protocol = new URL(value, base).protocol;
    if (protocol !== 'http:' && protocol !== 'https:') throw new Error(`Unsafe iframe src ${src}.`);
    return value;
}

/**
 * Mount the full demo page as an embeddable iframe-backed widget.
 *
 * @param container - container input used by this operation (HTMLElement).
 * @param options - Options that tune this operation.
 * @returns Computed MountedDemoApp value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * mountDemoApp(container, options);
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
