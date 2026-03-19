/** Construction-time options for mounting the full demo app in an iframe. */
export type DemoAppOptions = {
    src?: string;
    title?: string;
    className?: string;
};

/** Handle returned by {@link mountDemoApp} for lifecycle control. */
export type MountedDemoApp = {
    iframe: HTMLIFrameElement;
    destroy: () => void;
};

/** Mount the full demo page as an embeddable iframe-backed widget. */
export function mountDemoApp(container: HTMLElement, options: DemoAppOptions = {}): MountedDemoApp {
    const iframe = document.createElement('iframe');
    iframe.src = options.src ?? '/';
    iframe.title = options.title ?? 'tensor-viz';
    iframe.className = options.className ?? '';
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
