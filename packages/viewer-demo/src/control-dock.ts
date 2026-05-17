/**
 * Declarative description of one action button in the demo viewer's vertical control dock.
 *
 * The dock renderer uses the text fields for tooltips and accessibility, `content` for the
 * trusted icon or label markup inside the button, and `onClick` for the command to run when
 * the enabled button is activated.
 *
 * @example
 * const panControl: ControlSpec = {
 *     id: 'pan',
 *     label: 'Pan',
 *     description: 'Drag the tensor view without changing the selected cell.',
 *     shortcut: 'H',
 *     active: true,
 *     content: '<span aria-hidden="true">↔</span>',
 *     onClick: () => setInteractionMode('pan'),
 * };
 */
export type ControlSpec = {
    id: string;
    label: string;
    description: string;
    shortcut: string;
    active: boolean;
    disabled?: boolean;
    content: string;
    onClick: () => void | Promise<void>;
};

// dividers are keyed by control id instead of position so adding/removing
// controls does not silently move visual groups.
// if a new command starts a separate workflow, add its id here instead of
// baking another divider into the render loop.
const DIVIDER_BEFORE_IDS = new Set(['2d', 'dim-lines', 'gaps', 'mapping-contiguous']);

/** shared dock/widget icons so controls can move without duplicating svg markup. */
export const controlIcons = {
    selection: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 5h3v3H5zM16 5h3v3h-3zM5 16h3v3H5zM16 16h3v3h-3z" />
        <path d="M9.5 6.5h5M9.5 17.5h5M6.5 9.5v5M17.5 9.5v5" stroke-dasharray="2.2 2.2" />
      </svg>
    `,
    rotate: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 4.8A7.2 7.2 0 1 1 4.8 12" />
        <path d="M4.8 9.6l1.92 3.6H2.88z" fill="currentColor" stroke="none" />
      </svg>
    `,
    dimensionLines: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <polyline points="7,7 7,4 17,4 17,7" stroke="#ef4444" />
        <polyline points="7,7 4,7 4,17 7,17" stroke="#22c55e" />
        <rect x="7" y="7" width="10" height="10" />
      </svg>
    `,
    tensorNames: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 6h14" />
        <path d="M12 6v12" />
        <path d="M8 18h8" />
      </svg>
    `,
    gaps: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="6" width="4" height="12" />
        <rect x="10" y="6" width="4" height="12" />
        <rect x="16" y="6" width="4" height="12" />
      </svg>
    `,
    contiguousMapping: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <polyline points="4,5.5 20,5.5 4,10.5 20,10.5 4,15.5 20,15.5 4,20.5 20,20.5" />
      </svg>
    `,
    zOrderMapping: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <polyline points="4,4 9,4 4,9 9,9 15,4 20,4 15,9 20,9 4,15 9,15 4,20 9,20 15,15 20,15 15,20 20,20" />
      </svg>
    `,
    propagateOutputs: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3" y="2" width="6" height="6" />
        <rect x="3" y="16" width="6" height="6" />
        <rect x="15" y="9" width="6" height="6" />
        <line x1="18" y1="9" x2="12.655" y2="6.624" />
        <polygon points="9,5 13,3 13,7" fill="currentColor" stroke="none" transform="rotate(23.96 9 5)" />
        <line x1="18" y1="15" x2="12.655" y2="17.376" />
        <polygon points="9,19 13,17 13,21" fill="currentColor" stroke="none" transform="rotate(-23.96 9 19)" />
      </svg>
    `,
    pan: `
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M8 13v-7.5a1.5 1.5 0 0 1 3 0v6.5M11 5.5v-2a1.5 1.5 0 1 1 3 0v8.5M14 5.5a1.5 1.5 0 0 1 3 0v6.5M17 7.5a1.5 1.5 0 0 1 3 0v8.5a6 6 0 0 1-6 6h-2h.208a6 6 0 0 1-5.012-2.7a69.74 69.74 0 0 1-.196-.3c-.312-.479-1.407-2.388-3.286-5.728a1.5 1.5 0 0 1 .536-2.022a1.867 1.867 0 0 1 2.28.28l1.47 1.47" />
      </svg>
    `,
} as const;

/**
 * Rebuilds the vertical control dock from declarative control specs.
 *
 * Existing dock children are replaced with one button per control, with visual divider
 * elements inserted before the configured control groups.
 *
 * @param controlDock - Container element that owns the control-dock buttons.
 * @param controls - Ordered control definitions collected from the app shell and registered extensions.
 * @returns Nothing; callers observe the updated `controlDock` child list.
 * @noThrows Replaces DOM children from trusted control descriptors and does not validate external input.
 * @example
 * const controlDock = document.createElement('div');
 * renderControlDockControls(controlDock, [{
 *     id: 'pan',
 *     label: 'Pan',
 *     description: 'Move around the tensor view.',
 *     shortcut: 'H',
 *     active: true,
 *     content: 'Pan',
 *     onClick: () => undefined,
 * }]);
 *
 * console.assert(controlDock.querySelectorAll('button.control-button').length === 1);
 * console.assert(controlDock.querySelector('button')?.getAttribute('aria-label') === 'Pan');
 */
export function renderControlDockControls(controlDock: HTMLElement, controls: ControlSpec[]): void {
    controlDock.replaceChildren(...controls.map((control) => {
        const button = controlButton(control);
        if (!DIVIDER_BEFORE_IDS.has(control.id)) return button;
        const fragment = document.createDocumentFragment();
        const divider = document.createElement('div');
        divider.className = 'control-dock-divider';
        fragment.append(divider, button);
        return fragment;
    }));
}

/**
 * Creates the DOM button for a single control-dock action.
 *
 * The returned button receives tooltip data attributes, an accessible label, active/disabled
 * classes, trusted icon markup, and a click listener that skips disabled controls.
 *
 * @param control - Control definition containing the dock id, tooltip text, shortcut label, trusted button markup, enabled state, and click handler.
 * @returns Button element ready to insert into the control dock.
 * @noThrows Builds one DOM button from a trusted control descriptor; disabled controls simply ignore clicks.
 * @example
 * let clicked = false;
 * const button = controlButton({
 *     id: 'select',
 *     label: 'Select',
 *     description: 'Select tensor cells.',
 *     shortcut: 'V',
 *     active: false,
 *     content: '<span aria-hidden="true">□</span>',
 *     onClick: () => { clicked = true; },
 * });
 *
 * console.assert(button.type === 'button');
 * console.assert(button.dataset.tooltipShortcut === 'V');
 * console.assert(button.getAttribute('aria-label') === 'Select');
 * button.click();
 * console.assert(clicked === true);
 */
function controlButton(control: ControlSpec): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `control-button${control.active ? ' active' : ''}${control.disabled ? ' disabled' : ''}`;
    button.dataset.tooltipLabel = control.label;
    button.dataset.tooltipDescription = control.description;
    button.dataset.tooltipShortcut = control.shortcut;
    // control content is trusted markup from controlIcons or app-entry text
    // labels; user-provided strings should go into data attributes instead.
    button.innerHTML = control.content;
    button.disabled = Boolean(control.disabled);
    button.setAttribute('aria-label', control.label);
    button.addEventListener('click', async () => {
        if (control.disabled) return;
        await control.onClick();
    });
    return button;
}
