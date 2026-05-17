import {
    Box3,
    BoxGeometry,
    Color,
    EdgesGeometry,
    Group,
    InstancedBufferAttribute,
    InstancedMesh,
    LineBasicMaterial,
    LineSegments,
    Matrix4,
    MeshBasicMaterial,
    MOUSE,
    NoToneMapping,
    OrthographicCamera,
    PlaneGeometry,
    PerspectiveCamera,
    Raycaster,
    SRGBColorSpace,
    Scene,
    Sphere,
    Vector2,
    Vector3,
    Vector4,
    WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
    axisWorldKeyForMode,
    DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
    displayExtent,
    displayExtent2D,
    displayHitForPoint2D,
    displayPositionForCoord,
    displayPositionForCoord2D,
    unravelIndex,
} from './layout.js';
import {
    buildTensorViewExpression,
    clearTensorViewSlices,
    defaultTensorViewEditor,
    layoutAxisLabels,
    layoutCoordIsVisible,
    layoutShape,
    mapLayoutCoordToViewCoord,
    mapViewCoordToLayoutCoord,
    mapViewCoordToTensorCoord,
    parseTensorView,
    product,
    serializeTensorViewEditor,
    supportsContiguousSelectionFastPath2D,
} from './view.js';
import {
    ACTIVE_COLOR,
    AUTO_FIT_2D_SCALE,
    AUTO_FIT_3D_DISTANCE_SCALE,
    BASE_COLOR,
    CANVAS_WORLD_SCALE,
    DEFAULT_TENSOR_SPACING,
    HOVER_COLOR,
    MAX_CANVAS_FIT_INSET,
    MIN_CANVAS_FIT_INSET,
    MeshMeta,
    normalizeCanvasZoom,
    PickMesh,
    SelectionDragState,
    SelectionPreviewUniforms,
    ViewerOptions,
    logEvent,
} from './viewer-config.js';
import { axisFamilyColor, createLine, initializeVertexColors } from './viewer-graphics.js';
import { buildTensorGroup, updateSliceMesh } from './viewer-mesh.js';
import {
    boxesIntersect,
    colorFromHueSaturation,
    colorFromRgb,
    computeMinMax,
    coordFromKey,
    coordKey,
    dtypeFromArray,
    numericValue,
    parseCustomColor,
    quantile,
    signedLog1p,
    tupleFromVector,
    vectorFromTuple,
} from './viewer-utils.js';
import type {
    BundleManifest,
    ColorInstruction,
    DType,
    DimensionMappingScheme,
    HoverInfo,
    HueSaturation,
    InteractionMode,
    NumericArray,
    RGB,
    SelectionCoords,
    TensorDataRequestReason,
    TensorHandle,
    TensorRecord,
    TensorStatus,
    TensorViewSpec,
    TensorViewSnapshot,
    Vec3,
    ViewerSnapshot,
    ViewerState,
} from './types.js';
import {
    validateBundleManifest,
    validateColorInstructions,
    validateTensorPayload,
    validateTensorShape,
} from './validation.js';
export type { ViewerOptions } from './viewer-config.js';

const SELECTION_TINT_ALPHA = 0.4;
const CELL_LABEL_DARK = 'rgba(15, 23, 42, 0.96)';
const CELL_LABEL_LIGHT = 'rgba(255, 255, 255, 0.98)';
const MIN_VISIBLE_CELL_LABEL_FONT_SIZE = 3;
const MIN_SVG_CELL_LABEL_FONT_SIZE = 1;
const TENSOR_NAME_FONT_FAMILY = '"IBM Plex Sans", "Segoe UI", sans-serif';

/**
 * Imperative tensor viewer that owns its own renderer, cameras, input handlers, and public viewer state.
 *
 * @example
 * const viewport = document.createElement('div');
 * viewport.style.width = '640px';
 * viewport.style.height = '480px';
 * document.body.appendChild(viewport);
 *
 * const viewer = new TensorViewer(viewport);
 *
 * console.assert(viewer.getSnapshot().displayMode === '2d');
 * console.assert(viewport.querySelector('canvas') !== null);
 */
export class TensorViewer {
    private readonly container: HTMLElement;
    // the webgl scene owns tensor meshes in both modes; 2d still renders meshes
    // first, then paints text/markers into a pixel canvas overlay.
    private readonly scene = new Scene();
    private readonly renderer: WebGLRenderer;
    private readonly flatCanvas: HTMLCanvasElement;
    private readonly flatContext: CanvasRenderingContext2D;
    private readonly flatOverlay: SVGSVGElement;
    private readonly selectionBox: SVGRectElement;
    private readonly perspectiveCamera: PerspectiveCamera;
    private readonly orthographicCamera: OrthographicCamera;
    private camera: PerspectiveCamera | OrthographicCamera;
    private controls: OrbitControls;
    private readonly raycaster = new Raycaster();
    private readonly pointer = new Vector2();
    private readonly cubeGeometry = new BoxGeometry(1, 1, 1);
    private readonly planeGeometry = new PlaneGeometry(1, 1);
    private readonly hoverOutline = new LineSegments(
        new EdgesGeometry(new BoxGeometry(1.1, 1.1, 1.1)),
        new LineBasicMaterial({ color: HOVER_COLOR }),
    );
    private readonly hoverOutline2D = createLine([
        new Vector3(-0.55, 0.55, 0.05),
        new Vector3(0.55, 0.55, 0.05),
        new Vector3(0.55, -0.55, 0.05),
        new Vector3(-0.55, -0.55, 0.05),
        new Vector3(-0.55, 0.55, 0.05),
    ], `#${HOVER_COLOR.getHexString()}`);
    private readonly tensors = new Map<string, TensorRecord>();
    private readonly tensorMeshes = new Map<string, Group>();
    private readonly listeners = new Set<(snapshot: ViewerSnapshot) => void>();
    private readonly hoverListeners = new Set<(hover: HoverInfo | null) => void>();
    private readonly selectionListeners = new Set<(selection: SelectionCoords) => void>();
    private readonly selectionPreviewListeners = new Set<(selection: SelectionCoords) => void>();
    // pick meshes are rebuilt with tensor meshes so hit testing cannot use stale
    // bounds after a view, gap, slice, or display-mode change.
    private readonly pickMeshes: PickMesh[] = [];
    private readonly resizeObserver?: ResizeObserver;
    private readonly state: ViewerState = {
        displayMode: '2d',
        interactionMode: 'pan',
        heatmap: true,
        dimensionBlockGapMultiple: DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
        displayGaps: false,
        logScale: false,
        collapseHiddenAxes: false,
        dimensionMappingScheme: 'z-order',
        showDimensionLines: true,
        showTensorNames: true,
        showInspectorPanel: true,
        showSelectionPanel: true,
        showHoverDetailsPanel: true,
        activeTensorId: null,
        hover: null,
        lastHover: null,
    };
    private tensorCounter = 0;
    private renderPending = false;
    private canvasZoom = 1;
    private canvasPan = { x: 0, y: 0 };
    private isCanvasPanning = false;
    private lastCanvasPointer = { x: 0, y: 0 };
    private lastHoverLogKey: string | null = null;
    private readonly selectedCells = new Map<string, Set<string>>();
    private readonly previewSelectedCells = new Map<string, Set<string>>();
    private selectionDrag: SelectionDragState | null = null;
    private readonly requestTensorDataCallback?: ViewerOptions['requestTensorData'];
    private readonly pendingTensorDataRequests = new Map<string, Promise<boolean>>();

    /**
 * Creates the WebGL canvas, 2D overlay canvas, SVG selection overlay, cameras, controls, and resize handling inside a host element.
 *
 * @param container - Host DOM element that receives the viewer canvases and overlay; static-positioned elements are promoted to `position: relative` so the overlays align.
 * @param options - Viewer startup options such as the scene background color and optional tensor-data request callback used by lazy data loading.
 * @throws Error when the browser cannot provide a 2D rendering context for the flat overlay canvas.
 * @example
 * const viewport = document.createElement('div');
 * viewport.style.width = '640px';
 * viewport.style.height = '480px';
 * document.body.appendChild(viewport);
 *
 * const viewer = new TensorViewer(viewport, { background: '#ffffff' });
 *
 * console.assert(viewport.querySelectorAll('canvas').length === 2);
 * console.assert(viewer.getSnapshot().displayMode === '2d');
 *
 * @example
 * const originalCreateElement = document.createElement.bind(document);
 * document.createElement = ((tagName: string) => {
 *     const element = originalCreateElement(tagName);
 *     if (tagName === 'canvas') {
 *         Object.defineProperty(element, 'getContext', { value: () => null });
 *     }
 *     return element;
 * }) as typeof document.createElement;
 *
 * try {
 *     new TensorViewer(document.createElement('div'));
 * } catch (error) {
 *     console.assert(error instanceof Error);
 *     console.assert(error.message === 'Unable to create 2D canvas context.');
 * } finally {
 *     document.createElement = originalCreateElement;
 * }
 */
    public constructor(container: HTMLElement, options: ViewerOptions = {}) {
        this.container = container;
        this.requestTensorDataCallback = options.requestTensorData;
        if (getComputedStyle(container).position === 'static') this.container.style.position = 'relative';
        this.scene.background = new Color(options.background ?? '#e5e7eb');
        this.renderer = new WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
        if ('outputEncoding' in this.renderer) (this.renderer as WebGLRenderer & { outputEncoding: number }).outputEncoding = 3001;
        if ('outputColorSpace' in this.renderer) this.renderer.outputColorSpace = SRGBColorSpace;
        this.renderer.toneMapping = NoToneMapping;
        // hidpi scaling makes the webgl canvas and 2d overlay disagree about pixel-space,
        // which breaks picking/layout alignment on high-density displays, so keep 1:1 pixels
        this.renderer.setPixelRatio(1);
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        initializeVertexColors(this.cubeGeometry);
        initializeVertexColors(this.planeGeometry);
        this.flatCanvas = document.createElement('canvas');
        const context = this.flatCanvas.getContext('2d');
        if (!context) throw new Error('Unable to create 2D canvas context.');
        this.flatContext = context;
        this.flatContext.imageSmoothingEnabled = false;
        this.flatOverlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.flatOverlay.style.position = 'absolute';
        this.flatOverlay.style.inset = '0';
        this.flatOverlay.style.width = '100%';
        this.flatOverlay.style.height = '100%';
        this.flatOverlay.style.pointerEvents = 'none';
        this.flatOverlay.style.display = 'none';
        this.selectionBox = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        this.selectionBox.setAttribute('fill', '#1976d220');
        this.selectionBox.setAttribute('stroke', '#1976d2');
        this.selectionBox.setAttribute('stroke-width', '2');
        this.selectionBox.setAttribute('stroke-dasharray', '8 6');
        this.selectionBox.setAttribute('display', 'none');
        this.flatOverlay.appendChild(this.selectionBox);
        this.perspectiveCamera = new PerspectiveCamera(45, 1, 0.1, 2000);
        this.orthographicCamera = new OrthographicCamera(-10, 10, 10, -10, 0.1, 2000);
        this.camera = this.perspectiveCamera;
        this.controls = this.createControls(this.camera);
        this.renderer.domElement.style.position = 'absolute';
        this.renderer.domElement.style.inset = '0';
        this.flatCanvas.style.position = 'absolute';
        this.flatCanvas.style.inset = '0';
        this.container.appendChild(this.renderer.domElement);
        this.container.appendChild(this.flatCanvas);
        this.container.appendChild(this.flatOverlay);
        this.hoverOutline.visible = false;
        this.hoverOutline2D.visible = false;
        this.scene.add(this.hoverOutline);
        this.scene.add(this.hoverOutline2D);
        this.bindEvents();
        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.resize());
            this.resizeObserver.observe(this.container);
        }
        this.resize();
        this.setDisplayMode('2d');
        logEvent('viewer:init', { background: options.background ?? '#e5e7eb' });
    }

        /**
 * Builds the OrbitControls instance that lets the viewer pan and zoom the WebGL canvas for the active camera.
 *
 * @param camera - Perspective or orthographic camera currently assigned to the viewer's Three.js scene.
 * @returns OrbitControls attached to `this.renderer.domElement`, with damping disabled, panning and zooming enabled, left and right mouse buttons mapped to pan, and change events wired to request a render.
 * @noThrows The helper performs no input validation and only configures a controls object from the already-created renderer DOM element and supplied Three.js camera.
 * @example
 * const controls = viewer.createControls(viewer.perspectiveCamera);
 * expect(controls.enablePan).toBe(true);
 * expect(controls.enableZoom).toBe(true);
 * expect(controls.mouseButtons.LEFT).toBe(MOUSE.PAN);
 */
private createControls(camera: PerspectiveCamera | OrthographicCamera): OrbitControls {
        const controls = new OrbitControls(camera, this.renderer.domElement);
        controls.enableDamping = false;
        controls.enablePan = true;
        controls.enableZoom = true;
        controls.screenSpacePanning = true;
        controls.mouseButtons.LEFT = MOUSE.PAN;
        controls.mouseButtons.RIGHT = MOUSE.PAN;
        if ('zoomToCursor' in controls) {
            (controls as OrbitControls & { zoomToCursor: boolean }).zoomToCursor = true;
        }
        controls.addEventListener('change', () => this.requestRender());
        return controls;
    }

        /**
 * Rewrites unsupported interaction modes to pan so the current display mode never exposes rotate or selection gestures that the view cannot honor.
 *
 * @returns No value; mutates `this.state.interactionMode` when the current display mode or selection layout cannot support the requested mode.
 * @noThrows The method only compares and assigns existing viewer state fields and calls `selectionEnabled()`, so it has no expected error path for valid initialized viewer state.
 * @example
 * viewer.state.displayMode = '2d';
 * viewer.state.interactionMode = 'rotate';
 * viewer.normalizeInteractionMode();
 * expect(viewer.state.interactionMode).toBe('pan');
 */
private normalizeInteractionMode(): void {
        // selection currently depends on contiguous 2d screen order; normalize here
        // so callers can request a mode without checking every display constraint.
        if (this.state.displayMode === '2d') {
            if (this.state.interactionMode === 'rotate') this.state.interactionMode = 'pan';
            if (this.state.interactionMode === 'select' && !this.selectionEnabled()) this.state.interactionMode = 'pan';
            return;
        }
        if (this.state.interactionMode === 'select') this.state.interactionMode = 'pan';
    }

        /**
 * Applies the normalized viewer interaction mode to OrbitControls so mouse gestures match the active 2-D or 3-D display.
 *
 * @returns No value; may normalize `this.state.interactionMode`, toggles `controls.enableRotate`, and maps the left mouse button to rotate only for 3-D rotate mode.
 * @noThrows The method updates properties on the viewer's initialized OrbitControls instance and does not allocate resources or validate external input.
 * @example
 * viewer.state.displayMode = '3d';
 * viewer.state.interactionMode = 'rotate';
 * viewer.syncInteractionMode();
 * expect(viewer.controls.enableRotate).toBe(true);
 * expect(viewer.controls.mouseButtons.LEFT).toBe(MOUSE.ROTATE);
 */
private syncInteractionMode(): void {
        this.normalizeInteractionMode();
        const leftButton = this.state.displayMode === '3d' && this.state.interactionMode === 'rotate' ? MOUSE.ROTATE : MOUSE.PAN;
        this.controls.enableRotate = this.state.displayMode === '3d';
        this.controls.mouseButtons.LEFT = leftButton;
        this.controls.mouseButtons.RIGHT = MOUSE.PAN;
    }

        /**
 * Registers the viewer's window, WebGL canvas, and flat 2-D canvas event handlers for resizing, pointer interaction, wheel zooming, keyboard shortcuts, and context-menu suppression.
 *
 * @returns No value; attaches DOM event listeners to `window`, `this.renderer.domElement`, and `this.flatCanvas`.
 * @noThrows The method delegates to `addEventListener` on DOM objects created during viewer construction and does not inspect event payloads while binding.
 * @example
 * const addEventListener = vi.spyOn(viewer.renderer.domElement, 'addEventListener');
 * viewer.bindEvents();
 * expect(addEventListener).toHaveBeenCalledWith('pointerdown', viewer.onPointerDown, { capture: true });
 * expect(addEventListener).toHaveBeenCalledWith('click', viewer.onClick);
 */
private bindEvents(): void {
        window.addEventListener('resize', this.resize);
        this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown, { capture: true });
        this.renderer.domElement.addEventListener('pointermove', this.onPointerMove);
        this.renderer.domElement.addEventListener('pointerleave', this.onPointerLeave);
        this.renderer.domElement.addEventListener('pointerup', this.onPointerUp);
        this.renderer.domElement.addEventListener('click', this.onClick);
        this.renderer.domElement.addEventListener('contextmenu', (event) => event.preventDefault());
        this.flatCanvas.addEventListener('pointermove', this.onCanvasPointerMove);
        this.flatCanvas.addEventListener('pointerleave', this.onCanvasPointerLeave);
        this.flatCanvas.addEventListener('click', this.onCanvasClick);
        this.flatCanvas.addEventListener('pointerdown', this.onCanvasPointerDown);
        this.flatCanvas.addEventListener('pointerup', this.onCanvasPointerUp);
        this.flatCanvas.addEventListener('wheel', this.onCanvasWheel, { passive: false });
        this.flatCanvas.addEventListener('contextmenu', (event) => event.preventDefault());
        window.addEventListener('keydown', this.onKeyDown);
    }

    public readonly resize = (): void => {
        const previousWidth = this.flatCanvas.width;
        const previousHeight = this.flatCanvas.height;
        const width = this.container.clientWidth || 1;
        const height = this.container.clientHeight || 1;
        this.renderer.setSize(width, height);
        const pixelRatio = 1;
        const nextWidth = Math.floor(width * pixelRatio);
        const nextHeight = Math.floor(height * pixelRatio);
        this.flatCanvas.width = nextWidth;
        this.flatCanvas.height = nextHeight;
        this.flatCanvas.style.width = `${width}px`;
        this.flatCanvas.style.height = `${height}px`;
        this.flatOverlay.setAttribute('viewBox', `0 0 ${this.flatCanvas.width} ${this.flatCanvas.height}`);
        this.flatOverlay.setAttribute('width', String(this.flatCanvas.width));
        this.flatOverlay.setAttribute('height', String(this.flatCanvas.height));
        this.perspectiveCamera.aspect = width / height;
        this.perspectiveCamera.updateProjectionMatrix();
        this.orthographicCamera.left = -(this.flatCanvas.width / CANVAS_WORLD_SCALE) / 2;
        this.orthographicCamera.right = (this.flatCanvas.width / CANVAS_WORLD_SCALE) / 2;
        this.orthographicCamera.top = (this.flatCanvas.height / CANVAS_WORLD_SCALE) / 2;
        this.orthographicCamera.bottom = -(this.flatCanvas.height / CANVAS_WORLD_SCALE) / 2;
        this.orthographicCamera.updateProjectionMatrix();
        if (this.state.displayMode === '2d') {
            if (this.tensors.size !== 0) {
                // browser resizes should keep the same world point under the viewport
                // center; otherwise window chrome changes visibly shove the tensor.
                this.canvasPan.x += (previousWidth - nextWidth) / 2;
                this.canvasPan.y += (previousHeight - nextHeight) / 2;
            }
            this.sync2DCamera();
        }
        logEvent('viewport:resize', { width, height });
        this.requestRender();
    };

        /**
 * Computes the ratio between the canvas backing-store width and its displayed CSS width.
 *
 * Pointer and hit-test code use this scale to convert browser client coordinates into canvas pixel coordinates, including high-DPI canvases.
 *
 * @returns The backing-pixel-per-CSS-pixel scale; returns 1 when the canvas has no measurable client width.
 * @noThrows Reads numeric width fields from the owned canvas and clamps the divisor to at least 1, so zero or missing client dimensions do not create a throw path.
 * @example
 * const viewer = { flatCanvas: { width: 1200, clientWidth: 600 } } as TensorViewer;
 * expect(viewer['canvasScale']()).toBe(2);
 *
 * viewer.flatCanvas.clientWidth = 0;
 * viewer.flatCanvas.width = 800;
 * expect(viewer['canvasScale']()).toBe(1);
 */
private canvasScale(): number {
        return this.flatCanvas.width / Math.max(1, this.flatCanvas.clientWidth || this.flatCanvas.width || 1);
    }

        /**
 * Resolves the spacing multiplier applied between tensor layout blocks for the current viewer state.
 *
 * Rendering and outline placement pass this value into layout helpers; disabling gaps forces contiguous cells regardless of the configured multiplier.
 *
 * @returns `state.dimensionBlockGapMultiple` when `state.displayGaps` is enabled, otherwise `0`.
 * @noThrows Only reads already-normalized viewer state flags and returns one of two numeric values.
 * @example
 * const viewer = {
 *   state: { displayGaps: true, dimensionBlockGapMultiple: 0.5 },
 * } as TensorViewer;
 * expect(viewer['layoutGapMultiple']()).toBe(0.5);
 *
 * viewer.state.displayGaps = false;
 * expect(viewer['layoutGapMultiple']()).toBe(0);
 */
private layoutGapMultiple(): number {
        return this.state.displayGaps ? this.state.dimensionBlockGapMultiple : 0;
    }

        /**
 * Returns the visible tensor-view dimensions used to allocate and index mesh instances.
 *
 * Scalar views have an empty `viewShape`, but the renderer still needs one mesh instance, so scalars are represented as `[1]`.
 *
 * @param spec - Parsed tensor view specification whose `viewShape` contains the visible, non-hidden dimensions of the view.
 * @returns The visible view shape for vectors and higher-rank views, or `[1]` for a scalar view so callers can render one cell.
 * @noThrows Only checks `spec.viewShape.length` and returns the existing shape array or the scalar fallback; it does not parse view expressions or map coordinates.
 * @example
 * const matrixSpec = { viewShape: [2, 3] } as TensorViewSpec;
 * expect(viewer['instanceShape'](matrixSpec)).toEqual([2, 3]);
 *
 * const scalarSpec = { viewShape: [] } as TensorViewSpec;
 * expect(viewer['instanceShape'](scalarSpec)).toEqual([1]);
 */
private instanceShape(spec: TensorViewSpec): number[] {
        return spec.viewShape.length === 0 ? [1] : spec.viewShape;
    }

        /**
 * Returns the dimensions used to position cells, outlines, and axis labels for a parsed tensor view.
 *
 * When hidden axes are collapsed, layout uses only the visible `viewShape`; otherwise it preserves the full parsed `layoutShape` so sliced or hidden axes still reserve layout space.
 *
 * @param spec - Parsed tensor view specification containing both full layout dimensions and visible view dimensions.
 * @returns The layout dimensions selected for the current `state.collapseHiddenAxes` setting, with scalar layouts normalized to `[1]` by the shared view helper.
 * @noThrows This wrapper only forwards the parsed spec and a boolean viewer-state flag to the pure `layoutShape` helper and performs no validation itself.
 * @example
 * const spec = { layoutShape: [2, 1, 4], viewShape: [2, 4] } as TensorViewSpec;
 * const viewer = { state: { collapseHiddenAxes: false } } as TensorViewer;
 * expect(viewer['layoutShape'](spec)).toEqual([2, 1, 4]);
 *
 * viewer.state.collapseHiddenAxes = true;
 * expect(viewer['layoutShape'](spec)).toEqual([2, 4]);
 */
private layoutShape(spec: TensorViewSpec): number[] {
        return layoutShape(spec, this.state.collapseHiddenAxes);
    }

        /**
 * Converts a coordinate in a tensor view's rendered instance order into the layout coordinate used to position that cell on screen.
 *
 * @param viewCoord - Coordinate whose entries follow `spec.viewShape`, such as the instance coordinate produced while iterating visible cells.
 * @param spec - Parsed tensor view specification that describes grouped axes, hidden axes, layout shape, and any fixed slice indices.
 * @returns Layout-space coordinate for display placement, with hidden sliced axes inserted unless the viewer is configured to collapse hidden axes.
 * @noThrows Delegates to pure coordinate mapping with the viewer's `collapseHiddenAxes` flag; the wrapper performs no lookup or validation that introduces an additional throw path.
 * @example
 * // For a view expression like tensor.view(A=2048).view(*A0=32, *A1=64)[1, :],
 * // the rendered view has one visible axis and the fixed A0 slice is restored in layout space.
 * const layoutCoord = this.mapViewCoordToLayoutCoord([3], spec);
 * expect(layoutCoord).toEqual([1, 3]);
 */
private mapViewCoordToLayoutCoord(viewCoord: number[], spec: TensorViewSpec): number[] {
        return mapViewCoordToLayoutCoord(viewCoord, spec, this.state.collapseHiddenAxes);
    }

        /**
 * Converts a layout/display coordinate from hit testing or cell placement back into the tensor view coordinate used for data lookup.
 *
 * @param layoutCoord - Coordinate in `spec.layoutShape` order, including positions for layout axes that may be fixed by view slices.
 * @param spec - Parsed tensor view specification that defines the relationship between layout axes and view axes.
 * @returns View-space coordinate that can be passed to tensor-coordinate mapping and hover or selection logic.
 * @noThrows Delegates to pure coordinate mapping with the viewer's `collapseHiddenAxes` flag; the wrapper does not read tensor data or require external state beyond that flag.
 * @example
 * // With hidden axes collapsed for a sliced view whose visible layout axes are A and C,
 * // the display coordinate is already the view coordinate.
 * this.state.collapseHiddenAxes = true;
 * const viewCoord = this.mapLayoutCoordToViewCoord([1, 2], spec);
 * expect(viewCoord).toEqual([1, 2]);
 */
private mapLayoutCoordToViewCoord(layoutCoord: number[], spec: TensorViewSpec): number[] {
        return mapLayoutCoordToViewCoord(layoutCoord, spec, this.state.collapseHiddenAxes);
    }

        /**
 * Reports whether a layout cell should participate in rendering and hit testing for the current hidden-axis display mode.
 *
 * @param coord - Layout-space coordinate to test against the slice constraints encoded in `spec`.
 * @param spec - Parsed tensor view specification whose fixed slice indices determine which expanded layout cells are visible.
 * @returns `true` when the coordinate is drawable/selectable; `false` when it belongs to a hidden sliced position that should be skipped.
 * @noThrows Uses only slice matching and the viewer's `collapseHiddenAxes` flag, so normal in-bounds coordinate checks do not introduce an expected throw path.
 * @example
 * const visible = this.layoutCoordIsVisible([1, 2], spec);
 * expect(visible).toBe(true);
 *
 * const hiddenBySlice = this.layoutCoordIsVisible([0, 2], spec);
 * expect(hiddenBySlice).toBe(false);
 */
private layoutCoordIsVisible(coord: number[], spec: TensorViewSpec): boolean {
        return layoutCoordIsVisible(coord, spec, this.state.collapseHiddenAxes);
    }

        /**
 * Applies a tensor record's optional coordinate allow-list before rendering, labeling, hovering, or selecting a tensor cell.
 *
 * @param tensor - Tensor metadata and data record; when `visibleCoords` is present, it contains serialized tensor coordinates that remain visible.
 * @param tensorCoord - Coordinate in the tensor's original index space to check against `tensor.visibleCoords`.
 * @returns `true` when the tensor has no visibility filter or the coordinate key is included in that filter; otherwise `false`.
 * @noThrows The check only tests for an optional `Set` and performs a membership lookup using the serialized coordinate key.
 * @example
 * const tensor = { visibleCoords: new Set([coordKey([0, 1])]) } as TensorRecord;
 *
 * expect(this.tensorCoordVisible(tensor, [0, 1])).toBe(true);
 * expect(this.tensorCoordVisible(tensor, [1, 1])).toBe(false);
 */
private tensorCoordVisible(tensor: TensorRecord, tensorCoord: number[]): boolean {
        return !tensor.visibleCoords || tensor.visibleCoords.has(coordKey(tensorCoord));
    }

        /**
 * Returns the axis labels that should be drawn for a tensor view using the viewer's current hidden-axis collapse setting.
 *
 * @param spec - Parsed tensor view specification whose tokens contain the layout-axis labels and visibility flags.
 * @returns Ordered labels for the rendered layout axes; when `state.collapseHiddenAxes` is enabled, labels for hidden tokens are omitted.
 * @noThrows Reads normalized fields from the supplied view spec and delegates to the pure view helper, so there is no expected throw path for parsed viewer specs.
 * @example
 * ```ts
 * const result = parseTensorView('tensor.view(A=2, B=4)[1, :]');
 * if (!result.ok) throw new Error(result.error);
 * viewer.state.collapseHiddenAxes = true;
 * expect(viewer['layoutAxisLabels'](result.spec)).toEqual(['B']);
 * ```
 */
private layoutAxisLabels(spec: TensorViewSpec): string[] {
        return layoutAxisLabels(spec, this.state.collapseHiddenAxes);
    }

        /**
 * Builds the narrow callback surface that stateless mesh builders use to read viewer render state and invoke viewer coordinate, color, selection, and render helpers.
 *
 * @returns Mesh-construction context containing shared geometries, current viewer state, tensor mesh storage, and bound callbacks used by `viewer-mesh` to build or update Three.js objects.
 * @noThrows This method only packages existing viewer fields and creates bound closures; any failures would occur later when a returned callback is invoked with invalid mesh or tensor data.
 * @example
 * ```ts
 * const context = viewer['meshContext']();
 * expect(context.state).toBe(viewer.state);
 * expect(context.tensorMeshes).toBe(viewer['tensorMeshes']);
 * expect(context.layoutAxisLabels(tensor.view)).toEqual(viewer['layoutAxisLabels'](tensor.view));
 * ```
 */
private meshContext() {
        // viewer-mesh is deliberately stateless; this context is the narrow bridge
        // that lets mesh construction ask the viewer for current render settings.
        return {
            cubeGeometry: this.cubeGeometry,
            planeGeometry: this.planeGeometry,
            state: this.state,
            tensorMeshes: this.tensorMeshes,
            instanceShape: (spec: TensorViewSpec) => this.instanceShape(spec),
            layoutShape: (spec: TensorViewSpec) => this.layoutShape(spec),
            layoutAxisLabels: (spec: TensorViewSpec) => this.layoutAxisLabels(spec),
            layoutGapMultiple: () => this.layoutGapMultiple(),
            mapViewCoordToLayoutCoord: (viewCoord: number[], spec: TensorViewSpec) => this.mapViewCoordToLayoutCoord(viewCoord, spec),
            selectionStateAttribute: (mesh: InstancedMesh) => this.selectionStateAttribute(mesh),
            installSelectionPreviewShader: (mesh: InstancedMesh) => this.installSelectionPreviewShader(mesh),
            heatmapNormalizedValue: (value: number, min: number, max: number) => this.heatmapNormalizedValue(value, min, max),
            baseCellColor: (
                tensor: TensorRecord,
                tensorCoord: number[],
                value: number,
                heatmapRange: { min: number; max: number } | null,
            ) => this.baseCellColor(tensor, tensorCoord, value, heatmapRange),
            tensorCoordVisible: (tensor: TensorRecord, tensorCoord: number[]) => this.tensorCoordVisible(tensor, tensorCoord),
            isSelectedCell: (tensorId: string, tensorCoord: number[]) => this.isHighlightedCell(tensorId, tensorCoord),
            selectedColor: (color: Color) => this.selectedColor(color),
            linearIndex: (coord: number[], shape: number[]) => this.linearIndex(coord, shape),
            clearHover: () => this.clearHover(),
            requestRender: () => this.requestRender(),
            emit: () => this.emit(),
        };
    }

        /**
 * Computes the rendered world-space span of a tensor for the requested display mode, using its parsed view shape and the viewer's gap and dimension-mapping settings.
 *
 * @param tensor - Normalized tensor record whose `view` describes the visible layout shape to measure.
 * @param mode - Display mode to measure for; `'2d'` returns a flat extent with z set to `0`, while `'3d'` includes depth from cube layout.
 * @returns `[x, y, z]` extent used to place tensors, preserve scene bounds, and space automatic offsets.
 * @noThrows The calculation reads normalized tensor view data and viewer state and chooses between pure extent helpers, so parsed tensor records have no expected throw path.
 * @example
 * ```ts
 * const tensor = makeTensorRecord({ view: parseOk('tensor.view(A=2, B=3)') });
 * const extent2d = viewer['tensorExtentForMode'](tensor, '2d');
 * expect(extent2d[0]).toBeGreaterThan(0);
 * expect(extent2d[1]).toBeGreaterThan(0);
 * expect(extent2d[2]).toBe(0);
 * ```
 */
private tensorExtentForMode(tensor: TensorRecord, mode: '2d' | '3d'): Vec3 {
        const shape = layoutShape(tensor.view, this.state.collapseHiddenAxes);
        if (mode === '2d') {
            const extent = displayExtent2D(shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
            return [extent.x, extent.y, 0];
        }
        const extent = displayExtent(shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        return [extent.x, extent.y, extent.z];
    }

        /**
 * Chooses the default world offset for a newly added tensor by placing it to the right of the current tensor bounds.
 *
 * @param tensor - New tensor record being inserted; its rendered width is included so the returned x offset positions its center without overlapping existing tensors.
 * @param mode - Display mode used for width and spacing calculations; `'3d'` doubles the default spacing between tensors.
 * @returns Offset vector for the new tensor. Returns `[0, 0, 0]` when the scene has no existing finite tensor bounds; otherwise returns an x-axis placement after the current right edge with y and z at `0`.
 * @noThrows Iterates the viewer's normalized tensor map and uses extent math only, so there is no expected throw path for tensors already accepted by the viewer.
 * @example
 * ```ts
 * viewer['tensors'].clear();
 * const tensor = makeTensorRecord({ view: parseOk('tensor.view(A=2, B=2)') });
 * expect(viewer['autoTensorOffset'](tensor, '2d')).toEqual([0, 0, 0]);
 * ```
 */
private autoTensorOffset(tensor: TensorRecord, mode: '2d' | '3d'): Vec3 {
        // new tensors are placed to the right of the current world extent so demos
        // can append tensors without manually managing offsets.
        const spacing = mode === '3d' ? DEFAULT_TENSOR_SPACING * 2 : DEFAULT_TENSOR_SPACING;
        const [width] = this.tensorExtentForMode(tensor, mode);
        let maxRight = Number.NEGATIVE_INFINITY;
        for (const existing of this.tensors.values()) {
            const [existingWidth] = this.tensorExtentForMode(existing, mode);
            maxRight = Math.max(maxRight, existing.offset[0] + existingWidth / 2);
        }
        if (!Number.isFinite(maxRight)) return [0, 0, 0];
        return [maxRight + spacing + width / 2, 0, 0];
    }

        /**
 * Repositions all loaded tensors in a horizontal row after a display-mode or view-size change while keeping the row's previous visual center fixed.
 *
 * @param mode - Display mode whose tensor extents should be used for the relayout; 3D mode doubles the inter-tensor spacing, while 2D mode uses the default spacing.
 * @returns Nothing; each TensorRecord.offset is rewritten in place, or the only tensor is reset to the origin.
 * @noThrows Uses the existing tensor map, numeric extents, and finite fallbacks only; empty, single-tensor, and non-finite center cases are handled without raising an error.
 * @example
 * ```ts
 * const viewer = tensorViewer as any;
 * viewer.tensors = new Map([
 *   ['a', { offset: [-10, 4, 2] }],
 *   ['b', { offset: [10, -1, 3] }],
 * ]);
 * viewer.tensorExtentForMode = () => [4, 1, 1];
 *
 * viewer.relayoutTensorOffsets('2d');
 *
 * // The y/z offsets are reset for the row layout and the tensors remain centered around x=0.
 * expect(viewer.tensors.get('a').offset).toEqual([-DEFAULT_TENSOR_SPACING / 2 - 2, 0, 0]);
 * expect(viewer.tensors.get('b').offset).toEqual([DEFAULT_TENSOR_SPACING / 2 + 2, 0, 0]);
 * ```
 */
private relayoutTensorOffsets(mode: '2d' | '3d' = this.state.displayMode): void {
        const tensors = Array.from(this.tensors.values());
        if (tensors.length < 2) {
            if (tensors[0]) tensors[0].offset = [0, 0, 0];
            return;
        }
        const spacing = mode === '3d' ? DEFAULT_TENSOR_SPACING * 2 : DEFAULT_TENSOR_SPACING;
        let minLeft = Number.POSITIVE_INFINITY;
        let maxRight = Number.NEGATIVE_INFINITY;
        tensors.forEach((tensor) => {
            const [width] = this.tensorExtentForMode(tensor, mode);
            minLeft = Math.min(minLeft, tensor.offset[0] - width / 2);
            maxRight = Math.max(maxRight, tensor.offset[0] + width / 2);
        });
        const previousCenter = Number.isFinite(minLeft) && Number.isFinite(maxRight) ? (minLeft + maxRight) / 2 : 0;
        // preserve the visual center while recomputing spacing, otherwise toggling
        // 2d/3d or gaps makes the whole scene jump under the camera.
        const widths = tensors.map((tensor) => this.tensorExtentForMode(tensor, mode)[0]);
        const totalWidth = widths.reduce((sum, width) => sum + width, 0) + spacing * (tensors.length - 1);
        let left = previousCenter - totalWidth / 2;
        tensors.forEach((tensor, index) => {
            const width = widths[index] ?? 0;
            tensor.offset = [left + width / 2, 0, 0];
            left += width + spacing;
        });
    }

        /**
 * Places tensors marked with autoOffset immediately to the right of the widest occupied edge seen so far, leaving manually positioned tensors at their current coordinates.
 *
 * @returns Nothing; only TensorRecord.offset[0] is updated for records whose autoOffset flag is set.
 * @noThrows Iterates over the current tensor records and uses the active display mode to measure widths; the first auto-offset tensor falls back to x=0 when no prior right edge exists.
 * @example
 * ```ts
 * const viewer = tensorViewer as any;
 * viewer.state = { displayMode: '2d' };
 * viewer.tensorExtentForMode = () => [10, 1, 1];
 * viewer.tensors = new Map([
 *   ['manual', { autoOffset: false, offset: [20, 5, 0] }],
 *   ['auto', { autoOffset: true, offset: [0, 7, 1] }],
 * ]);
 *
 * viewer.relayoutAutoOffsets();
 *
 * expect(viewer.tensors.get('manual').offset).toEqual([20, 5, 0]);
 * expect(viewer.tensors.get('auto').offset).toEqual([25 + DEFAULT_TENSOR_SPACING + 5, 7, 1]);
 * ```
 */
private relayoutAutoOffsets(): void {
        let maxRight = Number.NEGATIVE_INFINITY;
        this.tensors.forEach((tensor) => {
            const [width] = this.tensorExtentForMode(tensor, this.state.displayMode);
            if (tensor.autoOffset) {
                const x = !Number.isFinite(maxRight) ? 0 : maxRight + DEFAULT_TENSOR_SPACING + width / 2;
                tensor.offset = [x, tensor.offset[1], tensor.offset[2]];
            }
            maxRight = Math.max(maxRight, tensor.offset[0] + width / 2);
        });
    }

        /**
 * Builds the value portion of hover metadata for a tensor cell, including whether the displayed color came from a custom override, the heatmap, or the base color.
 *
 * @param tensor - Tensor record for the hovered cell, including its shape, optional numeric data buffer, custom color coordinates, and value range.
 * @param tensorCoord - Tensor-space coordinate of the hovered cell after view-coordinate mapping.
 * @returns The numeric cell value when tensor data is loaded, otherwise null, plus the color source label consumed by hover rendering.
 * @noThrows Assumes the caller has already mapped and visibility-checked the coordinate; missing tensor data is represented as null instead of throwing.
 * @example
 * ```ts
 * const viewer = tensorViewer as any;
 * viewer.state = { heatmap: true };
 * viewer.linearIndex = () => 3;
 * const tensor = {
 *   hasData: true,
 *   data: new Float32Array([0, 1, 2, 42]),
 *   shape: [2, 2],
 *   customColors: new Map(),
 *   valueRange: { min: 0, max: 42 },
 * };
 *
 * expect(viewer.hoverValue(tensor, [1, 1])).toEqual({ value: 42, colorSource: 'heatmap' });
 * ```
 */
private hoverValue(tensor: TensorRecord, tensorCoord: number[]): Pick<HoverInfo, 'value' | 'colorSource'> {
        const value = tensor.hasData ? numericValue(tensor.data, this.linearIndex(tensorCoord, tensor.shape)) : null;
        return {
            value,
            colorSource: tensor.customColors.has(coordKey(tensorCoord))
                ? 'custom'
                : this.state.heatmap && tensor.valueRange
                    ? 'heatmap'
                    : 'base',
        };
    }

        /**
 * Converts a tensor cell value into the clamped 0..1 fraction used for grayscale heatmaps and custom-color saturation.
 *
 * @param value - Numeric tensor cell value being colored.
 * @param min - Lower bound of the active heatmap range for the tensor.
 * @param max - Upper bound of the active heatmap range for the tensor.
 * @returns A normalized heatmap fraction where 0 maps to the low end, 1 maps to the high end, and out-of-range values are clamped.
 * @noThrows A zero-width range is treated as a range of 1, and log-scale mode applies signed log1p to all three inputs before normalization.
 * @example
 * ```ts
 * const viewer = tensorViewer as any;
 * viewer.state = { logScale: false };
 *
 * expect(viewer.heatmapNormalizedValue(15, 10, 20)).toBe(0.5);
 * expect(viewer.heatmapNormalizedValue(30, 10, 20)).toBe(1);
 * expect(viewer.heatmapNormalizedValue(5, 10, 20)).toBe(0);
 * ```
 */
private heatmapNormalizedValue(value: number, min: number, max: number): number {
        const scaledValue = this.state.logScale ? signedLog1p(value) : value;
        const scaledMin = this.state.logScale ? signedLog1p(min) : min;
        const scaledMax = this.state.logScale ? signedLog1p(max) : max;
        const range = scaledMax - scaledMin || 1;
        return Math.max(0, Math.min(1, (scaledValue - scaledMin) / range));
    }

        /**
 * Applies the current 2D pan and zoom state to the orthographic camera used for
 * flat tensor-cell rendering.
 *
 * @returns Nothing; `canvasZoom`, `orthographicCamera.zoom`, camera position, look target, and projection matrix are updated in place.
 * @noThrows Uses numeric viewer fields and Three.js camera mutators only; invalid zoom values are normalized before they are written to the camera.
 * @example
 * // With canvasPan = { x: 200, y: -100 } and canvasZoom = 2, the 2D camera is
 * // moved to the pan-adjusted world position and its projection matrix is refreshed.
 * viewer.sync2DCamera();
 * expect(viewer.orthographicCamera.zoom).toBe(2);
 * expect(viewer.orthographicCamera.position.z).toBe(30);
 */
private sync2DCamera(): void {
        const zoom = normalizeCanvasZoom(this.canvasZoom);
        this.canvasZoom = zoom;
        this.orthographicCamera.zoom = zoom;
        this.orthographicCamera.position.set(
            -this.canvasPan.x / (CANVAS_WORLD_SCALE * zoom),
            this.canvasPan.y / (CANVAS_WORLD_SCALE * zoom),
            30,
        );
        this.orthographicCamera.lookAt(this.orthographicCamera.position.x, this.orthographicCamera.position.y, 0);
        this.orthographicCamera.updateProjectionMatrix();
    }

        /**
 * Converts a browser pointer position over the flat canvas into the 2D tensor-layout
 * world coordinate used by hit testing, accounting for canvas CSS scaling, pan, and zoom.
 *
 * @param clientX - `PointerEvent.clientX` or `MouseEvent.clientX` measured in viewport pixels.
 * @param clientY - `PointerEvent.clientY` or `MouseEvent.clientY` measured in viewport pixels.
 * @returns The `{ x, y }` world coordinate under the pointer in the flat 2D tensor layout; callers pass it to selection and hover hit tests.
 * @noThrows Canvas bounds with zero CSS width or height are clamped with `Math.max(1, ...)`, so the conversion avoids division by zero and only reads local viewer/canvas state.
 * @example
 * // A pointer at the visual center of an unpanned, unzoomed 800x600 canvas maps
 * // to the origin of the 2D tensor world.
 * viewer.flatCanvas.getBoundingClientRect = () => ({ left: 10, top: 20, width: 800, height: 600 } as DOMRect);
 * viewer.flatCanvas.width = 800;
 * viewer.flatCanvas.height = 600;
 * viewer.canvasPan = { x: 0, y: 0 };
 * viewer.canvasZoom = 1;
 * expect(viewer.canvasPointerToWorld(410, 320)).toEqual({ x: 0, y: -0 });
 */
private canvasPointerToWorld(clientX: number, clientY: number): { x: number; y: number } {
        const rect = this.flatCanvas.getBoundingClientRect();
        const scaleX = this.flatCanvas.width / Math.max(1, rect.width);
        const scaleY = this.flatCanvas.height / Math.max(1, rect.height);
        return {
            x: ((clientX - rect.left) * scaleX - this.flatCanvas.width / 2 - this.canvasPan.x) / (CANVAS_WORLD_SCALE * this.canvasZoom),
            y: -(((clientY - rect.top) * scaleY - this.flatCanvas.height / 2 - this.canvasPan.y) / (CANVAS_WORLD_SCALE * this.canvasZoom)),
        };
    }

        /**
 * Finds pickable tensor meshes under the pointer for the active display mode and
 * keeps the current hover or active tensor first when multiple tensors overlap.
 *
 * @param clientX - Pointer x coordinate in viewport pixels, used for 3D ray picking or for computing a 2D point when `point2D` is omitted.
 * @param clientY - Pointer y coordinate in viewport pixels, used for 3D ray picking or for computing a 2D point when `point2D` is omitted.
 * @param point2D - Optional precomputed flat-layout world coordinate for 2D picking, supplied by callers that already converted the pointer position.
 * @returns Matching `PickMesh` records whose 2D rectangles contain `point2D` or whose 3D bounds intersect the current raycaster ray, sorted so the hovered tensor outranks the active tensor, which outranks other hits.
 * @noThrows The method filters and sorts the existing `pickMeshes` array and reads the current raycaster/state; it does not allocate external resources or validate caller data by throwing.
 * @example
 * viewer.state.displayMode = '2d';
 * viewer.state.hover = { tensorId: 'weights' } as HoverInfo;
 * viewer.state.activeTensorId = 'bias';
 * viewer.pickMeshes = [biasEntry, weightsEntry, otherEntry];
 *
 * const hits = viewer.pickEntries(0, 0, { x: 4, y: 3 });
 * expect(hits.map((entry) => entry.tensorId)).toEqual(['weights', 'bias']);
 */
private pickEntries(clientX: number, clientY: number, point2D?: { x: number; y: number }): PickMesh[] {
        const currentTensorId = this.state.hover?.tensorId;
        const activeTensorId = this.state.activeTensorId;
        const candidates = this.state.displayMode === '2d'
            ? (() => {
                const point = point2D ?? this.canvasPointerToWorld(clientX, clientY);
                return this.pickMeshes.filter((entry) => point.x >= entry.rect2D.minX
                    && point.x <= entry.rect2D.maxX
                    && point.y >= entry.rect2D.minY
                    && point.y <= entry.rect2D.maxY);
            })()
            : this.pickMeshes.filter((entry) => this.raycaster.ray.intersectsBox(entry.bounds));
        // overlapping tensors should keep interacting with the current or active
        // tensor instead of flickering between candidates as the pointer moves.
        candidates.sort((left, right) => {
            const leftPriority = left.tensorId === currentTensorId ? 2 : left.tensorId === activeTensorId ? 1 : 0;
            const rightPriority = right.tensorId === currentTensorId ? 2 : right.tensorId === activeTensorId ? 1 : 0;
            return rightPriority - leftPriority;
        });
        return candidates;
    }

        /**
 * Resolves a pointer position on the flat canvas to the visible tensor cell that
 * should receive hover feedback.
 *
 * @param clientX - `PointerEvent.clientX` or `MouseEvent.clientX` for the canvas pointer location in viewport pixels.
 * @param clientY - `PointerEvent.clientY` or `MouseEvent.clientY` for the canvas pointer location in viewport pixels.
 * @returns Hover metadata and the cell's world-space label/tooltip anchor when the pointer is over a visible tensor cell, or `null` when the pointer misses all visible cells.
 * @noThrows Pointer misses are represented as `null`; candidate tensor ids come from the viewer's own pick meshes, so normal hover handling does not expect missing tensor lookups.
 * @example
 * const hit = viewer.canvasPointerToHover(event.clientX, event.clientY);
 * if (hit) {
 *   expect(hit.hover.tensorId).toBe('activations');
 *   expect(hit.hover.tensorCoord).toEqual([0, 2]);
 *   expect(hit.position.z).toBe(0);
 * } else {
 *   expect(hit).toBeNull();
 * }
 */
private canvasPointerToHover(clientX: number, clientY: number): { hover: HoverInfo; position: Vector3 } | null {
        const point = this.canvasPointerToWorld(clientX, clientY);
        const entries = this.pickEntries(clientX, clientY, point);
        for (const entry of entries) {
            const tensor = this.requireTensor(entry.tensorId);
            const shape = this.layoutShape(tensor.view);
            const hit = displayHitForPoint2D(
                point.x - tensor.offset[0],
                point.y - tensor.offset[1],
                shape,
                this.layoutGapMultiple(),
                this.state.dimensionMappingScheme,
            );
            if (!hit) continue;
            if (!this.layoutCoordIsVisible(hit.coord, tensor.view)) continue;
            const viewCoord = this.mapLayoutCoordToViewCoord(hit.coord, tensor.view);
            const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
            if (!this.tensorCoordVisible(tensor, tensorCoord)) continue;
            const { value, colorSource } = this.hoverValue(tensor, tensorCoord);
            return {
                hover: {
                    tensorId: tensor.id,
                    tensorName: tensor.name,
                    viewCoord: viewCoord.slice(),
                    layoutCoord: this.mapViewCoordToLayoutCoord(viewCoord, tensor.view),
                    tensorCoord: tensorCoord.slice(),
                    value,
                    colorSource,
                },
                position: new Vector3(tensor.offset[0] + hit.position.x, tensor.offset[1] + hit.position.y, 0),
            };
        }
        return null;
    }

        /**
 * Converts a browser pointer position over the WebGL canvas into hover data for the first visible tensor cell hit by the 3D raycaster.
 *
 * @param clientX - Pointer `MouseEvent.clientX` coordinate in viewport pixels, measured against the renderer canvas bounds.
 * @param clientY - Pointer `MouseEvent.clientY` coordinate in viewport pixels, measured against the renderer canvas bounds.
 * @returns The hovered tensor id/name, view/layout/tensor coordinates, displayed value, color source, and instance position for hover outline placement, or `null` when the ray misses selectable tensor meshes or hits a hidden coordinate.
 * @noThrows Uses the current renderer, raycaster, and prebuilt mesh metadata; misses and hidden coordinates return `null` instead of throwing.
 * @example
 * ```ts
 * // With the renderer canvas at { left: 100, top: 50, width: 400, height: 300 },
 * // a pointer event at the center of a rendered tensor cell can resolve to hover metadata.
 * const hit = viewer.scenePointerToHover(300, 200);
 * expect(hit?.hover.tensorId).toBe('weights');
 * expect(hit?.hover.tensorCoord).toEqual([2, 1]);
 * expect(hit?.position).toBeInstanceOf(Vector3);
 *
 * // A pointer outside all pickable cell meshes clears hover state.
 * expect(viewer.scenePointerToHover(20, 20)).toBeNull();
 * ```
 */
private scenePointerToHover(clientX: number, clientY: number): { hover: HoverInfo; position: Vector3 } | null {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
        this.pointer.y = -(((clientY - rect.top) / rect.height) * 2 - 1);
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const entries = this.pickEntries(clientX, clientY);
        if (entries.length === 0) return null;
        const hits = this.raycaster.intersectObjects(entries.map((entry) => entry.mesh), false);
        const hit = hits[0];
        if (!hit || hit.instanceId === undefined) return null;

        const mesh = hit.object as InstancedMesh;
        const meta = mesh.userData.meta as MeshMeta;
        const tensor = this.requireTensor(meta.tensorId);
        const viewCoord = tensor.view.viewShape.length === 0 ? [] : unravelIndex(hit.instanceId, meta.instanceShape);
        const layoutCoord = this.mapViewCoordToLayoutCoord(viewCoord, tensor.view);
        const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
        if (!this.tensorCoordVisible(tensor, tensorCoord)) return null;
        const { value, colorSource } = this.hoverValue(tensor, tensorCoord);
        const instanceMatrix = new Matrix4();
        const position = new Vector3();
        mesh.getMatrixAt(hit.instanceId, instanceMatrix);
        position.setFromMatrixPosition(instanceMatrix);
        return {
            hover: {
                tensorId: meta.tensorId,
                tensorName: this.tensors.get(meta.tensorId)?.name ?? meta.tensorId,
                viewCoord: viewCoord.slice(),
                layoutCoord: layoutCoord.slice(),
                tensorCoord: tensorCoord.slice(),
                value,
                colorSource,
            },
            position,
        };
    }

        /**
 * Tests whether two hover records identify the same tensor cell so hover updates can skip redundant state changes.
 *
 * @param left - Existing hover record from viewer state, or `null` when no cell is currently hovered.
 * @param right - Candidate hover record from 2D or 3D picking, or `null` when the pointer is not over a visible cell.
 * @returns `true` only when both records are non-null, have the same `tensorId`, and contain identical `tensorCoord` values in the same order.
 * @noThrows Compares nullable objects, strings, coordinate lengths, and numeric array entries without performing lookups or mutations.
 * @example
 * ```ts
 * const a = { tensorId: 'weights', tensorCoord: [1, 2] } as HoverInfo;
 * const sameCell = { tensorId: 'weights', tensorCoord: [1, 2] } as HoverInfo;
 * const otherCell = { tensorId: 'weights', tensorCoord: [1, 3] } as HoverInfo;
 *
 * expect(viewer.sameHover(a, sameCell)).toBe(true);
 * expect(viewer.sameHover(a, otherCell)).toBe(false);
 * expect(viewer.sameHover(a, null)).toBe(false);
 * ```
 */
private sameHover(left: HoverInfo | null, right: HoverInfo | null): boolean {
        return !!left
            && !!right
            && left.tensorId === right.tensorId
            && left.tensorCoord.length === right.tensorCoord.length
            && left.tensorCoord.every((value, index) => value === right.tensorCoord[index]);
    }

        /**
 * Counts all selected tensor-cell coordinate keys currently tracked by the viewer selection state.
 *
 * @returns The total number of selected cells across every tensor id in `selectedCells`; each coordinate key in a tensor's set contributes one cell.
 * @noThrows Iterates the in-memory selection `Map` and reads each `Set.size` without parsing coordinates or accessing tensor data.
 * @example
 * ```ts
 * viewer.selectedCells = new Map([
 *   ['weights', new Set(['0,0', '0,1'])],
 *   ['bias', new Set(['2'])],
 * ]);
 *
 * expect(viewer.selectionCount()).toBe(3);
 * ```
 */
private selectionCount(): number {
        let count = 0;
        this.selectionEntries().forEach((entries) => {
            count += entries.size;
        });
        return count;
    }

        /**
 * Provides the viewer's selected-cell storage grouped by tensor id for selection rendering and summary calculations.
 *
 * @returns The current `selectedCells` map, where each key is a tensor id and each value is the set of serialized tensor-coordinate keys selected for that tensor.
 * @noThrows Returns the existing in-memory map reference without allocating, parsing coordinate keys, or querying tensor data.
 * @example
 * ```ts
 * viewer.selectedCells = new Map([
 *   ['weights', new Set(['1,0', '1,1'])],
 * ]);
 *
 * const entries = viewer.selectionEntries();
 * expect(entries.get('weights')).toEqual(new Set(['1,0', '1,1']));
 * expect(entries).toBe(viewer.selectedCells);
 * ```
 */
private selectionEntries(): Map<string, Set<string>> {
        return this.selectedCells;
    }

        /**
 * Converts the viewer's internal selection map into the public SelectionCoords shape emitted to selection listeners and returned by selection APIs.
 *
 * @param entries - Map whose keys are tensor ids and whose values are encoded coordinate keys from the committed selection set or a drag-preview selection set.
 * @returns A new map with the same tensor ids and decoded tensor coordinate arrays, suitable for callbacks such as selection preview listeners and getSelectedCoords consumers.
 * @noThrows The helper only iterates the provided in-memory map and decodes coordinate keys that were produced by the viewer's own selection-key serializer.
 * @example
 * const entries = new Map([
 *   ['weights', new Set(['0,1', '2,3'])],
 * ]);
 *
 * const coords = this.selectionCoords(entries);
 *
 * // coords.get('weights') is [[0, 1], [2, 3]].
 */
private selectionCoords(entries: Map<string, Set<string>> = this.selectionEntries()): SelectionCoords {
        return new Map(Array.from(entries.entries(), ([tensorId, coords]) => [
            tensorId,
            Array.from(coords, (key) => coordFromKey(key)),
        ]));
    }

        /**
 * Reports whether cell selection controls may operate for the given viewer mode and dimension mapping.
 *
 * @param displayMode - Display mode to evaluate; selection is only supported while the viewer is rendering the flat 2D canvas.
 * @param dimensionMappingScheme - Active dimension mapping scheme; selection requires contiguous mapping so screen rectangles correspond to contiguous tensor coordinates.
 * @returns True when selection gestures, selected-coordinate APIs, and selection summaries should be enabled; false when selection state should be ignored or cleared.
 * @noThrows The method only compares the supplied mode and mapping literals and does not inspect tensor data or DOM state.
 * @example
 * this.selectionEnabled('2d', 'contiguous'); // true
 * this.selectionEnabled('3d', 'contiguous'); // false
 * this.selectionEnabled('2d', 'z-order'); // false
 */
private selectionEnabled(
        displayMode: '2d' | '3d' = this.state.displayMode,
        dimensionMappingScheme: DimensionMappingScheme = this.state.dimensionMappingScheme,
    ): boolean {
        return displayMode === '2d' && dimensionMappingScheme === 'contiguous';
    }

        /**
 * Looks up whether a tensor coordinate is part of the committed selection set.
 *
 * @param tensorId - Id of the tensor whose selected-coordinate set should be queried.
 * @param tensorCoord - Canonical tensor coordinate, such as the row and column coordinate produced while rendering a 2D tensor cell.
 * @returns True when the committed selection entries contain the coordinate key for that tensor; otherwise false, including when the tensor has no selected cells.
 * @noThrows Missing tensor entries are treated as empty selection sets, and the lookup only serializes the provided coordinate for an in-memory map query.
 * @example
 * // If the committed selection for 'weights' contains the key for [1, 2]:
 * this.isSelectedCell('weights', [1, 2]); // true
 * this.isSelectedCell('weights', [0, 0]); // false
 */
private isSelectedCell(tensorId: string, tensorCoord: number[]): boolean {
        return this.selectionEntries().get(tensorId)?.has(coordKey(tensorCoord)) ?? false;
    }

        /**
 * Determines whether a rendered cell should use selection highlighting from committed selection state or the active drag preview.
 *
 * @param tensorId - Id of the tensor whose committed and preview selection sets should be queried.
 * @param tensorCoord - Canonical tensor coordinate for the rendered cell, converted to the same key format used by selection storage.
 * @returns True when the cell is in the active drag-preview set during a drag, or when it is in either the committed selection or preview set outside the drag-only branch.
 * @noThrows Missing committed or preview maps are treated as empty sets, so lookup misses return false instead of throwing.
 * @example
 * // With no active drag, either committed selection or preview selection highlights the cell.
 * this.isHighlightedCell('weights', [1, 2]); // true when [1, 2] is selected or previewed
 *
 * // During a selection drag, only the current preview set controls highlighting.
 * this.isHighlightedCell('weights', [0, 0]); // false when [0, 0] is not in the drag preview
 */
private isHighlightedCell(tensorId: string, tensorCoord: number[]): boolean {
        const key = coordKey(tensorCoord);
        if (this.selectionDrag) return this.previewSelectedCells.get(tensorId)?.has(key) ?? false;
        return (this.selectionEntries().get(tensorId)?.has(key) ?? false)
            || (this.previewSelectedCells.get(tensorId)?.has(key) ?? false);
    }

        /**
 * Return the display color for a selected cell by tinting its base color toward the viewer's active-selection color.
 *
 * @param baseColor - Unselected cell color produced by the heatmap or custom-color pipeline; the color is cloned before tinting.
 * @returns A new `Color` instance with `SELECTION_TINT_ALPHA` of `ACTIVE_COLOR` blended in, suitable for writing to mesh or canvas color buffers.
 * @noThrows Cloning and linear interpolation are deterministic `Color` operations for a valid `Color` instance and this method does not read viewer state.
 * @example
 * const baseColor = new Color(0.2, 0.4, 0.6);
 * const tinted = viewer.selectedColor(baseColor);
 *
 * expect(tinted).not.toBe(baseColor);
 * expect(tinted).toEqual(baseColor.clone().lerp(ACTIVE_COLOR, SELECTION_TINT_ALPHA));
 * expect(baseColor).toEqual(new Color(0.2, 0.4, 0.6));
 */
private selectedColor(baseColor: Color): Color {
        return baseColor.clone().lerp(ACTIVE_COLOR, SELECTION_TINT_ALPHA);
    }

    /**
 * Resolve the unselected color for one tensor cell before committed-selection or drag-preview tinting is applied.
 *
 * @param tensor - Tensor record whose `customColors` map may contain an `rgb` or hue/saturation override for `tensorCoord`.
 * @param tensorCoord - Tensor-space coordinate for the cell, encoded with `coordKey` when looking up custom colors.
 * @param value - Numeric cell value used to compute grayscale heatmap intensity or hue/saturation brightness.
 * @param heatmapRange - Inclusive numeric range for normalizing `value`; `null` disables heatmap shading unless a hue/saturation override needs brightness.
 * @returns The color that renderers should use as the cell's base color: custom RGB, custom hue/saturation, heatmap grayscale, or the default base color.
 * @noThrows The method only performs map lookup, color construction, and numeric normalization for the supplied tensor cell inputs.
 * @example
 * const tensor = { customColors: new Map([[coordKey([1, 2]), { kind: 'rgb', value: [255, 0, 0] }]]) } as TensorRecord;
 * const color = viewer.baseCellColor(tensor, [1, 2], 0.5, { min: 0, max: 1 });
 *
 * expect(color).toEqual(new Color(1, 0, 0));
 *
 * const heatmapped = viewer.baseCellColor({ customColors: new Map() } as TensorRecord, [0, 0], 0.25, { min: 0, max: 1 });
 * expect(heatmapped.r).toBeCloseTo(0.25);
 * expect(heatmapped.g).toBeCloseTo(0.25);
 * expect(heatmapped.b).toBeCloseTo(0.25);
 */
    private baseCellColor(
        tensor: TensorRecord,
        tensorCoord: number[],
        value: number,
        heatmapRange: { min: number; max: number } | null,
    ): Color {
        const customColor = tensor.customColors.get(coordKey(tensorCoord));
        if (customColor?.kind === 'rgb') return colorFromRgb(customColor.value);
        if (customColor?.kind === 'hs') {
            return colorFromHueSaturation(
                customColor.value,
                heatmapRange ? this.heatmapNormalizedValue(value, heatmapRange.min, heatmapRange.max) : 1,
            );
        }
        if (!heatmapRange) return BASE_COLOR.clone();
        const gray = this.heatmapNormalizedValue(value, heatmapRange.min, heatmapRange.max);
        return new Color().setRGB(gray, gray, gray);
    }

        /**
 * Copy a selection map so drag previews can modify tensor coordinate sets without mutating the committed selection snapshot.
 *
 * @param entries - Map from tensor id to the set of selected coordinate keys, such as values produced by `coordKey`.
 * @returns A new map containing new `Set` instances for each tensor id while preserving the same coordinate-key strings.
 * @noThrows Constructing `Map` and `Set` copies from the provided iterable selection collections has no viewer-state dependency or validation branch.
 * @example
 * const entries = new Map([['tensor-a', new Set(['0,0', '0,1'])]]);
 * const clone = viewer.cloneSelectionEntries(entries);
 * clone.get('tensor-a')!.add('1,0');
 *
 * expect(clone).not.toBe(entries);
 * expect(clone.get('tensor-a')).not.toBe(entries.get('tensor-a'));
 * expect(entries.get('tensor-a')!.has('1,0')).toBe(false);
 */
private cloneSelectionEntries(entries: Map<string, Set<string>>): Map<string, Set<string>> {
        return new Map(Array.from(entries.entries(), ([tensorId, coords]) => [tensorId, new Set(coords)]));
    }

        /**
 * Convert the active selection drag into normalized overlay-space rectangle bounds for hit testing and preview rendering.
 *
 * @param drag - Selection drag state with the original pointer position, current client position, and optional 2D world start point.
 * @returns Overlay-space `left`, `right`, `top`, and `bottom` edges ordered so callers can compare cell positions without checking drag direction.
 * @noThrows For a populated drag state, the method only projects the start point, converts the current pointer to overlay coordinates, and orders the two points.
 * @example
 * vi.spyOn(viewer, 'overlayPoint')
 *   .mockReturnValueOnce({ x: 120, y: 80 })
 *   .mockReturnValueOnce({ x: 40, y: 150 });
 *
 * const bounds = viewer.selectionBoxBounds({
 *   source: '3d',
 *   startClient: { x: 120, y: 80 },
 *   currentClient: { x: 40, y: 150 },
 *   baseSelections: new Map(),
 *   previewSelections: new Map(),
 * } as SelectionDragState);
 *
 * expect(bounds).toEqual({ left: 40, right: 120, top: 80, bottom: 150 });
 */
private selectionBoxBounds(drag: SelectionDragState): { left: number; right: number; top: number; bottom: number } {
        const start = drag.source === '2d' && drag.startWorld
            ? this.projectCanvasPoint(drag.startWorld.x, drag.startWorld.y)
            : this.overlayPoint(drag.startClient.x, drag.startClient.y);
        const current = this.overlayPoint(drag.currentClient.x, drag.currentClient.y);
        return {
            left: Math.min(start.x, current.x),
            right: Math.max(start.x, current.x),
            top: Math.min(start.y, current.y),
            bottom: Math.max(start.y, current.y),
        };
    }

        /**
 * Converts a pointer position in canvas pixels into the viewer's world coordinate system.
 *
 * The conversion recenters the canvas origin, subtracts the current pan offset, applies the
 * inverse canvas scale/zoom, and flips the y axis so selection and shader code can compare
 * pointer bounds against tensor layout positions.
 *
 * @param x - Horizontal pixel coordinate within `flatCanvas`, such as a mouse or drag-box edge.
 * @param y - Vertical pixel coordinate within `flatCanvas`, such as a mouse or drag-box edge.
 * @returns World-space `{ x, y }` point corresponding to the supplied canvas pixel.
 * @noThrows Performs only numeric arithmetic against existing canvas, pan, and zoom fields; it does not allocate resources or validate external data.
 * @example
 * ```ts
 * viewer.flatCanvas.width = 800;
 * viewer.flatCanvas.height = 600;
 * viewer.canvasPan = { x: 0, y: 0 };
 * viewer.canvasZoom = 1;
 *
 * // The canvas center maps to the world origin.
 * expect(viewer.canvasPixelToWorld(400, 300)).toEqual({ x: 0, y: 0 });
 * ```
 */
private canvasPixelToWorld(x: number, y: number): { x: number; y: number } {
        return {
            x: (x - this.flatCanvas.width / 2 - this.canvasPan.x) / (CANVAS_WORLD_SCALE * this.canvasZoom),
            y: -((y - this.flatCanvas.height / 2 - this.canvasPan.y) / (CANVAS_WORLD_SCALE * this.canvasZoom)),
        };
    }

        /**
 * Decodes a flat selection index into coordinates for the visible axes of a tensor view.
 *
 * Scalar or fully collapsed selections have no visible axes, so their only coordinate is the
 * empty coordinate array.
 *
 * @param index - Zero-based flat index within the row or column selection space being expanded.
 * @param shape - Sizes of the visible axes that define the multidimensional selection space.
 * @returns Coordinate components for `index` in `shape`, or `[]` when `shape` has no axes.
 * @noThrows The empty-shape case is handled directly, and normal calls delegate to index-unraveling for the provided numeric shape without mutating viewer state.
 * @example
 * ```ts
 * expect(viewer.decodeSelectionIndex(5, [2, 3])).toEqual([1, 2]);
 * expect(viewer.decodeSelectionIndex(0, [])).toEqual([]);
 * ```
 */
private decodeSelectionIndex(index: number, shape: number[]): number[] {
        return shape.length === 0 ? [] : unravelIndex(index, shape);
    }

        /**
 * Finds the inclusive index span whose monotonic values intersect a numeric interval.
 *
 * This is used by the 2D selection fast path to avoid scanning every rendered cell when row
 * or column display positions increase monotonically.
 *
 * @param length - Number of indices in the monotonic sequence to search.
 * @param valueAt - Accessor that returns the sorted numeric value for a zero-based index.
 * @param lower - Inclusive lower bound of the value interval to match.
 * @param upper - Inclusive upper bound of the value interval to match.
 * @returns Inclusive `{ start, end }` indices whose values are within `[lower, upper]`, or `null` when the sequence is empty, the bounds are reversed, or no value overlaps the interval.
 * @noThrows The method only performs bounded binary searches; callers are expected to provide a non-throwing monotonic `valueAt` accessor.
 * @example
 * ```ts
 * const values = [0, 4, 8, 12, 16];
 * expect(viewer.monotonicIndexRange(values.length, (i) => values[i], 5, 12)).toEqual({ start: 2, end: 3 });
 * expect(viewer.monotonicIndexRange(values.length, (i) => values[i], 17, 20)).toBeNull();
 * expect(viewer.monotonicIndexRange(values.length, (i) => values[i], 10, 5)).toBeNull();
 * ```
 */
private monotonicIndexRange(
        length: number,
        valueAt: (index: number) => number,
        lower: number,
        upper: number,
    ): { start: number; end: number } | null {
        if (length <= 0 || lower > upper) return null;
        let start = 0;
        let end = length;
        while (start < end) {
            const middle = Math.floor((start + end) / 2);
            if (valueAt(middle) < lower) start = middle + 1;
            else end = middle;
        }
        const first = start;
        start = 0;
        end = length;
        while (start < end) {
            const middle = Math.floor((start + end) / 2);
            if (valueAt(middle) <= upper) start = middle + 1;
            else end = middle;
        }
        const last = start - 1;
        return first <= last ? { start: first, end: last } : null;
    }

        /**
 * Computes selected tensor coordinate keys for a 2D drag box when the tensor view supports the contiguous-layout fast path.
 *
 * Hidden visible-coordinate masks or non-contiguous layouts fall back to the slower per-cell
 * selection path by returning `null`. Supported layouts return a set, which may be empty when
 * the drag box misses all displayed cells.
 *
 * @param tensor - Rendered tensor record whose view, offset, and optional visible-coordinate mask determine whether fast selection is valid.
 * @param box - Selection rectangle in canvas pixels with `left`, `right`, `top`, and `bottom` edges from the current drag.
 * @returns Set of selected tensor coordinate keys for supported contiguous 2D views, an empty set for a supported view with no overlap, or `null` when the fast path is not applicable.
 * @noThrows The method derives ranges from existing viewer layout state and returns fallback values for unsupported fast-path cases instead of throwing.
 * @example
 * ```ts
 * const unsupportedTensor = {
 *   id: 'weights',
 *   visibleCoords: new Set(['0,0']),
 *   view,
 *   offset: [0, 0],
 * } as TensorRecord;
 *
 * const result = viewer.fastBoxSelectionEntries2D(unsupportedTensor, {
 *   left: 10,
 *   right: 80,
 *   top: 10,
 *   bottom: 80,
 * });
 *
 * // A visible-coordinate mask breaks the contiguous fast path, so callers should use the
 * // general selection scan instead.
 * expect(result).toBeNull();
 * ```
 */
private fastBoxSelectionEntries2D(
        tensor: TensorRecord,
        box: { left: number; right: number; top: number; bottom: number },
    ): Set<string> | null {
        // contiguous 2d views form monotonic rows/columns, so large selections can
        // avoid scanning every cell; hidden masks and z-order layouts break that.
        if (tensor.visibleCoords) return null;
        if (!supportsContiguousSelectionFastPath2D(tensor.view, this.state.collapseHiddenAxes)) return null;
        const split = Math.floor(tensor.view.tokens.length / 2);
        const ySizes = tensor.view.tokens.slice(0, split).filter((token) => token.visible).map((token) => token.size);
        const xSizes = tensor.view.tokens.slice(split).filter((token) => token.visible).map((token) => token.size);
        const yCount = ySizes.length === 0 ? 1 : product(ySizes);
        const xCount = xSizes.length === 0 ? 1 : product(xSizes);
        const yZero = new Array(ySizes.length).fill(0);
        const xZero = new Array(xSizes.length).fill(0);
        const shape = this.layoutShape(tensor.view);
        const worldLeft = this.canvasPixelToWorld(box.left, 0).x - tensor.offset[0] - 0.5;
        const worldRight = this.canvasPixelToWorld(box.right, 0).x - tensor.offset[0] + 0.5;
        const worldTop = this.canvasPixelToWorld(0, box.top).y - tensor.offset[1] + 0.5;
        const worldBottom = this.canvasPixelToWorld(0, box.bottom).y - tensor.offset[1] - 0.5;
        const xRange = this.monotonicIndexRange(
            xCount,
            (xIndex) => displayPositionForCoord2D(
                this.mapViewCoordToLayoutCoord([...yZero, ...this.decodeSelectionIndex(xIndex, xSizes)], tensor.view),
                shape,
                this.layoutGapMultiple(),
                this.state.dimensionMappingScheme,
            ).x,
            worldLeft,
            worldRight,
        );
        const yRange = this.monotonicIndexRange(
            yCount,
            (yIndex) => -displayPositionForCoord2D(
                this.mapViewCoordToLayoutCoord([...this.decodeSelectionIndex(yIndex, ySizes), ...xZero], tensor.view),
                shape,
                this.layoutGapMultiple(),
                this.state.dimensionMappingScheme,
            ).y,
            -worldTop,
            -worldBottom,
        );
        if (!xRange || !yRange) return new Set<string>();
        const selected = new Set<string>();
        for (let yIndex = yRange.start; yIndex <= yRange.end; yIndex += 1) {
            const yCoord = this.decodeSelectionIndex(yIndex, ySizes);
            for (let xIndex = xRange.start; xIndex <= xRange.end; xIndex += 1) {
                selected.add(coordKey(mapViewCoordToTensorCoord([...yCoord, ...this.decodeSelectionIndex(xIndex, xSizes)], tensor.view)));
            }
        }
        return selected;
    }

        /**
 * Projects a Three.js scene-space point through the viewer camera into flat-canvas pixel coordinates.
 *
 * @param point - Scene-space `Vector3` for a tensor cell corner or center; the vector is cloned before projection.
 * @returns Canvas pixel coordinates measured from the flat canvas top-left corner, or `null` when camera projection produces a non-finite x or y value.
 * @noThrows Uses `Vector3.clone().project()` and finite-number checks only; invalid projection results are reported as `null` instead of being thrown.
 * @example
 * ```ts
 * const viewer = makeViewerWithCanvas({ width: 200, height: 100 });
 * viewer.camera = makeCameraLookingAtOrigin();
 *
 * const screen = (viewer as any).projectScenePoint(new Vector3(0, 0, 0));
 *
 * expect(screen).toEqual({ x: 100, y: 50 });
 * ```
 * @example
 * ```ts
 * const viewer = makeViewerWithCanvas({ width: 200, height: 100 });
 * viewer.camera = makeCameraThatProjectsToInfinity();
 *
 * expect((viewer as any).projectScenePoint(new Vector3(0, 0, 0))).toBeNull();
 * ```
 */
private projectScenePoint(point: Vector3): { x: number; y: number } | null {
        const projected = point.clone().project(this.camera);
        if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y)) return null;
        return {
            x: ((projected.x + 1) * this.flatCanvas.width) / 2,
            y: ((1 - projected.y) * this.flatCanvas.height) / 2,
        };
    }

        /**
 * Computes the 2D canvas rectangle occupied by one visible tensor cell after layout spacing, tensor offset, and optional overlay bias are applied.
 *
 * @param tensor - Tensor record whose `view` determines the displayed layout shape and whose `offset` places the tensor on the canvas.
 * @param layoutCoord - Coordinate of the cell in layout space, typically produced from a visible view coordinate.
 * @param bias - Optional fractional `[x, y]` layout-cell offset used to draw shifted ghost layers or overlays around the same cell.
 * @returns Left, right, top, and bottom canvas pixel bounds used by 2D rendering, SVG export, and rectangle hit testing.
 * @noThrows Performs deterministic coordinate math against existing viewer state and does not allocate or validate external resources.
 * @example
 * ```ts
 * const tensor = makeTensorRecord({ id: 'weights', shape: [2, 2], offset: [0, 0, 0] });
 * const viewer = makeViewerWithIdentityCanvasProjection();
 *
 * const bounds = (viewer as any).canvasCellBounds(tensor, [1, 0]);
 *
 * expect(bounds).toEqual({ left: 50, right: 150, top: 50, bottom: 150 });
 * ```
 * @example
 * ```ts
 * const tensor = makeTensorRecord({ id: 'ghosted', shape: [1, 1], offset: [0, 0, 0] });
 * const viewer = makeViewerWithIdentityCanvasProjection();
 *
 * const shifted = (viewer as any).canvasCellBounds(tensor, [0, 0], [0.25, -0.25]);
 *
 * expect(shifted.left).toBeGreaterThan((viewer as any).canvasCellBounds(tensor, [0, 0]).left);
 * ```
 */
private canvasCellBounds(
        tensor: TensorRecord,
        layoutCoord: number[],
        bias: readonly [number, number] = [0, 0],
    ): { left: number; right: number; top: number; bottom: number } {
        const shape = this.layoutShape(tensor.view);
        const position = displayPositionForCoord2D(layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        const topLeft = this.projectCanvasPoint(
            tensor.offset[0] + position.x - 0.5 + (bias[0] ?? 0),
            tensor.offset[1] + position.y + 0.5 + (bias[1] ?? 0),
        );
        const bottomRight = this.projectCanvasPoint(
            tensor.offset[0] + position.x + 0.5 + (bias[0] ?? 0),
            tensor.offset[1] + position.y - 0.5 + (bias[1] ?? 0),
        );
        return {
            left: Math.min(topLeft.x, bottomRight.x),
            right: Math.max(topLeft.x, bottomRight.x),
            top: Math.min(topLeft.y, bottomRight.y),
            bottom: Math.max(topLeft.y, bottomRight.y),
        };
    }

        /**
 * Projects the eight corners of a 3D tensor cell and returns the screen-space rectangle that encloses the visible cube.
 *
 * @param tensor - Tensor record whose `view` determines the displayed 3D layout shape and whose `offset` positions the cube in scene space.
 * @param layoutCoord - Coordinate of the cell in layout space; the method projects the unit cube centered on this coordinate.
 * @returns Canvas pixel bounds enclosing all projected cube corners, or `null` when any corner cannot be projected to finite screen coordinates.
 * @noThrows Projection failures from the camera are converted to `null` by `projectScenePoint`, so callers can skip unprojectable cells without handling exceptions.
 * @example
 * ```ts
 * const tensor = makeTensorRecord({ id: 'activation', shape: [1, 1, 1], offset: [0, 0, 0] });
 * const viewer = makeViewerWithSceneProjection();
 *
 * const bounds = (viewer as any).sceneCellBounds(tensor, [0, 0, 0]);
 *
 * expect(bounds).toMatchObject({ left: expect.any(Number), right: expect.any(Number), top: expect.any(Number), bottom: expect.any(Number) });
 * expect(bounds!.left).toBeLessThan(bounds!.right);
 * expect(bounds!.top).toBeLessThan(bounds!.bottom);
 * ```
 * @example
 * ```ts
 * const tensor = makeTensorRecord({ id: 'activation', shape: [1, 1, 1], offset: [0, 0, 0] });
 * const viewer = makeViewerWithUnprojectableCamera();
 *
 * expect((viewer as any).sceneCellBounds(tensor, [0, 0, 0])).toBeNull();
 * ```
 */
private sceneCellBounds(
        tensor: TensorRecord,
        layoutCoord: number[],
    ): { left: number; right: number; top: number; bottom: number } | null {
        const shape = this.layoutShape(tensor.view);
        const center = displayPositionForCoord(layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme)
            .add(vectorFromTuple(tensor.offset));
        let left = Number.POSITIVE_INFINITY;
        let right = Number.NEGATIVE_INFINITY;
        let top = Number.POSITIVE_INFINITY;
        let bottom = Number.NEGATIVE_INFINITY;
        for (const x of [-0.5, 0.5]) {
            for (const y of [-0.5, 0.5]) {
                for (const z of [-0.5, 0.5]) {
                    const point = this.projectScenePoint(center.clone().add(new Vector3(x, y, z)));
                    if (!point) return null;
                    left = Math.min(left, point.x);
                    right = Math.max(right, point.x);
                    top = Math.min(top, point.y);
                    bottom = Math.max(bottom, point.y);
                }
            }
        }
        return { left, right, top, bottom };
    }

        /**
 * Finds the visible tensor cells whose 2D canvas bounds or projected 3D scene bounds intersect the active drag-selection rectangle.
 *
 * @param drag - Current selection drag state from pointer handling, including the start and current pointer positions used to build the selection box.
 * @returns Map from tensor id to a set of serialized tensor-coordinate keys for intersecting visible cells; returns an empty map when selection is disabled or no cells intersect.
 * @noThrows Disabled selection returns an empty map, and unprojectable 3D cells are skipped through `null` bounds instead of raising an exception.
 * @example
 * ```ts
 * const viewer = makeViewerWithTensorGrid({ tensorId: 'weights', shape: [2, 2] });
 * const drag = makeSelectionDrag({ start: { x: 0, y: 0 }, current: { x: 120, y: 120 } });
 *
 * const selected = (viewer as any).boxSelectionEntries(drag);
 *
 * expect(selected.get('weights')).toEqual(new Set(['0,0', '0,1']));
 * ```
 * @example
 * ```ts
 * const viewer = makeViewerWithTensorGrid({ tensorId: 'weights', shape: [2, 2], selectionEnabled: false });
 * const drag = makeSelectionDrag({ start: { x: 0, y: 0 }, current: { x: 120, y: 120 } });
 *
 * expect((viewer as any).boxSelectionEntries(drag)).toEqual(new Map());
 * ```
 */
private boxSelectionEntries(drag: SelectionDragState): Map<string, Set<string>> {
        if (!this.selectionEnabled()) return new Map();
        const entries = new Map<string, Set<string>>();
        const box = this.selectionBoxBounds(drag);
        this.tensors.forEach((tensor) => {
            const fastSelected = this.state.displayMode === '2d' ? this.fastBoxSelectionEntries2D(tensor, box) : null;
            if (fastSelected) {
                if (fastSelected.size !== 0) entries.set(tensor.id, fastSelected);
                return;
            }
            const instanceShape = this.instanceShape(tensor.view);
            const count = product(instanceShape);
            for (let index = 0; index < count; index += 1) {
                const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
                const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
                if (!this.tensorCoordVisible(tensor, tensorCoord)) continue;
                const layoutCoord = this.mapViewCoordToLayoutCoord(viewCoord, tensor.view);
                const cellBounds = this.state.displayMode === '2d'
                    ? this.canvasCellBounds(tensor, layoutCoord)
                    : this.sceneCellBounds(tensor, layoutCoord);
                if (!cellBounds || !boxesIntersect(box, cellBounds)) continue;
                const selected = entries.get(tensor.id) ?? new Set<string>();
                selected.add(coordKey(tensorCoord));
                entries.set(tensor.id, selected);
            }
        });
        return entries;
    }

        /**
 * Recomputes the cells covered by an active box-selection drag and stores the live preview map used by rendering and selection events.
 *
 * In replace mode the preview becomes exactly the cells inside the drag rectangle. In add and subtract modes the rectangle is merged with or removed from the drag's committed base selections. When the rectangle hits a tensor, the drag and viewer active tensor are moved to that tensor so the shader preview follows the source mesh.
 *
 * @param drag - Mutable selection drag state for the current pointer gesture, including the drag mode, source, base selections, current pointer location, and optional source tensor id.
 * @returns Nothing. The method updates `drag.previewSelections`, may update `drag.tensorId`, and may update `state.activeTensorId`.
 * @noThrows Selection preview updates are expected to be non-throwing because this method only derives maps from the current drag, clones existing selection sets, and assigns viewer state; it does not validate external input or allocate GPU resources.
 * @example
 * const drag = {
 *   mode: 'add',
 *   tensorId: 'weights',
 *   baseSelections: new Map([['weights', new Set(['0,0'])]]),
 *   previewSelections: new Map(),
 * } as SelectionDragState;
 * vi.spyOn(viewer as any, 'boxSelectionEntries').mockReturnValue(new Map([['weights', new Set(['0,1'])]]));
 *
 * (viewer as any).updateSelectionPreview(drag);
 *
 * expect([...drag.previewSelections.get('weights')!].sort()).toEqual(['0,0', '0,1']);
 */
private updateSelectionPreview(drag: SelectionDragState): void {
        const selected = this.boxSelectionEntries(drag);
        const sourceTensorId = this.selectionSourceTensorId(drag, selected);
        if (sourceTensorId) {
            drag.tensorId = sourceTensorId;
            this.state.activeTensorId = sourceTensorId;
        }
        if (drag.mode === 'replace') {
            drag.previewSelections = selected;
            return;
        }
        const nextPreviewSelections = this.cloneSelectionEntries(drag.baseSelections);
        selected.forEach((coords, tensorId) => {
            const preview = nextPreviewSelections.get(tensorId) ?? new Set<string>();
            coords.forEach((coord) => {
                if (drag.mode === 'add') preview.add(coord);
                else preview.delete(coord);
            });
            if (preview.size === 0) nextPreviewSelections.delete(tensorId);
            else nextPreviewSelections.set(tensorId, preview);
        });
        drag.previewSelections = nextPreviewSelections;
    }

        /**
 * Chooses which tensor should own a drag selection when the drag rectangle intersects zero, one, or several tensor meshes.
 *
 * Empty selections keep the existing drag tensor. A single non-empty tensor becomes the source. Multi-tensor 2D drags use the drag start point and cached mesh rectangles to pick the nearest tensor, falling back to the existing drag tensor and then the first non-empty tensor.
 *
 * @param drag - Current selection drag state, including the previous `tensorId`, drag `source`, and optional 2D `startWorld` point used for nearest-mesh tie breaking.
 * @param selected - Map from tensor id to the coordinate keys currently covered by the drag rectangle; empty coordinate sets are ignored.
 * @returns The tensor id that should receive preview focus, or `null` when there is no selected tensor and the drag did not already have a tensor id.
 * @noThrows Source selection is expected to be non-throwing because it only inspects in-memory maps, optional drag fields, and cached mesh bounds, with fallbacks for missing rectangles.
 * @example
 * const drag = { tensorId: 'bias', source: '2d' } as SelectionDragState;
 * const selected = new Map<string, Set<string>>([
 *   ['weights', new Set(['0,0', '0,1'])],
 *   ['bias', new Set()],
 * ]);
 *
 * const tensorId = (viewer as any).selectionSourceTensorId(drag, selected);
 *
 * expect(tensorId).toBe('weights');
 */
private selectionSourceTensorId(
        drag: SelectionDragState,
        selected: Map<string, Set<string>>,
    ): string | null {
        const tensorIds = Array.from(selected.entries())
            .filter(([_tensorId, coords]) => coords.size !== 0)
            .map(([tensorId]) => tensorId);
        if (tensorIds.length === 0) return drag.tensorId;
        if (tensorIds.length === 1) return tensorIds[0]!;
        if (drag.source !== '2d' || !drag.startWorld) return drag.tensorId ?? tensorIds[0]!;
        let bestTensorId: string | null = null;
        let bestDistance = Number.POSITIVE_INFINITY;
        tensorIds.forEach((tensorId) => {
            const rect = this.pickMeshes.find((entry) => entry.tensorId === tensorId)?.rect2D;
            if (!rect) return;
            const dx = drag.startWorld!.x < rect.minX
                ? rect.minX - drag.startWorld!.x
                : drag.startWorld!.x > rect.maxX
                    ? drag.startWorld!.x - rect.maxX
                    : 0;
            const dy = drag.startWorld!.y < rect.minY
                ? rect.minY - drag.startWorld!.y
                : drag.startWorld!.y > rect.maxY
                    ? drag.startWorld!.y - rect.maxY
                    : 0;
            const distance = (dx * dx) + (dy * dy);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestTensorId = tensorId;
            }
        });
        return bestTensorId ?? drag.tensorId ?? tensorIds[0]!;
    }

    /**
 * Returns the per-instance `selectionState` geometry attribute that stores committed selection flags for a tensor-cell mesh.
 *
 * @param mesh - Instanced tensor mesh whose geometry may include the `selectionState` attribute populated by the 2D mesh builder.
 * @returns The `InstancedBufferAttribute` containing one selection flag per rendered instance, or `null` when the geometry has no such attribute or the attribute is not instanced.
 * @noThrows Attribute lookup is expected to be non-throwing because the method only reads `mesh.geometry.getAttribute('selectionState')` and checks its runtime type.
 * @example
 * const attribute = new InstancedBufferAttribute(new Float32Array([1, 0]), 1);
 * mesh.geometry.setAttribute('selectionState', attribute);
 *
 * expect((viewer as any).selectionStateAttribute(mesh)).toBe(attribute);
 *
 * mesh.geometry.deleteAttribute('selectionState');
 * expect((viewer as any).selectionStateAttribute(mesh)).toBeNull();
 */
    private selectionStateAttribute(mesh: InstancedMesh): InstancedBufferAttribute | null {
        const attribute = mesh.geometry.getAttribute('selectionState');
        return attribute instanceof InstancedBufferAttribute ? attribute : null;
    }

    /**
 * Returns the shader uniform handles used to draw the live box-selection preview for a patched tensor mesh material.
 *
 * @param mesh - Instanced tensor mesh whose first material may have `userData.selectionPreviewUniforms` installed by `installSelectionPreviewShader`.
 * @returns The preview uniform object that callers mutate before rendering, or `null` when the mesh material has not been patched for selection preview.
 * @noThrows Uniform lookup is expected to be non-throwing because the method only reads the mesh material, supports material arrays by checking the first material, and treats missing `userData` metadata as no preview support.
 * @example
 * const uniforms = {
 *   selectionPreviewActive: { value: 0 },
 *   selectionPreviewMin: { value: new Vector2(0, 0) },
 *   selectionPreviewMax: { value: new Vector2(1, 1) },
 *   selectionColor: { value: ACTIVE_COLOR.clone() },
 * } satisfies SelectionPreviewUniforms;
 * (mesh.material as Material).userData.selectionPreviewUniforms = uniforms;
 *
 * expect((viewer as any).selectionPreviewUniforms(mesh)).toBe(uniforms);
 *
 * delete (mesh.material as Material).userData.selectionPreviewUniforms;
 * expect((viewer as any).selectionPreviewUniforms(mesh)).toBeNull();
 */
    private selectionPreviewUniforms(mesh: InstancedMesh): SelectionPreviewUniforms | null {
        const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
        const uniforms = material?.userData.selectionPreviewUniforms as SelectionPreviewUniforms | undefined;
        return uniforms ?? null;
    }

    /**
 * Adds the selection-preview shader hook used by 2D instanced tensor cells.
 *
 * @param mesh - Instanced tensor-cell mesh whose material is expected to be a `MeshBasicMaterial`; meshes with other material types are left unchanged.
 * @returns No value. For supported materials, stores the preview uniforms in `material.userData.selectionPreviewUniforms`, injects selection tint logic into `onBeforeCompile`, and marks the material for recompilation.
 * @noThrows Unsupported material types are handled as a no-op, and the method only assigns Three.js material callbacks and uniform objects.
 * @example
 * const mesh = new InstancedMesh(geometry, new MeshBasicMaterial({ color: 0xffffff }), 4);
 * viewer.installSelectionPreviewShader(mesh);
 *
 * const uniforms = mesh.material.userData.selectionPreviewUniforms;
 * expect(uniforms.selectionPreviewActive.value).toBe(0);
 * expect(uniforms.selectionPreviewBounds.value.toArray()).toEqual([0, 0, 0, 0]);
 * expect(mesh.material.needsUpdate).toBe(true);
 * expect(typeof mesh.material.onBeforeCompile).toBe('function');
 */
    private installSelectionPreviewShader(mesh: InstancedMesh): void {
        const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
        if (!(material instanceof MeshBasicMaterial)) return;
        const uniforms: SelectionPreviewUniforms = {
            selectionPreviewActive: { value: 0 },
            selectionPreviewBounds: { value: new Vector4(0, 0, 0, 0) },
            selectionPreviewMode: { value: 0 },
            selectionColor: { value: ACTIVE_COLOR.clone() },
        };
        material.userData.selectionPreviewUniforms = uniforms;
        material.onBeforeCompile = (shader): void => {
            shader.uniforms.selectionPreviewActive = uniforms.selectionPreviewActive;
            shader.uniforms.selectionPreviewBounds = uniforms.selectionPreviewBounds;
            shader.uniforms.selectionPreviewMode = uniforms.selectionPreviewMode;
            shader.uniforms.selectionColor = uniforms.selectionColor;
            shader.vertexShader = `
attribute float selectionState;
varying float vSelectionState;
varying vec2 vSelectionCenter;
${shader.vertexShader}`.replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>
vSelectionState = selectionState;
vSelectionCenter = (modelMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xy;`,
            );
            shader.fragmentShader = `
uniform float selectionPreviewActive;
uniform vec4 selectionPreviewBounds;
uniform float selectionPreviewMode;
uniform vec3 selectionColor;
varying float vSelectionState;
varying vec2 vSelectionCenter;
${shader.fragmentShader}`.replace(
                '#include <color_fragment>',
                `#include <color_fragment>
float selected = step(0.5, vSelectionState);
if (selectionPreviewActive > 0.5) {
    bool inPreview = vSelectionCenter.x >= selectionPreviewBounds.x - 0.5
        && vSelectionCenter.x <= selectionPreviewBounds.y + 0.5
        && vSelectionCenter.y >= selectionPreviewBounds.z - 0.5
        && vSelectionCenter.y <= selectionPreviewBounds.w + 0.5;
    if (selectionPreviewMode < 0.5) selected = inPreview ? 1.0 : 0.0;
    else if (selectionPreviewMode < 1.5) selected = max(selected, inPreview ? 1.0 : 0.0);
    else if (inPreview) selected = 0.0;
}
diffuseColor.rgb = mix(diffuseColor.rgb, selectionColor, ${SELECTION_TINT_ALPHA} * selected);`,
            );
        };
        material.needsUpdate = true;
    }

    /**
 * Copies the active selection drag rectangle into each 2D tensor mesh's shader-preview uniforms.
 *
 * @returns No value. Matching meshes receive active preview bounds and mode values; meshes outside the active drag tensor are marked inactive.
 * @noThrows Missing tensor meshes, non-instanced children, disabled selection, and meshes without preview uniforms are skipped before any uniform writes occur.
 * @example
 * viewer.selectionDrag = { tensorId: 'weights', mode: 'add', start: { x: 10, y: 20 }, current: { x: 50, y: 80 } };
 * viewer.syncSelectionPreviewState();
 *
 * const uniforms = viewer.selectionPreviewUniforms(weightsMesh)!;
 * expect(uniforms.selectionPreviewActive.value).toBe(1);
 * expect(uniforms.selectionPreviewMode.value).toBe(1);
 * expect(uniforms.selectionPreviewBounds.value.x).toBeLessThanOrEqual(uniforms.selectionPreviewBounds.value.y);
 */
    private syncSelectionPreviewState(): void {
        const drag = this.selectionEnabled() ? this.selectionDrag : null;
        const box = drag ? this.selectionBoxBounds(drag) : null;
        const left = box ? this.canvasPixelToWorld(box.left, 0).x : 0;
        const right = box ? this.canvasPixelToWorld(box.right, 0).x : 0;
        const bottom = box ? this.canvasPixelToWorld(0, box.bottom).y : 0;
        const top = box ? this.canvasPixelToWorld(0, box.top).y : 0;
        const mode = !drag ? 0 : drag.mode === 'add' ? 1 : drag.mode === 'remove' ? 2 : 0;
        // shader previews keep drag feedback cheap for large tensors; the committed
        // selection attribute remains the source of truth after pointerup.
        this.tensorMeshes.forEach((group, tensorId) => {
            const mesh = group.children[0];
            if (!(mesh instanceof InstancedMesh)) return;
            const uniforms = this.selectionPreviewUniforms(mesh);
            if (!uniforms) return;
            uniforms.selectionPreviewActive.value = drag && drag.tensorId === tensorId ? 1 : 0;
            uniforms.selectionPreviewBounds.value.set(left, right, bottom, top);
            uniforms.selectionPreviewMode.value = mode;
        });
    }

        /**
 * Recomputes the per-instance color data for one tensor mesh from tensor values, heatmap settings, and selected-cell state.
 *
 * @param tensorId - Id of a loaded tensor whose first mesh child contains `instanceColor`; unknown ids or tensors without an instanced color buffer are ignored.
 * @returns No value. Updates the mesh `instanceColor` buffer, updates the `selectionState` instanced attribute when present, and marks changed GPU attributes/materials dirty.
 * @noThrows The method returns before reading buffers when the tensor, mesh, instanced material, or color array is missing.
 * @example
 * viewer.selectedCells.set('weights', new Set(['0,1']));
 * viewer.refreshTensorColors('weights');
 *
 * expect(weightsMesh.instanceColor!.needsUpdate).toBe(true);
 * expect(viewer.selectionStateAttribute(weightsMesh)!.needsUpdate).toBe(true);
 * expect(weightsMesh.material.needsUpdate).toBe(true);
 */
private refreshTensorColors(tensorId: string): void {
        const tensor = this.tensors.get(tensorId);
        const group = this.tensorMeshes.get(tensorId);
        const mesh = group?.children[0];
        const colorArray = mesh instanceof InstancedMesh ? mesh.instanceColor?.array as Float32Array | undefined : undefined;
        const selectionAttribute = mesh instanceof InstancedMesh ? this.selectionStateAttribute(mesh) : null;
        const selectionState = selectionAttribute?.array as Float32Array | undefined;
        if (!tensor || !colorArray || !(mesh instanceof InstancedMesh)) return;
        const instanceShape = this.instanceShape(tensor.view);
        const count = product(instanceShape);
        const heatmapRange = this.state.heatmap ? tensor.valueRange : null;
        for (let index = 0; index < count; index += 1) {
            const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
            const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
            const value = numericValue(tensor.data, this.linearIndex(tensorCoord, tensor.shape));
            const baseColor = this.baseCellColor(tensor, tensorCoord, value, heatmapRange);
            const color = selectionState
                ? baseColor
                : this.isHighlightedCell(tensor.id, tensorCoord) ? this.selectedColor(baseColor) : baseColor;
            const offset = index * 3;
            colorArray[offset] = color.r;
            colorArray[offset + 1] = color.g;
            colorArray[offset + 2] = color.b;
            if (selectionState) selectionState[index] = this.isHighlightedCell(tensor.id, tensorCoord) ? 1 : 0;
        }
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        if (selectionAttribute) selectionAttribute.needsUpdate = true;
        mesh.material.needsUpdate = true;
    }

        /**
 * Refreshes selection-dependent colors for touched tensors and schedules the canvas to redraw.
 *
 * @param tensorIds - Tensor ids whose selected-cell highlights may have changed; duplicate ids are collapsed before colors are refreshed.
 * @returns No value. Calls `refreshTensorColors` once for each distinct id and then requests a render.
 * @noThrows Empty argument lists and duplicate tensor ids are valid; each id is delegated to `refreshTensorColors`, which treats missing tensors as a no-op.
 * @example
 * viewer.refreshSelectionVisuals('weights', 'weights', 'bias');
 *
 * expect(refreshTensorColorsSpy).toHaveBeenCalledTimes(2);
 * expect(refreshTensorColorsSpy).toHaveBeenCalledWith('weights');
 * expect(refreshTensorColorsSpy).toHaveBeenCalledWith('bias');
 * expect(requestRenderSpy).toHaveBeenCalledTimes(1);
 */
private refreshSelectionVisuals(...tensorIds: string[]): void {
        const ids = new Set(tensorIds);
        ids.forEach((tensorId) => this.refreshTensorColors(tensorId));
        this.requestRender();
    }

        /**
 * Removes any active cell selection and drag preview, then refreshes the affected tensor selection overlays and selection notifications.
 *
 * @param emit - Whether to emit the general viewer change event after the selection-specific events; pass `false` when a larger state transition will emit once after clearing.
 * @returns Nothing. The viewer's `selectedCells`, `selectionDrag`, preview overlay, selection box, and selection listeners are updated in place.
 * @noThrows Selection clearing only mutates viewer-owned maps and returns early when no selection or drag exists; it does not validate caller data or allocate from external sources.
 * @example
 * ```ts
 * const viewer = makeViewerWithSelectedCells([['weights', '0,1']]);
 * const emitSpy = vi.spyOn(viewer as any, 'emit');
 *
 * (viewer as any).clearSelection(false);
 *
 * expect((viewer as any).selectedCells.size).toBe(0);
 * expect((viewer as any).selectionDrag).toBeNull();
 * expect(emitSpy).not.toHaveBeenCalled();
 * ```
 */
private clearSelection(emit = true): void {
        if (!this.selectionDrag && this.selectedCells.size === 0) return;
        const tensorIds = new Set(this.selectedCells.keys());
        this.selectionDrag?.baseSelections.forEach((_coords, tensorId) => tensorIds.add(tensorId));
        this.selectionDrag?.previewSelections.forEach((_coords, tensorId) => tensorIds.add(tensorId));
        this.selectedCells.clear();
        this.selectionDrag = null;
        this.emitSelectionPreview();
        this.syncSelectionBox();
        if (tensorIds.size !== 0) this.refreshSelectionVisuals(...tensorIds);
        this.emitSelection();
        if (emit) this.emit();
    }

        /**
 * Starts a rectangular cell-selection drag from a 2D canvas or 3D renderer pointer event, preserving the existing selection as the merge base.
 *
 * @param source - Pointer surface that owns the drag: `'2d'` for the flat canvas or `'3d'` for the Three.js renderer.
 * @param hover - Hit-test result under the pointer, or `null` to start a box selection against the current active tensor.
 * @param mode - How the drag preview should combine with the existing selection: replace it, add cells, or remove cells.
 * @param clientX - Browser `PointerEvent.clientX` coordinate where the drag begins.
 * @param clientY - Browser `PointerEvent.clientY` coordinate where the drag begins.
 * @returns Nothing. When selection is enabled, `selectionDrag` is initialized, hover state may become active, and a render/change notification is requested.
 * @noThrows The method returns before doing work when selection is disabled; otherwise it stores pointer coordinates and existing viewer state without rejecting input.
 * @example
 * ```ts
 * const hover = { tensorId: 'weights', coords: [0, 1] } as HoverInfo;
 * const viewer = makeViewerWithSelectionEnabled();
 *
 * (viewer as any).beginSelectionDrag('2d', hover, 'add', 120, 64);
 *
 * expect((viewer as any).selectionDrag).toMatchObject({
 *   source: '2d',
 *   mode: 'add',
 *   tensorId: 'weights',
 *   startClient: { x: 120, y: 64 },
 *   currentClient: { x: 120, y: 64 },
 * });
 * expect((viewer as any).state.activeTensorId).toBe('weights');
 * ```
 */
private beginSelectionDrag(
        source: '2d' | '3d',
        hover: HoverInfo | null,
        mode: 'replace' | 'add' | 'remove',
        clientX: number,
        clientY: number,
    ): void {
        if (!this.selectionEnabled()) return;
        // store both client and world coordinates because 2d panning/zooming can
        // change the projection while the drag overlay still uses screen pixels.
        const startPosition = source === '2d' ? this.canvasPointerToWorld(clientX, clientY) : null;
        this.selectionDrag = {
            source,
            mode,
            tensorId: hover?.tensorId ?? this.state.activeTensorId,
            startClient: { x: clientX, y: clientY },
            startWorld: startPosition,
            currentClient: { x: clientX, y: clientY },
            baseSelections: this.cloneSelectionEntries(this.selectedCells),
            previewSelections: new Map(),
        };
        if (hover) {
            this.state.activeTensorId = hover.tensorId;
            this.state.hover = hover;
            this.state.lastHover = hover;
        }
        this.requestRender();
        this.emit();
    }

        /**
 * Moves the active selection drag for the matching pointer surface and recomputes the live selection preview.
 *
 * @param source - Pointer surface reporting movement; it must match the active drag's `'2d'` or `'3d'` source or the update is ignored.
 * @param clientX - Latest browser `PointerEvent.clientX` coordinate for the drag pointer.
 * @param clientY - Latest browser `PointerEvent.clientY` coordinate for the drag pointer.
 * @returns Nothing. A matching active drag receives the new client position, preview selections are recalculated, and preview/render notifications are sent.
 * @noThrows The method is a no-op when no drag exists or the event comes from the other surface; the update path only mutates the current drag and preview state.
 * @example
 * ```ts
 * const viewer = makeViewerWithActiveDrag({ source: '3d', startClient: { x: 10, y: 20 } });
 * const previewSpy = vi.spyOn(viewer as any, 'emitSelectionPreview');
 *
 * (viewer as any).updateSelectionDrag('3d', 48, 96);
 *
 * expect((viewer as any).selectionDrag.currentClient).toEqual({ x: 48, y: 96 });
 * expect(previewSpy).toHaveBeenCalled();
 * ```
 */
private updateSelectionDrag(
        source: '2d' | '3d',
        clientX: number,
        clientY: number,
    ): void {
        const drag = this.selectionDrag;
        if (!drag || drag.source !== source) return;
        drag.currentClient = { x: clientX, y: clientY };
        this.updateSelectionPreview(drag);
        this.emitSelectionPreview();
        this.requestRender();
    }

        /**
 * Commits the current drag preview as the viewer's selected cells and ends the selection-drag interaction.
 *
 * @returns Nothing. Preview selections are copied into `selectedCells`, preview state is cleared, the active tensor is updated, and selection/render events are emitted.
 * @noThrows The method returns when there is no active drag; with a drag, it copies entries from viewer-owned preview maps and does not validate external input.
 * @example
 * ```ts
 * const viewer = makeViewerWithActiveDrag({
 *   tensorId: 'weights',
 *   mode: 'replace',
 *   previewSelections: new Map([['weights', new Set(['0,1', '0,2'])]]),
 * });
 *
 * (viewer as any).finalizeSelectionDrag();
 *
 * expect((viewer as any).selectionDrag).toBeNull();
 * expect((viewer as any).previewSelectedCells.size).toBe(0);
 * expect([...(viewer as any).selectedCells.get('weights')]).toEqual(['0,1', '0,2']);
 * expect((viewer as any).state.activeTensorId).toBe('weights');
 * ```
 */
private finalizeSelectionDrag(): void {
        const drag = this.selectionDrag;
        if (!drag) return;
        this.updateSelectionPreview(drag);
        this.previewSelectedCells.clear();
        this.emitSelectionPreview();
        this.selectionDrag = null;
        this.selectedCells.clear();
        drag.previewSelections.forEach((coords, tensorId) => {
            if (coords.size !== 0) this.selectedCells.set(tensorId, new Set(coords));
        });
        this.state.activeTensorId = drag.tensorId ?? this.selectedCells.keys().next().value ?? this.state.activeTensorId;
        logEvent('selection:update', {
            tensorId: drag.tensorId ?? 'box',
            mode: drag.mode,
            count: this.selectionCount(),
        });
        this.refreshSelectionVisuals(...new Set([
            ...drag.baseSelections.keys(),
            ...this.selectedCells.keys(),
        ]));
        this.emitSelection();
        this.emit();
    }

        /**
 * Converts a browser pointer location into pixel coordinates in the flat SVG/canvas overlay.
 *
 * @param clientX - Viewport-relative x coordinate from a mouse or pointer event.
 * @param clientY - Viewport-relative y coordinate from a mouse or pointer event.
 * @returns Overlay-space coordinates scaled from the container's CSS bounds to the flat canvas backing size.
 * @noThrows Reads the container bounds and performs arithmetic guarded against zero-sized CSS bounds; it does not validate input or call throwing viewer APIs.
 * @example
 * // With a container at left=10, top=20, CSS size 200x100, and a 400x200 backing canvas:
 * viewer.overlayPoint(60, 70);
 * // => { x: 100, y: 100 }
 */
private overlayPoint(clientX: number, clientY: number): { x: number; y: number } {
        const rect = this.container.getBoundingClientRect();
        const scaleX = this.flatCanvas.width / Math.max(1, rect.width);
        const scaleY = this.flatCanvas.height / Math.max(1, rect.height);
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY,
        };
    }

        /**
 * Mirrors the active drag selection into the flat overlay rectangle, or hides the overlay when no drag is active.
 *
 * @returns Nothing; updates `selectionBox` SVG attributes and `flatOverlay.style.display` to match `selectionDrag`.
 * @noThrows Uses cached drag coordinates and DOM attribute/style writes only; a missing drag is handled by hiding the overlay.
 * @example
 * viewer.selectionDrag = null;
 * viewer.syncSelectionBox();
 * // selectionBox.getAttribute('display') === 'none'
 * // flatOverlay.style.display === 'none'
 *
 * @example
 * viewer.selectionDrag = {
 *   source: '2d',
 *   startClient: { x: 10, y: 20 },
 *   currentClient: { x: 30, y: 50 }
 * };
 * viewer.syncSelectionBox();
 * // selectionBox has x/y at the upper-left drag corner, width/height equal to the drag span,
 * // and flatOverlay.style.display === 'block'.
 */
private syncSelectionBox(): void {
        const drag = this.selectionDrag;
        if (!drag) {
            this.selectionBox.setAttribute('display', 'none');
            this.flatOverlay.style.display = 'none';
            return;
        }
        const start = drag.source === '2d' && drag.startWorld
            ? this.projectCanvasPoint(drag.startWorld.x, drag.startWorld.y)
            : this.overlayPoint(drag.startClient.x, drag.startClient.y);
        const current = this.overlayPoint(drag.currentClient.x, drag.currentClient.y);
        this.selectionBox.setAttribute('x', String(Math.min(start.x, current.x)));
        this.selectionBox.setAttribute('y', String(Math.min(start.y, current.y)));
        this.selectionBox.setAttribute('width', String(Math.abs(current.x - start.x)));
        this.selectionBox.setAttribute('height', String(Math.abs(current.y - start.y)));
        this.selectionBox.removeAttribute('display');
        this.flatOverlay.style.display = 'block';
    }

        /**
 * Resolves a hovered tensor cell to the display position used for hover outlines in the active 2D or 3D mode.
 *
 * @param hover - Hover record containing the tensor id and layout coordinate of the visible cell under the pointer.
 * @returns The cell's world/display position including the tensor offset, or `null` when the hover references a tensor that is no longer loaded.
 * @noThrows Stale hover records are handled by returning `null`; otherwise the method delegates to layout coordinate math using viewer state already held by the instance.
 * @example
 * const hover = { tensorId: 'weights', layoutCoord: [1, 2] };
 * const position = viewer.hoverPosition(hover);
 * // position is a Vector3 at the displayed cell location, offset by the 'weights' tensor origin.
 *
 * @example
 * const missing = viewer.hoverPosition({ tensorId: 'removed-tensor', layoutCoord: [0, 0] });
 * // missing === null
 */
private hoverPosition(hover: HoverInfo): Vector3 | null {
        const tensor = this.tensors.get(hover.tensorId);
        if (!tensor) return null;
        const shape = this.layoutShape(tensor.view);
        const layoutCoord = hover.layoutCoord;
        if (this.state.displayMode === '2d') {
            const position = displayPositionForCoord2D(layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
            return new Vector3(tensor.offset[0] + position.x, tensor.offset[1] + position.y, 0);
        }
        return displayPositionForCoord(layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme).add(vectorFromTuple(tensor.offset));
    }

        /**
 * Shows and positions the hover outline for one renderer while hiding the outline owned by the other renderer.
 *
 * @param hover - Hovered tensor cell to outline, or `null` to clear the outline for the selected source.
 * @param source - Renderer that owns the active outline: `'2d'` uses `hoverOutline2D`, and `'3d'` uses `hoverOutline`.
 * @param outlinePosition - Optional precomputed world/display position for the outline; when omitted, the position is derived from `hover`.
 * @returns `true` when outline visibility or position changed and callers should request a render; otherwise `false`.
 * @noThrows A null hover and an unresolvable hover position are treated as non-error states, so the method only toggles visibility and copies positions.
 * @example
 * const changed = viewer.syncHoverOutline(hover, '3d', new Vector3(4, 5, 6));
 * // hoverOutline.visible === true
 * // hoverOutline2D.visible === false
 * // hoverOutline.position.equals(new Vector3(4, 5, 6)) === true
 * // changed === true when the outline was hidden or at a different position.
 *
 * @example
 * viewer.syncHoverOutline(null, '2d');
 * // hoverOutline2D.visible === false
 * // hoverOutline.visible === false
 */
private syncHoverOutline(hover: HoverInfo | null, source: '2d' | '3d', outlinePosition?: Vector3): boolean {
        const outline = source === '3d' ? this.hoverOutline : this.hoverOutline2D;
        const otherOutline = source === '3d' ? this.hoverOutline2D : this.hoverOutline;
        let changed = outline.visible !== !!hover || otherOutline.visible;
        outline.visible = !!hover;
        otherOutline.visible = false;
        if (!hover) return changed;
        const position = outlinePosition ?? this.hoverPosition(hover);
        if (!position) return changed;
        if (!outline.position.equals(position)) {
            outline.position.copy(position);
            changed = true;
        }
        return changed;
    }

        /**
 * Checks whether a pointer event is still inside the cell described by the current hover record.
 *
 * Uses the active display mode to choose the 2D canvas or 3D scene hit-test path, and treats a missing
 * hover as no hit so pointer handlers can decide whether to reuse the existing hover or compute a new one.
 *
 * @param clientX - Pointer x coordinate from the DOM PointerEvent, in viewport client pixels.
 * @param clientY - Pointer y coordinate from the DOM PointerEvent, in viewport client pixels.
 * @param hover - Current hovered tensor cell, or null when no cell is currently hovered.
 * @returns True when the pointer lies inside the screen-space bounds of the hovered cell; false when there is no hover or the pointer has moved outside it.
 * @noThrows Null hover records are handled with a false result, and display-mode-specific hit testing returns false for missing tensors instead of throwing.
 * @example
 * // With no active hover, pointer handlers know they must perform a fresh hit test.
 * expect(viewer.hoveredCellContainsPointer(240, 180, null)).toBe(false);
 *
 * // With an existing hover, the result tells the pointer-move handler whether it can keep the outline.
 * viewer.state.displayMode = '2d';
 * expect(viewer.hoveredCellContainsPointer(112, 96, { tensorId: 'weights', layoutCoord: [0, 1] })).toBe(true);
 */
private hoveredCellContainsPointer(clientX: number, clientY: number, hover: HoverInfo | null): boolean {
        if (!hover) return false;
        return this.state.displayMode === '2d'
            ? this.canvasHoveredCellContainsPointer(clientX, clientY, hover)
            : this.sceneHoveredCellContainsPointer(clientX, clientY, hover);
    }

        /**
 * Tests a pointer location against the 2D canvas rectangle occupied by a hovered tensor cell.
 *
 * The hovered layout coordinate is converted to a display position, offset by the tensor's 2D placement,
 * projected through the current canvas pan and zoom, and compared with the canvas element's client rect.
 *
 * @param clientX - Pointer x coordinate from the DOM PointerEvent, in viewport client pixels.
 * @param clientY - Pointer y coordinate from the DOM PointerEvent, in viewport client pixels.
 * @param hover - Hover record containing the tensor id and layout coordinate of the cell being checked.
 * @returns True when the pointer is inside that cell's client-space canvas bounds; false when the tensor id is no longer present or the pointer is outside the bounds.
 * @noThrows A stale hover whose tensor was removed is treated as false, and the remaining work is deterministic coordinate arithmetic against initialized canvas state.
 * @example
 * viewer.flatCanvas.getBoundingClientRect = () => ({ left: 100, top: 50, width: 400, height: 300 } as DOMRect);
 * viewer.flatCanvas.width = 400;
 * viewer.flatCanvas.height = 300;
 * viewer.canvasPan = { x: 0, y: 0 };
 * viewer.canvasZoom = 1;
 *
 * const hover = { tensorId: 'activation', layoutCoord: [2, 3] };
 * expect(viewer.canvasHoveredCellContainsPointer(168, 124, hover)).toBe(true);
 * expect(viewer.canvasHoveredCellContainsPointer(20, 20, hover)).toBe(false);
 */
private canvasHoveredCellContainsPointer(clientX: number, clientY: number, hover: HoverInfo): boolean {
        const tensor = this.tensors.get(hover.tensorId);
        if (!tensor) return false;
        const shape = this.layoutShape(tensor.view);
        const position = displayPositionForCoord2D(hover.layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        const topLeft = this.projectCanvasPoint(tensor.offset[0] + position.x - 0.5, tensor.offset[1] + position.y + 0.5);
        const bottomRight = this.projectCanvasPoint(tensor.offset[0] + position.x + 0.5, tensor.offset[1] + position.y - 0.5);
        const rect = this.flatCanvas.getBoundingClientRect();
        const scaleX = rect.width / this.flatCanvas.width;
        const scaleY = rect.height / this.flatCanvas.height;
        const left = rect.left + Math.min(topLeft.x, bottomRight.x) * scaleX;
        const right = rect.left + Math.max(topLeft.x, bottomRight.x) * scaleX;
        const top = rect.top + Math.min(topLeft.y, bottomRight.y) * scaleY;
        const bottom = rect.top + Math.max(topLeft.y, bottomRight.y) * scaleY;
        return clientX >= left && clientX <= right && clientY >= top && clientY <= bottom;
    }

        /**
 * Tests a pointer location against the projected screen-space bounds of a hovered 3D tensor cell.
 *
 * The method projects the eight corners of the unit cell through the active camera and checks whether the
 * pointer falls inside the resulting client-pixel bounding box on the renderer's DOM element.
 *
 * @param clientX - Pointer x coordinate from the DOM PointerEvent, in viewport client pixels.
 * @param clientY - Pointer y coordinate from the DOM PointerEvent, in viewport client pixels.
 * @param hover - Hover record containing the tensor id and layout coordinate of the 3D cell being checked.
 * @returns True when the pointer is inside the projected client-space bounding box for the hovered cell; false for a stale tensor id or an outside pointer.
 * @noThrows Stale hover records return false, and projection uses the viewer's already-created renderer element and camera state.
 * @example
 * viewer.renderer.domElement.getBoundingClientRect = () => ({ left: 0, top: 0, width: 800, height: 600 } as DOMRect);
 * const hover = { tensorId: 'embedding', layoutCoord: [1, 0, 2] };
 *
 * expect(viewer.sceneHoveredCellContainsPointer(402, 298, hover)).toBe(true);
 * expect(viewer.sceneHoveredCellContainsPointer(799, 10, hover)).toBe(false);
 */
private sceneHoveredCellContainsPointer(clientX: number, clientY: number, hover: HoverInfo): boolean {
        const tensor = this.tensors.get(hover.tensorId);
        if (!tensor) return false;
        const shape = this.layoutShape(tensor.view);
        const center = displayPositionForCoord(hover.layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme)
            .add(vectorFromTuple(tensor.offset));
        const rect = this.renderer.domElement.getBoundingClientRect();
        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;

        for (const x of [-0.5, 0.5]) {
            for (const y of [-0.5, 0.5]) {
                for (const z of [-0.5, 0.5]) {
                    const projected = center.clone().add(new Vector3(x, y, z)).project(this.camera);
                    const screenX = rect.left + ((projected.x + 1) * rect.width) / 2;
                    const screenY = rect.top + ((1 - projected.y) * rect.height) / 2;
                    minX = Math.min(minX, screenX);
                    maxX = Math.max(maxX, screenX);
                    minY = Math.min(minY, screenY);
                    maxY = Math.max(maxY, screenY);
                }
            }
        }

        return clientX >= minX && clientX <= maxX && clientY >= minY && clientY <= maxY;
    }

        /**
 * Converts a 2D viewer world coordinate into a pixel coordinate on the flat canvas.
 *
 * The projection centers world space in the canvas, applies the current pan and zoom, scales by
 * CANVAS_WORLD_SCALE, and flips the y axis so positive world y appears upward on screen.
 *
 * @param x - Horizontal world coordinate in the 2D tensor layout.
 * @param y - Vertical world coordinate in the 2D tensor layout.
 * @returns Canvas pixel coordinates used to draw cells, tensor outlines, guides, labels, and selection boxes.
 * @noThrows The conversion only reads initialized canvas dimensions and pan/zoom fields and performs numeric arithmetic.
 * @example
 * viewer.flatCanvas.width = 800;
 * viewer.flatCanvas.height = 600;
 * viewer.canvasPan = { x: 10, y: -20 };
 * viewer.canvasZoom = 1;
 *
 * // The world origin projects to the canvas center plus pan.
 * expect(viewer.projectCanvasPoint(0, 0)).toEqual({ x: 410, y: 280 });
 */
private projectCanvasPoint(x: number, y: number): { x: number; y: number } {
        return {
            x: this.flatCanvas.width / 2 + this.canvasPan.x + (x * CANVAS_WORLD_SCALE * this.canvasZoom),
            y: this.flatCanvas.height / 2 + this.canvasPan.y - (y * CANVAS_WORLD_SCALE * this.canvasZoom),
        };
    }

    /**
 * Choose the largest tensor-name font size that fits inside the tensor's visible 2D outline width.
 *
 * @param name - Tensor title text to measure with the 2D canvas context.
 * @param preferredFontSize - Desired bold title size, in canvas pixels, before width fitting is applied.
 * @param outlineExtent2D - Visible tensor outline size in world units; the x extent determines the available label width.
 * @returns The preferred size when the measured title fits, otherwise a proportionally reduced font size clamped to at least 1 pixel.
 * @noThrows Uses numeric canvas measurements and clamps the available width, so normal string and extent inputs do not introduce an expected validation or throw path.
 * @example
 * viewer.canvasZoom = 1;
 * viewer.flatContext.measureText = () => ({ width: 200 }) as TextMetrics;
 * const fitted = viewer.fitTensorNameFontSize('attention.weights', 20, { x: 1, y: 0.25 });
 * expect(fitted).toBeLessThan(20);
 * expect(fitted).toBeGreaterThanOrEqual(1);
 */
    private fitTensorNameFontSize(
        name: string,
        preferredFontSize: number,
        outlineExtent2D: { x: number; y: number },
    ): number {
        const maxWidth = Math.max(1, outlineExtent2D.x * CANVAS_WORLD_SCALE * this.canvasZoom - 12);
        this.flatContext.font = `700 ${preferredFontSize}px ${TENSOR_NAME_FONT_FAMILY}`;
        const measuredWidth = this.flatContext.measureText(name).width;
        return measuredWidth > maxWidth ? Math.max(1, preferredFontSize * (maxWidth / measuredWidth)) : preferredFontSize;
    }

        /**
 * Center the 2D canvas on the supplied scene bounds and choose a zoom that keeps the bounds inside the canvas inset.
 *
 * @param bounds - Three.js bounding box for the visible tensor layout, or null to reset the 2D pan to the origin and the zoom to 1.
 * @returns Nothing; updates `canvasPan` and `canvasZoom` so the next 2D render shows the requested bounds.
 * @noThrows Accepts either null or an already-constructed `Box3` and only reads canvas dimensions and box size/center values, so there is no expected validation failure for normal viewer state.
 * @example
 * viewer.canvasPan = { x: 42, y: -7 };
 * viewer.canvasZoom = 3;
 * viewer.fitCanvasView(null);
 * expect(viewer.canvasPan).toEqual({ x: 0, y: 0 });
 * expect(viewer.canvasZoom).toBe(1);
 */
private fitCanvasView(bounds: Box3 | null = null): void {
        if (!bounds) {
            this.canvasPan = { x: 0, y: 0 };
            this.canvasZoom = 1;
            return;
        }
        const size = bounds.getSize(new Vector3());
        const center = bounds.getCenter(new Vector3());
        const width = Math.max(1, size.x * CANVAS_WORLD_SCALE);
        const height = Math.max(1, size.y * CANVAS_WORLD_SCALE);
        const inset = Math.min(
            MAX_CANVAS_FIT_INSET,
            Math.max(MIN_CANVAS_FIT_INSET, Math.min(this.flatCanvas.width, this.flatCanvas.height) * 0.035),
        ) * this.canvasScale();
        this.canvasZoom = normalizeCanvasZoom(
            Math.min(
                (this.flatCanvas.width - inset * 2) / width,
                (this.flatCanvas.height - inset * 2) / height,
            ) * AUTO_FIT_2D_SCALE,
        );
        this.canvasPan = {
            x: -(center.x * CANVAS_WORLD_SCALE * this.canvasZoom),
            y: center.y * CANVAS_WORLD_SCALE * this.canvasZoom,
        };
    }

    private readonly onCanvasPointerDown = (event: PointerEvent): void => {
        const hit = this.hoveredCellContainsPointer(event.clientX, event.clientY, this.state.hover)
            && this.state.hover
            ? { hover: this.state.hover, position: this.hoverPosition(this.state.hover) ?? new Vector3() }
            : this.canvasPointerToHover(event.clientX, event.clientY);
        if (event.button === 0) {
            if (this.state.interactionMode === 'select') {
                if (!this.selectionEnabled()) return;
                this.flatCanvas.setPointerCapture(event.pointerId);
                this.beginSelectionDrag(
                    '2d',
                    hit?.hover ?? null,
                    event.ctrlKey ? 'remove' : event.shiftKey ? 'add' : 'replace',
                    event.clientX,
                    event.clientY,
                );
                return;
            }
            this.flatCanvas.setPointerCapture(event.pointerId);
            this.isCanvasPanning = true;
            this.lastCanvasPointer = { x: event.clientX, y: event.clientY };
            return;
        }
        if (event.button !== 2) return;
        event.preventDefault();
    };

    private readonly onCanvasPointerUp = (event: PointerEvent): void => {
        if (this.flatCanvas.hasPointerCapture(event.pointerId)) this.flatCanvas.releasePointerCapture(event.pointerId);
        if (event.button === 0 && this.selectionDrag?.source === '2d') {
            this.finalizeSelectionDrag();
            return;
        }
        this.isCanvasPanning = false;
        logEvent('2d:pointerup');
    };

    private readonly onCanvasWheel = (event: WheelEvent): void => {
        if (this.state.displayMode !== '2d') return;
        event.preventDefault();
        const rect = this.flatCanvas.getBoundingClientRect();
        const scale = event.deltaY < 0 ? 1.1 : 1 / 1.1;
        const nextZoom = normalizeCanvasZoom(this.canvasZoom * scale);
        const cursorX = (event.clientX - rect.left) * (this.flatCanvas.width / rect.width) - this.flatCanvas.width / 2 - this.canvasPan.x;
        const cursorY = (event.clientY - rect.top) * (this.flatCanvas.height / rect.height) - this.flatCanvas.height / 2 - this.canvasPan.y;
        this.canvasPan.x -= cursorX * ((nextZoom / this.canvasZoom) - 1);
        this.canvasPan.y -= cursorY * ((nextZoom / this.canvasZoom) - 1);
        this.canvasZoom = nextZoom;
        logEvent('2d:zoom', { zoom: this.canvasZoom });
        this.requestRender();
    };

    private readonly onCanvasPointerMove = (event: PointerEvent): void => {
        if (this.selectionDrag?.source === '2d') {
            this.updateSelectionDrag('2d', event.clientX, event.clientY);
            return;
        }
        if (this.isCanvasPanning) {
            this.canvasPan.x += (event.clientX - this.lastCanvasPointer.x) * (this.flatCanvas.width / this.flatCanvas.getBoundingClientRect().width);
            this.canvasPan.y += (event.clientY - this.lastCanvasPointer.y) * (this.flatCanvas.height / this.flatCanvas.getBoundingClientRect().height);
            this.lastCanvasPointer = { x: event.clientX, y: event.clientY };
            this.requestRender();
            return;
        }
        const currentHover = this.state.hover;
        if (currentHover && this.hoveredCellContainsPointer(event.clientX, event.clientY, currentHover)) {
            if (this.syncHoverOutline(currentHover, '2d')) this.requestRender();
            return;
        }
        const hit = this.canvasPointerToHover(event.clientX, event.clientY);
        this.updateHover(hit?.hover ?? null, '2d', hit?.position);
    };

    private readonly onCanvasPointerLeave = (): void => {
        if (this.selectionDrag?.source === '2d') return;
        this.updateHover(null, '2d');
    };

    private readonly onCanvasClick = (event: PointerEvent): void => {
        const hover = this.hoveredCellContainsPointer(event.clientX, event.clientY, this.state.hover)
            ? this.state.hover
            : this.canvasPointerToHover(event.clientX, event.clientY)?.hover ?? null;
        if (!hover) return;
        this.state.activeTensorId = hover.tensorId;
        this.state.hover = hover;
        this.state.lastHover = hover;
        logEvent('2d:select', hover);
        this.emit();
    };

    private readonly onPointerDown = (event: PointerEvent): void => {
        const hit = this.hoveredCellContainsPointer(event.clientX, event.clientY, this.state.hover)
            && this.state.hover
            ? { hover: this.state.hover, position: this.hoverPosition(this.state.hover) ?? new Vector3() }
            : this.scenePointerToHover(event.clientX, event.clientY);
        if (event.button === 0) {
            if (this.state.interactionMode !== 'select') return;
            if (!this.selectionEnabled()) return;
            this.renderer.domElement.setPointerCapture(event.pointerId);
            this.controls.enabled = false;
            this.beginSelectionDrag(
                '3d',
                hit?.hover ?? null,
                event.ctrlKey ? 'remove' : event.shiftKey ? 'add' : 'replace',
                event.clientX,
                event.clientY,
            );
            return;
        }
        if (event.button !== 2) return;
        event.preventDefault();
        event.stopPropagation();
    };

    private readonly onPointerMove = (event: PointerEvent): void => {
        if (this.selectionDrag?.source === '3d') {
            this.updateSelectionDrag('3d', event.clientX, event.clientY);
            return;
        }
        const currentHover = this.state.hover;
        if (currentHover && this.hoveredCellContainsPointer(event.clientX, event.clientY, currentHover)) {
            if (this.syncHoverOutline(currentHover, '3d')) this.requestRender();
            return;
        }
        const hit = this.scenePointerToHover(event.clientX, event.clientY);
        if (!hit) {
            this.updateHover(null, '3d');
            return;
        }
        this.updateHover(hit.hover, '3d', hit.position);
    };

    private readonly onPointerLeave = (): void => {
        if (this.selectionDrag?.source === '3d') return;
        this.updateHover(null, '3d');
    };

    private readonly onPointerUp = (event: PointerEvent): void => {
        if (event.button !== 0 && event.button !== 2) return;
        if (this.selectionDrag?.source !== '3d') return;
        if (this.renderer.domElement.hasPointerCapture(event.pointerId)) this.renderer.domElement.releasePointerCapture(event.pointerId);
        this.controls.enabled = true;
        this.finalizeSelectionDrag();
    };

    private readonly onClick = (): void => {
        if (!this.state.hover) return;
        this.state.activeTensorId = this.state.hover.tensorId;
        logEvent('3d:select', this.state.hover);
        this.emit();
    };

    private readonly onKeyDown = (event: KeyboardEvent): void => {
        if (!event.ctrlKey) return;
        if (event.key === '+' || event.key === '=') {
            event.preventDefault();
            this.zoomBy(0.9);
        } else if (event.key === '-') {
            event.preventDefault();
            this.zoomBy(1.1);
        }
    };

        /**
 * Apply a keyboard-style zoom step to the active camera and notify the viewer that the viewport changed.
 *
 * @param scale - Multiplicative zoom step: values below 1 zoom in, values above 1 zoom out.
 * @returns Nothing; updates `canvasZoom` for the orthographic 2D camera or moves the 3D camera relative to the controls target, then schedules a render and emits state.
 * @noThrows With an initialized camera, controls target, and finite numeric scale, the method only performs Three.js vector arithmetic and render scheduling.
 * @example
 * viewer.camera = viewer.orthographicCamera;
 * viewer.canvasZoom = 2;
 * viewer.zoomBy(0.5);
 * expect(viewer.canvasZoom).toBe(normalizeCanvasZoom(4));
 * expect(requestRenderSpy).toHaveBeenCalled();
 * expect(emitSpy).toHaveBeenCalled();
 */
private zoomBy(scale: number): void {
        if (this.camera instanceof OrthographicCamera) {
            this.canvasZoom = normalizeCanvasZoom(this.canvasZoom / scale);
            this.sync2DCamera();
        } else {
            this.camera.position.sub(this.controls.target).multiplyScalar(scale).add(this.controls.target);
        }
        this.requestRender();
        this.emit();
    }

        /**
 * Schedule one canvas render on the next animation frame, coalescing repeated state changes while a frame is pending.
 *
 * @returns Nothing; sets `renderPending` until the animation-frame callback renders the active 2D or 3D view and synchronizes the selection box.
 * @noThrows Repeated calls return early while a frame is pending, and the scheduled callback uses the viewer's already-initialized renderer/canvas state.
 * @example
 * viewer.renderPending = false;
 * viewer.requestRender();
 * viewer.requestRender();
 * expect(viewer.renderPending).toBe(true);
 * expect(requestAnimationFrameSpy).toHaveBeenCalledTimes(1);
 */
private requestRender(): void {
        if (this.renderPending) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
            // coalescing all state changes into one animation frame prevents slider
            // drags and selection previews from queueing redundant full scene renders.
            this.renderPending = false;
            if (this.state.displayMode === '2d') {
                this.render2D();
            } else {
                this.controls.update();
                this.renderer.render(this.scene, this.camera);
            }
            this.syncSelectionBox();
        });
    }

        /**
 * Computes the world-space bounding box that encloses every tensor mesh group currently attached to the viewer.
 *
 * @returns A `Box3` expanded around all rendered tensor groups, or `null` when the viewer has no tensor mesh groups to frame.
 * @noThrows Reads the existing `tensorMeshes` map and asks each Three.js group to update its world matrix; it performs no caller-input validation and has no expected viewer-level error case.
 * @example
 * ```ts
 * viewer.tensorMeshes.clear();
 * expect(viewer.sceneBounds()).toBeNull();
 *
 * viewer.tensorMeshes.set('activation', activationGroup);
 * const bounds = viewer.sceneBounds();
 * expect(bounds).toBeInstanceOf(Box3);
 * expect(bounds!.isEmpty()).toBe(false);
 * ```
 */
private sceneBounds(): Box3 | null {
        const groups = Array.from(this.tensorMeshes.values());
        if (groups.length === 0) return null;
        const bounds = new Box3();
        groups.forEach((group) => {
            group.updateWorldMatrix(true, true);
            bounds.expandByObject(group);
        });
        return bounds;
    }

        /**
 * Repositions the active camera controls so the current tensor scene is visible, using canvas fitting in 2D and a diagonal 3D view for populated 3D scenes.
 *
 * @returns Nothing. The method updates camera positions, the controls target, and queues a render instead of producing a value.
 * @noThrows Camera fitting is derived from the viewer's current display mode and optional scene bounds; an empty scene is handled by resetting both cameras to a default `(0, 0, 30)` position.
 * @example
 * ```ts
 * viewer.tensorMeshes.clear();
 * viewer.state.displayMode = '3d';
 * viewer.fitCamera();
 *
 * expect(viewer.controls.target.toArray()).toEqual([0, 0, 0]);
 * expect(viewer.perspectiveCamera.position.toArray()).toEqual([0, 0, 30]);
 * expect(viewer.orthographicCamera.position.toArray()).toEqual([0, 0, 30]);
 * expect(requestRenderSpy).toHaveBeenCalled();
 * ```
 */
private fitCamera(): void {
        const bounds = this.sceneBounds();
        if (this.state.displayMode === '2d') {
            this.fitCanvasView(bounds);
            this.requestRender();
            return;
        }
        if (!bounds) {
            this.controls.target.set(0, 0, 0);
            this.perspectiveCamera.position.set(0, 0, 30);
            this.orthographicCamera.position.set(0, 0, 30);
            this.requestRender();
            return;
        }

        const center = bounds.getCenter(new Vector3());
        const sphere = bounds.getBoundingSphere(new Sphere());
        const radius = Math.max(sphere.radius, 8);
        const halfVerticalFov = (this.perspectiveCamera.fov * Math.PI) / 360;
        const halfHorizontalFov = Math.atan(Math.tan(halfVerticalFov) * this.perspectiveCamera.aspect);
        const fitDistance = (radius / Math.sin(Math.min(halfVerticalFov, halfHorizontalFov))) * AUTO_FIT_3D_DISTANCE_SCALE;
        const viewDirection = new Vector3(1, 1, 1.5).normalize();
        this.controls.target.copy(center);
        this.perspectiveCamera.position.copy(center.clone().add(viewDirection.multiplyScalar(fitDistance)));
        this.perspectiveCamera.lookAt(center);
        this.orthographicCamera.position.set(center.x, center.y, center.z + fitDistance * 1.5);
        this.orthographicCamera.lookAt(center);
        this.requestRender();
    }

        /**
 * Regenerates all Three.js tensor groups and coarse hit-test records from the viewer's tensor records after layout, data, or display settings change.
 *
 * @param options - Rebuild controls. Set `fitCamera: true` when the rebuilt scene should be reframed immediately; omit it or pass `false` to keep the current camera and only request a render.
 * @returns Nothing. The method removes old groups from the scene, repopulates `tensorMeshes` and `pickMeshes`, requests or refits rendering, and emits a viewer update event.
 * @noThrows The optional rebuild flag is defaulted locally, and the method operates on already-normalized tensor records owned by the viewer, so there is no expected validation failure for callers.
 * @example
 * ```ts
 * viewer.tensors.set('weights', weightsTensor);
 * viewer.rebuildAllMeshes({ fitCamera: true });
 *
 * expect(viewer.tensorMeshes.has('weights')).toBe(true);
 * expect(viewer.scene.children).toContain(viewer.tensorMeshes.get('weights'));
 * expect(viewer.pickMeshes.some((entry) => entry.tensorId === 'weights')).toBe(true);
 * expect(fitCameraSpy).toHaveBeenCalled();
 * expect(emitSpy).toHaveBeenCalled();
 * ```
 */
private rebuildAllMeshes(options: { fitCamera?: boolean } = {}): void {
        const shouldFitCamera = options.fitCamera ?? false;
        this.relayoutAutoOffsets();
        Array.from(this.tensorMeshes.values()).forEach((group) => this.scene.remove(group));
        this.tensorMeshes.clear();
        this.pickMeshes.length = 0;
        this.tensors.forEach((tensor) => {
            const group = this.buildTensorGroup(tensor);
            this.tensorMeshes.set(tensor.id, group);
            this.scene.add(group);
            const mesh = group.children[0];
            if (mesh instanceof InstancedMesh) {
                // hit testing uses a cheap coarse bounds pass before per-cell math,
                // so it must be regenerated whenever meshes are rebuilt.
                const shape = this.layoutShape(tensor.view);
                const extent2D = displayExtent2D(shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
                const bounds = (mesh.boundingBox ?? new Box3().setFromObject(mesh)).clone();
                this.pickMeshes.push({
                    tensorId: tensor.id,
                    mesh,
                    bounds,
                    rect2D: {
                        minX: tensor.offset[0] - extent2D.x / 2,
                        maxX: tensor.offset[0] + extent2D.x / 2,
                        minY: tensor.offset[1] - extent2D.y / 2,
                        maxY: tensor.offset[1] + extent2D.y / 2,
                    },
                });
            }
        });
        if (shouldFitCamera) this.fitCamera();
        else this.requestRender();
        this.emit();
    }

        /**
 * Creates the renderable Three.js group for one tensor by passing the viewer's mesh context and tensor record to the mesh builder.
 *
 * @param tensor - Normalized tensor record containing the tensor id, parsed view, shape, offsets, data/color state, and display metadata used to build its cells and labels.
 * @returns A `Group` containing the mesh objects that represent the tensor in the current viewer display mode.
 * @noThrows This wrapper only assembles the current mesh context and delegates to the mesh builder; it does not parse user input or perform additional validation.
 * @example
 * ```ts
 * const group = viewer.buildTensorGroup(weightsTensor);
 *
 * expect(group).toBeInstanceOf(Group);
 * expect(group.children.some((child) => child instanceof InstancedMesh)).toBe(true);
 * ```
 */
private buildTensorGroup(tensor: TensorRecord): Group {
        return buildTensorGroup(this.meshContext(), tensor);
    }

    /**
 * Recolors an existing tensor mesh after a slice-only view change so the viewer can avoid rebuilding the Three.js instances.
 *
 * @param tensor - Tensor record whose current `view` contains the newly selected hidden indices and whose mesh was built for the same canonical view expression.
 * @param previousView - Tensor view captured before the edit; its canonical expression and hidden indices are compared with `tensor.view`.
 * @returns `true` when the existing mesh was updated for changed slice indices; `false` when the view shape/canonical expression did not match or no slice index changed, so the caller should leave the mesh alone or rebuild it through the normal path.
 * @noThrows The wrapper only passes the viewer mesh context and tensor records to the mesh updater; unsupported view changes are reported with `false` rather than by throwing from this method.
 * @example
 * const previousView = { canonical: 'activation[:, hidden=0]', hiddenIndices: [0] } as TensorViewSpec;
 * tensor.view = { ...previousView, hiddenIndices: [3] };
 *
 * const updatedInPlace = viewer.updateSliceMesh(tensor, previousView);
 * expect(updatedInPlace).toBe(true);
 *
 * tensor.view = { ...tensor.view, canonical: 'activation.T[:, hidden=3]' };
 * expect(viewer.updateSliceMesh(tensor, previousView)).toBe(false);
 */
    private updateSliceMesh(tensor: TensorRecord, previousView: TensorViewSpec): boolean {
        return updateSliceMesh(this.meshContext(), tensor, previousView);
    }

        /**
 * Runs the hybrid 2D frame render: synchronizes camera and selection preview state, renders the WebGL cell layer, then clears and repaints the canvas overlay annotations.
 *
 * @returns Nothing; the WebGL renderer and flat overlay canvas are updated to match the viewer's current 2D tensor state.
 * @noThrows An initialized viewer owns the renderer, scene, camera, and 2D canvas context used here; this pass delegates to drawing helpers and does not validate caller input or intentionally raise errors.
 * @example
 * viewer.state.displayMode = '2d';
 * viewer.render2D();
 *
 * expect(renderer.render).toHaveBeenCalledWith(viewer.scene, viewer.camera);
 * expect(flatContext.clearRect).toHaveBeenCalledWith(0, 0, flatCanvas.width, flatCanvas.height);
 * expect(flatContext.fillText).toHaveBeenCalledWith('layer 0', expect.any(Number), expect.any(Number));
 */
private render2D(): void {
        // 2d is a hybrid render: webgl draws cells for speed, then canvas draws
        // labels and annotations that would be expensive as thousands of meshes.
        this.sync2DCamera();
        this.syncSelectionPreviewState();
        this.renderer.render(this.scene, this.camera);
        this.flatContext.setTransform(1, 0, 0, 1, 0, 0);
        this.flatContext.clearRect(0, 0, this.flatCanvas.width, this.flatCanvas.height);
        this.draw2DGhostLayers();
        this.draw2DLayerTips();
        this.draw2DMarkers();
        this.draw2DCellLabels();
    }

        /**
 * Paints crossed marker boxes over visible 2D cells whose tensor coordinates appear in each tensor's `markerCoords` set.
 *
 * @returns Nothing; matching cells on the flat overlay canvas receive the marker fill, border, and diagonal cross strokes.
 * @noThrows Tensors without marker coordinates are skipped, hidden coordinates are ignored, and cells without canvas bounds are not drawn, so missing or non-visible markers are handled as no-op cases.
 * @example
 * tensor.markerCoords = new Set(['0,2']);
 * tensor.view = parseTensorView('weights[row, col]');
 * jest.spyOn(viewer, 'tensorCoordVisible').mockReturnValue(true);
 * jest.spyOn(viewer, 'canvasCellBounds').mockReturnValue({ left: 10, top: 20, right: 30, bottom: 40 });
 *
 * viewer.draw2DMarkers();
 *
 * expect(flatContext.fillRect).toHaveBeenCalledWith(11.5, 21.5, 17, 17);
 * expect(flatContext.strokeRect).toHaveBeenCalledWith(11.5, 21.5, 17, 17);
 * expect(flatContext.lineTo).toHaveBeenCalledWith(27, 37);
 */
private draw2DMarkers(): void {
        this.flatContext.save();
        this.tensors.forEach((tensor) => {
            if (!tensor.markerCoords || tensor.markerCoords.size === 0) return;
            const instanceShape = this.instanceShape(tensor.view);
            const count = product(instanceShape);
            for (let index = 0; index < count; index += 1) {
                const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
                const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
                if (!tensor.markerCoords.has(coordKey(tensorCoord)) || !this.tensorCoordVisible(tensor, tensorCoord)) continue;
                const layoutCoord = this.mapViewCoordToLayoutCoord(viewCoord, tensor.view);
                const bounds = this.canvasCellBounds(tensor, layoutCoord);
                if (!bounds) continue;
                const outerInset = 1.5;
                const innerInset = 3;
                const outerLeft = bounds.left + outerInset;
                const outerTop = bounds.top + outerInset;
                const outerWidth = Math.max(0, bounds.right - bounds.left - outerInset * 2);
                const outerHeight = Math.max(0, bounds.bottom - bounds.top - outerInset * 2);
                const innerLeft = bounds.left + innerInset;
                const innerTop = bounds.top + innerInset;
                const innerWidth = Math.max(0, bounds.right - bounds.left - innerInset * 2);
                const innerHeight = Math.max(0, bounds.bottom - bounds.top - innerInset * 2);
                this.flatContext.fillStyle = 'rgba(229, 231, 235, 1)';
                this.flatContext.fillRect(outerLeft, outerTop, outerWidth, outerHeight);
                this.flatContext.strokeStyle = 'rgba(15, 23, 42, 0.65)';
                this.flatContext.lineWidth = 2;
                this.flatContext.strokeRect(outerLeft, outerTop, outerWidth, outerHeight);
                this.flatContext.beginPath();
                this.flatContext.moveTo(innerLeft, innerTop);
                this.flatContext.lineTo(innerLeft + innerWidth, innerTop + innerHeight);
                this.flatContext.moveTo(innerLeft + innerWidth, innerTop);
                this.flatContext.lineTo(innerLeft, innerTop + innerHeight);
                this.flatContext.stroke();
                this.flatContext.strokeStyle = 'rgba(241, 245, 249, 0.8)';
                this.flatContext.lineWidth = 1;
                this.flatContext.strokeRect(innerLeft, innerTop, innerWidth, innerHeight);
            }
        });
        this.flatContext.restore();
    }

        /**
 * Draws translucent ghost-layer rectangles and optional labels for tensors in 2D mode, using higher layer numbers behind lower layer numbers.
 *
 * @returns Nothing; the flat overlay canvas is filled with each ghost layer's color and any layer text is rendered inside that cell.
 * @noThrows Tensors without `ghostLayers` are skipped by optional chaining, and renderer-created ghost layers are expected to reference layout coordinates that resolve to canvas cell bounds.
 * @example
 * tensor.ghostLayers = [
 *   { layer: 0, coord: [0, 0], color: [255, 0, 0], text: 'current' },
 *   { layer: 2, coord: [0, 0], color: [0, 0, 255], text: 'previous' },
 * ];
 * jest.spyOn(viewer, 'canvasCellBounds').mockReturnValue({ left: 4, top: 8, right: 24, bottom: 28 });
 *
 * viewer.draw2DGhostLayers();
 *
 * expect(flatContext.fillRect).toHaveBeenNthCalledWith(1, 4, 8, 20, 20); // layer 2 is painted first.
 * expect(flatContext.fillRect).toHaveBeenNthCalledWith(2, 4, 8, 20, 20); // layer 0 is painted on top.
 * expect(flatContext.fillText).toHaveBeenCalledWith('current', expect.any(Number), expect.any(Number));
 */
private draw2DGhostLayers(): void {
        this.flatContext.save();
        this.flatContext.textAlign = 'center';
        this.flatContext.textBaseline = 'middle';
        this.tensors.forEach((tensor) => {
            tensor.ghostLayers?.slice().sort((left, right) => right.layer - left.layer).forEach((layer) => {
                const bounds = this.canvasCellBounds(tensor, layer.coord, layer.bias);
                this.flatContext.fillStyle = colorFromRgb(layer.color).getStyle();
                this.flatContext.fillRect(
                    bounds.left,
                    bounds.top,
                    Math.max(0, bounds.right - bounds.left),
                    Math.max(0, bounds.bottom - bounds.top),
                );
                if (layer.text) this.draw2DCellText(tensor, layer.coord, layer.text, bounds);
            });
        });
        this.flatContext.restore();
    }

        /**
 * Paints one 2D canvas cell for each ghost-layer coordinate so hidden or stacked
 * layer tips remain visible in the flat tensor view.
 *
 * @returns No value; matching ghost-layer cells are filled on `flatContext` using the tensor heatmap color when heatmap mode is active.
 * @noThrows Iterates existing tensor metadata and canvas bounds only; tensors without ghost layers are skipped instead of treated as errors.
 * @example
 * // Given a tensor with ghostLayers at coordinate [1, 0], the render pass fills
 * // that cell's canvas rectangle with the same color calculation used for data cells.
 * viewer.draw2DLayerTips();
 * expect(flatContext.fillRect).toHaveBeenCalledWith(cell.left, cell.top, cell.right - cell.left, cell.bottom - cell.top);
 */
private draw2DLayerTips(): void {
        this.flatContext.save();
        this.tensors.forEach((tensor) => {
            if (!tensor.ghostLayers?.length) return;
            const heatmapRange = this.state.heatmap ? tensor.valueRange : null;
            const coords = new Set(tensor.ghostLayers.map((layer) => coordKey(layer.coord)));
            coords.forEach((key) => {
                const tensorCoord = coordFromKey(key);
                const bounds = this.canvasCellBounds(tensor, tensorCoord);
                const value = tensor.hasData ? numericValue(tensor.data, this.linearIndex(tensorCoord, tensor.shape)) : 0;
                const color = this.cellColor(tensor, tensorCoord, value, heatmapRange);
                this.flatContext.fillStyle = color.getStyle();
                this.flatContext.fillRect(
                    bounds.left,
                    bounds.top,
                    Math.max(0, bounds.right - bounds.left),
                    Math.max(0, bounds.bottom - bounds.top),
                );
            });
        });
        this.flatContext.restore();
    }

        /**
 * Draws configured per-cell labels for every visible coordinate in the current 2D tensor views.
 *
 * @returns No value; visible entries from each tensor's `cellLabels` map are centered into their canvas cells.
 * @noThrows Missing label maps, empty label maps, blank labels, and off-screen coordinates are skipped during the render pass.
 * @example
 * // With cellLabels containing { "0,1": "max" } for a visible tensor cell,
 * // the label render pass delegates to draw2DCellText for that cell only.
 * viewer.draw2DCellLabels();
 * expect(viewer.draw2DCellText).toHaveBeenCalledWith(tensor, [0, 1], 'max', expectedBounds);
 */
private draw2DCellLabels(): void {
        this.flatContext.save();
        this.flatContext.textAlign = 'center';
        this.flatContext.textBaseline = 'middle';
        this.tensors.forEach((tensor) => {
            if (!tensor.cellLabels || tensor.cellLabels.size === 0) return;
            const instanceShape = this.instanceShape(tensor.view);
            const count = product(instanceShape);
            for (let index = 0; index < count; index += 1) {
                const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
                const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
                const text = tensor.cellLabels.get(coordKey(tensorCoord));
                if (!text || !this.tensorCoordVisible(tensor, tensorCoord)) continue;
                const bounds = this.canvasCellBounds(tensor, this.mapViewCoordToLayoutCoord(viewCoord, tensor.view));
                this.draw2DCellText(tensor, tensorCoord, text, bounds);
            }
        });
        this.flatContext.restore();
    }

        /**
 * Centers a single cell label in a 2D canvas cell, choosing a monospace font size
 * that fits the label's line count and longest line.
 *
 * @param tensor - Tensor whose label color should be resolved for the rendered coordinate.
 * @param tensorCoord - Coordinate in the tensor's data space for the labeled cell.
 * @param text - Cell label text; newline-separated non-empty lines are drawn as stacked lines.
 * @param bounds - Canvas-space rectangle for the target cell, with left/right x edges and top/bottom y edges.
 * @returns No value; draws one `fillText` call per visible label line, or skips drawing when the text is empty or the fitted font would be too small.
 * @noThrows Empty labels and undersized cells are handled with early returns, and the method only updates the existing 2D canvas context.
 * @example
 * const bounds = { left: 10, right: 90, top: 20, bottom: 60 };
 * viewer.draw2DCellText(tensor, [0, 1], 'peak\n42', bounds);
 * expect(flatContext.fillText).toHaveBeenCalledWith('peak', 50, expect.any(Number));
 * expect(flatContext.fillText).toHaveBeenCalledWith('42', 50, expect.any(Number));
 */
private draw2DCellText(
        tensor: TensorRecord,
        tensorCoord: number[],
        text: string,
        bounds: { left: number; right: number; top: number; bottom: number },
    ): void {
        const lines = text.split('\n').filter(Boolean);
        if (lines.length === 0) return;
        const width = bounds.right - bounds.left;
        const height = bounds.bottom - bounds.top;
        const maxChars = Math.max(...lines.map((line) => line.length), 1);
        const fontSize = Math.floor(Math.min(72, width / Math.max(1.8, maxChars * 0.72), height / Math.max(1.6, lines.length * 1.15)));
        if (fontSize < MIN_VISIBLE_CELL_LABEL_FONT_SIZE) return;
        const lineHeight = Math.max(fontSize, Math.floor(fontSize * 1.05));
        const centerX = (bounds.left + bounds.right) / 2;
        const centerY = (bounds.top + bounds.bottom) / 2;
        const startY = centerY - ((lines.length - 1) * lineHeight) / 2;
        this.flatContext.font = `${fontSize}px "IBM Plex Mono", "SFMono-Regular", monospace`;
        this.flatContext.fillStyle = this.cellLabelColor(tensor, tensorCoord);
        lines.forEach((line, lineIndex) => {
            const y = startY + (lineIndex * lineHeight);
            this.flatContext.fillText(line, centerX, y);
        });
    }

        /**
 * Escapes text before inserting it into generated SVG text nodes.
 *
 * @param text - Raw tensor name, axis label, or cell label content that will be embedded between SVG tags.
 * @returns The same string with `&`, `<`, `>`, and `"` replaced by their XML entity forms so callers can concatenate safe SVG markup.
 * @noThrows Performs deterministic string replacements on an already-typed string and does not parse or access external state.
 * @example
 * const escaped = viewer.escapeSvgText('A < B & "C"');
 * expect(escaped).toBe('A &lt; B &amp; &quot;C&quot;');
 */
private escapeSvgText(text: string): string {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

        /**
 * Converts a tensor-space coordinate into the row-major offset used to read the tensor's flat data buffer.
 *
 * @param coord - Zero-based coordinate with one entry per tensor axis, such as `[row, column]` for a 2-D tensor.
 * @param shape - Extent of each tensor axis in the same order as `coord`.
 * @returns The flat row-major data index for `coord`, suitable for `numericValue(tensor.data, index)`.
 * @noThrows Performs only arithmetic over the supplied arrays and does not validate bounds or tensor existence.
 * @example
 * ```ts
 * // In a tensor with shape [2, 3, 4], coordinate [1, 2, 3]
 * // is ((1 * 3) + 2) * 4 + 3.
 * expect(viewer.linearIndex([1, 2, 3], [2, 3, 4])).toBe(23);
 * ```
 */
private linearIndex(coord: number[], shape: number[]): number {
        let index = 0;
        coord.forEach((value, axis) => {
            index = (index * shape[axis]) + value;
        });
        return index;
    }

        /**
 * Resolves the final paint color for a tensor cell, then applies the selection highlight tint when that cell is highlighted.
 *
 * @param tensor - Tensor record whose id, custom colors, value range, and heatmap settings participate in the base color lookup.
 * @param tensorCoord - Tensor-space coordinate of the cell being rendered.
 * @param value - Numeric cell value read from the tensor data buffer for `tensorCoord`.
 * @param heatmapRange - Active heatmap minimum and maximum for scaling numeric values, or `null` to use non-heatmap coloring.
 * @returns The Three.js `Color` that canvas, SVG, or mesh rendering should use for the visible cell.
 * @noThrows Delegates to color helpers using an already-resolved tensor record and coordinate; it performs no tensor lookup or input validation itself.
 * @example
 * ```ts
 * const tensor = makeTensorRecord({ id: 'weights', shape: [2, 2] });
 * viewer.highlightedCells.set('weights', new Set(['1,0']));
 *
 * const base = viewer.baseCellColor(tensor, [1, 0], 0.75, { min: 0, max: 1 });
 * const color = viewer.cellColor(tensor, [1, 0], 0.75, { min: 0, max: 1 });
 *
 * expect(color.getHex()).toBe(viewer.selectedColor(base).getHex());
 * ```
 */
private cellColor(
        tensor: TensorRecord,
        tensorCoord: number[],
        value: number,
        heatmapRange: { min: number; max: number } | null,
    ): Color {
        const color = this.baseCellColor(tensor, tensorCoord, value, heatmapRange);
        return this.isHighlightedCell(tensor.id, tensorCoord) ? this.selectedColor(color) : color;
    }

        /**
 * Chooses a readable label color for text drawn over a tensor cell by measuring the luminance of that cell's rendered background.
 *
 * @param tensor - Tensor record containing shape, data presence, flat data, and heatmap value range for the cell background calculation.
 * @param tensorCoord - Tensor-space coordinate of the cell whose label is about to be drawn.
 * @returns The dark label color constant for light backgrounds, or the light label color constant for dark backgrounds.
 * @noThrows Uses an already-resolved tensor record and coordinate and only derives a value, color, and luminance from viewer state.
 * @example
 * ```ts
 * const tensor = makeTensorRecord({ id: 'activations', shape: [1], data: new Float32Array([0]) });
 * jest.spyOn(viewer, 'cellColor').mockReturnValue(new Color(1, 1, 1));
 *
 * expect(viewer.cellLabelColor(tensor, [0])).toBe(CELL_LABEL_DARK);
 * ```
 */
private cellLabelColor(tensor: TensorRecord, tensorCoord: number[]): string {
        const value = tensor.hasData ? numericValue(tensor.data, this.linearIndex(tensorCoord, tensor.shape)) : 0;
        const heatmapRange = this.state.heatmap ? tensor.valueRange : null;
        const color = this.cellColor(tensor, tensorCoord, value, heatmapRange);
        const luminance = (0.2126 * color.r) + (0.7152 * color.g) + (0.0722 * color.b);
        return luminance > 0.5 ? CELL_LABEL_DARK : CELL_LABEL_LIGHT;
    }

        /**
 * Replaces a tensor's custom cell colors with the validated dense, coordinate-list, or strided color instructions loaded for that tensor.
 *
 * @param tensorId - Id of an existing tensor in the current viewer session whose `customColors` map should be rebuilt.
 * @param instructions - Color instruction payloads from session restore or caller input; each instruction must match the target tensor shape.
 * @returns Nothing; the target tensor's `customColors` collection is cleared and repopulated from the valid instructions.
 * @throws Throws the viewer's missing-tensor error when `tensorId` does not identify a tensor in the current session.
 * @example
 * ```ts
 * const tensor = viewer.requireTensor('weights');
 * tensor.customColors.set('0,0', new Color('red'));
 *
 * viewer.applyColorInstructions('weights', [
 *   { kind: 'coords', coords: [[1, 0]], color: '#00ff00' },
 * ]);
 *
 * expect(tensor.customColors.has('0,0')).toBe(false);
 * expect(tensor.customColors.get('1,0')?.getHexString()).toBe('00ff00');
 * ```
 * @example
 * ```ts
 * expect(() => viewer.applyColorInstructions('missing-tensor', [])).toThrow(/missing-tensor/);
 * ```
 */
private applyColorInstructions(tensorId: string, instructions: ColorInstruction[]): void {
        const tensor = this.requireTensor(tensorId);
        tensor.customColors.clear();
        validateColorInstructions(instructions, tensor.shape)?.forEach((instruction) => {
            if (instruction.kind === 'dense') {
                this.applyColors(tensor, new Float32Array(instruction.values));
                return;
            }
            if (instruction.kind === 'coords') {
                this.applyColors(tensor, instruction.coords, instruction.color);
                return;
            }
            this.applyColors(tensor, instruction.base, instruction.shape, instruction.jumps, instruction.color);
        });
    }

        /**
 * Applies caller-supplied custom colors to tensor cells by writing entries into the tensor's coordinate-keyed color map.
 *
 * Accepts the three color instruction forms produced by validated tensor color instructions: a dense per-cell payload,
 * an explicit list of tensor coordinates sharing one color, or a strided rectangular region sharing one color.
 *
 * @param tensor - Loaded tensor record whose `shape` determines the dense cell count and whose `customColors` map receives the parsed colors.
 * @param arg1 - Either a dense color buffer with 2 or 3 channels per tensor cell, a list of coordinates to color, or the base coordinate for a rectangular region.
 * @param arg2 - For coordinate-list instructions, the shared color; for rectangular-region instructions, the region shape; omitted for dense payloads.
 * @param arg3 - For rectangular-region instructions, the per-axis coordinate jumps between cells in the region.
 * @param arg4 - For rectangular-region instructions, the shared color applied to every generated coordinate.
 * @returns Nothing; matching tensor coordinates are inserted or replaced in `tensor.customColors`.
 * @throws Error when a dense `Uint8ClampedArray` or `Float32Array` does not contain exactly 2 or 3 color channels for each tensor cell.
 * @example
 * const tensor = { shape: [2], customColors: new Map() } as TensorRecord;
 * viewer.applyColors(tensor, new Uint8ClampedArray([255, 0, 0, 0, 0, 255]));
 *
 * expect(tensor.customColors.size).toBe(2);
 * expect(tensor.customColors.has('0')).toBe(true);
 * expect(tensor.customColors.has('1')).toBe(true);
 *
 * @example
 * const tensor = { shape: [2, 2], customColors: new Map() } as TensorRecord;
 * viewer.applyColors(tensor, [0, 0], [2, 2], [1, 1], [255, 128, 0]);
 *
 * expect([...tensor.customColors.keys()]).toEqual(['0,0', '0,1', '1,0', '1,1']);
 *
 * @example
 * const tensor = { shape: [2], customColors: new Map() } as TensorRecord;
 *
 * expect(() => viewer.applyColors(tensor, new Uint8ClampedArray([255, 0, 0, 255]))).toThrow(
 *   'Expected dense color payload with 2 or 3 channels per cell, received 4.',
 * );
 */
private applyColors(
        tensor: TensorRecord,
        arg1: Uint8ClampedArray | Float32Array | number[][] | number[],
        arg2?: RGB | HueSaturation | number[],
        arg3?: number[] | RGB | HueSaturation,
        arg4?: RGB | HueSaturation | number[],
    ): void {
        if (arg1 instanceof Uint8ClampedArray || arg1 instanceof Float32Array) {
            const denseColorSize = arg1.length / product(tensor.shape);
            if (denseColorSize !== 2 && denseColorSize !== 3) {
                throw new Error(`Expected dense color payload with 2 or 3 channels per cell, received ${arg1.length}.`);
            }
            for (let index = 0; index < product(tensor.shape); index += 1) {
                const coord = unravelIndex(index, tensor.shape);
                const offset = index * denseColorSize;
                const values = Array.from(arg1.slice(offset, offset + denseColorSize));
                tensor.customColors.set(coordKey(coord), parseCustomColor(
                    denseColorSize === 3 && arg1 instanceof Float32Array
                        ? values.map((value) => Math.round(value * 255))
                        : values,
                ));
            }
            return;
        }

        if (Array.isArray(arg1) && Array.isArray(arg2) && typeof arg2[0] === 'number') {
            if (Array.isArray(arg1[0])) {
                const color = parseCustomColor(arg2 as number[]);
                (arg1 as number[][]).forEach((coord) => tensor.customColors.set(coordKey(coord), color));
                return;
            }
            if (Array.isArray(arg3) && Array.isArray(arg4)) {
                const base = arg1 as number[];
                const shape = arg2 as number[];
                const jumps = arg3 as number[];
                const color = parseCustomColor(arg4 as number[]);
                const ranges = shape.map((dim, axis) => Array.from({ length: dim }, (_entry, index) => base[axis] + index * jumps[axis]));
                const coord = new Array(shape.length).fill(0);
                /**
 * Walks each axis of the precomputed rectangular color ranges and writes the shared color for every generated tensor coordinate.
 *
 * @param axis - Zero-based recursion depth into `ranges`; `0` starts enumeration and `shape.length` commits the assembled coordinate.
 * @returns Nothing; each completed coordinate is copied into `tensor.customColors` with the parsed rectangular-region color.
 * @noThrows The recursion only iterates already-built numeric ranges and writes to a `Map`; validation and color parsing happened before `visit(0)` is called.
 * @example
 * // With base [1, 0], shape [2, 2], and jumps [1, 2], visit(0) colors:
 * // [1, 0], [1, 2], [2, 0], and [2, 2].
 * visit(0);
 * expect([...tensor.customColors.keys()]).toEqual(['1,0', '1,2', '2,0', '2,2']);
 */
                const visit = (axis: number): void => {
                    if (axis === shape.length) {
                        tensor.customColors.set(coordKey(coord.slice()), color);
                        return;
                    }
                    ranges[axis].forEach((value) => {
                        coord[axis] = value;
                        visit(axis + 1);
                    });
                };
                visit(0);
            }
        }
    }

        /**
 * Publishes the latest viewer snapshot to every registered state listener.
 *
 * @returns Nothing; subscribers observe the current snapshot, and viewer state is not changed by this method itself.
 * @throws Propagates any exception thrown by a registered snapshot listener because listeners are invoked synchronously.
 * @example
 * const snapshots: ViewerSnapshot[] = [];
 * viewer.subscribe((snapshot) => snapshots.push(snapshot));
 *
 * viewer.emit();
 *
 * expect(snapshots).toHaveLength(1);
 * expect(snapshots[0]).toEqual(viewer.getSnapshot());
 */
private emit(): void {
        const snapshot = this.getSnapshot();
        this.listeners.forEach((listener) => listener(snapshot));
    }

        /**
 * Publishes the current hover payload to every registered hover listener.
 *
 * @returns Nothing; subscribers receive the value returned by `getHover()`, and hover state is not changed by this method itself.
 * @throws Propagates any exception thrown by a registered hover listener because listeners are invoked synchronously.
 * @example
 * const hovers: HoverPayload[] = [];
 * viewer.onHover((hover) => hovers.push(hover));
 *
 * viewer.emitHover();
 *
 * expect(hovers).toEqual([viewer.getHover()]);
 */
private emitHover(): void {
        const hover = this.getHover();
        this.hoverListeners.forEach((listener) => listener(hover));
    }

        /**
 * Notifies every registered selection listener with the viewer's current selected tensor coordinates.
 *
 * The payload is the coordinate map produced by {@link getSelectedCoords}, so observers such as UI
 * panels can refresh their selected-cell display after a click, drag commit, tensor removal, or
 * programmatic selection change.
 *
 * @returns Void; selection observers receive the current selected-coordinate map through their callbacks.
 * @noThrows The method only reads the already-normalized selected-cell state and invokes the registered listener list; it does not validate caller input or perform I/O.
 * @example
 * const received: Array<Map<string, number[][]>> = [];
 * (viewer as any).selectionListeners.add((selection: Map<string, number[][]>) => {
 *   received.push(selection);
 * });
 * (viewer as any).getSelectedCoords = () => new Map([["activations", [[0, 1], [0, 2]]]]);
 *
 * (viewer as any).emitSelection();
 *
 * expect(received).toEqual([
 *   new Map([["activations", [[0, 1], [0, 2]]]]),
 * ]);
 */
private emitSelection(): void {
        const selection = this.getSelectedCoords();
        this.selectionListeners.forEach((listener) => listener(selection));
    }

        /**
 * Publishes the in-progress drag selection to preview listeners, or publishes an empty map when no selection drag is active.
 *
 * Preview subscribers use this transient coordinate map to highlight cells under the drag box before
 * those cells are committed to the viewer's selected-cell state.
 *
 * @returns Void; preview observers receive either the drag preview coordinate map or an empty map through their callbacks.
 * @noThrows The method only branches on the current `selectionDrag`, converts its stored preview selections when present, and invokes registered listeners without accepting external input.
 * @example
 * const previews: Array<Map<string, number[][]>> = [];
 * (viewer as any).selectionPreviewListeners.add((selection: Map<string, number[][]>) => {
 *   previews.push(selection);
 * });
 * (viewer as any).selectionDrag = null;
 *
 * (viewer as any).emitSelectionPreview();
 *
 * expect(previews).toEqual([new Map()]);
 */
private emitSelectionPreview(): void {
        const selection = this.selectionDrag ? this.selectionCoords(this.selectionDrag.previewSelections) : new Map();
        this.selectionPreviewListeners.forEach((listener) => listener(selection));
    }

        /**
 * Clears the active and previous cell hover records, hides both hover outlines, and emits the cleared hover state.
 *
 * This is used when tensor geometry, visibility, or data changes make the existing hover target stale.
 *
 * @returns Void; the viewer state is updated so `state.hover` and `state.lastHover` are `null`, hover outlines are hidden, and hover listeners are notified.
 * @noThrows The method assigns local viewer fields, toggles owned outline visibility flags, and emits the resulting hover state without validating caller-provided data.
 * @example
 * (viewer as any).state.hover = { tensorId: "weights", coord: [1, 2] };
 * (viewer as any).state.lastHover = { tensorId: "weights", coord: [1, 1] };
 * (viewer as any).hoverOutline.visible = true;
 * (viewer as any).hoverOutline2D.visible = true;
 *
 * (viewer as any).clearHover();
 *
 * expect((viewer as any).state.hover).toBeNull();
 * expect((viewer as any).state.lastHover).toBeNull();
 * expect((viewer as any).hoverOutline.visible).toBe(false);
 * expect((viewer as any).hoverOutline2D.visible).toBe(false);
 */
private clearHover(): void {
        this.state.hover = null;
        this.state.lastHover = null;
        this.hoverOutline.visible = false;
        this.hoverOutline2D.visible = false;
        this.lastHoverLogKey = null;
        this.emitHover();
    }

        /**
 * Removes all loaded tensor render objects and returns the viewer to an empty, no-selection, no-hover tensor state.
 *
 * The reset clears mesh groups from the Three.js scene, pending tensor-data requests, preview and
 * committed selections, tensor records, the active tensor id, and hover outlines. It is the shared
 * cleanup step before loading a new manifest and when the public `clear()` API empties the viewer.
 *
 * @returns Void; owned collections and viewer state are cleared in place and selection-preview/selection observers are updated during the reset.
 * @noThrows The method operates on viewer-owned collections and scene objects that were previously registered by the viewer, so it has no expected validation or data-loading failure path.
 * @example
 * const meshGroup = new Group();
 * (viewer as any).tensorMeshes.set("weights", meshGroup);
 * (viewer as any).scene.add(meshGroup);
 * (viewer as any).tensors.set("weights", { id: "weights" });
 * (viewer as any).pendingTensorDataRequests.set("weights", new AbortController());
 * (viewer as any).state.activeTensorId = "weights";
 * (viewer as any).state.hover = { tensorId: "weights", coord: [0, 0] };
 *
 * (viewer as any).resetLoadedState();
 *
 * expect((viewer as any).tensorMeshes.size).toBe(0);
 * expect((viewer as any).tensors.size).toBe(0);
 * expect((viewer as any).pendingTensorDataRequests.size).toBe(0);
 * expect((viewer as any).state.activeTensorId).toBeNull();
 * expect((viewer as any).state.hover).toBeNull();
 * expect((viewer as any).scene.children).not.toContain(meshGroup);
 */
private resetLoadedState(): void {
        Array.from(this.tensorMeshes.values()).forEach((group) => this.scene.remove(group));
        this.tensorMeshes.clear();
        this.pickMeshes.length = 0;
        this.pendingTensorDataRequests.clear();
        this.previewSelectedCells.clear();
        this.emitSelectionPreview();
        this.clearSelection(false);
        this.tensors.clear();
        this.state.activeTensorId = null;
        this.state.hover = null;
        this.state.lastHover = null;
        this.hoverOutline.visible = false;
        this.hoverOutline2D.visible = false;
        this.lastHoverLogKey = null;
    }

        /**
 * Switches the viewer between the flat 2D canvas workflow and the 3D orbit workflow.
 *
 * The method preserves the current orbit target, recreates the camera controls for the selected
 * camera, updates the flat canvas/overlay visibility, and synchronizes the 2D camera when entering
 * 2D mode.
 *
 * @param mode - Display pipeline to activate: `'2d'` shows the flat canvas with the orthographic camera, while `'3d'` uses the perspective camera and repositions it around the preserved controls target.
 * @returns Nothing; the viewer camera, controls, display-mode state, and backing DOM element visibility are updated in place.
 * @noThrows The accepted mode is restricted to `'2d'` or `'3d'`, and this method only copies existing viewer state, recreates controls, and updates DOM style fields; it performs no parsing or tensor lookup itself.
 * @example
 * viewer.applyDisplayMode('2d');
 * expect(viewer.state.displayMode).toBe('2d');
 * expect(viewer.flatCanvas.style.display).toBe('block');
 * expect(viewer.flatOverlay.style.display).toBe('none');
 *
 * viewer.applyDisplayMode('3d');
 * expect(viewer.state.displayMode).toBe('3d');
 * expect(viewer.flatCanvas.style.display).toBe('none');
 */
private applyDisplayMode(mode: '2d' | '3d'): void {
        const previousTarget = this.controls.target.clone();
        this.state.displayMode = mode;
        if (!this.selectionEnabled()) this.clearSelection(false);
        this.controls.dispose();
        this.camera = mode === '2d' ? this.orthographicCamera : this.perspectiveCamera;
        this.controls = this.createControls(this.camera);
        this.syncInteractionMode();
        this.renderer.domElement.style.display = 'block';
        this.flatCanvas.style.display = mode === '2d' ? 'block' : 'none';
        this.flatOverlay.style.display = 'none';
        this.controls.target.copy(previousTarget);
        if (mode === '3d') {
            // keep the same orbit target when switching modes so the user lands on
            // the tensor they were already inspecting.
            this.camera.position.set(previousTarget.x + 20, previousTarget.y + 16, previousTarget.z + 24);
            this.camera.lookAt(previousTarget);
            return;
        }
        this.sync2DCamera();
    }

        /**
 * Parses a tensor-view expression for a tensor and stores the normalized view on that tensor record.
 *
 * The parser validates the expression against the tensor shape, existing or supplied hidden-axis
 * indices, and optional axis labels before the tensor's current view is replaced.
 *
 * @param tensor - Tensor record whose shape, axis labels, and current hidden indices define the valid view grammar for this assignment.
 * @param spec - Tensor-view expression to parse, such as a grouping, projection, or slice expression accepted by `parseTensorView`.
 * @param hiddenIndices - Optional axis indices to hide while parsing `spec`; when omitted, the tensor's current `view.hiddenIndices` are reused.
 * @returns Snapshot of the assigned view containing the normalized editor model and a copy of the hidden-axis indices for callers that need to update UI state.
 * @throws Error when `spec` cannot be parsed for `tensor.shape`, references invalid axes or labels, or `hiddenIndices` contains indices that are not valid for the tensor rank.
 * @example
 * const tensor = viewer.requireTensor('attention');
 * const snapshot = viewer.assignTensorView(tensor, 'batch,head | token', [3]);
 * expect(tensor.view.editor).toEqual(snapshot.editor);
 * expect(snapshot.hiddenIndices).toEqual([3]);
 *
 * expect(() => viewer.assignTensorView(tensor, 'missing_axis')).toThrow(Error);
 */
private assignTensorView(tensor: TensorRecord, spec: string, hiddenIndices?: number[]): TensorViewSnapshot {
        const parsed = parseTensorView(tensor.shape, spec, hiddenIndices ?? tensor.view.hiddenIndices, tensor.axisLabels);
        if (!parsed.ok) throw new Error(parsed.errors.join(' '));
        tensor.view = parsed.spec;
        return {
            editor: tensor.view.editor,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        };
    }

        /**
 * Enforces the inspector rule that only one tensor keeps active slice tokens at a time.
 *
 * Tensors other than `activeTensorId` have their slice clauses removed from the stored tensor-view
 * editor, while the active tensor's current sliced view is left unchanged.
 *
 * @param activeTensorId - Identifier of the tensor whose current slice tokens should be preserved.
 * @returns Nothing; non-active tensor records with slice tokens are rewritten in place to their unsliced view expressions.
 * @throws Error when a non-active tensor's cleared view expression cannot be parsed back against that tensor's shape or hidden-axis state.
 * @example
 * viewer.assignTensorView(viewer.requireTensor('query'), 'batch | token=3');
 * viewer.assignTensorView(viewer.requireTensor('key'), 'batch | token=5');
 *
 * viewer.clearSliceStateFromOtherTensors('key');
 *
 * expect(viewer.requireTensor('key').view.sliceTokens).toHaveLength(1);
 * expect(viewer.requireTensor('query').view.sliceTokens).toHaveLength(0);
 */
private clearSliceStateFromOtherTensors(activeTensorId: string): void {
        this.tensors.forEach((tensor) => {
            if (tensor.id === activeTensorId || tensor.view.sliceTokens.length === 0) return;
            this.assignTensorView(tensor, serializeTensorViewEditor(clearTensorViewSlices(tensor.view.editor)));
        });
    }

        /**
 * Restores saved viewer state from a snapshot without rebuilding meshes.
 *
 * The snapshot drives global display flags, display mode, camera or 2D pan/zoom state, tensor
 * offsets, tensor-view editors, hidden-axis selections, and the active tensor fallback used by the
 * inspector panels.
 *
 * @param snapshot - Saved viewer snapshot produced by the viewer, including display settings, camera state, tensor entries, tensor-view editor snapshots, and optional legacy fields.
 * @returns Nothing; viewer state, camera controls, canvas pan/zoom, tensor offsets, tensor views, and active tensor selection are updated in place.
 * @throws Error when a tensor entry in the snapshot contains a view editor or hidden-axis list that cannot be parsed for the matching tensor in the current session.
 * @example
 * viewer.applySnapshot({
 *   displayMode: '2d',
 *   camera: { position: [10, -5, 0], target: [0, 0, 0], rotation: [0, 0, 0], zoom: 2 },
 *   tensors: [{ id: 'logits', view: savedLogitsView, offset: [0, 0, 0] }],
 *   activeTensorId: 'missing-tensor',
 *   heatmap: 'magma',
 *   showDimensionLines: true,
 *   showInspectorPanel: true,
 *   showHoverDetailsPanel: false,
 * });
 * expect(viewer.state.displayMode).toBe('2d');
 * expect(viewer.state.activeTensorId).toBe('logits');
 * expect(viewer.canvasZoom).toBe(2);
 *
 * const invalidSnapshot = { ...snapshot, tensors: [{ ...snapshot.tensors[0], view: invalidView }] };
 * expect(() => viewer.applySnapshot(invalidSnapshot)).toThrow(Error);
 */
private applySnapshot(snapshot: ViewerSnapshot): void {
        this.clearSelection(false);
        this.state.heatmap = snapshot.heatmap;
        this.state.dimensionBlockGapMultiple = snapshot.dimensionBlockGapMultiple ?? DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
        this.state.displayGaps = snapshot.displayGaps ?? false;
        this.state.logScale = snapshot.logScale ?? false;
        this.state.collapseHiddenAxes = snapshot.collapseHiddenAxes ?? snapshot.showSlicesInSamePlace ?? false;
        this.state.dimensionMappingScheme = snapshot.dimensionMappingScheme ?? 'z-order';
        this.state.showDimensionLines = snapshot.showDimensionLines;
        this.state.showTensorNames = snapshot.showTensorNames ?? true;
        this.state.showInspectorPanel = snapshot.showInspectorPanel;
        this.state.showSelectionPanel = snapshot.showSelectionPanel ?? true;
        this.state.showHoverDetailsPanel = snapshot.showHoverDetailsPanel;
        this.state.activeTensorId = snapshot.activeTensorId;
        this.state.interactionMode = snapshot.interactionMode ?? (snapshot.displayMode === '3d' ? 'rotate' : 'pan');
        this.applyDisplayMode(snapshot.displayMode);

        if (snapshot.displayMode === '2d') {
            // snapshots store camera position in world coordinates, while the 2d
            // renderer drives pan in canvas pixels.
            this.canvasZoom = normalizeCanvasZoom(snapshot.camera.zoom);
            this.canvasPan = {
                x: -snapshot.camera.position[0] * CANVAS_WORLD_SCALE * this.canvasZoom,
                y: snapshot.camera.position[1] * CANVAS_WORLD_SCALE * this.canvasZoom,
            };
            this.sync2DCamera();
        } else {
            this.camera.position.copy(vectorFromTuple(snapshot.camera.position));
            this.controls.target.copy(vectorFromTuple(snapshot.camera.target));
            this.camera.rotation.set(...snapshot.camera.rotation);
            if ('zoom' in this.camera) this.camera.zoom = snapshot.camera.zoom;
            this.camera.updateProjectionMatrix();
        }

        snapshot.tensors.forEach((entry) => {
            const tensor = this.tensors.get(entry.id);
            if (!tensor) return;
            if (entry.offset) tensor.offset = entry.offset;
            this.assignTensorView(tensor, serializeTensorViewEditor(entry.view.editor), entry.view.hiddenIndices);
        });
        let slicedTensorId: string | null = null;
        // old snapshots may contain multiple sliced tensors; the current inspector
        // model allows one active slice set, so the last saved sliced tensor wins.
        for (let index = snapshot.tensors.length - 1; index >= 0; index -= 1) {
            const tensorId = snapshot.tensors[index]!.id;
            const tensor = this.tensors.get(tensorId);
            if ((tensor?.view.sliceTokens.length ?? 0) === 0) continue;
            slicedTensorId = tensorId;
            break;
        }
        if (slicedTensorId) this.clearSliceStateFromOtherTensors(slicedTensorId);

        if (!this.state.activeTensorId || !this.tensors.has(this.state.activeTensorId)) {
            this.state.activeTensorId = snapshot.tensors[0]?.id ?? this.tensors.keys().next().value ?? null;
        }
    }

        /**
 * Detects legacy snapshots whose camera was serialized as the viewer's untouched default pose.
 *
 * Generated demos used to save the default camera even when no fit had occurred; those snapshots
 * should be treated as missing camera state so bundle loading can fit the scene to the tensors.
 *
 * @param snapshot - Validated viewer snapshot whose camera zoom, position, target, and rotation are compared against the default pose.
 * @returns True when the snapshot camera is exactly zoom 1, position [0, 0, 30], target [0, 0, 0], and rotation [0, 0, 0]; otherwise false so the saved camera is preserved.
 * @noThrows Only reads numeric camera fields from a normalized ViewerSnapshot supplied by bundle validation.
 * @example
 * const snapshot = {
 *   camera: {
 *     zoom: 1,
 *     position: [0, 0, 30],
 *     target: [0, 0, 0],
 *     rotation: [0, 0, 0],
 *   },
 * } as ViewerSnapshot;
 *
 * expect(viewer.shouldAutoFitSnapshot(snapshot)).toBe(true);
 */
private shouldAutoFitSnapshot(snapshot: ViewerSnapshot): boolean {
        // generated demos historically wrote the default camera even when the view
        // had never been fitted, so treat that exact pose as "no camera saved".
        return snapshot.camera.zoom === 1
            && snapshot.camera.position[0] === 0
            && snapshot.camera.position[1] === 0
            && snapshot.camera.position[2] === 30
            && snapshot.camera.target[0] === 0
            && snapshot.camera.target[1] === 0
            && snapshot.camera.target[2] === 0
            && snapshot.camera.rotation[0] === 0
            && snapshot.camera.rotation[1] === 0
            && snapshot.camera.rotation[2] === 0;
    }

        /**
 * Applies the latest pointer hit to the viewer hover model and refreshes dependent UI state.
 *
 * The method keeps the 2D/3D hover outline in sync, records the most recent non-null hover,
 * logs hover changes without flooding repeated pointer moves, requests a render, and emits the
 * public hover notification when the hovered tensor cell changes.
 *
 * @param hover - Tensor cell hit information from canvas or scene hit testing, or null when the pointer leaves or misses all cells.
 * @param source - Interaction surface that produced the hit; used in hover log events as either "2d:hover" or "3d:hover".
 * @param outlinePosition - Optional world-space position for the hover outline when the hit came from rendered geometry.
 * @returns Nothing; mutates this.state.hover/lastHover and schedules rendering or hover emission when the visible hover changes.
 * @noThrows Expects internally produced HoverInfo and Vector3 values; the method only compares them, updates viewer state, and delegates rendering/event notifications.
 * @example
 * const hover = {
 *   tensorId: 'activation',
 *   tensorCoord: [2, 1],
 *   value: 0.75,
 * } as HoverInfo;
 *
 * viewer.updateHover(hover, '2d');
 *
 * expect(viewer.state.hover).toEqual(hover);
 * expect(viewer.state.lastHover).toEqual(hover);
 */
private updateHover(hover: HoverInfo | null, source: '2d' | '3d', outlinePosition?: Vector3): void {
        const outlineChanged = this.syncHoverOutline(hover, source, outlinePosition);
        if (this.sameHover(this.state.hover, hover)) {
            if (outlineChanged) this.requestRender();
            return;
        }
        this.state.hover = hover;
        if (hover) this.state.lastHover = hover;
        const hoverKey = hover ? `${hover.tensorId}:${hover.tensorCoord.join(',')}:${hover.value}` : null;
        // logging every pointermove floods tests and devtools, but value changes on
        // the same coord should still be visible after async tensor data arrives.
        if (hoverKey !== this.lastHoverLogKey) {
            this.lastHoverLogKey = hoverKey;
            logEvent(`${source}:hover`, hover ?? 'none');
        }
        this.requestRender();
        this.emitHover();
    }

    /**
 * Builds the public status object that describes one tensor without exposing its dense data buffer.
 *
 * @param tensor - Internal tensor record containing the stable id/name/shape metadata and the current dense-data flags to publish.
 * @returns TensorStatus returned from tensor APIs and data-request callbacks, including rank, shape, axis labels, dtype, hasData, and valueRange.
 * @noThrows Copies fields from an already-normalized TensorRecord and derives rank from tensor.shape.length without validation or I/O.
 * @example
 * const status = viewer.tensorStatus({
 *   id: 'weights',
 *   name: 'Layer weights',
 *   shape: [2, 3],
 *   axisLabels: [['row0', 'row1'], ['c0', 'c1', 'c2']],
 *   dtype: 'float32',
 *   data: new Float32Array(6),
 *   hasData: true,
 *   valueRange: [-1, 1],
 * } as TensorRecord);
 *
 * expect(status).toEqual({
 *   id: 'weights',
 *   name: 'Layer weights',
 *   rank: 2,
 *   shape: [2, 3],
 *   axisLabels: [['row0', 'row1'], ['c0', 'c1', 'c2']],
 *   dtype: 'float32',
 *   hasData: true,
 *   valueRange: [-1, 1],
 * });
 */
    private tensorStatus(tensor: TensorRecord): TensorStatus {
        return {
            id: tensor.id,
            name: tensor.name,
            rank: tensor.shape.length,
            shape: tensor.shape,
            axisLabels: tensor.axisLabels,
            dtype: tensor.dtype,
            hasData: tensor.hasData,
            valueRange: tensor.valueRange,
        };
    }

    /**
 * Stores a tensor's dense numeric payload or marks the tensor as metadata-only while preserving its identity and shape.
 *
 * @param tensor - Existing tensor record whose dtype, data, hasData, and valueRange fields will be updated in place.
 * @param data - Dense numeric array containing exactly the bytes required by tensor.shape and dtype, or null to clear the payload.
 * @param dtype - Element type to store on the tensor and use when validating data.byteLength; defaults to the tensor's current dtype.
 * @returns Nothing; mutates the tensor record with the new payload state and recomputed min/max range.
 * @throws Error when a non-null data buffer's byteLength is incompatible with the tensor shape and dtype expected by validateTensorPayload.
 * @example
 * const tensor = { shape: [2], dtype: 'float32', data: null, hasData: false, valueRange: null } as TensorRecord;
 *
 * viewer.assignTensorData(tensor, new Float32Array([3, -1]), 'float32');
 *
 * expect(tensor.hasData).toBe(true);
 * expect(tensor.valueRange).toEqual([-1, 3]);
 *
 * @example
 * const tensor = { shape: [2], dtype: 'float32', data: null, hasData: false, valueRange: null } as TensorRecord;
 *
 * expect(() => viewer.assignTensorData(tensor, new Float32Array([1]), 'float32')).toThrow(Error);
 */
    private assignTensorData(tensor: TensorRecord, data: NumericArray | null, dtype: DType = tensor.dtype): void {
        if (data) validateTensorPayload(dtype, tensor.shape, data.byteLength);
        tensor.dtype = dtype;
        tensor.data = data;
        tensor.hasData = data !== null;
        tensor.valueRange = data ? computeMinMax(data) : null;
    }

    /**
 * Store a numeric tensor in the viewer and rebuild the scene so its cells are rendered immediately.
 *
 * @param shape - Positive integer dimensions for the tensor, in axis order; the value count in `data` must match this shape.
 * @param data - Typed numeric array containing the tensor's cell values in row-major order.
 * @param name - Optional label shown for the tensor; when omitted the viewer assigns a default name such as `Tensor 1`.
 * @param offset - Optional `[x, y, z]` scene position for this tensor; when omitted the viewer chooses an automatic offset.
 * @param dtype - Optional element type to record for the tensor; when omitted it is inferred from the numeric array.
 * @returns A handle describing the inserted tensor, including its generated id, display metadata, and current view status for later viewer operations.
 * @throws Error when `shape` is not a valid tensor shape, when `data` is incompatible with the requested shape or dtype, or when the generated default view cannot be parsed.
 * @example
 * const handle = viewer.addTensor([2, 2], new Float32Array([1, 2, 3, 4]), 'Weights');
 * console.assert(handle.name === 'Weights');
 * console.assert(handle.shape.join('x') === '2x2');
 *
 * @example
 * try {
 *   viewer.addTensor([2, 2], new Float32Array([1, 2]), 'Incomplete weights');
 * } catch (error) {
 *   console.assert(error instanceof Error);
 * }
 */
    public addTensor(shape: number[], data: NumericArray, name?: string, offset?: Vec3, dtype?: DType): TensorHandle {
        logEvent('tensor:add', { shape, name, offset, dtype });
        return this.insertTensor(shape, data, {
            name,
            offset,
            dtype,
        });
    }

    /**
 * Register a tensor shape and dtype without loading cell values, then rebuild the scene using metadata-only rendering.
 *
 * @param shape - Positive integer dimensions for the tensor whose data bytes are not available in the viewer.
 * @param dtype - Element type to display in tensor metadata and use when data is loaded later by session workflows.
 * @param name - Optional label shown for the tensor; when omitted the viewer assigns a default name such as `Tensor 1`.
 * @param offset - Optional `[x, y, z]` scene position for this tensor; when omitted the viewer chooses an automatic offset.
 * @param axisLabels - Optional labels for the tensor axes, in the same order as `shape`, used by the generated tensor view.
 * @returns A handle for the inserted metadata tensor, including its id, shape, dtype, axis labels, and data-availability status.
 * @throws Error when `shape` is not a valid tensor shape, when axis labels cannot be applied to the generated view, or when the generated default view cannot be parsed.
 * @example
 * const handle = viewer.addMetadataTensor([64, 128], 'float32', 'Activation map', undefined, ['batch', 'feature']);
 * console.assert(handle.name === 'Activation map');
 * console.assert(handle.hasData === false);
 *
 * @example
 * try {
 *   viewer.addMetadataTensor([], 'float32', 'Invalid metadata tensor');
 * } catch (error) {
 *   console.assert(error instanceof Error);
 * }
 */
    public addMetadataTensor(shape: number[], dtype: DType, name?: string, offset?: Vec3, axisLabels?: string[]): TensorHandle {
        logEvent('tensor:add-metadata', { shape, dtype, name, offset, axisLabels });
        return this.insertTensor(shape, null, {
            name,
            offset,
            dtype,
            axisLabels,
        });
    }

        /**
 * Normalize tensor metadata, attach optional numeric data, store the record, and either rebuild meshes or emit a state update.
 *
 * @param shape - Positive integer dimensions that define the tensor rank and cell count.
 * @param data - Typed numeric cell payload for a data-backed tensor, or `null` to create a metadata-only tensor.
 * @param options - Insertion controls for restoring ids, assigning display names, setting scene offsets, applying axis labels, choosing display mode, and suppressing rebuild or event emission during bulk loads.
 * @returns The viewer status handle for the stored tensor, including the id that callers use to select, update, or remove it later.
 * @throws Error when shape validation fails, when tensor data cannot be assigned to the normalized shape and dtype, or when the generated tensor-view specification cannot be parsed.
 * @example
 * const handle = this.insertTensor([2, 3], new Float32Array([0, 1, 2, 3, 4, 5]), {
 *   id: 'layer-0',
 *   name: 'Layer 0 activations',
 *   axisLabels: ['row', 'column'],
 * });
 * console.assert(handle.id === 'layer-0');
 * console.assert(handle.hasData === true);
 *
 * @example
 * try {
 *   this.insertTensor([2, 3], new Float32Array([1, 2]), { name: 'truncated payload' });
 * } catch (error) {
 *   console.assert(error instanceof Error);
 * }
 */
private insertTensor(
        shape: number[],
        data: NumericArray | null,
        options: {
            id?: string;
            name?: string;
            offset?: Vec3;
            dtype?: DType;
            axisLabels?: string[];
            displayMode?: '2d' | '3d';
            rebuild?: boolean;
            emit?: boolean;
        } = {},
    ): TensorHandle {
        const normalizedShape = validateTensorShape(shape);
        const parsed = parseTensorView(
            normalizedShape,
            serializeTensorViewEditor(defaultTensorViewEditor(normalizedShape, options.axisLabels)),
        );
        if (!parsed.ok) throw new Error(parsed.errors.join(' '));
        const id = options.id ?? (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `tensor-${this.tensorCounter += 1}`);
        const dtype = options.dtype ?? (data ? dtypeFromArray(data) : 'float32');
        const tensor: TensorRecord = {
            id,
            name: options.name ?? `Tensor ${this.tensors.size + 1}`,
            shape: normalizedShape,
            axisLabels: parsed.spec.axisLabels.slice(),
            dtype,
            data: null,
            hasData: false,
            valueRange: null,
            offset: [0, 0, 0],
            view: parsed.spec,
            customColors: new Map(),
            markerCoords: null,
            visibleCoords: null,
            cellLabels: null,
            ghostLayers: null,
            autoOffset: options.offset === undefined,
        };
        this.assignTensorData(tensor, data, dtype);
        tensor.offset = options.offset ?? this.autoTensorOffset(tensor, options.displayMode ?? this.state.displayMode);
        this.tensors.set(id, tensor);
        this.state.activeTensorId ??= id;
        if (options.rebuild === false) {
            if (options.emit !== false) this.emit();
        } else {
            this.relayoutTensorOffsets(options.displayMode ?? this.state.displayMode);
            this.rebuildAllMeshes({ fitCamera: true });
        }
        return this.tensorStatus(tensor);
    }

    /**
 * Delete a tensor from the viewer and clear any active, hover, selection, preview, or drag state that references it before rebuilding the scene.
 *
 * @param tensorId - Identifier returned in a tensor handle from `addTensor`, `addMetadataTensor`, or session loading.
 * @returns Nothing; the tensor map and related viewer state are updated in place.
 * @noThrows Missing tensor ids are treated as already removed: `Map.delete` and the related selection cleanup paths are idempotent for absent entries.
 * @example
 * const handle = viewer.addTensor([1], new Float32Array([42]), 'Temporary');
 * viewer.removeTensor(handle.id);
 * console.assert(viewer.getTensor(handle.id) === null);
 *
 * @example
 * viewer.removeTensor('tensor-that-was-never-added');
 * // No exception is thrown for an unknown id.
 */
    public removeTensor(tensorId: string): void {
        logEvent('tensor:remove', tensorId);
        this.tensors.delete(tensorId);
        const previewChanged = this.previewSelectedCells.delete(tensorId);
        const selectionChanged = this.selectedCells.delete(tensorId)
            || (this.selectionDrag?.baseSelections.delete(tensorId) ?? false)
            || (this.selectionDrag?.previewSelections.delete(tensorId) ?? false);
        if (this.selectionDrag?.tensorId === tensorId) {
            this.selectionDrag = null;
            this.emitSelectionPreview();
            this.syncSelectionBox();
        }
        if (selectionChanged) this.emitSelection();
        else if (previewChanged) {
            this.emitSelectionPreview();
            this.refreshSelectionVisuals(tensorId);
        }
        if (this.state.activeTensorId === tensorId) {
            this.state.activeTensorId = this.tensors.keys().next().value ?? null;
        }
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
    }

    /**
 * Unload all tensors from the viewer and notify subscribers that hover, selection, and active-tensor state are empty.
 *
 * @returns Nothing; the viewer mutates its in-memory scene and emits the cleared snapshot to listeners.
 * @noThrows Clearing only removes existing viewer-owned meshes, maps, selections, and hover records; it does not validate caller data or require a tensor to be loaded.
 * @example
 * const snapshots: ViewerSnapshot[] = [];
 * viewer.subscribe((snapshot) => snapshots.push(snapshot));
 * viewer.clear();
 *
 * const snapshot = snapshots.at(-1)!;
 * expect(snapshot.tensors).toEqual([]);
 * expect(snapshot.activeTensorId).toBeNull();
 */
    public clear(): void {
        logEvent('tensor:clear');
        this.resetLoadedState();
        this.requestRender();
        this.emitHover();
        this.emit();
    }

    /**
 * Capture the viewer configuration, camera, tensor metadata, tensor-view editors, offsets, and active tensor id in a serializable object.
 *
 * @returns A `ViewerSnapshot` that hosts can persist, clone into a new tab, pass to extensions, or later apply to restore the same viewer state.
 * @noThrows Snapshot capture reads normalized viewer state already owned by the instance and copies arrays before returning them; it performs no parsing, loading, or caller-input validation.
 * @example
 * const snapshot = viewer.getSnapshot();
 *
 * expect(snapshot.version).toBe(1);
 * expect(snapshot.camera.position).toHaveLength(3);
 * expect(snapshot.tensors[0]).toMatchObject({
 *   id: 'weights',
 *   name: 'weights',
 *   view: { hiddenIndices: [] },
 * });
 */
    public getSnapshot(): ViewerSnapshot {
        return {
            version: 1,
            displayMode: this.state.displayMode,
            interactionMode: this.state.interactionMode,
            heatmap: this.state.heatmap,
            dimensionBlockGapMultiple: this.state.dimensionBlockGapMultiple,
            displayGaps: this.state.displayGaps,
            logScale: this.state.logScale,
            collapseHiddenAxes: this.state.collapseHiddenAxes,
            dimensionMappingScheme: this.state.dimensionMappingScheme,
            showDimensionLines: this.state.showDimensionLines,
            showTensorNames: this.state.showTensorNames,
            showInspectorPanel: this.state.showInspectorPanel,
            showSelectionPanel: this.state.showSelectionPanel,
            showHoverDetailsPanel: this.state.showHoverDetailsPanel,
            camera: {
                position: tupleFromVector(this.camera.position),
                target: tupleFromVector(this.controls.target),
                rotation: tupleFromVector(new Vector3(this.camera.rotation.x, this.camera.rotation.y, this.camera.rotation.z)),
                zoom: 'zoom' in this.camera ? this.camera.zoom : 1,
            },
            tensors: Array.from(this.tensors.values()).map((tensor) => ({
                id: tensor.id,
                name: tensor.name,
                offset: tensor.offset,
                view: {
                    editor: tensor.view.editor,
                    hiddenIndices: tensor.view.hiddenIndices.slice(),
                },
            })),
            activeTensorId: this.state.activeTensorId,
        };
    }

    /**
 * Render the active viewport and serialize it as an SVG document suitable for download or copying into reports.
 *
 * @returns A complete SVG document string. In 2D mode it contains vector primitives for cells, outlines, labels, and guides; in 3D mode it embeds the rendered canvas as a PNG image inside an `<image>` element.
 * @noThrows Export uses the viewer's existing renderer, camera, canvases, and tensor state; it does not accept external input or reject empty scenes.
 * @example
 * viewer.setDisplayMode('3d');
 * const svg = viewer.exportCurrentViewSvg();
 *
 * expect(svg).toContain('<svg xmlns="http://www.w3.org/2000/svg"');
 * expect(svg).toContain('<image href="data:image/png;base64,');
 */
    public exportCurrentViewSvg(): string {
        if (this.state.displayMode === '2d') {
            this.render2D();
            return this.exportCurrentViewSvg2D();
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
        this.syncSelectionBox();
        const canvas = this.renderer.domElement;
        const width = canvas.width;
        const height = canvas.height;
        const imageHref = canvas.toDataURL('image/png');
        return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <image href="${imageHref}" width="${width}" height="${height}" preserveAspectRatio="none" />
</svg>`;
    }

    /**
 * Serialize the 2D canvas view as SVG primitives instead of a raster screenshot.
 *
 * @returns A complete SVG document string whose dimensions match the flat canvas and whose elements describe visible tensor cells, outlines, optional dimension guides, tensor names, hover details, and selection overlays.
 * @noThrows The method derives SVG markup from normalized tensors and the existing 2D canvas projection; it performs no caller-input parsing and represents an empty viewer as an empty SVG body.
 * @example
 * viewer.setDisplayMode('2d');
 * const svg = viewer.exportCurrentViewSvg();
 *
 * expect(svg).toContain('<svg xmlns="http://www.w3.org/2000/svg"');
 * expect(svg).toContain('<rect');
 * expect(svg).not.toContain('<image href="data:image/png');
 */
    private exportCurrentViewSvg2D(): string {
        const width = this.flatCanvas.width;
        const height = this.flatCanvas.height;
        const worldScale = CANVAS_WORLD_SCALE * this.canvasZoom;
        const lineWidth = Math.max(1.5, worldScale * 0.06);
        const outlineColor = '#334155';
        const parts: string[] = [];

        this.tensors.forEach((tensor) => {
            const shape = this.layoutShape(tensor.view);
            const instanceShape = this.instanceShape(tensor.view);
            const labels = this.layoutAxisLabels(tensor.view);
            const extent = displayExtent2D(shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
            const outlineSpan = Math.max(extent.x, extent.y);
            const baseOutlineLabelScale2D = Math.max(1.25, Math.min(10, outlineSpan * 0.05));
            const guideLabelScale2D = baseOutlineLabelScale2D / 5;
            const guideStartOffset2D = Math.max(1.15, guideLabelScale2D * 2.5);
            const guideLevelStep2D = Math.max(0.75, guideLabelScale2D * 3.5);
            const guideLabelOffset2D = Math.max(0.3, guideLabelScale2D * 1.2);
            const tensorNameScale2D = (baseOutlineLabelScale2D * 1.25) / 2;
            const count = product(instanceShape);
            const heatmapRange = this.state.heatmap && tensor.hasData ? tensor.valueRange : null;

            for (let index = 0; index < count; index += 1) {
                const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
                const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
                const layoutCoord = this.mapViewCoordToLayoutCoord(viewCoord, tensor.view);
                const position = displayPositionForCoord2D(layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
                const topLeft = this.projectCanvasPoint(tensor.offset[0] + position.x - 0.5, tensor.offset[1] + position.y + 0.5);
                const bottomRight = this.projectCanvasPoint(tensor.offset[0] + position.x + 0.5, tensor.offset[1] + position.y - 0.5);
                const x = Math.min(topLeft.x, bottomRight.x);
                const y = Math.min(topLeft.y, bottomRight.y);
                const rectWidth = Math.abs(bottomRight.x - topLeft.x);
                const rectHeight = Math.abs(bottomRight.y - topLeft.y);
                const value = tensor.hasData ? numericValue(tensor.data, this.linearIndex(tensorCoord, tensor.shape)) : 0;
                const color = this.cellColor(tensor, tensorCoord, value, heatmapRange);
                parts.push(`<rect x="${x}" y="${y}" width="${rectWidth}" height="${rectHeight}" fill="${this.svgColor(color)}" />`);
            }

            const outlineTopLeft = this.projectCanvasPoint(tensor.offset[0] - extent.x / 2, tensor.offset[1] + extent.y / 2);
            const outlineBottomRight = this.projectCanvasPoint(tensor.offset[0] + extent.x / 2, tensor.offset[1] - extent.y / 2);
            parts.push(
                `<rect x="${Math.min(outlineTopLeft.x, outlineBottomRight.x)}" y="${Math.min(outlineTopLeft.y, outlineBottomRight.y)}" width="${Math.abs(outlineBottomRight.x - outlineTopLeft.x)}" height="${Math.abs(outlineBottomRight.y - outlineTopLeft.y)}" fill="none" stroke="${outlineColor}" stroke-width="${lineWidth}" />`,
            );

            if (this.state.showDimensionLines && labels.length > 0) {
                const rank = shape.length;
                const families = new Map<number, number[]>();
                for (let axis = 0; axis < rank; axis += 1) {
                    const key = axisWorldKeyForMode('2d', rank, axis, this.state.dimensionMappingScheme) as 0 | 1;
                    const family = families.get(key) ?? [];
                    family.push(axis);
                    families.set(key, family);
                }
                shape.forEach((size, axis) => {
                    const familyKey = axisWorldKeyForMode('2d', rank, axis, this.state.dimensionMappingScheme) as 0 | 1;
                    const family = families.get(familyKey) ?? [axis];
                    const familyPos = Math.max(0, family.indexOf(axis));
                    const start = new Array(rank).fill(0);
                    const end = start.slice();
                    family.forEach((familyAxis) => {
                        if (familyAxis >= axis) end[familyAxis] = Math.max(0, shape[familyAxis] - 1);
                    });
                    const startPos = displayPositionForCoord2D(start, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
                    const endPos = displayPositionForCoord2D(end, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
                    const delta = { x: endPos.x - startPos.x, y: endPos.y - startPos.y };
                    const length = Math.hypot(delta.x, delta.y) || 1;
                    const axisDir = { x: delta.x / length, y: delta.y / length };
                    const extentStart = { x: tensor.offset[0] + startPos.x - axisDir.x * 0.5, y: tensor.offset[1] + startPos.y - axisDir.y * 0.5 };
                    const extentEnd = { x: tensor.offset[0] + endPos.x + axisDir.x * 0.5, y: tensor.offset[1] + endPos.y + axisDir.y * 0.5 };
                    const color = axisFamilyColor(familyKey as 0 | 1 | 2, familyPos, family.length);
                    const dir = familyKey === 0 ? { x: 0, y: 1 } : { x: -1, y: 0 };
                    const reverseIndex = family.length - 1 - familyPos;
                    const worldOffset = guideStartOffset2D + reverseIndex * guideLevelStep2D;
                    const startGuide = { x: extentStart.x + dir.x * worldOffset, y: extentStart.y + dir.y * worldOffset };
                    const endGuide = { x: extentEnd.x + dir.x * worldOffset, y: extentEnd.y + dir.y * worldOffset };
                    const projectedPoints = [extentStart, startGuide, endGuide, extentEnd].map((point) => this.projectCanvasPoint(point.x, point.y));
                    parts.push(
                        `<path d="M ${projectedPoints[0].x} ${projectedPoints[0].y} L ${projectedPoints[1].x} ${projectedPoints[1].y} L ${projectedPoints[2].x} ${projectedPoints[2].y} L ${projectedPoints[3].x} ${projectedPoints[3].y}" fill="none" stroke="${color}" stroke-width="${lineWidth}" stroke-linecap="round" stroke-linejoin="round" />`,
                    );
                    const labelPoint = this.projectCanvasPoint(
                        (startGuide.x + endGuide.x) / 2 + dir.x * guideLabelOffset2D,
                        (startGuide.y + endGuide.y) / 2 + dir.y * guideLabelOffset2D,
                    );
                    parts.push(
                        `<text x="${labelPoint.x}" y="${labelPoint.y}" fill="${color}" font-family="IBM Plex Sans, Segoe UI, sans-serif" font-size="${guideLabelScale2D * worldScale}" font-weight="700" text-anchor="middle" dominant-baseline="middle">${this.svgEscape(`${labels[axis] ?? 'X'}: ${size}`)}</text>`,
                    );
                });
            }

            if (this.state.showTensorNames) {
                const tensorName = tensor.name || tensor.id;
                const topGuideCount = shape.reduce((countByAxis, _size, axis) => (
                    countByAxis + Number(axisWorldKeyForMode('2d', shape.length, axis, this.state.dimensionMappingScheme) === 0)
                ), 0);
                const fittedTensorNameScale2D = this.fittedSvgFontSize(
                    tensorName,
                    tensorNameScale2D * worldScale,
                    Math.max(1, extent.x * worldScale * 0.95),
                ) / worldScale;
                const guideClearance = this.state.showDimensionLines && labels.length > 0
                    ? guideStartOffset2D + Math.max(0, topGuideCount - 1) * guideLevelStep2D + guideLabelOffset2D + fittedTensorNameScale2D * 1.5
                    : fittedTensorNameScale2D * 1.75;
                const namePoint = this.projectCanvasPoint(tensor.offset[0], tensor.offset[1] + extent.y / 2 + guideClearance);
                parts.push(
                    `<text x="${namePoint.x}" y="${namePoint.y}" fill="#0f172a" font-family="IBM Plex Sans, Segoe UI, sans-serif" font-size="${fittedTensorNameScale2D * worldScale}" font-weight="700" text-anchor="middle" dominant-baseline="middle">${this.svgEscape(tensorName)}</text>`,
                );
            }
        });

        if (this.flatOverlay.style.display !== 'none' && this.selectionBox.getAttribute('display') !== 'none') {
            const x = this.selectionBox.getAttribute('x') ?? '0';
            const y = this.selectionBox.getAttribute('y') ?? '0';
            const rectWidth = this.selectionBox.getAttribute('width') ?? '0';
            const rectHeight = this.selectionBox.getAttribute('height') ?? '0';
            const fill = this.selectionBox.getAttribute('fill') ?? '#1976d220';
            const stroke = this.selectionBox.getAttribute('stroke') ?? '#1976d2';
            const strokeWidth = this.selectionBox.getAttribute('stroke-width') ?? '2';
            const dash = this.selectionBox.getAttribute('stroke-dasharray');
            parts.push(
                `<rect x="${x}" y="${y}" width="${rectWidth}" height="${rectHeight}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"${dash ? ` stroke-dasharray="${dash}"` : ''} />`,
            );
        }

        return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  ${parts.join('\n  ')}
</svg>`;
    }

    /**
 * Convert a normalized Three.js `Color` into the CSS color literal embedded in exported SVG rectangles.
 *
 * @param color - Three.js color whose `r`, `g`, and `b` channels are normalized floats in the 0..1 range.
 * @returns An `rgb(red, green, blue)` string with each channel rounded to an 8-bit 0..255 integer for SVG `fill` attributes.
 * @noThrows Reads numeric channel fields and formats a string without validating input, allocating graphics resources, or calling browser APIs.
 * @example
 * const color = new Color(0.1, 0.5, 1);
 * viewer['svgColor'](color);
 * // => 'rgb(26, 128, 255)'
 */
    private svgColor(color: Color): string {
        return `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
    }

    /**
 * Escape tensor names and guide labels before inserting them as SVG text-node content.
 *
 * @param value - Raw label text that may contain XML-sensitive characters such as `&`, `<`, `>`, double quotes, or apostrophes.
 * @returns The label with those characters replaced by SVG/XML entities so the text is displayed literally instead of being parsed as markup.
 * @noThrows Performs deterministic string replacements on the provided string and does not parse XML or access external state.
 * @example
 * viewer['svgEscape']('Q<K>&"bias"');
 * // => 'Q&lt;K&gt;&amp;&quot;bias&quot;'
 */
    private svgEscape(value: string): string {
        return value
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&apos;');
    }

    /**
 * Choose the SVG font size for a label so it does not exceed the horizontal space available in the 2D export.
 *
 * @param text - Tensor name or axis label that will be measured with the viewer's bold SVG label font.
 * @param baseSize - Preferred font size in CSS pixels before fitting.
 * @param maxWidth - Maximum allowed rendered label width in CSS pixels.
 * @returns The original `baseSize` when the measured text fits, otherwise a proportional smaller font size that fits within `maxWidth`.
 * @noThrows Updates the existing canvas text-measurement context and uses `measureText`; empty or zero-width measurements fall back to `baseSize`.
 * @example
 * viewer['flatContext'].measureText = () => ({ width: 200 }) as TextMetrics;
 * viewer['fittedSvgFontSize']('attention_scores', 20, 100);
 * // => 10
 */
    private fittedSvgFontSize(text: string, baseSize: number, maxWidth: number): number {
        this.flatContext.font = `700 ${baseSize}px "IBM Plex Sans", "Segoe UI", sans-serif`;
        const measuredWidth = this.flatContext.measureText(text).width;
        return measuredWidth > 0 ? Math.min(baseSize, (maxWidth / measuredWidth) * baseSize) : baseSize;
    }

    /**
 * Replace the viewer's current state with a captured snapshot and rebuild the rendered tensor meshes to match it.
 *
 * @param snapshot - Snapshot previously produced for this viewer session, including the saved camera, view, selection, and rendering state consumed by `applySnapshot`.
 * @returns Nothing; callers observe the restored viewer state and refreshed meshes after the method completes.
 * @noThrows Applies a snapshot already shaped as `ViewerSnapshot`; invalid tensor-view data is handled by snapshot application rather than by this wrapper.
 * @example
 * const snapshot = viewer.snapshot();
 * viewer.setHeatmapEnabled(false);
 * viewer.restoreSnapshot(snapshot);
 * expect(viewer.snapshot()).toEqual(snapshot);
 */
    public restoreSnapshot(snapshot: ViewerSnapshot): void {
        logEvent('snapshot:restore', snapshot);
        this.applySnapshot(snapshot);
        this.rebuildAllMeshes();
    }

    /**
 * Register a viewer-state observer and synchronously deliver the current snapshot before future updates.
 *
 * @param listener - Callback that receives the current `ViewerSnapshot` immediately and then each snapshot emitted after viewer state changes.
 * @returns Unsubscribe function that removes this listener from subsequent snapshot notifications.
 * @noThrows Adds/removes callbacks from an in-memory listener set and immediately forwards the current snapshot.
 * @example
 * const snapshots: ViewerSnapshot[] = [];
 * const unsubscribe = viewer.subscribe((snapshot) => snapshots.push(snapshot));
 *
 * // Subscription delivers the current snapshot synchronously.
 * expect(snapshots[0]).toBe(viewer.getSnapshot());
 *
 * unsubscribe();
 * // Later viewer updates no longer call this listener.
 */
    public subscribe(listener: (snapshot: ViewerSnapshot) => void): () => void {
        this.listeners.add(listener);
        listener(this.getSnapshot());
        return () => this.listeners.delete(listener);
    }

    /**
 * Register a hover observer and synchronously deliver the currently hovered tensor item, or `null` when nothing is hovered.
 *
 * @param listener - Callback that receives the current `HoverInfo | null` value immediately and then each hover change produced by pointer movement or hover clearing.
 * @returns Unsubscribe function that removes this listener from subsequent hover notifications.
 * @noThrows Adds/removes callbacks from an in-memory listener set and immediately forwards the current hover value.
 * @example
 * const hoverEvents: Array<HoverInfo | null> = [];
 * const unsubscribe = viewer.subscribeHover((hover) => hoverEvents.push(hover));
 *
 * // A newly registered listener is initialized with the viewer's current hover state.
 * expect(hoverEvents[0]).toBe(viewer.getHover());
 *
 * unsubscribe();
 */
    public subscribeHover(listener: (hover: HoverInfo | null) => void): () => void {
        this.hoverListeners.add(listener);
        listener(this.getHover());
        return () => this.hoverListeners.delete(listener);
    }

    /**
 * Register a committed-selection observer and synchronously deliver the viewer's selected tensor coordinates.
 *
 * @param listener - Callback that receives the current `SelectionCoords` map immediately and then each committed selection change.
 * @returns Unsubscribe function that removes this listener from subsequent committed-selection notifications.
 * @noThrows Adds/removes callbacks from an in-memory listener set and immediately forwards the current committed selection map.
 * @example
 * const selections: SelectionCoords[] = [];
 * const unsubscribe = viewer.subscribeSelection((selection) => selections.push(selection));
 *
 * // The listener starts with the same coordinates returned by the selection accessor.
 * expect(selections[0]).toBe(viewer.getSelectedCoords());
 *
 * unsubscribe();
 */
    public subscribeSelection(listener: (selection: SelectionCoords) => void): () => void {
        this.selectionListeners.add(listener);
        listener(this.getSelectedCoords());
        return () => this.selectionListeners.delete(listener);
    }

    /**
 * Register a drag-preview observer and synchronously deliver the coordinates currently previewed by an in-progress selection drag.
 *
 * @param listener - Callback that receives the current preview `SelectionCoords` map immediately; when no selection drag is active, the initial value is an empty map.
 * @returns Unsubscribe function that removes this listener from subsequent drag-preview selection notifications.
 * @noThrows Adds/removes callbacks from an in-memory listener set and uses an empty map when no drag preview exists.
 * @example
 * const previews: SelectionCoords[] = [];
 * const unsubscribe = viewer.subscribeSelectionPreview((selection) => previews.push(selection));
 *
 * // Without an active drag, subscribers are initialized with no previewed coordinates.
 * expect(previews[0].size).toBe(0);
 *
 * unsubscribe();
 */
    public subscribeSelectionPreview(listener: (selection: SelectionCoords) => void): () => void {
        this.selectionPreviewListeners.add(listener);
        listener(this.selectionDrag ? this.selectionCoords(this.selectionDrag.previewSelections) : new Map());
        return () => this.selectionPreviewListeners.delete(listener);
    }

    /**
 * Changes the viewer between the flat 2D tensor layout and the 3D mesh layout, then rebuilds rendered tensor meshes for that mode.
 *
 * @param mode - Display layout to apply: `'2d'` for the planar grid view or `'3d'` for the depth-rendered view.
 * @returns Nothing. The viewer records the new display mode, recomputes tensor offsets, and rebuilds meshes in place.
 * @noThrows The mode is limited to the TypeScript union accepted by the public API; the method only applies viewer state, layout, logging, and mesh rebuild steps and does not validate additional runtime inputs.
 * @example
 * viewer.setDisplayMode('3d');
 * expect(viewer.getSnapshot().displayMode).toBe('3d');
 *
 * viewer.setDisplayMode('2d');
 * expect(viewer.getSnapshot().displayMode).toBe('2d');
 */
    public setDisplayMode(mode: '2d' | '3d'): void {
        logEvent('display:mode', mode);
        this.applyDisplayMode(mode);
        this.relayoutTensorOffsets(mode);
        this.rebuildAllMeshes({ fitCamera: mode === '3d' });
    }

    /**
 * Reads which tool is currently assigned to primary left-drag gestures in the viewer canvas.
 *
 * @returns The active interaction mode, such as `'pan'`, `'select'`, or `'rotate'`, so UI controls can mark the matching tool as active.
 * @noThrows This is a direct read from the viewer state object and performs no parsing, rendering, or external I/O.
 * @example
 * viewer.setInteractionMode('select');
 * expect(viewer.getInteractionMode()).toBe('select');
 */
    public getInteractionMode(): InteractionMode {
        return this.state.interactionMode;
    }

    /**
 * Assigns the tool used for primary left-drag gestures and notifies viewer subscribers so controls can refresh.
 *
 * @param mode - Interaction tool to activate for left-drag input, for example `'pan'`, `'select'`, or `'rotate'`.
 * @returns The interaction mode stored on the viewer after synchronization.
 * @noThrows The mode is constrained by the `InteractionMode` type, and the method only updates in-memory viewer state, synchronizes controls, logs the change, and emits a viewer update.
 * @example
 * const activeMode = viewer.setInteractionMode('rotate');
 *
 * expect(activeMode).toBe('rotate');
 * expect(viewer.getInteractionMode()).toBe('rotate');
 */
    public setInteractionMode(mode: InteractionMode): InteractionMode {
        this.state.interactionMode = mode;
        this.syncInteractionMode();
        logEvent('interaction:mode', this.state.interactionMode);
        this.emit();
        return this.state.interactionMode;
    }

    /**
 * Enables or disables grayscale heatmap coloring for tensor values and rebuilds the rendered meshes to reflect the new color mode.
 *
 * @param force - Optional explicit heatmap state. Pass `true` to enable, `false` to disable, or omit it to invert the current setting.
 * @returns The final heatmap-enabled state after applying the optional override or toggle.
 * @noThrows The method writes an in-memory boolean flag and starts any missing tensor-data loads asynchronously; it has no expected synchronous validation or parsing failure path.
 * @example
 * expect(viewer.toggleHeatmap(true)).toBe(true);
 * expect(viewer.getSnapshot().heatmap).toBe(true);
 *
 * expect(viewer.toggleHeatmap()).toBe(false);
 * expect(viewer.getSnapshot().heatmap).toBe(false);
 */
    public toggleHeatmap(force?: boolean): boolean {
        this.state.heatmap = force ?? !this.state.heatmap;
        logEvent('display:heatmap', this.state.heatmap);
        if (this.state.heatmap) {
            this.tensors.forEach((tensor) => {
                if (!tensor.hasData) void this.ensureTensorData(tensor.id, 'heatmap');
            });
        }
        this.rebuildAllMeshes();
        return this.state.heatmap;
    }

    /**
 * Updates the multiplier used to widen visual gaps between nested dimension blocks, then relayouts and rebuilds the tensor meshes when the stored value changes.
 *
 * @param value - Requested gap multiplier from the UI or host application; finite values are clamped to the inclusive range 1..100, and non-finite values reset to the default multiplier.
 * @returns The multiplier actually stored on the viewer after clamping or defaulting, so callers can synchronize numeric inputs with the accepted value.
 * @noThrows Non-finite and out-of-range numbers are normalized instead of rejected, and the method only updates viewer display state and mesh layout.
 * @example
 * const stored = viewer.setDimensionBlockGapMultiple(250);
 * console.assert(stored === 100);
 * console.assert(viewer.getState().dimensionBlockGapMultiple === 100);
 */
    public setDimensionBlockGapMultiple(value: number): number {
        const nextValue = Number.isFinite(value) ? Math.max(1, Math.min(100, value)) : DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
        if (nextValue === this.state.dimensionBlockGapMultiple) return nextValue;
        this.state.dimensionBlockGapMultiple = nextValue;
        logEvent('display:dimension-block-gap-multiple', nextValue);
        if (this.state.hover) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return nextValue;
    }

    /**
 * Enables or disables rendering of visual gaps between tensor dimension blocks and rebuilds the layout-dependent meshes.
 *
 * @param force - Optional explicit display-gaps state from a checkbox or host control; omit it to invert the current state.
 * @returns The new `displayGaps` state after applying the forced value or toggle, for synchronizing UI controls.
 * @noThrows The optional boolean is assigned directly and the method has no validation branch that rejects caller input.
 * @example
 * const enabled = viewer.toggleDisplayGaps(true);
 * console.assert(enabled === true);
 * console.assert(viewer.getState().displayGaps === true);
 */
    public toggleDisplayGaps(force?: boolean): boolean {
        this.state.displayGaps = force ?? !this.state.displayGaps;
        logEvent('display:gaps', this.state.displayGaps);
        if (this.state.hover) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return this.state.displayGaps;
    }

    /**
 * Shows or hides the dimension-guide line overlays that help identify tensor axes, then rebuilds the rendered meshes.
 *
 * @param force - Optional explicit visibility for dimension-guide overlays; omit it to invert the current overlay visibility.
 * @returns The new `showDimensionLines` state, suitable for updating toolbar button or menu checked state.
 * @noThrows The method only stores a boolean visibility flag and rebuilds meshes; it does not parse input or reject viewer state.
 * @example
 * const visible = viewer.toggleDimensionLines(false);
 * console.assert(visible === false);
 * console.assert(viewer.getState().showDimensionLines === false);
 */
    public toggleDimensionLines(force?: boolean): boolean {
        this.state.showDimensionLines = force ?? !this.state.showDimensionLines;
        logEvent('display:dimension-lines', this.state.showDimensionLines);
        this.rebuildAllMeshes();
        return this.state.showDimensionLines;
    }

    /**
 * Shows or hides tensor-name labels in the rendered viewer and rebuilds meshes so label visibility matches the setting.
 *
 * @param force - Optional explicit tensor-label visibility from a toolbar, menu, or host control; omit it to invert the current label visibility.
 * @returns The new `showTensorNames` state, which callers can use to keep label controls in sync.
 * @noThrows The method only updates a boolean display flag and schedules a mesh rebuild, with no input parsing or validation failure path.
 * @example
 * const visible = viewer.toggleTensorNames(true);
 * console.assert(visible === true);
 * console.assert(viewer.getState().showTensorNames === true);
 */
    public toggleTensorNames(force?: boolean): boolean {
        this.state.showTensorNames = force ?? !this.state.showTensorNames;
        logEvent('display:tensor-names', this.state.showTensorNames);
        this.rebuildAllMeshes();
        return this.state.showTensorNames;
    }

    /**
 * Enable, disable, or invert signed-log heatmap normalization and rebuild the rendered tensor meshes.
 *
 * @param force - Optional target state from a display control: `true` enables signed-log color scaling, `false` restores the linear heatmap, and `undefined` toggles the current setting.
 * @returns The active log-scale state after the update, suitable for writing back to a checkbox or settings snapshot.
 * @noThrows The method only stores the requested boolean state, records a display event, and schedules mesh rebuilding; it performs no input parsing or validation that would intentionally reject the optional flag.
 * @example
 * const enabled = viewer.toggleLogScale(true);
 * expect(enabled).toBe(true);
 * expect(viewer.getSnapshot().logScale).toBe(true);
 *
 * const disabled = viewer.toggleLogScale();
 * expect(disabled).toBe(false);
 */
    public toggleLogScale(force?: boolean): boolean {
        this.state.logScale = force ?? !this.state.logScale;
        logEvent('display:log-scale', this.state.logScale);
        this.rebuildAllMeshes();
        return this.state.logScale;
    }

    /**
 * Enable, disable, or invert rendering sliced/hidden axes at the same display position, then relayout and rebuild the tensor view.
 *
 * @param force - Optional target state from the collapse-hidden-axes control: `true` collapses hidden axes into shared rendered positions, `false` preserves their spacing, and `undefined` toggles the current setting.
 * @returns The active collapse-hidden-axes state after hover cleanup, relayout, and mesh rebuilding, so callers can resynchronize the originating control.
 * @noThrows The method only applies an optional boolean display preference and updates derived layout/rendering state; it has no explicit validation branch for the flag.
 * @example
 * const collapsed = viewer.toggleCollapseHiddenAxes(true);
 * expect(collapsed).toBe(true);
 * expect(viewer.getSnapshot().collapseHiddenAxes).toBe(true);
 *
 * const expanded = viewer.toggleCollapseHiddenAxes(false);
 * expect(expanded).toBe(false);
 */
    public toggleCollapseHiddenAxes(force?: boolean): boolean {
        this.state.collapseHiddenAxes = force ?? !this.state.collapseHiddenAxes;
        logEvent('display:collapse-hidden-axes', this.state.collapseHiddenAxes);
        if (this.state.hover) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return this.state.collapseHiddenAxes;
    }

    /**
 * Legacy name for {@link toggleCollapseHiddenAxes} that keeps older callers controlling the same sliced-axis rendering behavior.
 *
 * @param force - Optional target state passed through to `toggleCollapseHiddenAxes`: `true` renders hidden/sliced axes in the same place, `false` keeps them separated, and `undefined` toggles the current setting.
 * @returns The resulting collapse-hidden-axes state returned by `toggleCollapseHiddenAxes`.
 * @noThrows This alias performs no work other than delegating the optional boolean to `toggleCollapseHiddenAxes`, so it adds no validation or independent throw path.
 * @example
 * const shownInSamePlace = viewer.toggleShowSlicesInSamePlace(true);
 * expect(shownInSamePlace).toBe(true);
 * expect(viewer.getSnapshot().collapseHiddenAxes).toBe(true);
 */
    public toggleShowSlicesInSamePlace(force?: boolean): boolean {
        return this.toggleCollapseHiddenAxes(force);
    }

    /**
 * Select the layout strategy that assigns tensor dimensions to the viewer's x, y, and z axis families.
 *
 * @param value - Requested mapping scheme from a manifest, settings select, shortcut, or command: `'contiguous'` keeps adjacent dimensions together, while any other runtime value is normalized to `'z-order'`.
 * @returns The normalized scheme now stored on the viewer, which callers can write back to controls whose raw value may have been corrected.
 * @noThrows Invalid runtime strings are coerced to `'z-order'` instead of rejected, and the method has no explicit error branch for changing this display preference.
 * @example
 * const selected = viewer.setDimensionMappingScheme('contiguous');
 * expect(selected).toBe('contiguous');
 * expect(viewer.getSnapshot().dimensionMappingScheme).toBe('contiguous');
 *
 * const normalized = viewer.setDimensionMappingScheme('diagonal' as DimensionMappingScheme);
 * expect(normalized).toBe('z-order');
 */
    public setDimensionMappingScheme(value: DimensionMappingScheme): DimensionMappingScheme {
        const nextValue = value === 'contiguous' ? 'contiguous' : 'z-order';
        if (nextValue === this.state.dimensionMappingScheme) return nextValue;
        this.state.dimensionMappingScheme = nextValue;
        if (!this.selectionEnabled()) this.clearSelection(false);
        this.syncInteractionMode();
        logEvent('display:dimension-mapping-scheme', nextValue);
        if (this.state.hover) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return nextValue;
    }

    /**
 * Show, hide, or invert the inspector panel flag used by host UI widgets.
 *
 * @param force - Optional visibility override: pass `true` to show the inspector panel, `false` to hide it, or omit it to invert the current flag.
 * @returns The inspector panel visibility after the update, suitable for synchronizing host UI controls.
 * @noThrows The method only updates the viewer's boolean panel state, records the widget event, emits the state change, and returns the stored flag; it performs no input validation that raises viewer errors.
 * @example
 * const shown = viewer.toggleInspectorPanel(true);
 * expect(shown).toBe(true);
 * expect(viewer.state.showInspectorPanel).toBe(true);
 *
 * const hidden = viewer.toggleInspectorPanel();
 * expect(hidden).toBe(false);
 */
    public toggleInspectorPanel(force?: boolean): boolean {
        this.state.showInspectorPanel = force ?? !this.state.showInspectorPanel;
        logEvent('widget:inspector', this.state.showInspectorPanel);
        this.emit();
        return this.state.showInspectorPanel;
    }

    /**
 * Show, hide, or invert the selection-summary panel flag used by host UI widgets.
 *
 * @param force - Optional visibility override: pass `true` to show the selection panel, `false` to hide it, or omit it to invert the current flag.
 * @returns The selection panel visibility after the update, suitable for keeping menu checkmarks or widget buttons in sync.
 * @noThrows The method only updates the viewer's boolean panel state, records the widget event, emits the state change, and returns the stored flag; it performs no input validation that raises viewer errors.
 * @example
 * const hidden = viewer.toggleSelectionPanel(false);
 * expect(hidden).toBe(false);
 * expect(viewer.state.showSelectionPanel).toBe(false);
 *
 * const shown = viewer.toggleSelectionPanel();
 * expect(shown).toBe(true);
 */
    public toggleSelectionPanel(force?: boolean): boolean {
        this.state.showSelectionPanel = force ?? !this.state.showSelectionPanel;
        logEvent('widget:selection', this.state.showSelectionPanel);
        this.emit();
        return this.state.showSelectionPanel;
    }

    /**
 * Show, hide, or invert the hover-details widget flag used by host UI overlays.
 *
 * @param force - Optional visibility override: pass `true` to show hover details, `false` to hide them, or omit it to invert the current flag.
 * @returns The hover-details visibility after the update, suitable for synchronizing overlay controls.
 * @noThrows The method only updates the viewer's boolean hover-details state, emits the state change, and returns the stored flag; it performs no input validation that raises viewer errors.
 * @example
 * const shown = viewer.toggleHoverDetailsPanel(true);
 * expect(shown).toBe(true);
 * expect(viewer.state.showHoverDetailsPanel).toBe(true);
 *
 * const hidden = viewer.toggleHoverDetailsPanel();
 * expect(hidden).toBe(false);
 */
    public toggleHoverDetailsPanel(force?: boolean): boolean {
        this.state.showHoverDetailsPanel = force ?? !this.state.showHoverDetailsPanel;
        this.emit();
        return this.state.showHoverDetailsPanel;
    }

    /**
 * Compute the rendered world-space extent of a loaded tensor's current view.
 *
 * @param tensorId - Identifier of a tensor already present in the viewer's tensor map, such as an id from the loaded session manifest.
 * @returns A `[width, height, depth]` vector describing the display extent produced from the tensor view's layout shape, layout gap, and active dimension-mapping scheme.
 * @throws Error when `tensorId` does not match any tensor loaded in the viewer; the message is `Unknown tensor ${tensorId}.`.
 * @example
 * const dims = viewer.getViewDims('activations.layer1');
 * expect(dims).toEqual([12, 8, 1]);
 *
 * expect(() => viewer.getViewDims('missing.tensor')).toThrow(
 *   new Error('Unknown tensor missing.tensor.'),
 * );
 */
    public getViewDims(tensorId: string): Vec3 {
        const tensor = this.tensors.get(tensorId);
        if (!tensor) throw new Error(`Unknown tensor ${tensorId}.`);
        const extent = displayExtent(this.layoutShape(tensor.view), this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        return [extent.x, extent.y, extent.z];
    }

    /**
 * Report the manifest metadata and dense-data availability for a tensor in the loaded viewer session.
 *
 * @param tensorId - Id of an existing tensor from the session manifest, such as the id used by hover or selection state.
 * @returns Snapshot of the tensor status, including its shape, dtype, labels/metadata, and whether dense numeric data is currently attached.
 * @throws If `tensorId` does not match any tensor registered in the viewer.
 * @example
 * const status = viewer.getTensorStatus('attention_scores');
 * expect(status.shape).toEqual([2, 8, 128, 128]);
 * expect(status.hasData).toBe(true);
 *
 * expect(() => viewer.getTensorStatus('missing_tensor')).toThrow();
 */
    public getTensorStatus(tensorId: string): TensorStatus {
        return this.tensorStatus(this.requireTensor(tensorId));
    }

    /**
 * Read the tensor-view editor snapshot that controls how one tensor is displayed.
 *
 * @param tensorId - Id of an existing tensor from the loaded session whose view expression should be inspected.
 * @returns The canonical view expression/editor state plus a copied `hiddenIndices` array, so callers can display or edit the view without mutating viewer state directly.
 * @throws If `tensorId` does not match any tensor registered in the viewer.
 * @example
 * const view = viewer.getTensorView('attention_scores');
 * expect(view.editor.canonical).toBe('b,h,q,k');
 * expect(view.hiddenIndices).toEqual([0]);
 *
 * view.hiddenIndices.push(3);
 * expect(viewer.getTensorView('attention_scores').hiddenIndices).toEqual([0]);
 *
 * expect(() => viewer.getTensorView('missing_tensor')).toThrow();
 */
    public getTensorView(tensorId: string): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        return {
            editor: tensor.view.editor,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        };
    }

    /**
 * Parse and apply a tensor-view expression that changes how one tensor is rendered and inspected.
 *
 * This is a virtual view transform. The viewer does not transpose or rewrite
 * the stored tensor buffer; it reparses the axis order and remaps displayed
 * coordinates back to the original tensor coordinates during render, hover,
 * and value lookup.
 *
 * @param tensorId - Id of an existing tensor whose display view should be replaced.
 * @param spec - Tensor-view expression to parse, for example an axis order, grouping, or slicing expression accepted by the viewer grammar.
 * @param hiddenIndices - Optional dimension indices that should be hidden in the rendered view after the new expression is applied.
 * @returns Canonicalized view snapshot stored on the tensor after parsing, including the editor representation and copied hidden indices.
 * @throws If `tensorId` is unknown, `spec` is not valid for the tensor shape/view grammar, or `hiddenIndices` contains indices outside the tensor dimensions.
 * @example
 * const snapshot = viewer.setTensorView('attention_scores', 'b,h,q,k', [0]);
 * expect(snapshot.editor.canonical).toBe('b,h,q,k');
 * expect(snapshot.hiddenIndices).toEqual([0]);
 * expect(viewer.getTensorView('attention_scores').hiddenIndices).toEqual([0]);
 *
 * expect(() => viewer.setTensorView('attention_scores', 'not a valid view')).toThrow();
 */
    public setTensorView(tensorId: string, spec: string, hiddenIndices?: number[]): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        const snapshot = this.assignTensorView(tensor, spec, hiddenIndices);
        if (tensor.view.sliceTokens.length > 0) this.clearSliceStateFromOtherTensors(tensorId);
        logEvent('tensor:view', { tensorId, view: tensor.view.canonical, hiddenIndices: tensor.view.hiddenIndices });
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return snapshot;
    }

    /**
 * Attach or replace the dense numeric payload for an existing tensor while preserving its shape and view expression.
 *
 * @param tensorId - Id of an existing tensor whose dense values should be attached or replaced.
 * @param data - Flat numeric array containing one value for each element in the tensor shape, in the tensor's storage order.
 * @param dtype - Optional dtype to record for the supplied payload; when omitted, the tensor keeps its current dtype metadata.
 * @returns Updated tensor status showing the tensor's dtype, shape, and `hasData` availability after the payload is stored.
 * @throws If `tensorId` is unknown or the supplied dense array is incompatible with the tensor's element count or dtype handling.
 * @example
 * const status = viewer.setTensorData('logits', new Float32Array([0.1, 0.9, 0.3, 0.7]), 'float32');
 * expect(status.hasData).toBe(true);
 * expect(status.dtype).toBe('float32');
 *
 * expect(() => viewer.setTensorData('logits', new Float32Array([1, 2, 3]), 'float32')).toThrow();
 */
    public setTensorData(tensorId: string, data: NumericArray, dtype?: DType): TensorStatus {
        const tensor = this.requireTensor(tensorId);
        this.assignTensorData(tensor, data, dtype ?? tensor.dtype);
        logEvent('tensor:data:set', { tensorId, dtype: tensor.dtype, hasData: tensor.hasData });
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.rebuildAllMeshes();
        return this.tensorStatus(tensor);
    }

    /**
 * Convert a loaded tensor back to a metadata-only tensor while preserving its view configuration and custom colors.
 *
 * The viewer drops the dense numeric payload, clears hover state if it referenced this tensor, rebuilds the meshes,
 * and returns the tensor's updated status so hosts can disable data-dependent UI until data is requested again.
 *
 * @param tensorId - Id of an existing tensor in this viewer whose dense data should be released.
 * @returns Status for the same tensor after clearing; callers can inspect `hasData` to confirm that only metadata remains.
 * @throws If `tensorId` does not identify a tensor registered with this viewer.
 * @example
 * const status = viewer.clearTensorData('activation.0');
 * expect(status.id).toBe('activation.0');
 * expect(status.hasData).toBe(false);
 *
 * expect(() => viewer.clearTensorData('missing-tensor')).toThrow();
 */
    public clearTensorData(tensorId: string): TensorStatus {
        const tensor = this.requireTensor(tensorId);
        this.assignTensorData(tensor, null, tensor.dtype);
        logEvent('tensor:data:clear', tensorId);
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.rebuildAllMeshes();
        return this.tensorStatus(tensor);
    }

    /**
 * Ensure that a tensor has a dense payload, asking the host data requester to hydrate metadata-only tensors when needed.
 *
 * Already-loaded tensors resolve immediately. Metadata-only tensors resolve to `false` when no requester is installed,
 * and concurrent requests for the same tensor share one pending host request.
 *
 * @param tensorId - Id of an existing tensor whose dense data may need to be loaded from the host application.
 * @param reason - Source of the hydration request, such as an explicit user action or viewer features like heatmap rendering.
 * @returns Promise resolving to `true` when the tensor has data after the check, or `false` when no data was available or supplied.
 * @throws If `tensorId` does not identify a tensor registered with this viewer, or rejects if the host requester rejects.
 * @example
 * const hydrated = await viewer.ensureTensorData('activation.0', 'explicit');
 * expect(hydrated).toBe(true);
 * expect(viewer.tensorStatus('activation.0').hasData).toBe(true);
 *
 * await expect(viewer.ensureTensorData('missing-tensor')).rejects.toThrow();
 */
    public async ensureTensorData(tensorId: string, reason: TensorDataRequestReason = 'explicit'): Promise<boolean> {
        const tensor = this.requireTensor(tensorId);
        if (tensor.hasData) return true;
        if (!this.requestTensorDataCallback) return false;
        const pending = this.pendingTensorDataRequests.get(tensorId);
        if (pending) return pending;
        // dedupe concurrent heatmap/hover/explicit requests so the host cannot race
        // two payloads into the same metadata-only tensor.
        const request = Promise.resolve(this.requestTensorDataCallback(this.tensorStatus(tensor), reason))
            .then((data) => {
                const current = this.tensors.get(tensorId);
                if (!current) return false;
                if (!current.hasData && data) this.setTensorData(tensorId, data);
                return this.tensors.get(tensorId)?.hasData ?? false;
            })
            .finally(() => this.pendingTensorDataRequests.delete(tensorId));
        this.pendingTensorDataRequests.set(tensorId, request);
        return request;
    }

        /**
 * Replace a tensor's custom colors with a dense color buffer that maps across the tensor's rendered cells.
 *
 * Use this overload when the host has already computed one color entry per element or visible element, for example
 * an extension-provided heatmap or classification overlay.
 *
 * @param tensorId - Id of the tensor that should receive the dense custom-color buffer.
 * @param colors - Uint8 or float color channel buffer produced by the host for the tensor's cells.
 * @returns Nothing; the viewer stores the color overlay and refreshes the rendered tensor in place.
 * @throws If `tensorId` does not identify a tensor registered with this viewer, or if the dense color buffer is invalid for the tensor.
 * @example
 * const rgb = new Uint8ClampedArray([
 *   255, 0, 0,
 *   0, 255, 0,
 *   0, 0, 255,
 *   255, 255, 0,
 * ]);
 * viewer.colorTensor('activation.0', rgb);
 * // The next render uses the supplied red, green, blue, and yellow overlay for activation.0 cells.
 */
public colorTensor(tensorId: string, colors: Uint8ClampedArray | Float32Array): void;
        /**
 * Paint specific tensor coordinates with one custom color while leaving other cells unchanged.
 *
 * Use this overload for sparse annotations such as selected cells, outliers, or coordinates returned by an analysis tool.
 *
 * @param tensorId - Id of the tensor whose coordinate cells should be colored.
 * @param coords - Tensor index tuples in dimension order; each tuple identifies one cell to paint.
 * @param color - RGB or hue/saturation color applied to every coordinate in `coords`.
 * @returns Nothing; the viewer records the sparse overlay and refreshes the rendered tensor in place.
 * @throws If `tensorId` does not identify a tensor registered with this viewer, or if a coordinate tuple is invalid for the tensor shape.
 * @example
 * viewer.colorTensor('activation.0', [[0, 1], [2, 3]], { r: 255, g: 80, b: 0 });
 * // Cells [0, 1] and [2, 3] in activation.0 render with the orange overlay.
 */
public colorTensor(tensorId: string, coords: number[][], color: RGB | HueSaturation): void;
        /**
 * Paint every cell in a strided rectangular tensor region with one custom color.
 *
 * @param tensorId - Id of the loaded tensor whose custom color overrides should be replaced.
 * @param base - Starting tensor coordinate for the region, with one index per tensor dimension.
 * @param shape - Number of positions to visit along each region dimension.
 * @param jumps - Coordinate increments used to move through the region from `base`.
 * @param color - RGB bytes or hue/saturation value to assign to each coordinate in the region.
 * @returns Nothing; clears the tensor's previous custom colors, stores the region color overrides, logs a tensor color event, and rebuilds the meshes.
 * @throws If `tensorId` does not name a tensor loaded in the viewer.
 * @example
 * ```ts
 * // Colors a 2x3 window of an activation tensor red, starting at row 1, column 0.
 * viewer.colorTensor('activation', [1, 0], [2, 3], [1, 1], { r: 255, g: 0, b: 0 });
 *
 * // Unknown tensor ids are rejected before any color overrides are applied.
 * expect(() => viewer.colorTensor('missing', [0, 0], [1, 1], [1, 1], { r: 255, g: 0, b: 0 })).toThrow();
 * ```
 */
public colorTensor(tensorId: string, base: number[], shape: number[], jumps: number[], color: RGB | HueSaturation): void;
    /**
 * Replace a tensor's custom color overrides from either a dense color buffer, an explicit coordinate list, or a strided rectangular region.
 *
 * @param tensorId - Id of the loaded tensor whose rendered cells should receive custom colors.
 * @param arg1 - Dense per-cell color buffer, coordinate tuples to color, or the base coordinate of a strided region.
 * @param arg2 - For coordinate mode, the RGB or hue/saturation color; for region mode, the region shape.
 * @param arg3 - For region mode, the coordinate jumps between positions; unused for dense and coordinate modes.
 * @param arg4 - For region mode, the RGB or hue/saturation color assigned to all visited coordinates.
 * @returns Nothing; clears existing custom colors for the tensor, applies the requested overrides, records a `tensor:color` event, and rebuilds all meshes.
 * @throws If `tensorId` does not name a tensor loaded in the viewer.
 * @example
 * ```ts
 * // Dense mode: one RGB triplet per rendered tensor cell.
 * viewer.colorTensor('weights', new Uint8ClampedArray([255, 0, 0, 0, 0, 255]));
 *
 * // Coordinate mode: color two tensor cells yellow.
 * viewer.colorTensor('weights', [[0, 1], [2, 3]], { r: 255, g: 255, b: 0 });
 *
 * // Region mode: color a 2x2 block green.
 * viewer.colorTensor('weights', [1, 1], [2, 2], [1, 1], { r: 0, g: 255, b: 0 });
 *
 * expect(() => viewer.colorTensor('missing', new Uint8ClampedArray([255, 0, 0]))).toThrow();
 * ```
 */
    public colorTensor(
        tensorId: string,
        arg1: Uint8ClampedArray | Float32Array | number[][] | number[],
        arg2?: RGB | HueSaturation | number[],
        arg3?: number[] | RGB | HueSaturation,
        arg4?: RGB | HueSaturation,
    ): void {
        const tensor = this.requireTensor(tensorId);
        tensor.customColors.clear();
        this.applyColors(tensor, arg1, arg2, arg3, arg4);
        logEvent('tensor:color', {
            tensorId,
            kind: arg1 instanceof Uint8ClampedArray || arg1 instanceof Float32Array
                ? 'dense'
                : Array.isArray(arg1[0])
                    ? 'coords'
                    : 'region',
        });
        this.rebuildAllMeshes();
    }

    /**
 * Remove every custom color override from a tensor so it renders with its normal color mapping again.
 *
 * @param tensorId - Id of the loaded tensor whose `customColors` map should be emptied.
 * @returns Nothing; clears the tensor's custom color map and rebuilds all meshes to show the default coloring.
 * @throws If `tensorId` does not name a tensor loaded in the viewer.
 * @example
 * ```ts
 * viewer.colorTensor('logits', [[0, 0]], { r: 255, g: 0, b: 0 });
 * viewer.clearTensorColors('logits'); // The highlighted cell returns to the tensor's default palette.
 *
 * expect(() => viewer.clearTensorColors('missing')).toThrow();
 * ```
 */
    public clearTensorColors(tensorId: string): void {
        this.requireTensor(tensorId).customColors.clear();
        this.rebuildAllMeshes();
    }

    /**
 * Limit a tensor to a set of visible tensor coordinates, or restore all coordinates by clearing the mask.
 *
 * @param tensorId - Id of the loaded tensor whose visibility mask should change.
 * @param coords - Coordinate tuples that remain renderable for this tensor; pass `null` to remove the mask and show all coordinates.
 * @returns Nothing; stores the coordinate mask, clears hover state for the tensor when necessary, and rebuilds all meshes.
 * @throws If `tensorId` does not name a tensor loaded in the viewer.
 * @example
 * ```ts
 * // Show only two cells from the activation tensor.
 * viewer.setTensorVisibleCoords('activation', [[0, 0], [0, 1]]);
 *
 * // Restore the tensor's full visible coordinate set.
 * viewer.setTensorVisibleCoords('activation', null);
 *
 * expect(() => viewer.setTensorVisibleCoords('missing', [[0, 0]])).toThrow();
 * ```
 */
    public setTensorVisibleCoords(tensorId: string, coords: number[][] | null): void {
        const tensor = this.requireTensor(tensorId);
        tensor.visibleCoords = coords ? new Set(coords.map((coord) => coordKey(coord))) : null;
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.rebuildAllMeshes();
    }

    /**
 * Sets the text drawn on individual 2D tensor cells, or removes all custom cell labels for that tensor.
 *
 * @param tensorId - Id of a tensor already loaded in this viewer session.
 * @param labels - Per-cell label records keyed by tensor coordinate; each `coord` is converted to the same coordinate key used by rendering, and `text` is drawn on that cell. Pass `null` or an empty array to clear the tensor's cell-label overlay.
 * @returns No value; the tensor's label overlay is replaced and a render is requested.
 * @throws If `tensorId` does not match a tensor in the current viewer session.
 * @example
 * viewer.setTensorCellLabels('activation', [
 *   { coord: [0, 1], text: 'input A' },
 *   { coord: [1, 1], text: 'input B' },
 * ]);
 * // The next render draws "input A" on cell [0, 1] and "input B" on cell [1, 1].
 *
 * viewer.setTensorCellLabels('activation', null);
 * // The next render shows the tensor without custom per-cell text labels.
 *
 * expect(() => viewer.setTensorCellLabels('missing-tensor', [])).toThrow();
 */
    public setTensorCellLabels(tensorId: string, labels: Array<{ coord: number[]; text: string }> | null): void {
        const tensor = this.requireTensor(tensorId);
        tensor.cellLabels = labels?.length
            ? new Map(labels.map(({ coord, text }) => [coordKey(coord), text]))
            : null;
        this.requestRender();
    }

    /**
 * Configures additional 2D ghost copies of tensor cells used to visualize propagated or related coordinates.
 *
 * @param tensorId - Id of a tensor already loaded in this viewer session.
 * @param layers - Ghost-cell records keyed by tensor coordinate. `color` supplies the RGB tint, `bias` offsets the copy in screen space, `layer` controls stacking order, and `text` optionally labels the ghost copy. Pass `null` or an empty array to remove all ghost layers for the tensor.
 * @returns No value; the tensor's ghost-layer overlay is replaced, then the viewer either rebuilds 3D meshes or requests a 2D render.
 * @throws If `tensorId` does not match a tensor in the current viewer session.
 * @example
 * viewer.setTensorGhostLayers('activation', [
 *   { coord: [2, 0], color: [255, 128, 0], bias: [0.18, -0.18], layer: 1, text: 'propagated' },
 * ]);
 * // In 2D mode, the next render shows an orange offset copy of cell [2, 0] labeled "propagated".
 *
 * viewer.setTensorGhostLayers('activation', null);
 * // The tensor renders without ghost copies.
 *
 * expect(() => viewer.setTensorGhostLayers('missing-tensor', [])).toThrow();
 */
    public setTensorGhostLayers(
        tensorId: string,
        layers: Array<{ coord: number[]; color: RGB; bias: readonly [number, number]; layer: number; text?: string | null }> | null,
    ): void {
        const tensor = this.requireTensor(tensorId);
        tensor.ghostLayers = layers?.length
            ? layers.map((layer) => ({
                coord: layer.coord.slice(),
                color: [...layer.color] as RGB,
                bias: [layer.bias[0] ?? 0, layer.bias[1] ?? 0] as const,
                layer: layer.layer,
                text: layer.text ?? null,
            }))
            : null;
        if (this.state.displayMode === '3d') this.rebuildAllMeshes();
        else this.requestRender();
    }

    /**
 * Returns the same immutable viewer snapshot exposed by {@link getSnapshot}.
 *
 * @returns A read-only snapshot of public viewer state, such as the active tensor id, display settings, and other fields that UI integrations inspect without mutating the viewer.
 * @noThrows Reading the snapshot delegates to `getSnapshot`, which clones state from memory and does not validate caller input or require external resources.
 * @example
 * const state = viewer.getState();
 * if (state.activeTensorId) {
 *   console.log(`Inspector should show ${state.activeTensorId}`);
 * }
 */
    public getState(): Readonly<ViewerSnapshot> {
        return this.getSnapshot();
    }

    /**
 * Returns the tensor cell currently under the pointer, falling back to the most recent hover payload immediately after the pointer leaves.
 *
 * @returns A shallow copy of the active or last `HoverInfo` payload for inspector and popup UI, or `null` when the viewer has not recorded a hover target.
 * @noThrows The method only reads in-memory hover fields and returns a copied object or `null`; it does not look up tensors or process caller input.
 * @example
 * const hover = viewer.getHover();
 * if (hover) {
 *   console.log(`Pointer is over tensor ${hover.tensorId}`);
 * } else {
 *   console.log('No tensor cell is hovered.');
 * }
 */
    public getHover(): HoverInfo | null {
        const hover = this.state.hover ?? this.state.lastHover;
        return hover ? { ...hover } : null;
    }

    /**
 * Read the tensor cell that is under the pointer right now.
 *
 * Unlike hover consumers that may keep a previous cell for UI continuity, this method reports only the live hit-test result stored in viewer state.
 *
 * @returns A shallow copy of the current hover cell metadata, or `null` when the pointer is not over a tensor cell.
 * @noThrows Reads an optional state field and clones it when present; it does not perform hit testing, parse coordinates, or notify listeners.
 * @example
 * const hover = viewer.getLiveHover();
 * if (hover) {
 *   console.log(`Pointer is over tensor ${hover.tensorId}.`);
 * } else {
 *   console.log("Pointer is not over a tensor cell.");
 * }
 */
    public getLiveHover(): HoverInfo | null {
        return this.state.hover ? { ...this.state.hover } : null;
    }

    /**
 * Read the committed tensor-cell selection grouped by tensor id.
 *
 * @returns A `SelectionCoords` map whose keys are tensor ids and whose values are the selected coordinates for each tensor; returns an empty map when selection is disabled for the current viewer state.
 * @noThrows Only checks whether selection is enabled and serializes the viewer's in-memory selected-cell keys back to coordinates.
 * @example
 * const selection = viewer.getSelectedCoords();
 * const selectedInWeights = selection.get("weights") ?? [];
 * console.log(`Selected cells in weights: ${selectedInWeights.length}`);
 */
    public getSelectedCoords(): SelectionCoords {
        if (!this.selectionEnabled()) return new Map();
        return this.selectionCoords();
    }

    /**
 * Replace the committed tensor-cell selection used by selection listeners and rendered selection highlights.
 *
 * Empty coordinate arrays are ignored. When selection is disabled, the existing selection is cleared instead of applying the supplied map.
 *
 * @param selection - Map from tensor id to the tensor coordinates that should become selected for that tensor.
 * @param emit - Whether to emit the viewer-level change notification after updating selection visuals and selection listeners.
 * @returns Nothing. The viewer's committed selected cells, active tensor choice, selection preview, selection box, and selection visuals are updated in place.
 * @noThrows The method normalizes caller-provided coordinate arrays into internal keys and updates local viewer state; it does not validate tensor ids against external data or perform asynchronous work.
 * @example
 * const selection = new Map([
 *   ["weights", [[0, 1], [0, 2]]],
 * ]);
 *
 * viewer.setSelectedCoords(selection, false);
 *
 * const selected = viewer.getSelectedCoords().get("weights") ?? [];
 * console.assert(selected.length === 2);
 */
    public setSelectedCoords(selection: SelectionCoords, emit = true): void {
        if (!this.selectionEnabled()) {
            this.clearSelection(false);
            return;
        }
        const nextEntries = new Map<string, Set<string>>();
        selection.forEach((coords, tensorId) => {
            if (coords.length === 0) return;
            nextEntries.set(tensorId, new Set(coords.map((coord) => coordKey(coord))));
        });
        const touched = new Set([...this.selectedCells.keys(), ...nextEntries.keys()]);
        this.selectedCells.clear();
        nextEntries.forEach((coords, tensorId) => this.selectedCells.set(tensorId, coords));
        if (this.selectedCells.size !== 0) {
            this.state.activeTensorId = this.state.activeTensorId && this.selectedCells.has(this.state.activeTensorId)
                ? this.state.activeTensorId
                : this.selectedCells.keys().next().value ?? this.state.activeTensorId;
        }
        this.selectionDrag = null;
        this.emitSelectionPreview();
        this.syncSelectionBox();
        if (touched.size !== 0) this.refreshSelectionVisuals(...touched);
        this.emitSelection();
        if (emit) this.emit();
    }

    /**
 * Replace the temporary tensor-cell highlight overlay without changing the committed selection.
 *
 * This is used for hover or external-preview workflows that need to show prospective selected cells while leaving selection listeners unchanged.
 *
 * @param selection - Map from tensor id to tensor coordinates that should be highlighted as the preview overlay; tensors with no coordinates are omitted.
 * @returns Nothing. The preview-highlight cells are replaced and affected tensor visuals are refreshed in place.
 * @noThrows The method converts the supplied coordinate arrays to internal keys and refreshes existing visuals; it does not parse expressions, load tensor data, or notify selection listeners.
 * @example
 * const committedBefore = viewer.getSelectedCoords();
 *
 * viewer.setPreviewSelectedCoords(new Map([
 *   ["activations", [[3, 4]]],
 * ]));
 *
 * console.assert(viewer.getSelectedCoords() === committedBefore || viewer.getSelectedCoords().size === committedBefore.size);
 */
    public setPreviewSelectedCoords(selection: SelectionCoords): void {
        const nextEntries = new Map<string, Set<string>>();
        selection.forEach((coords, tensorId) => {
            if (coords.length === 0) return;
            nextEntries.set(tensorId, new Set(coords.map((coord) => coordKey(coord))));
        });
        const touched = new Set([...this.previewSelectedCells.keys(), ...nextEntries.keys()]);
        this.previewSelectedCells.clear();
        nextEntries.forEach((coords, tensorId) => this.previewSelectedCells.set(tensorId, coords));
        if (touched.size !== 0) this.refreshSelectionVisuals(...touched);
    }

    /**
 * Summarize the active cell selection for selection widgets and status readouts.
 *
 * @returns Selection totals where `count` is the number of selected coordinates, `availableCount` is the subset whose tensor data is loaded, and `stats` contains min/quartile/mean/std values for loaded numeric cells or `null` when selection is disabled or no selected value is available.
 * @noThrows Reads selection sets and already-loaded tensor buffers without parsing caller input or requiring a tensor to exist; unloaded or removed tensors are skipped.
 * @example
 * // With selection mode disabled, callers can render selection statistics as unavailable.
 * viewer.setInteractionMode('inspect');
 * expect(viewer.getSelectionSummary()).toEqual({ count: 0, availableCount: 0, stats: null });
 */
    public getSelectionSummary(): {
        count: number;
        availableCount: number;
        stats: null | {
            min: number;
            p25: number;
            p50: number;
            p75: number;
            max: number;
            mean: number;
            std: number;
        };
    } {
        if (!this.selectionEnabled()) return { count: 0, availableCount: 0, stats: null };
        const count = this.selectionCount();
        const values: number[] = [];
        this.selectionEntries().forEach((coords, tensorId) => {
            const tensor = this.tensors.get(tensorId);
            if (!tensor?.data) return;
            coords.forEach((key) => {
                values.push(numericValue(tensor.data, this.linearIndex(coordFromKey(key), tensor.shape)));
            });
        });
        if (values.length === 0) return { count, availableCount: 0, stats: null };
        const sorted = values.slice().sort((left, right) => left - right);
        const mean = values.reduce((total, value) => total + value, 0) / values.length;
        const variance = values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
        return {
            count,
            availableCount: values.length,
            stats: {
                min: sorted[0] ?? 0,
                p25: quantile(sorted, 0.25),
                p50: quantile(sorted, 0.5),
                p75: quantile(sorted, 0.75),
                max: sorted[sorted.length - 1] ?? 0,
                mean,
                std: Math.sqrt(variance),
            },
        };
    }

    /**
 * Replace the viewer contents with a normalized session manifest and its decoded tensor bytes.
 *
 * @param manifest - Bundle manifest describing the tensors, their ids/shapes/dtypes, saved viewer snapshot, marker coordinates, and color instructions to restore.
 * @param tensors - Decoded numeric buffers keyed by manifest tensor id; each non-placeholder manifest tensor must have a matching entry.
 * @returns Nothing. The viewer clears previously loaded tensors, inserts the manifest tensors, restores the saved viewer snapshot, rebuilds meshes, and optionally refits the camera.
 * @throws Error when a manifest tensor that is not marked as placeholder data has no decoded buffer in `tensors`.
 * @example
 * const manifest = {
 *   tensors: [{ id: 'logits', name: 'Logits', shape: [2], dtype: 'float32' }],
 *   viewer: { activeTensorId: 'logits' },
 * } as BundleManifest;
 * const tensors = new Map([['logits', new Float32Array([0.25, 0.75])]]);
 *
 * viewer.loadBundleData(manifest, tensors);
 * expect(viewer.getInspectorModel().tensors).toEqual([{ id: 'logits', name: 'Logits' }]);
 *
 * @example
 * const manifest = {
 *   tensors: [{ id: 'weights', name: 'Weights', shape: [2], dtype: 'float32' }],
 *   viewer: {},
 * } as BundleManifest;
 *
 * expect(() => viewer.loadBundleData(manifest, new Map())).toThrow('Session tensor weights is missing bytes.');
 */
    public loadBundleData(manifest: BundleManifest, tensors: Map<string, NumericArray>): void {
        const safeManifest = validateBundleManifest(manifest);
        const shouldFitCamera = this.shouldAutoFitSnapshot(safeManifest.viewer);
        this.resetLoadedState();
        this.state.dimensionBlockGapMultiple = safeManifest.viewer.dimensionBlockGapMultiple ?? DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
        this.state.displayGaps = safeManifest.viewer.displayGaps ?? false;
        this.state.collapseHiddenAxes = safeManifest.viewer.collapseHiddenAxes ?? safeManifest.viewer.showSlicesInSamePlace ?? false;
        this.state.dimensionMappingScheme = safeManifest.viewer.dimensionMappingScheme ?? 'z-order';
        safeManifest.tensors.forEach((entry) => {
            const data = tensors.get(entry.id) ?? null;
            if (!data && !entry.placeholderData) {
                throw new Error(`Session tensor ${entry.id} is missing bytes.`);
            }
            this.insertTensor(entry.shape, data, {
                id: entry.id,
                name: entry.name,
                offset: entry.offset,
                dtype: entry.dtype,
                axisLabels: entry.axisLabels,
                displayMode: safeManifest.viewer.displayMode,
                rebuild: false,
                emit: false,
            });
            const tensor = this.requireTensor(entry.id);
            tensor.markerCoords = entry.markerCoords ? new Set(entry.markerCoords.map((coord) => coordKey(coord))) : null;
            if (entry.colorInstructions?.length) this.applyColorInstructions(entry.id, entry.colorInstructions);
        });
        // restore after insertion because tensor views and active ids reference ids
        // that only exist once the manifest tensors have been created.
        this.applySnapshot(safeManifest.viewer);
        this.rebuildAllMeshes({ fitCamera: shouldFitCamera });
    }

    /**
 * Recompute the camera framing for the currently loaded tensor meshes and publish a viewer update.
 *
 * @returns Nothing. After the call, subscribers receive the emitted viewer state and the camera is fitted to the current viewport and visible tensor bounds.
 * @noThrows Delegates to the viewer's internal camera-fit and emit routines using existing renderer state; callers provide no input that can fail validation.
 * @example
 * const updates: ViewerSnapshot[] = [];
 * viewer.subscribe((snapshot) => updates.push(snapshot));
 *
 * viewer.refitView();
 * expect(updates.length).toBeGreaterThan(0);
 */
    public refitView(): void {
        this.fitCamera();
        this.emit();
    }

    /**
 * Build the tensor-inspector view model for the active tensor and the tensor-view editor controls.
 *
 * @returns Inspector data containing the active tensor handle, selectable tensor names, tensors with color ranges, canonical and preview tensor-view expressions, view/slice tokens, and the active tensor's color range. When no tensor is loaded, `handle` is `null` and editor-specific fields are empty.
 * @noThrows Chooses an existing active tensor id or the first loaded tensor before reading tensor details, so it does not require callers to preselect a valid tensor.
 * @example
 * expect(new TensorViewer().getInspectorModel()).toMatchObject({
 *   handle: null,
 *   tensors: [],
 *   colorRanges: [],
 *   viewInput: '',
 *   preview: '',
 * });
 */
    public getInspectorModel(): {
        handle: TensorHandle | null;
        tensors: Array<{ id: string; name: string }>;
        colorRanges: Array<{ id: string; name: string; min: number; max: number }>;
        viewInput: string;
        preview: string;
        viewEditor: TensorViewSpec['editor'] | null;
        viewTokens: Array<{ kind: 'axis_group' | 'singleton'; token: string; key: string; size: number; sliced: boolean }>;
        sliceTokens: Array<{ token: string; key: string; size: number; value: number }>;
        colorRange: { min: number; max: number } | null;
    } {
        const tensors = Array.from(this.tensors.values()).map((tensor) => ({ id: tensor.id, name: tensor.name }));
        const colorRanges = Array.from(this.tensors.values())
            .filter((tensor) => tensor.valueRange)
            .map((tensor) => ({
                id: tensor.id,
                name: tensor.name,
                ...tensor.valueRange!,
            }));
        const activeTensorId = this.state.activeTensorId && this.tensors.has(this.state.activeTensorId)
            ? this.state.activeTensorId
            : this.tensors.keys().next().value ?? null;
        if (!activeTensorId) {
            return { handle: null, tensors, colorRanges, viewInput: '', preview: '', viewEditor: null, viewTokens: [], sliceTokens: [], colorRange: null };
        }
        const tensor = this.requireTensor(activeTensorId);
        return {
            handle: {
                ...this.tensorStatus(tensor),
            },
            tensors,
            colorRanges,
            viewInput: tensor.view.canonical,
            preview: buildTensorViewExpression(tensor.view),
            viewEditor: tensor.view.editor,
            viewTokens: tensor.view.tokens.map((token) => ({
                kind: token.kind,
                token: token.label,
                key: token.key,
                size: token.size,
                sliced: !token.visible,
            })),
            sliceTokens: tensor.view.sliceTokens.map((token) => ({
                token: token.token,
                key: token.key,
                size: token.size,
                value: token.value,
            })),
            colorRange: tensor.valueRange,
        };
    }

    /**
 * Select an already-loaded tensor as the viewer's active tensor and notify listeners without changing that tensor's data or view expression.
 *
 * @param tensorId - ID of a tensor present in this viewer's tensor registry, such as the value chosen from the tensor selector UI.
 * @returns Nothing; updates the viewer snapshot so `activeTensorId` is `tensorId` and emits a change event.
 * @throws Error when `tensorId` does not identify a tensor loaded in this viewer.
 * @example
 * viewer.setActiveTensor('attention.q_proj');
 * expect(viewer.getSnapshot().activeTensorId).toBe('attention.q_proj');
 *
 * expect(() => viewer.setActiveTensor('missing.tensor')).toThrow();
 */
    public setActiveTensor(tensorId: string): void {
        this.requireTensor(tensorId);
        this.state.activeTensorId = tensorId;
        logEvent('tensor:active', tensorId);
        this.emit();
    }

    /**
 * Change the selected index for one slice token in a tensor view, then return the resulting view snapshot.
 *
 * @param tensorId - ID of the loaded tensor whose view contains the slice token.
 * @param token - Slice token key or rendered token text from `tensor.view.sliceTokens`.
 * @param value - Requested slice index; fractional values are floored and the result is clamped into the token's valid range.
 * @returns Snapshot of the tensor view after the slice update, including the serialized editor state and hidden indices.
 * @throws Error when `tensorId` is not loaded or when `token` does not match any slice token key or token text for that tensor.
 * @example
 * const snapshot = viewer.setSliceTokenValue('logits', 'batch', 2.9);
 * expect(snapshot.editor.sliceValues.batch).toBe(2);
 *
 * const clamped = viewer.setSliceTokenValue('logits', 'batch', 999);
 * expect(clamped.editor.sliceValues.batch).toBeLessThan(batchToken.size);
 *
 * expect(() => viewer.setSliceTokenValue('logits', 'unknown', 0))
 *   .toThrow('Unknown slice token unknown.');
 */
    public setSliceTokenValue(tensorId: string, token: string, value: number): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        const sliceToken = tensor.view.sliceTokens.find((entry) => entry.key === token || entry.token === token);
        if (!sliceToken) throw new Error(`Unknown slice token ${token}.`);
        const clamped = Math.max(0, Math.min(sliceToken.size - 1, Math.floor(value)));
        if (sliceToken.value === clamped) {
            return {
                editor: tensor.view.editor,
                hiddenIndices: tensor.view.hiddenIndices.slice(),
            };
        }
        logEvent('tensor:slice-token', { tensorId, token, value: clamped });
        const editor = tensor.view.editor;
        return this.setTensorView(tensorId, serializeTensorViewEditor({
            ...editor,
            sliceValues: { ...editor.sliceValues, [sliceToken.key]: clamped },
        }));
    }

    /**
 * Update a legacy hidden-dimension token by delegating to {@link setSliceTokenValue}.
 *
 * @param tensorId - ID of the loaded tensor whose view contains the legacy hidden or slice token.
 * @param token - Slice token key or rendered token text accepted by {@link setSliceTokenValue}.
 * @param value - Requested token index; delegated handling floors fractional values and clamps to the token's valid range.
 * @returns Snapshot of the tensor view after applying the delegated slice-token update.
 * @throws Error when `tensorId` is not loaded or `token` is not a slice token for that tensor.
 * @example
 * const snapshot = viewer.setHiddenTokenValue('activations', 'layer', 3);
 * expect(snapshot.editor.sliceValues.layer).toBe(3);
 *
 * expect(() => viewer.setHiddenTokenValue('activations', 'not-a-token', 0))
 *   .toThrow('Unknown slice token not-a-token.');
 */
    public setHiddenTokenValue(tensorId: string, token: string, value: number): TensorViewSnapshot {
        return this.setSliceTokenValue(tensorId, token, value);
    }

    /**
 * Serialize the rendered 2D tensor viewport, including cell colors, labels, markers, axes, and ghost layers, into an SVG image blob.
 *
 * @returns SVG `Blob` that callers can download, copy, or read with `blob.text()` as an XML document for the current 2D viewport.
 * @throws Error with message `SVG export is only available in 2D.` when the viewer is currently in a non-2D display mode.
 * @example
 * const svg = viewer.saveSvg();
 * expect(svg.type).toBe('image/svg+xml');
 * await expect(svg.text()).resolves.toContain('<svg');
 *
 * viewer.setDisplayMode('3d');
 * expect(() => viewer.saveSvg()).toThrow('SVG export is only available in 2D.');
 */
    public saveSvg(): Blob {
        if (this.state.displayMode !== '2d') {
            throw new Error('SVG export is only available in 2D.');
        }
        const parts: string[] = [];
        parts.push(
            '<?xml version="1.0" encoding="UTF-8"?>',
            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${this.flatCanvas.width} ${this.flatCanvas.height}" width="${this.flatCanvas.width}" height="${this.flatCanvas.height}">`,
            `<rect width="${this.flatCanvas.width}" height="${this.flatCanvas.height}" fill="#${this.scene.background instanceof Color ? this.scene.background.getHexString() : 'e5e7eb'}" />`,
        );
        this.tensors.forEach((tensor) => {
            tensor.ghostLayers?.slice().sort((left, right) => right.layer - left.layer).forEach((layer) => {
                const bounds = this.canvasCellBounds(tensor, layer.coord, layer.bias);
                parts.push(
                    `<rect x="${bounds.left}" y="${bounds.top}" width="${Math.max(0, bounds.right - bounds.left)}" height="${Math.max(0, bounds.bottom - bounds.top)}" fill="${this.svgColor(colorFromRgb(layer.color))}" />`,
                );
                if (!layer.text) return;
                const lines = layer.text.split('\n').filter(Boolean);
                if (lines.length === 0) return;
                const width = bounds.right - bounds.left;
                const height = bounds.bottom - bounds.top;
                const maxChars = Math.max(...lines.map((line) => line.length), 1);
                const fontSize = Math.floor(Math.min(72, width / Math.max(1.8, maxChars * 0.72), height / Math.max(1.6, lines.length * 1.15)));
                if (fontSize < MIN_SVG_CELL_LABEL_FONT_SIZE) return;
                const lineHeight = Math.max(fontSize, Math.floor(fontSize * 1.05));
                const centerX = (bounds.left + bounds.right) / 2;
                const centerY = (bounds.top + bounds.bottom) / 2;
                const startY = centerY - ((lines.length - 1) * lineHeight) / 2;
                const fill = this.cellLabelColor(tensor, layer.coord);
                const tspans = lines.map((line, lineIndex) => (
                    `<tspan x="${centerX}" y="${startY + (lineIndex * lineHeight)}">${this.escapeSvgText(line)}</tspan>`
                )).join('');
                parts.push(
                    `<text text-anchor="middle" dominant-baseline="middle" font-family="IBM Plex Mono, SFMono-Regular, monospace" font-size="${fontSize}" fill="${fill}">${tspans}</text>`,
                );
            });
            if (tensor.ghostLayers?.length) {
                const heatmapRange = this.state.heatmap ? tensor.valueRange : null;
                const coords = new Set(tensor.ghostLayers.map((layer) => coordKey(layer.coord)));
                coords.forEach((key) => {
                    const tensorCoord = coordFromKey(key);
                    const bounds = this.canvasCellBounds(tensor, tensorCoord);
                    const value = tensor.hasData ? numericValue(tensor.data, this.linearIndex(tensorCoord, tensor.shape)) : 0;
                    const color = this.cellColor(tensor, tensorCoord, value, heatmapRange);
                    parts.push(
                        `<rect x="${bounds.left}" y="${bounds.top}" width="${Math.max(0, bounds.right - bounds.left)}" height="${Math.max(0, bounds.bottom - bounds.top)}" fill="${this.svgColor(color)}" />`,
                    );
                });
            }
            const instanceShape = this.instanceShape(tensor.view);
            const shape = this.layoutShape(tensor.view);
            const labels = this.layoutAxisLabels(tensor.view);
            const count = product(instanceShape);
            const heatmapRange = this.state.heatmap ? tensor.valueRange : null;
            for (let index = 0; index < count; index += 1) {
                const viewCoord = count === 1 && tensor.view.viewShape.length === 0 ? [] : unravelIndex(index, instanceShape);
                const tensorCoord = mapViewCoordToTensorCoord(viewCoord, tensor.view);
                if (!this.tensorCoordVisible(tensor, tensorCoord)) continue;
                const bounds = this.canvasCellBounds(tensor, this.mapViewCoordToLayoutCoord(viewCoord, tensor.view));
                const value = tensor.hasData ? numericValue(tensor.data, this.linearIndex(tensorCoord, tensor.shape)) : 0;
                const color = this.cellColor(tensor, tensorCoord, value, heatmapRange);
                parts.push(
                    `<rect x="${bounds.left}" y="${bounds.top}" width="${Math.max(0, bounds.right - bounds.left)}" height="${Math.max(0, bounds.bottom - bounds.top)}" fill="#${color.getHexString()}" />`,
                );
                if (tensor.markerCoords?.has(coordKey(tensorCoord))) {
                    const outerInset = 1.5;
                    const innerInset = 3;
                    const outerLeft = bounds.left + outerInset;
                    const outerTop = bounds.top + outerInset;
                    const outerWidth = Math.max(0, bounds.right - bounds.left - outerInset * 2);
                    const outerHeight = Math.max(0, bounds.bottom - bounds.top - outerInset * 2);
                    const innerLeft = bounds.left + innerInset;
                    const innerTop = bounds.top + innerInset;
                    const innerWidth = Math.max(0, bounds.right - bounds.left - innerInset * 2);
                    const innerHeight = Math.max(0, bounds.bottom - bounds.top - innerInset * 2);
                    parts.push(
                        `<rect x="${outerLeft}" y="${outerTop}" width="${outerWidth}" height="${outerHeight}" fill="#e5e7eb" stroke="rgba(15, 23, 42, 0.65)" stroke-width="2" />`,
                        `<line x1="${innerLeft}" y1="${innerTop}" x2="${innerLeft + innerWidth}" y2="${innerTop + innerHeight}" stroke="rgba(15, 23, 42, 0.65)" stroke-width="2" />`,
                        `<line x1="${innerLeft + innerWidth}" y1="${innerTop}" x2="${innerLeft}" y2="${innerTop + innerHeight}" stroke="rgba(15, 23, 42, 0.65)" stroke-width="2" />`,
                        `<rect x="${innerLeft}" y="${innerTop}" width="${innerWidth}" height="${innerHeight}" fill="none" stroke="rgba(241, 245, 249, 0.8)" stroke-width="1" />`,
                    );
                }
                const text = tensor.cellLabels?.get(coordKey(tensorCoord));
                if (!text) continue;
                const lines = text.split('\n').filter(Boolean);
                if (lines.length === 0) continue;
                const width = bounds.right - bounds.left;
                const height = bounds.bottom - bounds.top;
                const maxChars = Math.max(...lines.map((line) => line.length), 1);
                const fontSize = Math.floor(Math.min(72, width / Math.max(1.8, maxChars * 0.72), height / Math.max(1.6, lines.length * 1.15)));
                if (fontSize < MIN_SVG_CELL_LABEL_FONT_SIZE) continue;
                const lineHeight = Math.max(fontSize, Math.floor(fontSize * 1.05));
                const centerX = (bounds.left + bounds.right) / 2;
                const centerY = (bounds.top + bounds.bottom) / 2;
                const startY = centerY - ((lines.length - 1) * lineHeight) / 2;
                const fill = this.cellLabelColor(tensor, tensorCoord);
                const tspans = lines.map((line, lineIndex) => (
                    `<tspan x="${centerX}" y="${startY + (lineIndex * lineHeight)}">${this.escapeSvgText(line)}</tspan>`
                )).join('');
                parts.push(
                    `<text text-anchor="middle" dominant-baseline="middle" font-family="IBM Plex Mono, SFMono-Regular, monospace" font-size="${fontSize}" fill="${fill}">${tspans}</text>`,
                );
            }
            const outlineExtent2D = displayExtent2D(shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
            const outlineLeft = tensor.offset[0] - outlineExtent2D.x / 2;
            const outlineTop = tensor.offset[1] + outlineExtent2D.y / 2;
            const outlineBottom = tensor.offset[1] - outlineExtent2D.y / 2;
            const outlineStart = this.projectCanvasPoint(outlineLeft, outlineTop);
            const outlineEnd = this.projectCanvasPoint(tensor.offset[0] + outlineExtent2D.x / 2, outlineBottom);
            parts.push(
                `<rect x="${outlineStart.x}" y="${outlineStart.y}" width="${Math.max(0, outlineEnd.x - outlineStart.x)}" height="${Math.max(0, outlineEnd.y - outlineStart.y)}" fill="none" stroke="#94a3b8" stroke-width="1.5" />`,
            );
            if (this.state.showDimensionLines && labels.length > 0) {
                const guideLabelScale2D = Math.max(1.25, Math.min(10, Math.max(outlineExtent2D.x, outlineExtent2D.y) * 0.05)) / 5;
                const guideStartOffset2D = Math.max(1.15, guideLabelScale2D * 2.5);
                const guideLevelStep2D = Math.max(0.75, guideLabelScale2D * 3.5);
                const guideLabelOffset2D = Math.max(0.3, guideLabelScale2D * 1.2);
                const families = new Map<number, number[]>();
                for (let axis = 0; axis < shape.length; axis += 1) {
                    const key = axisWorldKeyForMode('2d', shape.length, axis, this.state.dimensionMappingScheme) as 0 | 1;
                    const family = families.get(key) ?? [];
                    family.push(axis);
                    families.set(key, family);
                }
                shape.forEach((size, axis) => {
                    const familyKey = axisWorldKeyForMode('2d', shape.length, axis, this.state.dimensionMappingScheme) as 0 | 1;
                    const family = families.get(familyKey) ?? [axis];
                    const familyPos = Math.max(0, family.indexOf(axis));
                    const start = new Array(shape.length).fill(0);
                    const end = start.slice();
                    family.forEach((familyAxis) => {
                        if (familyAxis >= axis) end[familyAxis] = Math.max(0, shape[familyAxis] - 1);
                    });
                    const startPos = displayPositionForCoord2D(start, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
                    const endPos = displayPositionForCoord2D(end, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
                    const delta = { x: endPos.x - startPos.x, y: endPos.y - startPos.y };
                    const length = Math.hypot(delta.x, delta.y) || 1;
                    const axisDir = { x: delta.x / length, y: delta.y / length };
                    const extentStart = {
                        x: tensor.offset[0] + startPos.x - axisDir.x * 0.5,
                        y: tensor.offset[1] + startPos.y - axisDir.y * 0.5,
                    };
                    const extentEnd = {
                        x: tensor.offset[0] + endPos.x + axisDir.x * 0.5,
                        y: tensor.offset[1] + endPos.y + axisDir.y * 0.5,
                    };
                    const dir = familyKey === 0 ? { x: 0, y: 1 } : { x: -1, y: 0 };
                    const reverseIndex = family.length - 1 - familyPos;
                    const worldOffset = guideStartOffset2D + reverseIndex * guideLevelStep2D;
                    const startGuide = { x: extentStart.x + dir.x * worldOffset, y: extentStart.y + dir.y * worldOffset };
                    const endGuide = { x: extentEnd.x + dir.x * worldOffset, y: extentEnd.y + dir.y * worldOffset };
                    const labelPos = {
                        x: (startGuide.x + endGuide.x) / 2 + dir.x * guideLabelOffset2D,
                        y: (startGuide.y + endGuide.y) / 2 + dir.y * guideLabelOffset2D,
                    };
                    const color = axisFamilyColor(familyKey as 0 | 1 | 2, familyPos, family.length);
                    const startCanvas = this.projectCanvasPoint(extentStart.x, extentStart.y);
                    const endCanvas = this.projectCanvasPoint(extentEnd.x, extentEnd.y);
                    const startGuideCanvas = this.projectCanvasPoint(startGuide.x, startGuide.y);
                    const endGuideCanvas = this.projectCanvasPoint(endGuide.x, endGuide.y);
                    const labelCanvas = this.projectCanvasPoint(labelPos.x, labelPos.y);
                    const guideFontSize = Math.max(10, guideLabelScale2D * CANVAS_WORLD_SCALE * this.canvasZoom * 0.8);
                    parts.push(
                        `<line x1="${startCanvas.x}" y1="${startCanvas.y}" x2="${startGuideCanvas.x}" y2="${startGuideCanvas.y}" stroke="${color}" stroke-width="1.5" />`,
                        `<line x1="${endCanvas.x}" y1="${endCanvas.y}" x2="${endGuideCanvas.x}" y2="${endGuideCanvas.y}" stroke="${color}" stroke-width="1.5" />`,
                        `<line x1="${startGuideCanvas.x}" y1="${startGuideCanvas.y}" x2="${endGuideCanvas.x}" y2="${endGuideCanvas.y}" stroke="${color}" stroke-width="1.5" />`,
                    `<text x="${labelCanvas.x}" y="${labelCanvas.y}" text-anchor="middle" dominant-baseline="middle" font-family="Helvetica, Arial, sans-serif" font-size="${guideFontSize}" font-weight="700" fill="${color}">${this.escapeSvgText(`${labels[axis] ?? 'X'}: ${size}`)}</text>`,
                    );
                });
            }
            if (this.state.showTensorNames) {
                const tensorNameScale2D = (Math.max(1.25, Math.min(10, Math.max(outlineExtent2D.x, outlineExtent2D.y) * 0.05)) * 1.25) / 2;
                const guideLabelScale2D = Math.max(1.25, Math.min(10, Math.max(outlineExtent2D.x, outlineExtent2D.y) * 0.05)) / 5;
                const guideStartOffset2D = Math.max(1.15, guideLabelScale2D * 2.5);
                const guideLevelStep2D = Math.max(0.75, guideLabelScale2D * 3.5);
                const guideLabelOffset2D = Math.max(0.3, guideLabelScale2D * 1.2);
                const topGuideCount = shape.reduce((sum, _size, axis) => (
                    sum + Number(axisWorldKeyForMode('2d', shape.length, axis, this.state.dimensionMappingScheme) === 0)
                ), 0);
                const guideClearance = this.state.showDimensionLines && labels.length > 0
                    ? guideStartOffset2D + Math.max(0, topGuideCount - 1) * guideLevelStep2D + guideLabelOffset2D + tensorNameScale2D * 1.5
                    : tensorNameScale2D * 1.75;
                const nameCanvas = this.projectCanvasPoint(tensor.offset[0], tensor.offset[1] + outlineExtent2D.y / 2 + guideClearance);
                const tensorNameFontSize = this.fitTensorNameFontSize(
                    tensor.name,
                    Math.max(12, tensorNameScale2D * CANVAS_WORLD_SCALE * this.canvasZoom),
                    outlineExtent2D,
                );
                parts.push(
                    `<text x="${nameCanvas.x}" y="${nameCanvas.y}" text-anchor="middle" dominant-baseline="middle" font-family='${TENSOR_NAME_FONT_FAMILY}' font-size="${tensorNameFontSize}" font-weight="700" fill="#0f172a">${this.escapeSvgText(tensor.name)}</text>`,
                );
            }
        });
        parts.push('</svg>');
        return new Blob(parts, { type: 'image/svg+xml;charset=utf-8' });
    }

    /**
 * Tears down the viewer instance by unregistering its window listeners, disconnecting resize observation,
 * disposing Three.js controls and renderer resources, and removing the viewer-owned canvases from the
 * container element.
 *
 * @returns Nothing; after this call the instance no longer owns active DOM nodes or browser listeners.
 * @throws DOMException if one of the renderer or overlay elements has already been removed from the container before cleanup runs.
 * @example
 * const viewer = new TensorViewer(container, session);
 * expect(container.contains(viewerElement)).toBe(true);
 *
 * viewer.destroy();
 *
 * expect(container.contains(viewerElement)).toBe(false);
 */
    public destroy(): void {
        window.removeEventListener('resize', this.resize);
        window.removeEventListener('keydown', this.onKeyDown);
        this.resizeObserver?.disconnect();
        this.controls.dispose();
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
        this.container.removeChild(this.flatCanvas);
        this.container.removeChild(this.flatOverlay);
    }

        /**
 * Looks up a tensor record that has been registered in this viewer and fails fast when an event,
 * public API call, or render path references an unknown tensor id.
 *
 * @param tensorId - Tensor identifier from the loaded session manifest or viewer state map.
 * @returns The stored tensor record for that id, including its shape, view state, data status, and rendering metadata.
 * @throws Error when `tensorId` is not present in the viewer's tensor registry; the message is `Unknown tensor ${tensorId}.`.
 * @example
 * const tensor = viewer.requireTensor('activations');
 * expect(tensor.id).toBe('activations');
 *
 * expect(() => viewer.requireTensor('missing')).toThrow('Unknown tensor missing.');
 */
private requireTensor(tensorId: string): TensorRecord {
        const tensor = this.tensors.get(tensorId);
        if (!tensor) throw new Error(`Unknown tensor ${tensorId}.`);
        return tensor;
    }
}
