import {
    Box3,
    BoxGeometry,
    Color,
    EdgesGeometry,
    Group,
    InstancedBufferAttribute,
    InstancedMesh,
    Line,
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
    buildPreviewExpression,
    defaultTensorView,
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
import { axisFamilyColor, createLine, createTextLabel, initializeVertexColors } from './viewer-graphics.js';
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
export type { ViewerOptions } from './viewer-config.js';

/** Imperative tensor viewer that owns its own renderer, cameras, and input handling. */
export class TensorViewer {
    private readonly container: HTMLElement;
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
    private readonly pickMeshes: PickMesh[] = [];
    private readonly resizeObserver?: ResizeObserver;
    private readonly state: ViewerState = {
        displayMode: '2d',
        interactionMode: 'pan',
        heatmap: true,
        dimensionBlockGapMultiple: DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
        displayGaps: true,
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
    private selectionDrag: SelectionDragState | null = null;
    private readonly requestTensorDataCallback?: ViewerOptions['requestTensorData'];
    private readonly pendingTensorDataRequests = new Map<string, Promise<boolean>>();

    /** Create a viewer inside one host container element. */
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

    private normalizeInteractionMode(): void {
        if (this.state.displayMode === '2d') {
            if (this.state.interactionMode === 'rotate') this.state.interactionMode = 'pan';
            if (this.state.interactionMode === 'select' && !this.selectionEnabled()) this.state.interactionMode = 'pan';
            return;
        }
        if (this.state.interactionMode === 'select') this.state.interactionMode = 'pan';
    }

    private syncInteractionMode(): void {
        this.normalizeInteractionMode();
        const leftButton = this.state.displayMode === '3d' && this.state.interactionMode === 'rotate' ? MOUSE.ROTATE : MOUSE.PAN;
        this.controls.enableRotate = this.state.displayMode === '3d';
        this.controls.mouseButtons.LEFT = leftButton;
        this.controls.mouseButtons.RIGHT = MOUSE.PAN;
    }

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
                this.canvasPan.x += (previousWidth - nextWidth) / 2;
                this.canvasPan.y += (previousHeight - nextHeight) / 2;
            }
            this.sync2DCamera();
        }
        logEvent('viewport:resize', { width, height });
        this.requestRender();
    };

    private canvasScale(): number {
        return this.flatCanvas.width / Math.max(1, this.flatCanvas.clientWidth || this.flatCanvas.width || 1);
    }

    private layoutGapMultiple(): number {
        return this.state.displayGaps ? this.state.dimensionBlockGapMultiple : 0;
    }

    private instanceShape(spec: TensorViewSpec): number[] {
        return spec.viewShape.length === 0 ? [1] : spec.viewShape;
    }

    private layoutShape(spec: TensorViewSpec): number[] {
        return layoutShape(spec, this.state.collapseHiddenAxes);
    }

    private mapViewCoordToLayoutCoord(viewCoord: number[], spec: TensorViewSpec): number[] {
        return mapViewCoordToLayoutCoord(viewCoord, spec, this.state.collapseHiddenAxes);
    }

    private mapLayoutCoordToViewCoord(layoutCoord: number[], spec: TensorViewSpec): number[] {
        return mapLayoutCoordToViewCoord(layoutCoord, spec, this.state.collapseHiddenAxes);
    }

    private layoutCoordIsVisible(coord: number[], spec: TensorViewSpec): boolean {
        return layoutCoordIsVisible(coord, spec, this.state.collapseHiddenAxes);
    }

    private layoutAxisLabels(spec: TensorViewSpec): string[] {
        return layoutAxisLabels(spec, this.state.collapseHiddenAxes);
    }

    private meshContext() {
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
            isSelectedCell: (tensorId: string, tensorCoord: number[]) => this.isSelectedCell(tensorId, tensorCoord),
            selectedColor: (color: Color) => this.selectedColor(color),
            linearIndex: (coord: number[], shape: number[]) => this.linearIndex(coord, shape),
            clearHover: () => this.clearHover(),
            requestRender: () => this.requestRender(),
            emit: () => this.emit(),
        };
    }

    private tensorExtentForMode(tensor: TensorRecord, mode: '2d' | '3d'): Vec3 {
        const shape = layoutShape(tensor.view, this.state.collapseHiddenAxes);
        if (mode === '2d') {
            const extent = displayExtent2D(shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
            return [extent.x, extent.y, 0];
        }
        const extent = displayExtent(shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        return [extent.x, extent.y, extent.z];
    }

    private autoTensorOffset(tensor: TensorRecord, mode: '2d' | '3d'): Vec3 {
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
        const widths = tensors.map((tensor) => this.tensorExtentForMode(tensor, mode)[0]);
        const totalWidth = widths.reduce((sum, width) => sum + width, 0) + spacing * (tensors.length - 1);
        let left = previousCenter - totalWidth / 2;
        tensors.forEach((tensor, index) => {
            const width = widths[index] ?? 0;
            tensor.offset = [left + width / 2, 0, 0];
            left += width + spacing;
        });
    }

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

    private heatmapNormalizedValue(value: number, min: number, max: number): number {
        const scaledValue = this.state.logScale ? signedLog1p(value) : value;
        const scaledMin = this.state.logScale ? signedLog1p(min) : min;
        const scaledMax = this.state.logScale ? signedLog1p(max) : max;
        const range = scaledMax - scaledMin || 1;
        return Math.max(0, Math.min(1, (scaledValue - scaledMin) / range));
    }

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

    private canvasPointerToWorld(clientX: number, clientY: number): { x: number; y: number } {
        const rect = this.flatCanvas.getBoundingClientRect();
        const scaleX = this.flatCanvas.width / Math.max(1, rect.width);
        const scaleY = this.flatCanvas.height / Math.max(1, rect.height);
        return {
            x: ((clientX - rect.left) * scaleX - this.flatCanvas.width / 2 - this.canvasPan.x) / (CANVAS_WORLD_SCALE * this.canvasZoom),
            y: -(((clientY - rect.top) * scaleY - this.flatCanvas.height / 2 - this.canvasPan.y) / (CANVAS_WORLD_SCALE * this.canvasZoom)),
        };
    }

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
        candidates.sort((left, right) => {
            const leftPriority = left.tensorId === currentTensorId ? 2 : left.tensorId === activeTensorId ? 1 : 0;
            const rightPriority = right.tensorId === currentTensorId ? 2 : right.tensorId === activeTensorId ? 1 : 0;
            return rightPriority - leftPriority;
        });
        return candidates;
    }

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

    private sameHover(left: HoverInfo | null, right: HoverInfo | null): boolean {
        return !!left
            && !!right
            && left.tensorId === right.tensorId
            && left.tensorCoord.length === right.tensorCoord.length
            && left.tensorCoord.every((value, index) => value === right.tensorCoord[index]);
    }

    private selectionCount(): number {
        let count = 0;
        this.selectionEntries().forEach((entries) => {
            count += entries.size;
        });
        return count;
    }

    private selectionEntries(): Map<string, Set<string>> {
        return this.selectedCells;
    }

    private selectionCoords(entries: Map<string, Set<string>> = this.selectionEntries()): SelectionCoords {
        return new Map(Array.from(entries.entries(), ([tensorId, coords]) => [
            tensorId,
            Array.from(coords, (key) => coordFromKey(key)),
        ]));
    }

    private selectionEnabled(
        displayMode: '2d' | '3d' = this.state.displayMode,
        dimensionMappingScheme: DimensionMappingScheme = this.state.dimensionMappingScheme,
    ): boolean {
        return displayMode === '2d' && dimensionMappingScheme === 'contiguous';
    }

    private isSelectedCell(tensorId: string, tensorCoord: number[]): boolean {
        return this.selectionEntries().get(tensorId)?.has(coordKey(tensorCoord)) ?? false;
    }

    private selectedColor(baseColor: Color): Color {
        return baseColor.clone().lerp(ACTIVE_COLOR, 0.7);
    }

    /** Return one cell's base color before any committed or preview selection tint is applied. */
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

    private cloneSelectionEntries(entries: Map<string, Set<string>>): Map<string, Set<string>> {
        return new Map(Array.from(entries.entries(), ([tensorId, coords]) => [tensorId, new Set(coords)]));
    }

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

    private canvasPixelToWorld(x: number, y: number): { x: number; y: number } {
        return {
            x: (x - this.flatCanvas.width / 2 - this.canvasPan.x) / (CANVAS_WORLD_SCALE * this.canvasZoom),
            y: -((y - this.flatCanvas.height / 2 - this.canvasPan.y) / (CANVAS_WORLD_SCALE * this.canvasZoom)),
        };
    }

    private decodeSelectionIndex(index: number, shape: number[]): number[] {
        return shape.length === 0 ? [] : unravelIndex(index, shape);
    }

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

    private fastBoxSelectionEntries2D(
        tensor: TensorRecord,
        box: { left: number; right: number; top: number; bottom: number },
    ): Set<string> | null {
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

    private projectScenePoint(point: Vector3): { x: number; y: number } | null {
        const projected = point.clone().project(this.camera);
        if (!Number.isFinite(projected.x) || !Number.isFinite(projected.y)) return null;
        return {
            x: ((projected.x + 1) * this.flatCanvas.width) / 2,
            y: ((1 - projected.y) * this.flatCanvas.height) / 2,
        };
    }

    private canvasCellBounds(
        tensor: TensorRecord,
        layoutCoord: number[],
    ): { left: number; right: number; top: number; bottom: number } {
        const shape = this.layoutShape(tensor.view);
        const position = displayPositionForCoord2D(layoutCoord, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        const topLeft = this.projectCanvasPoint(tensor.offset[0] + position.x - 0.5, tensor.offset[1] + position.y + 0.5);
        const bottomRight = this.projectCanvasPoint(tensor.offset[0] + position.x + 0.5, tensor.offset[1] + position.y - 0.5);
        return {
            left: Math.min(topLeft.x, bottomRight.x),
            right: Math.max(topLeft.x, bottomRight.x),
            top: Math.min(topLeft.y, bottomRight.y),
            bottom: Math.max(topLeft.y, bottomRight.y),
        };
    }

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

    private updateSelectionPreview(drag: SelectionDragState): void {
        const selected = this.boxSelectionEntries(drag);
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

    /** Return the shader-side committed-selection attribute for one 2D mesh, if present. */
    private selectionStateAttribute(mesh: InstancedMesh): InstancedBufferAttribute | null {
        const attribute = mesh.geometry.getAttribute('selectionState');
        return attribute instanceof InstancedBufferAttribute ? attribute : null;
    }

    /** Return the live preview uniforms for one 2D mesh, if its material was patched for selection preview. */
    private selectionPreviewUniforms(mesh: InstancedMesh): SelectionPreviewUniforms | null {
        const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
        const uniforms = material?.userData.selectionPreviewUniforms as SelectionPreviewUniforms | undefined;
        return uniforms ?? null;
    }

    /** Patch one 2D instanced material so committed selection and drag-preview tint are computed in the shader. */
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
diffuseColor.rgb = mix(diffuseColor.rgb, selectionColor, 0.7 * selected);`,
            );
        };
        material.needsUpdate = true;
    }

    /** Update live preview uniforms for every 2D tensor mesh from the current drag box. */
    private syncSelectionPreviewState(): void {
        const drag = this.selectionEnabled() ? this.selectionDrag : null;
        const box = drag ? this.selectionBoxBounds(drag) : null;
        const left = box ? this.canvasPixelToWorld(box.left, 0).x : 0;
        const right = box ? this.canvasPixelToWorld(box.right, 0).x : 0;
        const bottom = box ? this.canvasPixelToWorld(0, box.bottom).y : 0;
        const top = box ? this.canvasPixelToWorld(0, box.top).y : 0;
        const mode = !drag ? 0 : drag.mode === 'add' ? 1 : drag.mode === 'remove' ? 2 : 0;
        this.tensorMeshes.forEach((group) => {
            const mesh = group.children[0];
            if (!(mesh instanceof InstancedMesh)) return;
            const uniforms = this.selectionPreviewUniforms(mesh);
            if (!uniforms) return;
            uniforms.selectionPreviewActive.value = drag ? 1 : 0;
            uniforms.selectionPreviewBounds.value.set(left, right, bottom, top);
            uniforms.selectionPreviewMode.value = mode;
        });
    }

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
                : this.isSelectedCell(tensor.id, tensorCoord) ? this.selectedColor(baseColor) : baseColor;
            const offset = index * 3;
            colorArray[offset] = color.r;
            colorArray[offset + 1] = color.g;
            colorArray[offset + 2] = color.b;
            if (selectionState) selectionState[index] = this.isSelectedCell(tensor.id, tensorCoord) ? 1 : 0;
        }
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        if (selectionAttribute) selectionAttribute.needsUpdate = true;
        mesh.material.needsUpdate = true;
    }

    private refreshSelectionVisuals(...tensorIds: string[]): void {
        const ids = new Set(tensorIds);
        ids.forEach((tensorId) => this.refreshTensorColors(tensorId));
        this.requestRender();
    }

    private clearSelection(emit = true): void {
        if (!this.selectionDrag && this.selectedCells.size === 0) return;
        const tensorIds = new Set(this.selectedCells.keys());
        this.selectionDrag?.baseSelections.forEach((_coords, tensorId) => tensorIds.add(tensorId));
        this.selectionDrag?.previewSelections.forEach((_coords, tensorId) => tensorIds.add(tensorId));
        this.selectedCells.clear();
        this.selectionDrag = null;
        this.syncSelectionBox();
        if (tensorIds.size !== 0) this.refreshSelectionVisuals(...tensorIds);
        this.emitSelection();
        if (emit) this.emit();
    }

    private beginSelectionDrag(
        source: '2d' | '3d',
        hover: HoverInfo | null,
        mode: 'replace' | 'add' | 'remove',
        clientX: number,
        clientY: number,
    ): void {
        if (!this.selectionEnabled()) return;
        const startPosition = source === '2d' ? this.canvasPointerToWorld(clientX, clientY) : null;
        this.selectionDrag = {
            source,
            mode,
            tensorId: hover?.tensorId ?? null,
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

    private updateSelectionDrag(
        source: '2d' | '3d',
        clientX: number,
        clientY: number,
    ): void {
        const drag = this.selectionDrag;
        if (!drag || drag.source !== source) return;
        drag.currentClient = { x: clientX, y: clientY };
        this.requestRender();
    }

    private finalizeSelectionDrag(): void {
        const drag = this.selectionDrag;
        if (!drag) return;
        this.updateSelectionPreview(drag);
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

    private overlayPoint(clientX: number, clientY: number): { x: number; y: number } {
        const rect = this.container.getBoundingClientRect();
        const scaleX = this.flatCanvas.width / Math.max(1, rect.width);
        const scaleY = this.flatCanvas.height / Math.max(1, rect.height);
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY,
        };
    }

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

    private hoveredCellContainsPointer(clientX: number, clientY: number, hover: HoverInfo | null): boolean {
        if (!hover) return false;
        return this.state.displayMode === '2d'
            ? this.canvasHoveredCellContainsPointer(clientX, clientY, hover)
            : this.sceneHoveredCellContainsPointer(clientX, clientY, hover);
    }

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

    private projectCanvasPoint(x: number, y: number): { x: number; y: number } {
        return {
            x: this.flatCanvas.width / 2 + this.canvasPan.x + (x * CANVAS_WORLD_SCALE * this.canvasZoom),
            y: this.flatCanvas.height / 2 + this.canvasPan.y - (y * CANVAS_WORLD_SCALE * this.canvasZoom),
        };
    }

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

    private requestRender(): void {
        if (this.renderPending) return;
        this.renderPending = true;
        requestAnimationFrame(() => {
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

    private rebuildAllMeshes(options: { fitCamera?: boolean } = {}): void {
        const shouldFitCamera = options.fitCamera ?? false;
        Array.from(this.tensorMeshes.values()).forEach((group) => this.scene.remove(group));
        this.tensorMeshes.clear();
        this.pickMeshes.length = 0;
        this.tensors.forEach((tensor) => {
            const group = this.buildTensorGroup(tensor);
            this.tensorMeshes.set(tensor.id, group);
            this.scene.add(group);
            const mesh = group.children[0];
            if (mesh instanceof InstancedMesh) {
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

    private populateFastMesh2D(
        tensor: TensorRecord,
        mesh: InstancedMesh,
        instanceShape: number[],
        heatmapRange: { min: number; max: number } | null,
    ): boolean {
        const colorArray = mesh.instanceColor?.array as Float32Array | undefined;
        const selectionAttribute = this.selectionStateAttribute(mesh);
        const selectionState = selectionAttribute?.array as Float32Array | undefined;
        const isIdentityView = this.state.displayMode === '2d'
            && tensor.shape.length <= 2
            && tensor.view.sliceTokens.length === 0
            && tensor.view.tokens.length === tensor.shape.length
            && tensor.view.tokens.every((token, index) => token.kind === 'axis_group'
                && token.visible
                && token.axes.length === 1
                && token.axes[0] === index);
        if (!isIdentityView || tensor.customColors.size !== 0 || !colorArray) return false;

        // default 1d/2d views are affine grids, so write instance buffers directly.
        const matrixArray = mesh.instanceMatrix.array as Float32Array;
        const extent = displayExtent2D(instanceShape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        const rowCount = instanceShape.length > 1 ? instanceShape[0] : 1;
        const columnCount = instanceShape.length > 1 ? instanceShape[1] : (instanceShape[0] ?? 1);
        const selected = this.selectedCells.get(tensor.id);
        const startX = tensor.offset[0] - extent.x / 2 + 0.5;
        const startY = tensor.offset[1] + extent.y / 2 - 0.5;
        const stepX = columnCount > 1 ? (extent.x - 1) / (columnCount - 1) : 0;
        const stepY = rowCount > 1 ? (extent.y - 1) / (rowCount - 1) : 0;
        let cellIndex = 0;

        for (let row = 0; row < rowCount; row += 1) {
            const y = startY - row * stepY;
            for (let column = 0; column < columnCount; column += 1) {
                const x = startX + column * stepX;
                const matrixOffset = cellIndex * 16;
                matrixArray[matrixOffset] = 1;
                matrixArray[matrixOffset + 1] = 0;
                matrixArray[matrixOffset + 2] = 0;
                matrixArray[matrixOffset + 3] = 0;
                matrixArray[matrixOffset + 4] = 0;
                matrixArray[matrixOffset + 5] = 1;
                matrixArray[matrixOffset + 6] = 0;
                matrixArray[matrixOffset + 7] = 0;
                matrixArray[matrixOffset + 8] = 0;
                matrixArray[matrixOffset + 9] = 0;
                matrixArray[matrixOffset + 10] = 1;
                matrixArray[matrixOffset + 11] = 0;
                matrixArray[matrixOffset + 12] = x;
                matrixArray[matrixOffset + 13] = y;
                matrixArray[matrixOffset + 14] = 0;
                matrixArray[matrixOffset + 15] = 1;

                const colorOffset = cellIndex * 3;
                if (heatmapRange) {
                    const gray = this.heatmapNormalizedValue(
                        numericValue(tensor.data, cellIndex),
                        heatmapRange.min,
                        heatmapRange.max,
                    );
                    colorArray[colorOffset] = gray;
                    colorArray[colorOffset + 1] = gray;
                    colorArray[colorOffset + 2] = gray;
                } else {
                    colorArray[colorOffset] = BASE_COLOR.r;
                    colorArray[colorOffset + 1] = BASE_COLOR.g;
                    colorArray[colorOffset + 2] = BASE_COLOR.b;
                }
                if (selectionState) {
                    const tensorCoord = instanceShape.length > 1 ? [row, column] : [column];
                    selectionState[cellIndex] = selected?.has(coordKey(tensorCoord)) ? 1 : 0;
                }
                cellIndex += 1;
            }
        }

        if (selectionAttribute) selectionAttribute.needsUpdate = true;
        return true;
    }

    private buildOutline(extent: Vector3, offset: Vec3): LineSegments {
        const outline = new LineSegments(
            new EdgesGeometry(new BoxGeometry(extent.x + 0.2, extent.y + 0.2, extent.z + 0.2)),
            new LineBasicMaterial({ color: '#334155' }),
        );
        outline.position.copy(vectorFromTuple(offset));
        return outline;
    }

    private buildOutline2D(extent: { x: number; y: number }, offset: Vec3): Line {
        const halfX = extent.x / 2;
        const halfY = extent.y / 2;
        const outline = createLine([
            new Vector3(-halfX, halfY, 0.02),
            new Vector3(halfX, halfY, 0.02),
            new Vector3(halfX, -halfY, 0.02),
            new Vector3(-halfX, -halfY, 0.02),
            new Vector3(-halfX, halfY, 0.02),
        ], '#334155');
        outline.position.copy(vectorFromTuple(offset));
        return outline;
    }

    private buildDimensionGuides2D(
        shape: number[],
        offset: Vec3,
        labels: string[],
        guideOffset: number,
        linearStep: number,
        labelOffset: number,
        labelScale: number,
    ): Group {
        const group = new Group();
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
            const extentStart = new Vector3(offset[0] + startPos.x - axisDir.x * 0.5, offset[1] + startPos.y - axisDir.y * 0.5, 0.02);
            const extentEnd = new Vector3(offset[0] + endPos.x + axisDir.x * 0.5, offset[1] + endPos.y + axisDir.y * 0.5, 0.02);
            const color = axisFamilyColor(familyKey as 0 | 1 | 2, familyPos, family.length);
            const dir = familyKey === 0 ? new Vector3(0, 1, 0) : new Vector3(-1, 0, 0);
            const reverseIndex = family.length - 1 - familyPos;
            const worldOffset = guideOffset + reverseIndex * linearStep;
            const startGuide = extentStart.clone().add(dir.clone().multiplyScalar(worldOffset));
            const endGuide = extentEnd.clone().add(dir.clone().multiplyScalar(worldOffset));
            group.add(createLine([extentStart, startGuide], color));
            group.add(createLine([extentEnd, endGuide], color));
            group.add(createLine([startGuide, endGuide], color));
            const label = createTextLabel(`${labels[axis] ?? 'X'}: ${size}`, color);
            label.position.copy(startGuide.clone().add(endGuide).multiplyScalar(0.5).add(dir.clone().multiplyScalar(labelOffset)));
            label.scale.setScalar(labelScale);
            group.add(label);
        });
        return group;
    }

    private buildDimensionGuides(extent: Vector3, shape: number[], offset: Vec3, labels: string[]): Group {
        const group = new Group();
        const halfX = extent.x / 2;
        const halfY = extent.y / 2;
        const halfZ = extent.z / 2;
        const rank = shape.length;
        const families = new Map<number, number[]>();
        for (let axis = 0; axis < rank; axis += 1) {
            const key = axisWorldKeyForMode('3d', rank, axis, this.state.dimensionMappingScheme);
            const family = families.get(key) ?? [];
            family.push(axis);
            families.set(key, family);
        }
        const entries = shape.map((size, axis) => {
            const worldKey = axisWorldKeyForMode('3d', rank, axis, this.state.dimensionMappingScheme);
            const family = families.get(worldKey) ?? [axis];
            const familyPos = Math.max(0, family.indexOf(axis));
            const start = new Array(rank).fill(0);
            const end = start.slice();
            if (worldKey === 2) {
                family.forEach((familyAxis, index) => {
                    end[familyAxis] = index >= familyPos ? Math.max(0, shape[familyAxis] - 1) : 0;
                });
            } else {
                family.forEach((familyAxis) => {
                    if (familyAxis >= axis) end[familyAxis] = Math.max(0, shape[familyAxis] - 1);
                });
            }
            const startPos = displayPositionForCoord(start, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
            const endPos = displayPositionForCoord(end, shape, this.layoutGapMultiple(), this.state.dimensionMappingScheme);
            const axisDir = endPos.clone().sub(startPos);
            const direction = axisDir.lengthSq() > 1e-9
                ? axisDir.normalize()
                : worldKey === 1
                    ? new Vector3(0, 1, 0)
                    : worldKey === 2
                        ? new Vector3(0, 0, 1)
                        : new Vector3(1, 0, 0);
            return {
                axis,
                worldKey,
                color: axisFamilyColor(worldKey, familyPos, family.length),
                label: `${labels[axis] ?? 'X'}: ${size}`,
                start: startPos.add(direction.clone().multiplyScalar(-0.5)),
                end: endPos.add(direction.clone().multiplyScalar(0.5)),
            };
        });
        const offsetBase = 2.25;
        const linearStep = 1.85;
        ([0, 1, 2] as const).forEach((worldKey) => {
            const familyEntries = entries.filter((entry) => entry.worldKey === worldKey);
            familyEntries.forEach((entry, index) => {
                const reverseIndex = familyEntries.length - 1 - index;
                const midpoint = entry.start.clone().add(entry.end).multiplyScalar(0.5);
                const extensionDirection = worldKey === 0
                    ? new Vector3(0, midpoint.y >= 0 ? 1 : -1, 0)
                    : worldKey === 1
                        ? new Vector3(midpoint.x >= 0 ? 1 : -1, 0, 0)
                        : new Vector3(midpoint.x >= 0 ? 1 : -1, midpoint.y >= 0 ? 1 : -1, 0).normalize();
                const boundaryOffset = worldKey === 0
                    ? Math.max(0, halfY - Math.abs(midpoint.y))
                    : worldKey === 1
                        ? Math.max(0, halfX - Math.abs(midpoint.x))
                        : Math.max(0, Math.max(halfX - Math.abs(midpoint.x), halfY - Math.abs(midpoint.y)));
                const guideOffset = extensionDirection.clone().multiplyScalar(boundaryOffset + offsetBase + reverseIndex * linearStep);
                const startGuide = entry.start.clone().add(guideOffset);
                const endGuide = entry.end.clone().add(guideOffset);
                group.add(createLine([entry.start.clone(), startGuide], entry.color));
                group.add(createLine([entry.end.clone(), endGuide], entry.color));
                group.add(createLine([startGuide, endGuide], entry.color));
                const label = createTextLabel(entry.label, entry.color);
                label.position.copy(startGuide.clone().add(endGuide).multiplyScalar(0.5).add(extensionDirection.clone().multiplyScalar(0.75)));
                label.renderOrder = 10_000;
                group.add(label);
            });
        });

        group.position.copy(vectorFromTuple(offset));
        return group;
    }

    private buildTensorGroup(tensor: TensorRecord): Group {
        return buildTensorGroup(this.meshContext(), tensor);
    }

    /** updates instance colors in place when only the sliced tensor indices changed. */
    private updateSliceMesh(tensor: TensorRecord, previousView: TensorViewSpec): boolean {
        return updateSliceMesh(this.meshContext(), tensor, previousView);
    }

    private render2D(): void {
        this.sync2DCamera();
        this.syncSelectionPreviewState();
        this.renderer.render(this.scene, this.camera);
        this.flatContext.setTransform(1, 0, 0, 1, 0, 0);
        this.flatContext.clearRect(0, 0, this.flatCanvas.width, this.flatCanvas.height);
    }

    private linearIndex(coord: number[], shape: number[]): number {
        let index = 0;
        coord.forEach((value, axis) => {
            index = (index * shape[axis]) + value;
        });
        return index;
    }

    private cellColor(
        tensor: TensorRecord,
        tensorCoord: number[],
        value: number,
        heatmapRange: { min: number; max: number } | null,
    ): Color {
        const color = this.baseCellColor(tensor, tensorCoord, value, heatmapRange);
        return this.isSelectedCell(tensor.id, tensorCoord) ? this.selectedColor(color) : color;
    }

    private applyColorInstructions(tensorId: string, instructions: ColorInstruction[]): void {
        const tensor = this.requireTensor(tensorId);
        tensor.customColors.clear();
        instructions.forEach((instruction) => {
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

    private emit(): void {
        const snapshot = this.getSnapshot();
        this.listeners.forEach((listener) => listener(snapshot));
    }

    private emitHover(): void {
        const hover = this.getHover();
        this.hoverListeners.forEach((listener) => listener(hover));
    }

    private emitSelection(): void {
        const selection = this.getSelectedCoords();
        this.selectionListeners.forEach((listener) => listener(selection));
    }

    private clearHover(): void {
        this.state.hover = null;
        this.state.lastHover = null;
        this.hoverOutline.visible = false;
        this.hoverOutline2D.visible = false;
        this.lastHoverLogKey = null;
        this.emitHover();
    }

    private resetLoadedState(): void {
        Array.from(this.tensorMeshes.values()).forEach((group) => this.scene.remove(group));
        this.tensorMeshes.clear();
        this.pickMeshes.length = 0;
        this.pendingTensorDataRequests.clear();
        this.clearSelection(false);
        this.tensors.clear();
        this.state.activeTensorId = null;
        this.state.hover = null;
        this.state.lastHover = null;
        this.hoverOutline.visible = false;
        this.hoverOutline2D.visible = false;
        this.lastHoverLogKey = null;
    }

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
            this.camera.position.set(previousTarget.x + 20, previousTarget.y + 16, previousTarget.z + 24);
            this.camera.lookAt(previousTarget);
            return;
        }
        this.sync2DCamera();
    }

    private assignTensorView(tensor: TensorRecord, spec: string, hiddenIndices?: number[]): TensorViewSnapshot {
        const parsed = parseTensorView(tensor.shape, spec, hiddenIndices ?? tensor.view.hiddenIndices, tensor.view.axisLabels);
        if (!parsed.ok) throw new Error(parsed.errors.join(' '));
        tensor.view = parsed.spec;
        return {
            view: tensor.view.canonical,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        };
    }

    private applySnapshot(snapshot: ViewerSnapshot): void {
        this.clearSelection(false);
        this.state.heatmap = snapshot.heatmap;
        this.state.dimensionBlockGapMultiple = snapshot.dimensionBlockGapMultiple ?? DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
        this.state.displayGaps = snapshot.displayGaps ?? true;
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
            this.assignTensorView(tensor, entry.view.view ?? entry.view.visible ?? '', entry.view.hiddenIndices);
        });

        if (!this.state.activeTensorId || !this.tensors.has(this.state.activeTensorId)) {
            this.state.activeTensorId = snapshot.tensors[0]?.id ?? this.tensors.keys().next().value ?? null;
        }
    }

    private shouldAutoFitSnapshot(snapshot: ViewerSnapshot): boolean {
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

    private updateHover(hover: HoverInfo | null, source: '2d' | '3d', outlinePosition?: Vector3): void {
        const outlineChanged = this.syncHoverOutline(hover, source, outlinePosition);
        if (this.sameHover(this.state.hover, hover)) {
            if (outlineChanged) this.requestRender();
            return;
        }
        this.state.hover = hover;
        if (hover) this.state.lastHover = hover;
        const hoverKey = hover ? `${hover.tensorId}:${hover.tensorCoord.join(',')}:${hover.value}` : null;
        if (hoverKey !== this.lastHoverLogKey) {
            this.lastHoverLogKey = hoverKey;
            logEvent(`${source}:hover`, hover ?? 'none');
        }
        this.requestRender();
        this.emitHover();
    }

    /** Build the public metadata and dense-data status payload for one tensor. */
    private tensorStatus(tensor: TensorRecord): TensorStatus {
        return {
            id: tensor.id,
            name: tensor.name,
            rank: tensor.shape.length,
            shape: tensor.shape,
            axisLabels: tensor.view.axisLabels,
            dtype: tensor.dtype,
            hasData: tensor.hasData,
            valueRange: tensor.valueRange,
        };
    }

    /** Replace or clear one tensor's dense payload while keeping its metadata stable. */
    private assignTensorData(tensor: TensorRecord, data: NumericArray | null, dtype: DType = tensor.dtype): void {
        if (data && product(tensor.shape) !== data.length) {
            throw new Error(`Tensor data length ${data.length} does not match shape ${tensor.shape.join('x')}.`);
        }
        tensor.dtype = dtype;
        tensor.data = data;
        tensor.hasData = data !== null;
        tensor.valueRange = data ? computeMinMax(data) : null;
    }

    /** Add one tensor and immediately rebuild the rendered scene. */
    public addTensor(shape: number[], data: NumericArray, name?: string, offset?: Vec3, dtype?: DType): TensorHandle {
        logEvent('tensor:add', { shape, name, offset, dtype });
        return this.insertTensor(shape, data, {
            name,
            offset,
            dtype,
        });
    }

    /** Add one metadata-only tensor and render it without loading per-cell numeric values. */
    public addMetadataTensor(shape: number[], dtype: DType, name?: string, offset?: Vec3, axisLabels?: string[]): TensorHandle {
        logEvent('tensor:add-metadata', { shape, dtype, name, offset, axisLabels });
        return this.insertTensor(shape, null, {
            name,
            offset,
            dtype,
            axisLabels,
        });
    }

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
        const normalizedShape = shape.map((dim) => Math.max(1, Math.floor(dim)));
        const parsed = parseTensorView(
            normalizedShape,
            defaultTensorView(normalizedShape, options.axisLabels),
            undefined,
            options.axisLabels,
        );
        if (!parsed.ok) throw new Error(parsed.errors.join(' '));
        const id = options.id ?? (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `tensor-${this.tensorCounter += 1}`);
        const dtype = options.dtype ?? (data ? dtypeFromArray(data) : 'float32');
        const tensor: TensorRecord = {
            id,
            name: options.name ?? `Tensor ${this.tensors.size + 1}`,
            shape: normalizedShape,
            dtype,
            data: null,
            hasData: false,
            valueRange: null,
            offset: [0, 0, 0],
            view: parsed.spec,
            customColors: new Map(),
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

    /** Remove one tensor by id. */
    public removeTensor(tensorId: string): void {
        logEvent('tensor:remove', tensorId);
        this.tensors.delete(tensorId);
        const selectionChanged = this.selectedCells.delete(tensorId)
            || (this.selectionDrag?.baseSelections.delete(tensorId) ?? false)
            || (this.selectionDrag?.previewSelections.delete(tensorId) ?? false);
        if (this.selectionDrag?.tensorId === tensorId) {
            this.selectionDrag = null;
            this.syncSelectionBox();
        }
        if (selectionChanged) this.emitSelection();
        if (this.state.activeTensorId === tensorId) {
            this.state.activeTensorId = this.tensors.keys().next().value ?? null;
        }
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
    }

    /** Remove every loaded tensor and reset hover and active state. */
    public clear(): void {
        logEvent('tensor:clear');
        this.resetLoadedState();
        this.requestRender();
        this.emitHover();
        this.emit();
    }

    /** Capture the full serializable viewer snapshot. */
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
                    view: tensor.view.canonical,
                    hiddenIndices: tensor.view.hiddenIndices.slice(),
                },
            })),
            activeTensorId: this.state.activeTensorId,
        };
    }

    /** Export the current viewport exactly as rendered, wrapped in an SVG file. */
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

    /** Export the current 2D viewport as true SVG primitives. */
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

    /** Convert one three.js color into an SVG rgb() string. */
    private svgColor(color: Color): string {
        return `rgb(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)})`;
    }

    /** Escape text so it is safe to place inside SVG text nodes. */
    private svgEscape(value: string): string {
        return value
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&apos;');
    }

    /** Fit a 2D SVG label font size into the given pixel width. */
    private fittedSvgFontSize(text: string, baseSize: number, maxWidth: number): number {
        this.flatContext.font = `700 ${baseSize}px "IBM Plex Sans", "Segoe UI", sans-serif`;
        const measuredWidth = this.flatContext.measureText(text).width;
        return measuredWidth > 0 ? Math.min(baseSize, (maxWidth / measuredWidth) * baseSize) : baseSize;
    }

    /** Restore a previously captured viewer snapshot. */
    public restoreSnapshot(snapshot: ViewerSnapshot): void {
        logEvent('snapshot:restore', snapshot);
        this.applySnapshot(snapshot);
        this.rebuildAllMeshes();
    }

    /** Subscribe to snapshot updates and immediately receive the current state. */
    public subscribe(listener: (snapshot: ViewerSnapshot) => void): () => void {
        this.listeners.add(listener);
        listener(this.getSnapshot());
        return () => this.listeners.delete(listener);
    }

    /** Subscribe to hover changes and immediately receive the current hover state. */
    public subscribeHover(listener: (hover: HoverInfo | null) => void): () => void {
        this.hoverListeners.add(listener);
        listener(this.getHover());
        return () => this.hoverListeners.delete(listener);
    }

    /** Subscribe to selection changes and immediately receive the current selection. */
    public subscribeSelection(listener: (selection: SelectionCoords) => void): () => void {
        this.selectionListeners.add(listener);
        listener(this.getSelectedCoords());
        return () => this.selectionListeners.delete(listener);
    }

    /** Switch between 2D and 3D display modes. */
    public setDisplayMode(mode: '2d' | '3d'): void {
        logEvent('display:mode', mode);
        this.applyDisplayMode(mode);
        this.relayoutTensorOffsets(mode);
        this.rebuildAllMeshes({ fitCamera: mode === '3d' });
    }

    /** Return the primary left-drag interaction mode. */
    public getInteractionMode(): InteractionMode {
        return this.state.interactionMode;
    }

    /** Set the primary left-drag interaction mode. */
    public setInteractionMode(mode: InteractionMode): InteractionMode {
        this.state.interactionMode = mode;
        this.syncInteractionMode();
        logEvent('interaction:mode', this.state.interactionMode);
        this.emit();
        return this.state.interactionMode;
    }

    /** Toggle grayscale heatmap coloring on or off. */
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

    /** Set the multiplicative gap growth between higher-level dimension blocks. */
    public setDimensionBlockGapMultiple(value: number): number {
        const nextValue = Number.isFinite(value) ? Math.max(1, value) : DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
        if (nextValue === this.state.dimensionBlockGapMultiple) return nextValue;
        this.state.dimensionBlockGapMultiple = nextValue;
        logEvent('display:dimension-block-gap-multiple', nextValue);
        if (this.state.hover) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return nextValue;
    }

    /** Toggle whether inter-block gaps are rendered at all. */
    public toggleDisplayGaps(force?: boolean): boolean {
        this.state.displayGaps = force ?? !this.state.displayGaps;
        logEvent('display:gaps', this.state.displayGaps);
        if (this.state.hover) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return this.state.displayGaps;
    }

    /** Toggle the dimension-guide overlays. */
    public toggleDimensionLines(force?: boolean): boolean {
        this.state.showDimensionLines = force ?? !this.state.showDimensionLines;
        logEvent('display:dimension-lines', this.state.showDimensionLines);
        this.rebuildAllMeshes();
        return this.state.showDimensionLines;
    }

    /** Toggle the tensor-name labels. */
    public toggleTensorNames(force?: boolean): boolean {
        this.state.showTensorNames = force ?? !this.state.showTensorNames;
        logEvent('display:tensor-names', this.state.showTensorNames);
        this.rebuildAllMeshes();
        return this.state.showTensorNames;
    }

    /** Toggle signed-log heatmap normalization. */
    public toggleLogScale(force?: boolean): boolean {
        this.state.logScale = force ?? !this.state.logScale;
        logEvent('display:log-scale', this.state.logScale);
        this.rebuildAllMeshes();
        return this.state.logScale;
    }

    /** Toggle whether sliced axes collapse into the same rendered position. */
    public toggleCollapseHiddenAxes(force?: boolean): boolean {
        this.state.collapseHiddenAxes = force ?? !this.state.collapseHiddenAxes;
        logEvent('display:collapse-hidden-axes', this.state.collapseHiddenAxes);
        if (this.state.hover) this.clearHover();
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return this.state.collapseHiddenAxes;
    }

    /** Backward-compatible alias for {@link toggleCollapseHiddenAxes}. */
    public toggleShowSlicesInSamePlace(force?: boolean): boolean {
        return this.toggleCollapseHiddenAxes(force);
    }

    /** Change how dimensions are assigned to the x, y, and z layout families. */
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

    /** Toggle whether the host UI should show an inspector panel. */
    public toggleInspectorPanel(force?: boolean): boolean {
        this.state.showInspectorPanel = force ?? !this.state.showInspectorPanel;
        logEvent('widget:inspector', this.state.showInspectorPanel);
        this.emit();
        return this.state.showInspectorPanel;
    }

    /** Toggle whether the host UI should show a selection-summary panel. */
    public toggleSelectionPanel(force?: boolean): boolean {
        this.state.showSelectionPanel = force ?? !this.state.showSelectionPanel;
        logEvent('widget:selection', this.state.showSelectionPanel);
        this.emit();
        return this.state.showSelectionPanel;
    }

    /** Toggle whether the host UI should show hover-detail widgets. */
    public toggleHoverDetailsPanel(force?: boolean): boolean {
        this.state.showHoverDetailsPanel = force ?? !this.state.showHoverDetailsPanel;
        this.emit();
        return this.state.showHoverDetailsPanel;
    }

    /** Return the current rendered extent for one tensor in world units. */
    public getViewDims(tensorId: string): Vec3 {
        const tensor = this.tensors.get(tensorId);
        if (!tensor) throw new Error(`Unknown tensor ${tensorId}.`);
        const extent = displayExtent(this.layoutShape(tensor.view), this.layoutGapMultiple(), this.state.dimensionMappingScheme);
        return [extent.x, extent.y, extent.z];
    }

    /** Return the current metadata and dense-data availability for one tensor. */
    public getTensorStatus(tensorId: string): TensorStatus {
        return this.tensorStatus(this.requireTensor(tensorId));
    }

    /** Return the current canonical view string and slice indices for one tensor. */
    public getTensorView(tensorId: string): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        return {
            view: tensor.view.canonical,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        };
    }

    /** Apply a new tensor-view string and optional sliced indices to one tensor.
     *
     * This is a virtual view transform. The viewer does not transpose or rewrite
     * the stored tensor buffer; it reparses the axis order and remaps displayed
     * coordinates back to the original tensor coordinates during render, hover,
     * and value lookup.
     */
    public setTensorView(tensorId: string, spec: string, hiddenIndices?: number[]): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        const snapshot = this.assignTensorView(tensor, spec, hiddenIndices);
        logEvent('tensor:view', { tensorId, view: tensor.view.canonical, hiddenIndices: tensor.view.hiddenIndices });
        this.relayoutTensorOffsets();
        this.rebuildAllMeshes();
        return snapshot;
    }

    /** Attach dense numeric data to one existing tensor without changing its shape or view. */
    public setTensorData(tensorId: string, data: NumericArray, dtype?: DType): TensorStatus {
        const tensor = this.requireTensor(tensorId);
        this.assignTensorData(tensor, data, dtype ?? tensor.dtype);
        logEvent('tensor:data:set', { tensorId, dtype: tensor.dtype, hasData: tensor.hasData });
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.rebuildAllMeshes();
        return this.tensorStatus(tensor);
    }

    /** Drop the dense payload for one tensor and keep only its metadata, view, and colors. */
    public clearTensorData(tensorId: string): TensorStatus {
        const tensor = this.requireTensor(tensorId);
        this.assignTensorData(tensor, null, tensor.dtype);
        logEvent('tensor:data:clear', tensorId);
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.rebuildAllMeshes();
        return this.tensorStatus(tensor);
    }

    /** Ask the host to hydrate a metadata-only tensor if a requester callback was provided. */
    public async ensureTensorData(tensorId: string, reason: TensorDataRequestReason = 'explicit'): Promise<boolean> {
        const tensor = this.requireTensor(tensorId);
        if (tensor.hasData) return true;
        if (!this.requestTensorDataCallback) return false;
        const pending = this.pendingTensorDataRequests.get(tensorId);
        if (pending) return pending;
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

    public colorTensor(tensorId: string, colors: Uint8ClampedArray | Float32Array): void;
    public colorTensor(tensorId: string, coords: number[][], color: RGB | HueSaturation): void;
    public colorTensor(tensorId: string, base: number[], shape: number[], jumps: number[], color: RGB | HueSaturation): void;
    /** Apply dense, coordinate, or region-based custom colors to one tensor. */
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

    /** Remove all custom colors from one tensor. */
    public clearTensorColors(tensorId: string): void {
        this.requireTensor(tensorId).customColors.clear();
        this.rebuildAllMeshes();
    }

    /** Alias for {@link getSnapshot}. */
    public getState(): Readonly<ViewerSnapshot> {
        return this.getSnapshot();
    }

    /** Return the current hover payload, or the last hover when the pointer just left. */
    public getHover(): HoverInfo | null {
        const hover = this.state.hover ?? this.state.lastHover;
        return hover ? { ...hover } : null;
    }

    /** Return the current selected coordinates grouped by tensor id. */
    public getSelectedCoords(): SelectionCoords {
        if (!this.selectionEnabled()) return new Map();
        return this.selectionCoords();
    }

    /** Return selection count plus summary stats across selected cells with loaded values. */
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

    /** Load a manifest plus decoded tensor buffers into the viewer. */
    public loadBundleData(manifest: BundleManifest, tensors: Map<string, NumericArray>): void {
        const shouldFitCamera = this.shouldAutoFitSnapshot(manifest.viewer);
        this.resetLoadedState();
        manifest.tensors.forEach((entry) => {
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
                displayMode: manifest.viewer.displayMode,
                rebuild: false,
                emit: false,
            });
            if (entry.colorInstructions?.length) this.applyColorInstructions(entry.id, entry.colorInstructions);
        });
        this.applySnapshot(manifest.viewer);
        this.rebuildAllMeshes({ fitCamera: shouldFitCamera });
    }

    /** Build the demo-side inspector model for the currently active tensor. */
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

    /** Mark one tensor as active without changing its data or view. */
    public setActiveTensor(tensorId: string): void {
        this.requireTensor(tensorId);
        this.state.activeTensorId = tensorId;
        logEvent('tensor:active', tensorId);
        this.emit();
    }

    /** Set the flattened value of one slice token. */
    public setSliceTokenValue(tensorId: string, token: string, value: number): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        const sliceToken = tensor.view.sliceTokens.find((entry) => entry.key === token || entry.token === token);
        if (!sliceToken) throw new Error(`Unknown slice token ${token}.`);
        const clamped = Math.max(0, Math.min(sliceToken.size - 1, Math.floor(value)));
        if (sliceToken.value === clamped) {
            return {
                view: tensor.view.canonical,
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

    /** Backward-compatible alias for {@link setSliceTokenValue}. */
    public setHiddenTokenValue(tensorId: string, token: string, value: number): TensorViewSnapshot {
        return this.setSliceTokenValue(tensorId, token, value);
    }

    /** Dispose the renderer, listeners, and DOM nodes created by this viewer. */
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

    private requireTensor(tensorId: string): TensorRecord {
        const tensor = this.tensors.get(tensorId);
        if (!tensor) throw new Error(`Unknown tensor ${tensorId}.`);
        return tensor;
    }
}
