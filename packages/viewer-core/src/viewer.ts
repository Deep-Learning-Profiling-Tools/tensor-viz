import {
    BufferAttribute,
    Box3,
    BoxGeometry,
    BufferGeometry,
    Color,
    DoubleSide,
    EdgesGeometry,
    Group,
    InstancedBufferAttribute,
    InstancedMesh,
    LineBasicMaterial,
    Line,
    LineSegments,
    Matrix4,
    Mesh,
    MeshBasicMaterial,
    NoToneMapping,
    OrthographicCamera,
    PlaneGeometry,
    PerspectiveCamera,
    Raycaster,
    SRGBColorSpace,
    Scene,
    ShapeGeometry,
    Sphere,
    Vector2,
    Vector3,
    WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import helvetikerBoldFont from 'three/examples/fonts/helvetiker_bold.typeface.json';
import { loadBundle } from './bundle.js';
import { isNpyFile, loadNpy, saveNpy } from './npy.js';
import {
    DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
    displayExtent,
    displayExtent2D,
    displayHitForPoint2D,
    displayPositionForCoord,
    displayPositionForCoord2D,
    unravelIndex,
} from './layout.js';
import {
    buildPreviewExpression,
    defaultTensorView,
    expandGroupedIndex,
    mapDisplayCoordToFullCoord,
    mapDisplayCoordToOutlineCoord,
    mapOutlineCoordToDisplayCoord,
    parseTensorView,
    product,
} from './view.js';
import type {
    BundleManifest,
    ColorInstruction,
    CustomColor,
    DType,
    HoverInfo,
    HueSaturation,
    NumericArray,
    RGBA,
    TensorHandle,
    TensorRecord,
    TensorViewSnapshot,
    Vec3,
    ViewerSnapshot,
    ViewerState,
} from './types.js';

const BASE_COLOR = new Color('#90a4ae');
const ACTIVE_COLOR = new Color('#1976d2');
const HOVER_COLOR = new Color('#f59e0b');

type ViewerOptions = {
    background?: string;
};

type MeshMeta = {
    tensorId: string;
    renderShape: number[];
};

type PickMesh = {
    tensorId: string;
    mesh: InstancedMesh;
    bounds: Box3;
    rect2D: {
        minX: number;
        maxX: number;
        minY: number;
        maxY: number;
    };
};

const CANVAS_WORLD_SCALE = 4;
const MAX_CANVAS_ZOOM = 96;
const MIN_CANVAS_FIT_INSET = 16;
const MAX_CANVAS_FIT_INSET = 48;
const AUTO_FIT_2D_SCALE = 0.75;
const AUTO_FIT_3D_DISTANCE_SCALE = 1.25;
const LOG_PREFIX = '[tensor-viz]';
const LABEL_FONT = new FontLoader().parse(helvetikerBoldFont as never);
const LOG_ENABLED = (() => {
    try {
        return typeof window !== 'undefined'
            && ((window as { __TENSOR_VIZ_DEBUG__?: boolean }).__TENSOR_VIZ_DEBUG__ === true
                || window.localStorage.getItem('tensor-viz-debug') === '1');
    } catch {
        return false;
    }
})();

function createLine(points: Vector3[], color: string): Line {
    const geometry = new BufferGeometry().setFromPoints(points);
    const line = new Line(geometry, new LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
    line.renderOrder = 5_000;
    return line;
}

function logEvent(event: string, details?: unknown): void {
    if (!LOG_ENABLED) return;
    if (details === undefined) console.log(LOG_PREFIX, event);
    else console.log(LOG_PREFIX, event, details);
}

function initializeVertexColors(geometry: BufferGeometry): void {
    const normals = geometry.attributes.normal;
    const colorArray = new Float32Array(geometry.attributes.position.count * 3);
    const lightDir = new Vector3(0.35, 0.6, 0.7).normalize();
    for (let index = 0; index < normals.count; index += 1) {
        const ndotl = Math.max(0, normals.getX(index) * lightDir.x + normals.getY(index) * lightDir.y + normals.getZ(index) * lightDir.z);
        const shade = 0.6 + 0.4 * ndotl;
        const base = index * 3;
        colorArray[base] = shade;
        colorArray[base + 1] = shade;
        colorArray[base + 2] = shade;
    }
    geometry.setAttribute('color', new BufferAttribute(colorArray, 3));
}

function createTextLabel(text: string, color = '#334155'): Group {
    // use shape text instead of troika so brave/linux stays on the same path as the working line geometry.
    const shapes = LABEL_FONT.generateShapes(text, 1.1);
    const frontGeometry = new ShapeGeometry(shapes);
    frontGeometry.center();
    const front = new Mesh(frontGeometry, new MeshBasicMaterial({ color, depthTest: false, depthWrite: false, side: DoubleSide }));

    const label = new Group();
    label.add(front);
    label.frustumCulled = false;
    label.renderOrder = 10_000;
    label.onBeforeRender = function onBeforeRender(_renderer: unknown, _scene: unknown, camera: { quaternion: unknown }): void {
        this.quaternion.copy(camera.quaternion as never);
    };
    front.frustumCulled = false;
    front.renderOrder = 10_000;
    return label;
}

function axisFamilyColor(worldKey: 0 | 1 | 2, familyIndex: number, familyCount: number): string {
    const t = Math.max(1, familyIndex + 1) / Math.max(1, familyCount);
    const color = new Color();
    if (worldKey === 1) color.setRGB(0, t, 0);
    else if (worldKey === 2) color.setRGB(0, 0, t);
    else color.setRGB(t, 0, 0);
    return `#${color.getHexString()}`;
}

function axisWorldKey(rank: number, axis: number): 0 | 1 | 2 {
    return ((rank - 1 - axis) % 3) as 0 | 1 | 2;
}

function axisWorldKey2D(rank: number, axis: number): 0 | 1 {
    return ((rank - 1 - axis) % 2) as 0 | 1;
}

function signedLog1p(value: number): number {
    return Math.sign(value) * Math.log1p(Math.abs(value));
}

function computeMinMax(data: NumericArray): { min: number; max: number } {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < data.length; index += 1) {
        const value = Number(data[index] ?? 0);
        if (value < min) min = value;
        if (value > max) max = value;
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
    return { min, max };
}

function vectorFromTuple(tuple: Vec3): Vector3 {
    return new Vector3(tuple[0], tuple[1], tuple[2]);
}

function tupleFromVector(vector: Vector3): Vec3 {
    return [vector.x, vector.y, vector.z];
}

function dtypeFromArray(data: NumericArray): DType {
    if (data instanceof Float32Array) return 'float32';
    if (data instanceof Int32Array) return 'int32';
    if (data instanceof Uint8Array) return 'uint8';
    return 'float64';
}

function numericValue(data: NumericArray, index: number): number {
    return Number(data[index] ?? 0);
}

function coordKey(coord: number[]): string {
    return coord.join(',');
}

function colorFromRgba(rgba: RGBA): Color {
    return new Color(rgba[0] / 255, rgba[1] / 255, rgba[2] / 255);
}

function normalizeHue(hue: number): number {
    const unit = Math.abs(hue) > 1 ? hue / 360 : hue;
    return ((unit % 1) + 1) % 1;
}

function normalizeSaturation(saturation: number): number {
    const unit = Math.abs(saturation) > 1 ? saturation / 100 : saturation;
    return Math.max(0, Math.min(1, unit));
}

function colorFromHueSaturation(color: HueSaturation, brightness: number): Color {
    const hue = normalizeHue(color[0]);
    const saturation = normalizeSaturation(color[1]);
    const value = Math.max(0, Math.min(1, brightness));
    const sector = hue * 6;
    const index = Math.floor(sector);
    const fraction = sector - index;
    const p = value * (1 - saturation);
    const q = value * (1 - saturation * fraction);
    const t = value * (1 - saturation * (1 - fraction));
    const components = [
        [value, t, p],
        [q, value, p],
        [p, value, t],
        [p, q, value],
        [t, p, value],
        [value, p, q],
    ][index % 6] ?? [value, value, value];
    return new Color(components[0], components[1], components[2]);
}

function parseCustomColor(value: number[]): CustomColor {
    if (value.length === 2) return { kind: 'hs', value: [value[0], value[1]] };
    if (value.length === 4) return { kind: 'rgba', value: [value[0], value[1], value[2], value[3]] };
    throw new Error(`Expected color tuple of length 2 or 4, received ${value.length}.`);
}

export class TensorViewer {
    private readonly container: HTMLElement;
    private readonly scene = new Scene();
    private readonly renderer: WebGLRenderer;
    private readonly flatCanvas: HTMLCanvasElement;
    private readonly flatContext: CanvasRenderingContext2D;
    private readonly flatOverlay: SVGSVGElement;
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
    private readonly pickMeshes: PickMesh[] = [];
    private readonly state: ViewerState = {
        displayMode: '2d',
        heatmap: true,
        dimensionBlockGapMultiple: DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE,
        displayGaps: true,
        logScale: false,
        showDimensionLines: true,
        showInspectorPanel: true,
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

    public constructor(container: HTMLElement, options: ViewerOptions = {}) {
        this.container = container;
        if (getComputedStyle(container).position === 'static') this.container.style.position = 'relative';
        this.scene.background = new Color(options.background ?? '#e5e7eb');
        this.renderer = new WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
        if ('outputEncoding' in this.renderer) (this.renderer as WebGLRenderer & { outputEncoding: number }).outputEncoding = 3001;
        if ('outputColorSpace' in this.renderer) this.renderer.outputColorSpace = SRGBColorSpace;
        this.renderer.toneMapping = NoToneMapping;
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
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
        controls.mouseButtons.LEFT = 2;
        controls.mouseButtons.RIGHT = 0;
        if ('zoomToCursor' in controls) {
            (controls as OrbitControls & { zoomToCursor: boolean }).zoomToCursor = true;
        }
        controls.addEventListener('change', () => this.requestRender());
        return controls;
    }

    private bindEvents(): void {
        window.addEventListener('resize', this.resize);
        this.renderer.domElement.addEventListener('pointermove', this.onPointerMove);
        this.renderer.domElement.addEventListener('pointerleave', this.onPointerLeave);
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

    private readonly resize = (): void => {
        const width = this.container.clientWidth || 1;
        const height = this.container.clientHeight || 1;
        this.renderer.setSize(width, height);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        this.flatCanvas.width = Math.floor(width * pixelRatio);
        this.flatCanvas.height = Math.floor(height * pixelRatio);
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
        logEvent('viewport:resize', { width, height });
        this.requestRender();
    };

    private canvasScale(): number {
        return this.flatCanvas.width / Math.max(1, this.flatCanvas.clientWidth || this.flatCanvas.width || 1);
    }

    private layoutGapMultiple(): number {
        return this.state.displayGaps ? this.state.dimensionBlockGapMultiple : 0;
    }

    private heatmapNormalizedValue(value: number, min: number, max: number): number {
        const scaledValue = this.state.logScale ? signedLog1p(value) : value;
        const scaledMin = this.state.logScale ? signedLog1p(min) : min;
        const scaledMax = this.state.logScale ? signedLog1p(max) : max;
        const range = scaledMax - scaledMin || 1;
        return Math.max(0, Math.min(1, (scaledValue - scaledMin) / range));
    }

    private sync2DCamera(): void {
        const zoom = Math.max(0.15, this.canvasZoom);
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

    private pickEntries(clientX: number, clientY: number): PickMesh[] {
        const currentTensorId = this.state.hover?.tensorId;
        const activeTensorId = this.state.activeTensorId;
        const candidates = this.state.displayMode === '2d'
            ? (() => {
                const point = this.canvasPointerToWorld(clientX, clientY);
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
        const entries = this.pickEntries(clientX, clientY);
        for (const entry of entries) {
            const tensor = this.requireTensor(entry.tensorId);
            const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
            const hit = displayHitForPoint2D(point.x - tensor.offset[0], point.y - tensor.offset[1], outlineShape, this.layoutGapMultiple());
            if (!hit) continue;
            const displayCoord = mapOutlineCoordToDisplayCoord(hit.coord, tensor.view);
            const fullCoord = mapDisplayCoordToFullCoord(displayCoord, tensor.view);
            const value = numericValue(tensor.data, this.linearIndex(fullCoord, tensor.shape));
            const colorSource: HoverInfo['colorSource'] = tensor.customColors.has(coordKey(fullCoord))
                ? 'custom'
                : this.state.heatmap
                    ? 'heatmap'
                    : 'base';
            return {
                hover: {
                    tensorId: tensor.id,
                    tensorName: tensor.name,
                    displayCoord: displayCoord.slice(),
                    viewedCoord: hit.coord.slice(),
                    fullCoord: fullCoord.slice(),
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
        const displayCoord = meta.renderShape.length === 0 ? [] : unravelIndex(hit.instanceId, meta.renderShape);
        const tensor = this.requireTensor(meta.tensorId);
        const viewedCoord = mapDisplayCoordToOutlineCoord(displayCoord, tensor.view);
        const fullCoord = mapDisplayCoordToFullCoord(displayCoord, tensor.view);
        const value = numericValue(tensor.data, this.linearIndex(fullCoord, tensor.shape));
        const colorSource: HoverInfo['colorSource'] = tensor.customColors.has(coordKey(fullCoord))
            ? 'custom'
            : this.state.heatmap
                ? 'heatmap'
                : 'base';
        const instanceMatrix = new Matrix4();
        const position = new Vector3();
        mesh.getMatrixAt(hit.instanceId, instanceMatrix);
        position.setFromMatrixPosition(instanceMatrix);
        return {
            hover: {
                tensorId: meta.tensorId,
                tensorName: this.tensors.get(meta.tensorId)?.name ?? meta.tensorId,
                displayCoord: displayCoord.slice(),
                viewedCoord: viewedCoord.slice(),
                fullCoord: fullCoord.slice(),
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
            && left.fullCoord.length === right.fullCoord.length
            && left.fullCoord.every((value, index) => value === right.fullCoord[index]);
    }

    private hoverPosition(hover: HoverInfo): Vector3 | null {
        const tensor = this.tensors.get(hover.tensorId);
        if (!tensor) return null;
        const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
        const outlineCoord = hover.viewedCoord;
        if (this.state.displayMode === '2d') {
            const position = displayPositionForCoord2D(outlineCoord, outlineShape, this.layoutGapMultiple());
            return new Vector3(tensor.offset[0] + position.x, tensor.offset[1] + position.y, 0);
        }
        return displayPositionForCoord(outlineCoord, outlineShape, this.layoutGapMultiple()).add(vectorFromTuple(tensor.offset));
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
        const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
        const outlineCoord = hover.viewedCoord;
        const position = displayPositionForCoord2D(outlineCoord, outlineShape, this.layoutGapMultiple());
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
        const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
        const outlineCoord = hover.viewedCoord;
        const center = displayPositionForCoord(outlineCoord, outlineShape, this.layoutGapMultiple()).add(vectorFromTuple(tensor.offset));
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

    private fitCanvasView(): void {
        if (this.tensors.size === 0) {
            this.canvasPan = { x: 0, y: 0 };
            this.canvasZoom = 1;
            return;
        }

        let minX = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        this.tensors.forEach((tensor) => {
            const extent = displayExtent2D(tensor.view.outlineShape, this.layoutGapMultiple());
            minX = Math.min(minX, tensor.offset[0] - extent.x / 2);
            maxX = Math.max(maxX, tensor.offset[0] + extent.x / 2);
            minY = Math.min(minY, tensor.offset[1] - extent.y / 2);
            maxY = Math.max(maxY, tensor.offset[1] + extent.y / 2);
        });

        const width = Math.max(1, (maxX - minX) * CANVAS_WORLD_SCALE);
        const height = Math.max(1, (maxY - minY) * CANVAS_WORLD_SCALE);
        const inset = Math.min(
            MAX_CANVAS_FIT_INSET,
            Math.max(MIN_CANVAS_FIT_INSET, Math.min(this.flatCanvas.width, this.flatCanvas.height) * 0.035),
        ) * this.canvasScale();
        this.canvasZoom = Math.max(
            0.15,
            Math.min(
                MAX_CANVAS_ZOOM,
                Math.min(
                    (this.flatCanvas.width - inset * 2) / width,
                    (this.flatCanvas.height - inset * 2) / height,
                ),
            ),
        ) * AUTO_FIT_2D_SCALE;
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        this.canvasPan = {
            x: -(centerX * CANVAS_WORLD_SCALE * this.canvasZoom),
            y: centerY * CANVAS_WORLD_SCALE * this.canvasZoom,
        };
    }

    private readonly onCanvasPointerDown = (event: PointerEvent): void => {
        if (event.button !== 0) return;
        this.isCanvasPanning = true;
        this.lastCanvasPointer = { x: event.clientX, y: event.clientY };
        logEvent('2d:pointerdown', { x: event.clientX, y: event.clientY, button: event.button });
    };

    private readonly onCanvasPointerUp = (): void => {
        this.isCanvasPanning = false;
        logEvent('2d:pointerup');
    };

    private readonly onCanvasWheel = (event: WheelEvent): void => {
        if (this.state.displayMode !== '2d') return;
        event.preventDefault();
        const rect = this.flatCanvas.getBoundingClientRect();
        const scale = event.deltaY < 0 ? 1.1 : 1 / 1.1;
        const nextZoom = Math.max(0.15, Math.min(MAX_CANVAS_ZOOM, this.canvasZoom * scale));
        const cursorX = (event.clientX - rect.left) * (this.flatCanvas.width / rect.width) - this.flatCanvas.width / 2 - this.canvasPan.x;
        const cursorY = (event.clientY - rect.top) * (this.flatCanvas.height / rect.height) - this.flatCanvas.height / 2 - this.canvasPan.y;
        this.canvasPan.x -= cursorX * ((nextZoom / this.canvasZoom) - 1);
        this.canvasPan.y -= cursorY * ((nextZoom / this.canvasZoom) - 1);
        this.canvasZoom = nextZoom;
        logEvent('2d:zoom', { zoom: this.canvasZoom });
        this.requestRender();
    };

    private readonly onCanvasPointerMove = (event: PointerEvent): void => {
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

    private readonly onPointerMove = (event: PointerEvent): void => {
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
        this.updateHover(null, '3d');
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
            this.camera.zoom = Math.max(0.1, this.camera.zoom / scale);
            this.camera.updateProjectionMatrix();
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
                return;
            }
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
        });
    }

    private fitCamera(): void {
        if (this.state.displayMode === '2d') {
            this.fitCanvasView();
            this.requestRender();
            return;
        }
        const groups = Array.from(this.tensorMeshes.values());
        if (groups.length === 0) {
            this.controls.target.set(0, 0, 0);
            this.perspectiveCamera.position.set(0, 0, 30);
            this.orthographicCamera.position.set(0, 0, 30);
            this.requestRender();
            return;
        }

        const bounds = new Box3();
        groups.forEach((group) => bounds.expandByObject(group));
        const center = bounds.getCenter(new Vector3());
        const size = bounds.getSize(new Vector3());
        const radius = Math.max(size.x, size.y, size.z, 8) * AUTO_FIT_3D_DISTANCE_SCALE;
        this.controls.target.copy(center);
        this.perspectiveCamera.position.set(center.x + radius, center.y + radius, center.z + radius * 1.5);
        this.perspectiveCamera.lookAt(center);
        this.orthographicCamera.position.set(center.x, center.y, center.z + radius * 2);
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
                const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
                const extent2D = displayExtent2D(outlineShape, this.layoutGapMultiple());
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
        renderShape: number[],
        min: number,
        max: number,
    ): boolean {
        const colorArray = mesh.instanceColor?.array as Float32Array | undefined;
        const isIdentityView = this.state.displayMode === '2d'
            && tensor.shape.length <= 2
            && tensor.view.hiddenTokens.length === 0
            && tensor.view.tokens.length === tensor.shape.length
            && tensor.view.tokens.every((token, index) => token.kind === 'axis_group'
                && token.visible
                && token.axes.length === 1
                && token.axes[0] === index);
        if (!isIdentityView || tensor.customColors.size !== 0 || !colorArray) return false;

        // default 1d/2d views are affine grids, so write instance buffers directly.
        const matrixArray = mesh.instanceMatrix.array as Float32Array;
        const extent = displayExtent2D(renderShape, this.layoutGapMultiple());
        const rowCount = renderShape.length > 1 ? renderShape[0] : 1;
        const columnCount = renderShape.length > 1 ? renderShape[1] : (renderShape[0] ?? 1);
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

                const value = numericValue(tensor.data, cellIndex);
                const colorOffset = cellIndex * 3;
                if (this.state.heatmap) {
                    const gray = this.heatmapNormalizedValue(value, min, max);
                    colorArray[colorOffset] = gray;
                    colorArray[colorOffset + 1] = gray;
                    colorArray[colorOffset + 2] = gray;
                } else {
                    colorArray[colorOffset] = BASE_COLOR.r;
                    colorArray[colorOffset + 1] = BASE_COLOR.g;
                    colorArray[colorOffset + 2] = BASE_COLOR.b;
                }
                cellIndex += 1;
            }
        }

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

    private buildDimensionGuides2D(shape: number[], offset: Vec3, labels: string[]): Group {
        const group = new Group();
        const rank = shape.length;
        const families = new Map<number, number[]>();
        for (let axis = 0; axis < rank; axis += 1) {
            const key = axisWorldKey2D(rank, axis);
            const family = families.get(key) ?? [];
            family.push(axis);
            families.set(key, family);
        }
        const guideOffset = 1.15;
        const linearStep = 0.75;
        const labelOffset = 0.5;
        const labelScale = 0.15;

        shape.forEach((size, axis) => {
            const familyKey = axisWorldKey2D(rank, axis);
            const family = families.get(familyKey) ?? [axis];
            const familyPos = Math.max(0, family.indexOf(axis));
            const start = new Array(rank).fill(0);
            const end = start.slice();
            family.forEach((familyAxis) => {
                if (familyAxis >= axis) end[familyAxis] = Math.max(0, shape[familyAxis] - 1);
            });
            const startPos = displayPositionForCoord2D(start, shape, this.layoutGapMultiple());
            const endPos = displayPositionForCoord2D(end, shape, this.layoutGapMultiple());
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
            const key = axisWorldKey(rank, axis);
            const family = families.get(key) ?? [];
            family.push(axis);
            families.set(key, family);
        }
        const entries = shape.map((size, axis) => {
            const worldKey = axisWorldKey(rank, axis);
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
            const startPos = displayPositionForCoord(start, shape, this.layoutGapMultiple());
            const endPos = displayPositionForCoord(end, shape, this.layoutGapMultiple());
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
        const group = new Group();
        const renderShape = tensor.view.displayShape.length === 0 ? [1] : tensor.view.displayShape;
        const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
        const count = product(renderShape);
        const mesh = new InstancedMesh(
            this.state.displayMode === '2d' ? this.planeGeometry : this.cubeGeometry,
            new MeshBasicMaterial({ color: 0xffffff, vertexColors: true, toneMapped: false }),
            count,
        );
        mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(count * 3), 3);
        const outlineExtent2D = displayExtent2D(outlineShape, this.layoutGapMultiple());
        const { min, max } = tensor.valueRange;
        if (!this.populateFastMesh2D(tensor, mesh, renderShape, min, max)) {
            const matrix = new Matrix4();
            const offset = vectorFromTuple(tensor.offset);
            for (let index = 0; index < count; index += 1) {
                const displayCoord = count === 1 && tensor.view.displayShape.length === 0 ? [] : unravelIndex(index, renderShape);
                const fullCoord = mapDisplayCoordToFullCoord(displayCoord, tensor.view);
                const outlineCoord = mapDisplayCoordToOutlineCoord(displayCoord, tensor.view);
                const fullLinear = this.linearIndex(fullCoord, tensor.shape);
                const value = numericValue(tensor.data, fullLinear);
                const position = this.state.displayMode === '2d'
                    ? (() => {
                        const flat = displayPositionForCoord2D(outlineCoord, outlineShape, this.layoutGapMultiple());
                        return new Vector3(tensor.offset[0] + flat.x, tensor.offset[1] + flat.y, 0);
                    })()
                    : displayPositionForCoord(outlineCoord, outlineShape, this.layoutGapMultiple()).add(offset);
                matrix.makeTranslation(position.x, position.y, position.z);
                mesh.setMatrixAt(index, matrix);
                mesh.setColorAt(index, this.cellColor(tensor, fullCoord, value, min, max));
            }
        }

        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        if (this.state.displayMode === '2d') {
            const halfX = outlineExtent2D.x / 2;
            const halfY = outlineExtent2D.y / 2;
            const center = new Vector3(tensor.offset[0], tensor.offset[1], 0);
            mesh.boundingBox = new Box3(
                new Vector3(center.x - halfX, center.y - halfY, 0),
                new Vector3(center.x + halfX, center.y + halfY, 0),
            );
            mesh.boundingSphere = new Sphere(center, Math.hypot(halfX, halfY));
        } else {
            const outlineExtent = displayExtent(outlineShape, this.layoutGapMultiple());
            const halfExtent = outlineExtent.clone().multiplyScalar(0.5);
            const center = vectorFromTuple(tensor.offset);
            mesh.boundingBox = new Box3(center.clone().sub(halfExtent), center.clone().add(halfExtent));
            mesh.boundingSphere = new Sphere(center, halfExtent.length());
        }
        mesh.material.needsUpdate = true;
        mesh.userData.meta = { tensorId: tensor.id, renderShape } satisfies MeshMeta;
        group.add(mesh);
        if (this.state.displayMode === '2d') {
            group.add(this.buildOutline2D(outlineExtent2D, tensor.offset));
            if (this.state.showDimensionLines) group.add(this.buildDimensionGuides2D(outlineShape, tensor.offset, tensor.view.tokens.map((token) => token.label.toUpperCase())));
        } else {
            const outlineExtent = displayExtent(outlineShape, this.layoutGapMultiple());
            group.add(this.buildOutline(outlineExtent, tensor.offset));
            if (this.state.showDimensionLines) group.add(this.buildDimensionGuides(outlineExtent, outlineShape, tensor.offset, tensor.view.tokens.map((token) => token.label.toUpperCase())));
        }
        return group;
    }

    private render2D(): void {
        this.sync2DCamera();
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

    private cellColor(tensor: TensorRecord, fullCoord: number[], value: number, min: number, max: number): Color {
        const customColor = tensor.customColors.get(coordKey(fullCoord));
        if (customColor?.kind === 'rgba') return colorFromRgba(customColor.value);
        if (customColor?.kind === 'hs') return colorFromHueSaturation(customColor.value, this.state.heatmap ? this.heatmapNormalizedValue(value, min, max) : 1);
        if (!this.state.heatmap) return BASE_COLOR.clone();
        const gray = this.heatmapNormalizedValue(value, min, max);
        return new Color().setRGB(gray, gray, gray);
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
        arg2?: RGBA | HueSaturation | number[],
        arg3?: number[] | RGBA | HueSaturation,
        arg4?: RGBA | HueSaturation | number[],
    ): void {
        if (arg1 instanceof Uint8ClampedArray || arg1 instanceof Float32Array) {
            const denseColorSize = arg1.length / product(tensor.shape);
            if (denseColorSize !== 2 && denseColorSize !== 4) {
                throw new Error(`Expected dense color payload with 2 or 4 channels per cell, received ${arg1.length}.`);
            }
            for (let index = 0; index < product(tensor.shape); index += 1) {
                const coord = unravelIndex(index, tensor.shape);
                const offset = index * denseColorSize;
                const values = Array.from(arg1.slice(offset, offset + denseColorSize));
                tensor.customColors.set(coordKey(coord), parseCustomColor(
                    denseColorSize === 4 && arg1 instanceof Float32Array
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
        this.controls.dispose();
        this.camera = mode === '2d' ? this.orthographicCamera : this.perspectiveCamera;
        this.controls = this.createControls(this.camera);
        this.controls.enableRotate = mode === '3d';
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
        const parsed = parseTensorView(tensor.shape, spec, hiddenIndices ?? tensor.view.hiddenIndices);
        if (!parsed.ok) throw new Error(parsed.errors.join(' '));
        tensor.view = parsed.spec;
        return {
            visible: tensor.view.canonical,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        };
    }

    private applySnapshot(snapshot: ViewerSnapshot): void {
        this.state.heatmap = snapshot.heatmap;
        this.state.dimensionBlockGapMultiple = snapshot.dimensionBlockGapMultiple ?? DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
        this.state.displayGaps = snapshot.displayGaps ?? true;
        this.state.logScale = snapshot.logScale ?? false;
        this.state.showDimensionLines = snapshot.showDimensionLines;
        this.state.showInspectorPanel = snapshot.showInspectorPanel;
        this.state.showHoverDetailsPanel = snapshot.showHoverDetailsPanel;
        this.state.activeTensorId = snapshot.activeTensorId;
        this.applyDisplayMode(snapshot.displayMode);

        if (snapshot.displayMode === '2d') {
            this.canvasZoom = snapshot.camera.zoom;
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
            tensor.offset = entry.offset;
            this.assignTensorView(tensor, entry.view.visible, entry.view.hiddenIndices);
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
        const hoverKey = hover ? `${hover.tensorId}:${hover.fullCoord.join(',')}:${hover.value}` : null;
        if (hoverKey !== this.lastHoverLogKey) {
            this.lastHoverLogKey = hoverKey;
            logEvent(`${source}:hover`, hover ?? 'none');
        }
        this.requestRender();
        this.emitHover();
    }

    public addTensor(shape: number[], data: NumericArray, name?: string, offset?: Vec3, dtype?: DType): TensorHandle {
        logEvent('tensor:add', { shape, name, offset, dtype });
        return this.insertTensor(shape, data, {
            name,
            offset,
            dtype,
        });
    }

    private insertTensor(
        shape: number[],
        data: NumericArray,
        options: {
            id?: string;
            name?: string;
            offset?: Vec3;
            dtype?: DType;
            rebuild?: boolean;
            emit?: boolean;
        } = {},
    ): TensorHandle {
        const normalizedShape = shape.map((dim) => Math.max(1, Math.floor(dim)));
        if (product(normalizedShape) !== data.length) {
            throw new Error(`Tensor data length ${data.length} does not match shape ${normalizedShape.join('x')}.`);
        }

        const parsed = parseTensorView(normalizedShape, defaultTensorView(normalizedShape));
        if (!parsed.ok) throw new Error(parsed.errors.join(' '));
        const id = options.id ?? (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `tensor-${this.tensorCounter += 1}`);
        const tensor: TensorRecord = {
            id,
            name: options.name ?? `Tensor ${this.tensors.size + 1}`,
            shape: normalizedShape,
            dtype: options.dtype ?? dtypeFromArray(data),
            data,
            valueRange: computeMinMax(data),
            offset: options.offset ?? (this.tensors.size === 0 ? [0, 0, 0] : [this.tensors.size * 6, 0, 0]),
            view: parsed.spec,
            customColors: new Map(),
        };
        this.tensors.set(id, tensor);
        this.state.activeTensorId ??= id;
        if (options.rebuild === false) {
            if (options.emit !== false) this.emit();
        } else {
            this.rebuildAllMeshes({ fitCamera: true });
        }
        return {
            id,
            name: tensor.name,
            rank: tensor.shape.length,
            shape: tensor.shape,
            dtype: tensor.dtype,
        };
    }

    public removeTensor(tensorId: string): void {
        logEvent('tensor:remove', tensorId);
        this.tensors.delete(tensorId);
        if (this.state.activeTensorId === tensorId) {
            this.state.activeTensorId = this.tensors.keys().next().value ?? null;
        }
        if (this.state.hover?.tensorId === tensorId || this.state.lastHover?.tensorId === tensorId) this.clearHover();
        this.rebuildAllMeshes();
    }

    public clear(): void {
        logEvent('tensor:clear');
        this.resetLoadedState();
        this.requestRender();
        this.emitHover();
        this.emit();
    }

    public getSnapshot(): ViewerSnapshot {
        return {
            version: 1,
            displayMode: this.state.displayMode,
            heatmap: this.state.heatmap,
            dimensionBlockGapMultiple: this.state.dimensionBlockGapMultiple,
            displayGaps: this.state.displayGaps,
            logScale: this.state.logScale,
            showDimensionLines: this.state.showDimensionLines,
            showInspectorPanel: this.state.showInspectorPanel,
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
                    visible: tensor.view.canonical,
                    hiddenIndices: tensor.view.hiddenIndices.slice(),
                },
            })),
            activeTensorId: this.state.activeTensorId,
        };
    }

    public restoreSnapshot(snapshot: ViewerSnapshot): void {
        logEvent('snapshot:restore', snapshot);
        this.applySnapshot(snapshot);
        this.rebuildAllMeshes();
    }

    public subscribe(listener: (snapshot: ViewerSnapshot) => void): () => void {
        this.listeners.add(listener);
        listener(this.getSnapshot());
        return () => this.listeners.delete(listener);
    }

    public subscribeHover(listener: (hover: HoverInfo | null) => void): () => void {
        this.hoverListeners.add(listener);
        listener(this.getHover());
        return () => this.hoverListeners.delete(listener);
    }

    public setDisplayMode(mode: '2d' | '3d'): void {
        logEvent('display:mode', mode);
        this.applyDisplayMode(mode);
        this.rebuildAllMeshes();
    }

    public toggleHeatmap(force?: boolean): boolean {
        this.state.heatmap = force ?? !this.state.heatmap;
        logEvent('display:heatmap', this.state.heatmap);
        this.rebuildAllMeshes();
        return this.state.heatmap;
    }

    public setDimensionBlockGapMultiple(value: number): number {
        const nextValue = Number.isFinite(value) ? Math.max(1, value) : DEFAULT_DIMENSION_BLOCK_GAP_MULTIPLE;
        if (nextValue === this.state.dimensionBlockGapMultiple) return nextValue;
        this.state.dimensionBlockGapMultiple = nextValue;
        logEvent('display:dimension-block-gap-multiple', nextValue);
        if (this.state.hover) this.clearHover();
        this.rebuildAllMeshes();
        return nextValue;
    }

    public toggleDisplayGaps(force?: boolean): boolean {
        this.state.displayGaps = force ?? !this.state.displayGaps;
        logEvent('display:gaps', this.state.displayGaps);
        if (this.state.hover) this.clearHover();
        this.rebuildAllMeshes();
        return this.state.displayGaps;
    }

    public toggleDimensionLines(force?: boolean): boolean {
        this.state.showDimensionLines = force ?? !this.state.showDimensionLines;
        logEvent('display:dimension-lines', this.state.showDimensionLines);
        this.rebuildAllMeshes();
        return this.state.showDimensionLines;
    }

    public toggleLogScale(force?: boolean): boolean {
        this.state.logScale = force ?? !this.state.logScale;
        logEvent('display:log-scale', this.state.logScale);
        this.rebuildAllMeshes();
        return this.state.logScale;
    }

    public toggleInspectorPanel(force?: boolean): boolean {
        this.state.showInspectorPanel = force ?? !this.state.showInspectorPanel;
        logEvent('widget:inspector', this.state.showInspectorPanel);
        this.emit();
        return this.state.showInspectorPanel;
    }

    public toggleHoverDetailsPanel(force?: boolean): boolean {
        this.state.showHoverDetailsPanel = force ?? !this.state.showHoverDetailsPanel;
        this.emit();
        return this.state.showHoverDetailsPanel;
    }

    public getViewDims(tensorId: string): Vec3 {
        const tensor = this.tensors.get(tensorId);
        if (!tensor) throw new Error(`Unknown tensor ${tensorId}.`);
        const extent = displayExtent(tensor.view.outlineShape, this.layoutGapMultiple());
        return [extent.x, extent.y, extent.z];
    }

    public getTensorView(tensorId: string): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        return {
            visible: tensor.view.canonical,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        };
    }

    public setTensorView(tensorId: string, spec: string, hiddenIndices?: number[]): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        const snapshot = this.assignTensorView(tensor, spec, hiddenIndices);
        logEvent('tensor:view', { tensorId, view: tensor.view.canonical, hiddenIndices: tensor.view.hiddenIndices });
        this.rebuildAllMeshes();
        return snapshot;
    }

    public colorTensor(tensorId: string, colors: Uint8ClampedArray | Float32Array): void;
    public colorTensor(tensorId: string, coords: number[][], color: RGBA | HueSaturation): void;
    public colorTensor(tensorId: string, base: number[], shape: number[], jumps: number[], color: RGBA | HueSaturation): void;
    public colorTensor(
        tensorId: string,
        arg1: Uint8ClampedArray | Float32Array | number[][] | number[],
        arg2?: RGBA | HueSaturation | number[],
        arg3?: number[] | RGBA | HueSaturation,
        arg4?: RGBA | HueSaturation,
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

    public clearTensorColors(tensorId: string): void {
        this.requireTensor(tensorId).customColors.clear();
        this.rebuildAllMeshes();
    }

    public getState(): Readonly<ViewerSnapshot> {
        return this.getSnapshot();
    }

    public getHover(): HoverInfo | null {
        const hover = this.state.hover ?? this.state.lastHover;
        return hover ? { ...hover } : null;
    }

    public async openFile(file: File): Promise<void> {
        logEvent('file:open', file.name);
        if (await isNpyFile(file)) {
            const array = await loadNpy(file);
            this.resetLoadedState();
            this.insertTensor(array.shape, array.data, {
                name: file.name.replace(/\.npy$/i, '') || 'Tensor',
                dtype: array.dtype,
            });
            return;
        }
        const { manifest, tensors } = await loadBundle(file);
        this.loadBundleData(manifest, tensors);
    }

    public loadBundleData(manifest: BundleManifest, tensors: Map<string, NumericArray>): void {
        const shouldFitCamera = this.shouldAutoFitSnapshot(manifest.viewer);
        this.resetLoadedState();
        manifest.tensors.forEach((entry) => {
            const data = tensors.get(entry.id);
            if (!data) throw new Error(`Bundle tensor ${entry.id} is missing bytes.`);
            this.insertTensor(entry.shape, data, {
                id: entry.id,
                name: entry.name,
                offset: entry.offset,
                dtype: entry.dtype,
                rebuild: false,
                emit: false,
            });
            if (entry.colorInstructions?.length) this.applyColorInstructions(entry.id, entry.colorInstructions);
        });
        this.applySnapshot(manifest.viewer);
        this.rebuildAllMeshes({ fitCamera: shouldFitCamera });
    }

    public async saveFile(): Promise<Blob> {
        logEvent('file:save');
        if (!this.state.activeTensorId) throw new Error('No tensor loaded.');
        const tensor = this.requireTensor(this.state.activeTensorId);
        return saveNpy(tensor.shape, tensor.data, tensor.dtype);
    }

    public getInspectorModel(): {
        handle: TensorHandle | null;
        tensors: Array<{ id: string; name: string }>;
        colorRanges: Array<{ id: string; name: string; min: number; max: number }>;
        viewInput: string;
        preview: string;
        hiddenTokens: Array<{ token: string; size: number; value: number }>;
        colorRange: { min: number; max: number } | null;
    } {
        const tensors = Array.from(this.tensors.values()).map((tensor) => ({ id: tensor.id, name: tensor.name }));
        const colorRanges = Array.from(this.tensors.values()).map((tensor) => ({
            id: tensor.id,
            name: tensor.name,
            ...tensor.valueRange,
        }));
        const activeTensorId = this.state.activeTensorId && this.tensors.has(this.state.activeTensorId)
            ? this.state.activeTensorId
            : this.tensors.keys().next().value ?? null;
        if (!activeTensorId) {
            return { handle: null, tensors, colorRanges, viewInput: '', preview: '', hiddenTokens: [], colorRange: null };
        }
        const tensor = this.requireTensor(activeTensorId);
        return {
            handle: {
                id: tensor.id,
                name: tensor.name,
                rank: tensor.shape.length,
                shape: tensor.shape,
                dtype: tensor.dtype,
            },
            tensors,
            colorRanges,
            viewInput: tensor.view.canonical,
            preview: buildPreviewExpression(tensor.view),
            hiddenTokens: tensor.view.hiddenTokens.map((token) => ({
                token: token.token,
                size: token.size,
                value: token.value,
            })),
            colorRange: tensor.valueRange,
        };
    }

    public setActiveTensor(tensorId: string): void {
        this.requireTensor(tensorId);
        this.state.activeTensorId = tensorId;
        logEvent('tensor:active', tensorId);
        this.emit();
    }

    public setHiddenTokenValue(tensorId: string, token: string, value: number): TensorViewSnapshot {
        const tensor = this.requireTensor(tensorId);
        const hidden = tensor.view.hiddenTokens.find((entry) => entry.token === token);
        if (!hidden) throw new Error(`Unknown hidden token ${token}.`);
        const nextHiddenIndices = tensor.view.hiddenIndices.slice();
        const clamped = Math.max(0, Math.min(hidden.size - 1, Math.floor(value)));
        const expanded = expandGroupedIndex(hidden.axes, clamped, tensor.shape);
        hidden.axes.forEach((axis, axisIndex) => {
            nextHiddenIndices[axis] = expanded[axisIndex];
        });
        logEvent('tensor:hidden-token', { tensorId, token, value: clamped });
        return this.setTensorView(tensorId, tensor.view.canonical, nextHiddenIndices);
    }

    public destroy(): void {
        window.removeEventListener('resize', this.resize);
        window.removeEventListener('keydown', this.onKeyDown);
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
