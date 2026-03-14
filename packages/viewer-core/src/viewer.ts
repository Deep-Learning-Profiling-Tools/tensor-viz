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
    Vector2,
    Vector3,
    WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import helvetikerBoldFont from 'three/examples/fonts/helvetiker_bold.typeface.json';
import { loadBundle, saveBundle } from './bundle.js';
import {
    displayExtent,
    displayExtent2D,
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
    parseTensorView,
    product,
} from './view.js';
import type {
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

type CanvasEntry = {
    x: number;
    y: number;
    size: number;
    hover: HoverInfo;
};

type CanvasSimpleGrid = {
    width: number;
    height: number;
    startX: number;
    startY: number;
    stepX: number;
    stepY: number;
    tensorId: string;
};

type CanvasCache = {
    canvas: HTMLCanvasElement;
    entries: CanvasEntry[] | null;
    simpleGrid: CanvasSimpleGrid | null;
    extent: { x: number; y: number };
};

const CANVAS_WORLD_SCALE = 4;
const SVG_CELL_LIMIT = 4_096;
const LOG_PREFIX = '[tensor-viz]';
const LABEL_FONT = new FontLoader().parse(helvetikerBoldFont as never);

function createLine(points: Vector3[], color: string): Line {
    const geometry = new BufferGeometry().setFromPoints(points);
    const line = new Line(geometry, new LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
    line.renderOrder = 5_000;
    return line;
}

function logEvent(event: string, details?: unknown): void {
    if (details === undefined) console.log(LOG_PREFIX, event);
    else console.log(LOG_PREFIX, event, details);
}

function initializeCubeVertexColors(geometry: BoxGeometry): void {
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

function heatmapGray(value: number, min: number, max: number): number {
    const range = max - min || 1;
    const t = Math.max(0, Math.min(1, (value - min) / range));
    return Math.round(t * 255);
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
    private readonly tensors = new Map<string, TensorRecord>();
    private readonly tensorMeshes = new Map<string, Group>();
    private readonly listeners = new Set<(snapshot: ViewerSnapshot) => void>();
    private readonly canvasCache = new Map<string, CanvasCache>();
    private readonly state: ViewerState = {
        displayMode: '2d',
        heatmap: true,
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
        initializeCubeVertexColors(this.cubeGeometry);
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
        this.scene.add(this.hoverOutline);
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
        const span = 20;
        this.orthographicCamera.left = -(span * width / height) / 2;
        this.orthographicCamera.right = (span * width / height) / 2;
        this.orthographicCamera.top = span / 2;
        this.orthographicCamera.bottom = -span / 2;
        this.orthographicCamera.updateProjectionMatrix();
        logEvent('viewport:resize', { width, height });
        this.requestRender();
    };

    private canvasScale(): number {
        return this.flatCanvas.width / Math.max(1, this.flatCanvas.clientWidth || this.flatCanvas.width || 1);
    }

    private sameHover(left: HoverInfo | null, right: HoverInfo | null): boolean {
        return !!left
            && !!right
            && left.tensorId === right.tensorId
            && left.fullCoord.length === right.fullCoord.length
            && left.fullCoord.every((value, index) => value === right.fullCoord[index]);
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
        const outlineCoord = mapDisplayCoordToOutlineCoord(hover.displayCoord, tensor.view);
        const position = displayPositionForCoord2D(outlineCoord, outlineShape);
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
        const outlineCoord = mapDisplayCoordToOutlineCoord(hover.displayCoord, tensor.view);
        const center = displayPositionForCoord(outlineCoord, outlineShape).add(vectorFromTuple(tensor.offset));
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
            const extent = displayExtent2D(tensor.view.outlineShape);
            minX = Math.min(minX, tensor.offset[0] - extent.x / 2);
            maxX = Math.max(maxX, tensor.offset[0] + extent.x / 2);
            minY = Math.min(minY, tensor.offset[1] - extent.y / 2);
            maxY = Math.max(maxY, tensor.offset[1] + extent.y / 2);
        });

        const width = Math.max(1, (maxX - minX) * CANVAS_WORLD_SCALE);
        const height = Math.max(1, (maxY - minY) * CANVAS_WORLD_SCALE);
        const inset = 48 * this.canvasScale();
        this.canvasZoom = Math.max(
            0.15,
            Math.min(
                24,
                Math.min(
                    (this.flatCanvas.width - inset * 2) / width,
                    (this.flatCanvas.height - inset * 2) / height,
                ),
            ),
        );
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
        const nextZoom = Math.max(0.15, Math.min(24, this.canvasZoom * scale));
        const cursorX = (event.clientX - rect.left) * (this.flatCanvas.width / rect.width) - this.flatCanvas.width / 2 - this.canvasPan.x;
        const cursorY = (event.clientY - rect.top) * (this.flatCanvas.height / rect.height) - this.flatCanvas.height / 2 - this.canvasPan.y;
        this.canvasPan.x -= cursorX * ((nextZoom / this.canvasZoom) - 1);
        this.canvasPan.y -= cursorY * ((nextZoom / this.canvasZoom) - 1);
        this.canvasZoom = nextZoom;
        logEvent('2d:zoom', { zoom: this.canvasZoom });
        this.requestRender();
        this.emit();
    };

    private canvasPointerToHover(clientX: number, clientY: number): HoverInfo | null {
        const rect = this.flatCanvas.getBoundingClientRect();
        const scaleX = this.flatCanvas.width / rect.width;
        const scaleY = this.flatCanvas.height / rect.height;
        const x = (clientX - rect.left) * scaleX - this.flatCanvas.width / 2 - this.canvasPan.x;
        const y = (clientY - rect.top) * scaleY - this.flatCanvas.height / 2 - this.canvasPan.y;
        const localX = x / this.canvasZoom;
        const localY = y / this.canvasZoom;

        for (const tensor of Array.from(this.tensors.values()).reverse()) {
            let cache = this.canvasCache.get(tensor.id);
            if (!cache) {
                cache = this.buildCanvasCache(tensor);
                this.canvasCache.set(tensor.id, cache);
            }
            if (!cache) continue;
            const offsetX = tensor.offset[0] * CANVAS_WORLD_SCALE;
            const offsetY = -tensor.offset[1] * CANVAS_WORLD_SCALE;
            const cacheX = localX - offsetX + cache.canvas.width / 2;
            const cacheY = localY - offsetY + cache.canvas.height / 2;

            if (!cache.entries) continue;
            for (const entry of cache.entries) {
                if (
                    cacheX >= entry.x
                    && cacheX <= entry.x + entry.size
                    && cacheY >= entry.y
                    && cacheY <= entry.y + entry.size
                ) return entry.hover;
            }
        }
        return null;
    }

    private readonly onCanvasPointerMove = (event: PointerEvent): void => {
        if (this.isCanvasPanning) {
            this.canvasPan.x += (event.clientX - this.lastCanvasPointer.x) * (this.flatCanvas.width / this.flatCanvas.getBoundingClientRect().width);
            this.canvasPan.y += (event.clientY - this.lastCanvasPointer.y) * (this.flatCanvas.height / this.flatCanvas.getBoundingClientRect().height);
            this.lastCanvasPointer = { x: event.clientX, y: event.clientY };
            this.requestRender();
            return;
        }
        const currentHover = this.state.hover;
        const nextHover = this.canvasPointerToHover(event.clientX, event.clientY);
        const hover = currentHover && this.hoveredCellContainsPointer(event.clientX, event.clientY, currentHover) ? currentHover : nextHover;
        this.state.hover = hover;
        if (hover) this.state.lastHover = hover;
        const hoverKey = hover ? `${hover.tensorId}:${hover.fullCoord.join(',')}:${hover.value}` : null;
        if (hoverKey !== this.lastHoverLogKey) {
            this.lastHoverLogKey = hoverKey;
            logEvent('2d:hover', hover ?? 'none');
        }
        this.requestRender();
        this.emit();
    };

    private readonly onCanvasPointerLeave = (): void => {
        this.state.hover = null;
        this.lastHoverLogKey = null;
        logEvent('2d:hover', 'leave');
        this.requestRender();
        this.emit();
    };

    private readonly onCanvasClick = (event: PointerEvent): void => {
        const hover = this.canvasPointerToHover(event.clientX, event.clientY);
        if (!hover) return;
        this.state.activeTensorId = hover.tensorId;
        this.state.hover = hover;
        this.state.lastHover = hover;
        logEvent('2d:select', hover);
        this.emit();
    };

    private readonly onPointerMove = (event: PointerEvent): void => {
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const meshes = Array.from(this.tensorMeshes.values()).map((group) => group.children[0]).filter(Boolean);
        const hits = this.raycaster.intersectObjects(meshes, false);
        const hit = hits[0];
        const currentHover = this.state.hover;
        if (!hit || hit.instanceId === undefined) {
            if (currentHover && this.hoveredCellContainsPointer(event.clientX, event.clientY, currentHover)) {
                this.requestRender();
                return;
            }
            this.state.hover = null;
            this.hoverOutline.visible = false;
            this.emit();
            this.requestRender();
            return;
        }

        const mesh = hit.object as InstancedMesh;
        const meta = mesh.userData.meta as MeshMeta;
        const displayCoord = meta.renderShape.length === 0 ? [] : unravelIndex(hit.instanceId, meta.renderShape);
        const tensor = this.requireTensor(meta.tensorId);
        const fullCoord = mapDisplayCoordToFullCoord(displayCoord, tensor.view);
        const value = numericValue(tensor.data, this.linearIndex(fullCoord, tensor.shape));
        const colorSource: HoverInfo['colorSource'] = tensor.customColors.has(coordKey(fullCoord))
            ? 'custom'
            : this.state.heatmap
                ? 'heatmap'
                : 'base';
        const hover = {
            tensorId: meta.tensorId,
            tensorName: this.tensors.get(meta.tensorId)?.name ?? meta.tensorId,
            displayCoord: displayCoord.slice(),
            fullCoord: fullCoord.slice(),
            value,
            colorSource,
        };
        if (currentHover && !this.sameHover(currentHover, hover) && this.hoveredCellContainsPointer(event.clientX, event.clientY, currentHover)) {
            this.requestRender();
            return;
        }
        this.state.hover = hover;
        this.state.lastHover = hover;
        const hoverKey = `${hover.tensorId}:${hover.fullCoord.join(',')}:${hover.value}`;
        if (hoverKey !== this.lastHoverLogKey) {
            this.lastHoverLogKey = hoverKey;
            logEvent('3d:hover', hover);
        }
        const instanceMatrix = new Matrix4();
        const position = new Vector3();
        mesh.getMatrixAt(hit.instanceId, instanceMatrix);
        position.setFromMatrixPosition(instanceMatrix);
        this.hoverOutline.position.copy(position);
        this.hoverOutline.visible = true;
        this.emit();
        this.requestRender();
    };

    private readonly onPointerLeave = (): void => {
        this.state.hover = null;
        this.hoverOutline.visible = false;
        this.lastHoverLogKey = null;
        logEvent('3d:hover', 'leave');
        this.emit();
        this.requestRender();
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
        const radius = Math.max(size.x, size.y, size.z, 8);
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
        this.canvasCache.clear();
        this.tensors.forEach((tensor) => {
            const group = this.buildTensorGroup(tensor);
            this.tensorMeshes.set(tensor.id, group);
            this.scene.add(group);
        });
        if (shouldFitCamera) this.fitCamera();
        else this.requestRender();
        this.emit();
    }

    private buildOutline(extent: Vector3, offset: Vec3): LineSegments {
        const outline = new LineSegments(
            new EdgesGeometry(new BoxGeometry(extent.x + 0.2, extent.y + 0.2, extent.z + 0.2)),
            new LineBasicMaterial({ color: '#334155' }),
        );
        outline.position.copy(vectorFromTuple(offset));
        return outline;
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
            const startPos = displayPositionForCoord(start, shape);
            const endPos = displayPositionForCoord(end, shape);
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
        const matrix = new Matrix4();
        const { min, max } = computeMinMax(tensor.data);

        for (let index = 0; index < count; index += 1) {
            const displayCoord = count === 1 && tensor.view.displayShape.length === 0 ? [] : unravelIndex(index, renderShape);
            const fullCoord = mapDisplayCoordToFullCoord(displayCoord, tensor.view);
            const outlineCoord = mapDisplayCoordToOutlineCoord(displayCoord, tensor.view);
            const fullLinear = this.linearIndex(fullCoord, tensor.shape);
            const value = numericValue(tensor.data, fullLinear);
            const position = displayPositionForCoord(outlineCoord, outlineShape).add(vectorFromTuple(tensor.offset));
            matrix.makeTranslation(position.x, position.y, position.z);
            mesh.setMatrixAt(index, matrix);
            mesh.setColorAt(index, this.cellColor(tensor, fullCoord, value, min, max));
        }

        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        mesh.computeBoundingBox();
        mesh.computeBoundingSphere();
        mesh.material.needsUpdate = true;
        mesh.userData.meta = { tensorId: tensor.id, renderShape } satisfies MeshMeta;
        group.add(mesh);
        const outlineExtent = displayExtent(outlineShape);
        group.add(this.buildOutline(outlineExtent, tensor.offset));
        if (this.state.showDimensionLines && this.state.displayMode === '3d') group.add(this.buildDimensionGuides(outlineExtent, outlineShape, tensor.offset, tensor.view.tokens.map((token) => token.label.toUpperCase())));
        return group;
    }

    private buildCanvasCache(tensor: TensorRecord): CanvasCache {
        const renderShape = tensor.view.displayShape.length === 0 ? [1] : tensor.view.displayShape;
        const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
        const extent = displayExtent2D(outlineShape);
        const canvas = document.createElement('canvas');
        const padding = CANVAS_WORLD_SCALE * 4;
        canvas.width = Math.max(8, Math.ceil(extent.x * CANVAS_WORLD_SCALE) + padding * 2);
        canvas.height = Math.max(8, Math.ceil(extent.y * CANVAS_WORLD_SCALE) + padding * 2);
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Unable to create offscreen canvas context.');
        context.imageSmoothingEnabled = false;
        context.clearRect(0, 0, canvas.width, canvas.height);

        const { min, max } = computeMinMax(tensor.data);
        const count = product(renderShape);
        const cellSize = Math.max(1, Math.round(CANVAS_WORLD_SCALE));
        const entries: CanvasEntry[] = [];
        const shouldTrackEntries = count <= 200_000;

        for (let index = 0; index < count; index += 1) {
            const displayCoord = count === 1 && tensor.view.displayShape.length === 0 ? [] : unravelIndex(index, renderShape);
            const fullCoord = mapDisplayCoordToFullCoord(displayCoord, tensor.view);
            const outlineCoord = mapDisplayCoordToOutlineCoord(displayCoord, tensor.view);
            const fullLinear = this.linearIndex(fullCoord, tensor.shape);
            const value = numericValue(tensor.data, fullLinear);
            const position = displayPositionForCoord2D(outlineCoord, outlineShape);
            const x = Math.round((position.x + extent.x / 2 - 0.5) * CANVAS_WORLD_SCALE) + padding;
            const y = Math.round((extent.y / 2 - position.y - 0.5) * CANVAS_WORLD_SCALE) + padding;
            context.fillStyle = `#${this.cellColor(tensor, fullCoord, value, min, max).getHexString()}`;
            context.fillRect(x, y, cellSize, cellSize);
            if (shouldTrackEntries) {
                const colorSource: HoverInfo['colorSource'] = tensor.customColors.has(coordKey(fullCoord))
                    ? 'custom'
                    : this.state.heatmap
                        ? 'heatmap'
                        : 'base';
                entries.push({
                    x,
                    y,
                    size: cellSize,
                    hover: {
                        tensorId: tensor.id,
                        tensorName: tensor.name,
                        displayCoord: displayCoord.slice(),
                        fullCoord: fullCoord.slice(),
                        value,
                        colorSource,
                    },
                });
            }
        }
        return {
            canvas,
            entries: shouldTrackEntries ? entries : null,
            simpleGrid: null,
            extent,
        };
    }

    private render2DOverlay(): void {
        this.flatOverlay.replaceChildren();
        const add = (tag: string, attrs: Record<string, string>): SVGElement => {
            const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
            Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
            this.flatOverlay.appendChild(element);
            return element;
        };

        this.tensors.forEach((tensor) => {
            const renderShape = tensor.view.displayShape.length === 0 ? [1] : tensor.view.displayShape;
            const outlineShape = tensor.view.outlineShape.length === 0 ? [1] : tensor.view.outlineShape;
            const count = product(renderShape);
            const { min, max } = computeMinMax(tensor.data);
            if (count <= SVG_CELL_LIMIT) {
                for (let index = 0; index < count; index += 1) {
                    const displayCoord = count === 1 && tensor.view.displayShape.length === 0 ? [] : unravelIndex(index, renderShape);
                    const fullCoord = mapDisplayCoordToFullCoord(displayCoord, tensor.view);
                    const outlineCoord = mapDisplayCoordToOutlineCoord(displayCoord, tensor.view);
                    const value = numericValue(tensor.data, this.linearIndex(fullCoord, tensor.shape));
                    const position = displayPositionForCoord2D(outlineCoord, outlineShape);
                    const topLeft = this.projectCanvasPoint(
                        tensor.offset[0] + position.x - 0.5,
                        tensor.offset[1] + position.y + 0.5,
                    );
                    const bottomRight = this.projectCanvasPoint(
                        tensor.offset[0] + position.x + 0.5,
                        tensor.offset[1] + position.y - 0.5,
                    );
                    add('rect', {
                        x: String(Math.min(topLeft.x, bottomRight.x)),
                        y: String(Math.min(topLeft.y, bottomRight.y)),
                        width: String(Math.abs(bottomRight.x - topLeft.x)),
                        height: String(Math.abs(bottomRight.y - topLeft.y)),
                        fill: `#${this.cellColor(tensor, fullCoord, value, min, max).getHexString()}`,
                    });
                }
            }

            const extent = displayExtent2D(tensor.view.outlineShape);
            const left = tensor.offset[0] - extent.x / 2;
            const right = tensor.offset[0] + extent.x / 2;
            const top = tensor.offset[1] + extent.y / 2;
            const bottom = tensor.offset[1] - extent.y / 2;
            const topLeft = this.projectCanvasPoint(left, top);
            const bottomRight = this.projectCanvasPoint(right, bottom);
            const rectX = Math.min(topLeft.x, bottomRight.x);
            const rectY = Math.min(topLeft.y, bottomRight.y);
            const rectWidth = Math.abs(bottomRight.x - topLeft.x);
            const rectHeight = Math.abs(bottomRight.y - topLeft.y);
            add('rect', {
                x: String(rectX),
                y: String(rectY),
                width: String(rectWidth),
                height: String(rectHeight),
                fill: 'none',
                stroke: '#334155',
                'stroke-width': String(Math.max(1, 1.5 * this.canvasScale())),
            });
            const rank = tensor.view.outlineShape.length;
            if (this.state.hover?.tensorId === tensor.id) {
                const hoverCoord = mapDisplayCoordToOutlineCoord(this.state.hover.displayCoord, tensor.view);
                const hoverPosition = displayPositionForCoord2D(hoverCoord, tensor.view.outlineShape);
                const hoverTopLeft = this.projectCanvasPoint(tensor.offset[0] + hoverPosition.x - 0.5, tensor.offset[1] + hoverPosition.y + 0.5);
                const hoverBottomRight = this.projectCanvasPoint(tensor.offset[0] + hoverPosition.x + 0.5, tensor.offset[1] + hoverPosition.y - 0.5);
                add('rect', {
                    x: String(Math.min(hoverTopLeft.x, hoverBottomRight.x)),
                    y: String(Math.min(hoverTopLeft.y, hoverBottomRight.y)),
                    width: String(Math.abs(hoverBottomRight.x - hoverTopLeft.x)),
                    height: String(Math.abs(hoverBottomRight.y - hoverTopLeft.y)),
                    fill: 'none',
                    stroke: '#f59e0b',
                    'stroke-width': String(Math.max(2, 2 * this.canvasScale())),
                });
            }
            if (!this.state.showDimensionLines) return;

            const labels = tensor.view.tokens.map((token) => token.label.toUpperCase());
            const families = new Map<number, number[]>();
            for (let axis = 0; axis < rank; axis += 1) {
                const key = axisWorldKey2D(rank, axis);
                const family = families.get(key) ?? [];
                family.push(axis);
                families.set(key, family);
            }
            const zoomScale = Math.max(1, Math.sqrt(this.canvasZoom));
            const fontSize = 16 * this.canvasScale() * zoomScale;
            const guide = 28 * this.canvasScale();
            const entries = tensor.view.outlineShape.map((size, axis) => {
                const familyKey = axisWorldKey2D(rank, axis);
                const family = families.get(familyKey) ?? [axis];
                const familyPos = Math.max(0, family.indexOf(axis));
                const start = new Array(rank).fill(0);
                const end = start.slice();
                family.forEach((familyAxis) => {
                    if (familyAxis >= axis) end[familyAxis] = Math.max(0, tensor.view.outlineShape[familyAxis] - 1);
                });
                const startPos = displayPositionForCoord2D(start, tensor.view.outlineShape);
                const endPos = displayPositionForCoord2D(end, tensor.view.outlineShape);
                const delta = { x: endPos.x - startPos.x, y: endPos.y - startPos.y };
                const length = Math.hypot(delta.x, delta.y) || 1;
                const axisDir = { x: delta.x / length, y: delta.y / length };
                const extentStart = { x: startPos.x - axisDir.x * 0.5, y: startPos.y - axisDir.y * 0.5 };
                const extentEnd = { x: endPos.x + axisDir.x * 0.5, y: endPos.y + axisDir.y * 0.5 };
                return {
                    axis,
                    familyKey,
                    color: axisFamilyColor(familyKey as 0 | 1 | 2, familyPos, family.length),
                    label: `${labels[axis] ?? 'X'}: ${size}`,
                    start: this.projectCanvasPoint(tensor.offset[0] + extentStart.x, tensor.offset[1] + extentStart.y),
                    end: this.projectCanvasPoint(tensor.offset[0] + extentEnd.x, tensor.offset[1] + extentEnd.y),
                };
            });
            ([0, 1] as const).forEach((familyKey) => {
                const familyEntries = entries.filter((entry) => entry.familyKey === familyKey);
                familyEntries.forEach((entry, index) => {
                    const reverseIndex = familyEntries.length - 1 - index;
                    const dir = familyKey === 0 ? { x: 0, y: 1 } : { x: -1, y: 0 };
                    const offsetScale = guide + reverseIndex * (guide * 0.9);
                    const startGuide = { x: entry.start.x + dir.x * offsetScale, y: entry.start.y + dir.y * offsetScale };
                    const endGuide = { x: entry.end.x + dir.x * offsetScale, y: entry.end.y + dir.y * offsetScale };
                    add('line', {
                        x1: String(entry.start.x),
                        y1: String(entry.start.y),
                        x2: String(startGuide.x),
                        y2: String(startGuide.y),
                        stroke: entry.color,
                        'stroke-width': String(Math.max(2, 2 * this.canvasScale())),
                    });
                    add('line', {
                        x1: String(entry.end.x),
                        y1: String(entry.end.y),
                        x2: String(endGuide.x),
                        y2: String(endGuide.y),
                        stroke: entry.color,
                        'stroke-width': String(Math.max(2, 2 * this.canvasScale())),
                    });
                    add('line', {
                        x1: String(startGuide.x),
                        y1: String(startGuide.y),
                        x2: String(endGuide.x),
                        y2: String(endGuide.y),
                        stroke: entry.color,
                        'stroke-width': String(Math.max(2, 2 * this.canvasScale())),
                    });
                    const text = add('text', {
                        x: String((startGuide.x + endGuide.x) / 2 + dir.x * 8 * this.canvasScale()),
                        y: String((startGuide.y + endGuide.y) / 2 + dir.y * (fontSize + 5 * this.canvasScale())),
                        fill: entry.color,
                        'font-size': String(fontSize * 1.3),
                        'font-weight': '700',
                        'font-family': '"IBM Plex Mono", monospace',
                        'text-anchor': familyKey === 0 ? 'middle' : 'end',
                        'dominant-baseline': 'middle',
                        stroke: '#ffffff',
                        'stroke-width': String(1.15 * this.canvasScale() * zoomScale),
                        'paint-order': 'stroke',
                    });
                    text.textContent = entry.label;
                });
            });
        });
    }

    private render2D(): void {
        this.flatContext.setTransform(1, 0, 0, 1, 0, 0);
        this.flatContext.clearRect(0, 0, this.flatCanvas.width, this.flatCanvas.height);
        this.flatContext.fillStyle = '#e5e7eb';
        this.flatContext.fillRect(0, 0, this.flatCanvas.width, this.flatCanvas.height);
        this.flatContext.save();
        this.flatContext.translate(this.flatCanvas.width / 2 + this.canvasPan.x, this.flatCanvas.height / 2 + this.canvasPan.y);
        this.flatContext.scale(this.canvasZoom, this.canvasZoom);
        this.tensors.forEach((tensor) => {
            const renderShape = tensor.view.displayShape.length === 0 ? [1] : tensor.view.displayShape;
            if (product(renderShape) <= SVG_CELL_LIMIT) return;
            let cache = this.canvasCache.get(tensor.id);
            if (!cache) {
                cache = this.buildCanvasCache(tensor);
                this.canvasCache.set(tensor.id, cache);
            }
            this.flatContext.drawImage(
                cache.canvas,
                Math.round(tensor.offset[0] * CANVAS_WORLD_SCALE - cache.canvas.width / 2),
                Math.round(-tensor.offset[1] * CANVAS_WORLD_SCALE - cache.canvas.height / 2),
            );
        });
        this.flatContext.restore();
        this.render2DOverlay();
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
        if (customColor?.kind === 'hs') return colorFromHueSaturation(customColor.value, this.state.heatmap ? (max === min ? 1 : Math.max(0, Math.min(1, (value - min) / (max - min)))) : 1);
        if (!this.state.heatmap) return BASE_COLOR.clone();
        const gray = heatmapGray(value, min, max) / 255;
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
            offset: options.offset ?? (this.tensors.size === 0 ? [0, 0, 0] : [this.tensors.size * 6, 0, 0]),
            view: parsed.spec,
            customColors: new Map(),
        };
        this.tensors.set(id, tensor);
        this.state.activeTensorId ??= id;
        if (options.rebuild === false) {
            this.emit();
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
        this.rebuildAllMeshes();
    }

    public clear(): void {
        logEvent('tensor:clear');
        this.tensors.clear();
        this.state.activeTensorId = null;
        this.state.hover = null;
        this.state.lastHover = null;
        this.rebuildAllMeshes();
    }

    public getSnapshot(): ViewerSnapshot {
        return {
            version: 1,
            displayMode: this.state.displayMode,
            heatmap: this.state.heatmap,
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
        this.state.displayMode = snapshot.displayMode;
        this.state.heatmap = snapshot.heatmap;
        this.state.showDimensionLines = snapshot.showDimensionLines;
        this.state.showInspectorPanel = snapshot.showInspectorPanel;
        this.state.showHoverDetailsPanel = snapshot.showHoverDetailsPanel;
        this.state.activeTensorId = snapshot.activeTensorId;
        this.setDisplayMode(snapshot.displayMode);
        this.camera.position.copy(vectorFromTuple(snapshot.camera.position));
        this.controls.target.copy(vectorFromTuple(snapshot.camera.target));
        this.camera.rotation.set(...snapshot.camera.rotation);
        if ('zoom' in this.camera) this.camera.zoom = snapshot.camera.zoom;
        this.camera.updateProjectionMatrix();

        snapshot.tensors.forEach((entry) => {
            const tensor = this.tensors.get(entry.id);
            if (!tensor) return;
            tensor.offset = entry.offset;
            const parsed = parseTensorView(tensor.shape, entry.view.visible, entry.view.hiddenIndices);
            if (parsed.ok) tensor.view = parsed.spec;
        });
        this.rebuildAllMeshes();
    }

    public subscribe(listener: (snapshot: ViewerSnapshot) => void): () => void {
        this.listeners.add(listener);
        listener(this.getSnapshot());
        return () => this.listeners.delete(listener);
    }

    public setDisplayMode(mode: '2d' | '3d'): void {
        logEvent('display:mode', mode);
        const previousTarget = this.controls.target.clone();
        this.state.displayMode = mode;
        this.controls.dispose();
        this.camera = mode === '2d' ? this.orthographicCamera : this.perspectiveCamera;
        this.controls = this.createControls(this.camera);
        this.controls.enableRotate = mode === '3d';
        this.renderer.domElement.style.display = mode === '3d' ? 'block' : 'none';
        this.flatCanvas.style.display = mode === '2d' ? 'block' : 'none';
        this.flatOverlay.style.display = mode === '2d' ? 'block' : 'none';
        this.controls.target.copy(previousTarget);
        if (mode === '3d') {
            this.camera.position.set(previousTarget.x + 20, previousTarget.y + 16, previousTarget.z + 24);
            this.camera.lookAt(previousTarget);
        }
        this.rebuildAllMeshes();
        this.requestRender();
        this.emit();
    }

    public toggleHeatmap(force?: boolean): boolean {
        this.state.heatmap = force ?? !this.state.heatmap;
        logEvent('display:heatmap', this.state.heatmap);
        this.rebuildAllMeshes();
        return this.state.heatmap;
    }

    public toggleDimensionLines(force?: boolean): boolean {
        this.state.showDimensionLines = force ?? !this.state.showDimensionLines;
        logEvent('display:dimension-lines', this.state.showDimensionLines);
        this.rebuildAllMeshes();
        return this.state.showDimensionLines;
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
        const extent = displayExtent(tensor.view.outlineShape);
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
        const parsed = parseTensorView(tensor.shape, spec, hiddenIndices ?? tensor.view.hiddenIndices);
        if (!parsed.ok) throw new Error(parsed.errors.join(' '));
        tensor.view = parsed.spec;
        logEvent('tensor:view', { tensorId, view: tensor.view.canonical, hiddenIndices: tensor.view.hiddenIndices });
        this.rebuildAllMeshes();
        return {
            visible: tensor.view.canonical,
            hiddenIndices: tensor.view.hiddenIndices.slice(),
        };
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
        const { manifest, tensors } = await loadBundle(file);
        this.clear();
        manifest.tensors.forEach((entry) => {
            const data = tensors.get(entry.id);
            if (!data) throw new Error(`Bundle tensor ${entry.id} is missing bytes.`);
            const handle = this.insertTensor(entry.shape, data, {
                id: entry.id,
                name: entry.name,
                offset: entry.offset,
                dtype: entry.dtype,
                rebuild: false,
            });
            this.setTensorView(handle.id, entry.view.visible, entry.view.hiddenIndices);
            if (entry.colorInstructions?.length) this.applyColorInstructions(handle.id, entry.colorInstructions);
        });
        this.restoreSnapshot(manifest.viewer);
    }

    public async saveFile(): Promise<Blob> {
        logEvent('file:save');
        return saveBundle(this.getSnapshot(), Array.from(this.tensors.values()));
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
            ...computeMinMax(tensor.data),
        }));
        if (!this.state.activeTensorId) {
            return { handle: null, tensors, colorRanges, viewInput: '', preview: '', hiddenTokens: [], colorRange: null };
        }
        const tensor = this.requireTensor(this.state.activeTensorId);
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
            colorRange: computeMinMax(tensor.data),
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
