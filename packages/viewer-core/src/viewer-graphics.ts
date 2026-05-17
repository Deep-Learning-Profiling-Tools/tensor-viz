import {
    BufferAttribute,
    BufferGeometry,
    Color,
    DoubleSide,
    Group,
    Line,
    LineBasicMaterial,
    Mesh,
    MeshBasicMaterial,
    ShapeGeometry,
    Vector3,
} from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import helvetikerBoldFont from 'three/examples/fonts/helvetiker_bold.typeface.json';
import { VIEWER_LIMITS } from './validation.js';

const LABEL_FONT = new FontLoader().parse(helvetikerBoldFont as never);

/**
 * create line for the current viewer state.
 *
 * @param points - points input used by this operation (Vector3[]).
 * @param color - color input used by this operation (string).
 * @returns Computed Line value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * createLine(points, color);
 */
export function createLine(points: Vector3[], color: string): Line {
    const geometry = new BufferGeometry().setFromPoints(points);
    const line = new Line(geometry, new LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
    line.renderOrder = 5_000;
    return line;
}

/**
 * initialize vertex colors for the current viewer state.
 *
 * @param geometry - geometry input used by this operation (BufferGeometry).
 * @returns Nothing; the function updates state in place.
 * @noThrows This function has no direct throw path.
 * @example
 * initializeVertexColors(geometry);
 */
export function initializeVertexColors(geometry: BufferGeometry): void {
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

/**
 * create text label for the current viewer state.
 *
 * @param text - Text supplied by the caller.
 * @param color - color input used by this operation (value).
 * @returns Computed Group value for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * createTextLabel(text, color);
 */
export function createTextLabel(text: string, color = '#334155'): Group {
    // use shape text instead of troika so brave/linux stays on the same path as the working line geometry.
    const boundedText = text.length > VIEWER_LIMITS.maxTextLength ? `${text.slice(0, VIEWER_LIMITS.maxTextLength)}...` : text;
    const shapes = LABEL_FONT.generateShapes(boundedText, 1.1);
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

/**
 * return axis family color for the current viewer state.
 *
 * @param worldKey - world key input used by this operation (0 | 1 | 2).
 * @param familyIndex - Index used by this operation.
 * @param familyCount - family count input used by this operation (number).
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * axisFamilyColor(worldKey, familyIndex, familyCount);
 */
export function axisFamilyColor(worldKey: 0 | 1 | 2, familyIndex: number, familyCount: number): string {
    const t = Math.max(1, familyIndex + 1) / Math.max(1, familyCount);
    const color = new Color();
    if (worldKey === 1) color.setRGB(0, t, 0);
    else if (worldKey === 2) color.setRGB(0, 0, t);
    else color.setRGB(t, 0, 0);
    return `#${color.getHexString()}`;
}
