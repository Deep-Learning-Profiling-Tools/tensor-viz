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
import helvetikerBoldFont from 'three/examples/fonts/helvetiker_bold.typeface.json' with { type: 'json' };
import { VIEWER_LIMITS } from './validation.js';

const LABEL_FONT = new FontLoader().parse(helvetikerBoldFont as never);

/**
 * Builds a Three.js polyline for viewer overlays such as hover outlines and dimension guides, using a high render order so the line draws above tensor meshes.
 *
 * @param points - Ordered world-space vertices that define the line segments to store in the `BufferGeometry`.
 * @param color - Three.js-compatible material color string, such as a hex color used for hover or selection outlines.
 * @returns A `Line` whose geometry contains the supplied vertices and whose transparent `LineBasicMaterial` disables depth testing and renders with order `5000`.
 * @noThrows This helper has no validation or explicit error branch; it directly passes caller-provided vertices and color to Three.js constructors.
 * @example
 * const outline = createLine([
 *     new Vector3(-0.5, -0.5, 0.05),
 *     new Vector3(0.5, -0.5, 0.05),
 * ], '#38bdf8');
 *
 * outline.renderOrder; // 5000
 * (outline.material as LineBasicMaterial).depthTest; // false
 */
export function createLine(points: Vector3[], color: string): Line {
    const geometry = new BufferGeometry().setFromPoints(points);
    const line = new Line(geometry, new LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
    line.renderOrder = 5_000;
    return line;
}

/**
 * Adds a grayscale vertex color attribute to a Three.js geometry from its normals.
 *
 * The viewer uses the generated colors as lightweight directional shading for
 * shared cube and plane geometries before they are instanced into tensor cells.
 * Each normal is dotted with the fixed viewer light direction and written as an
 * RGB triplet in `geometry.attributes.color`.
 *
 * @param geometry - BufferGeometry that already contains matching `position` and `normal` attributes for every vertex to shade.
 * @returns Nothing; the geometry is mutated by installing a `color` BufferAttribute with three float components per position vertex.
 * @noThrows With the viewer's cube and plane geometries the required attributes are created by Three.js before this helper runs, so the helper has no expected throw path for normal viewer setup.
 * @example
 * const geometry = new BoxGeometry(1, 1, 1);
 * initializeVertexColors(geometry);
 *
 * const colors = geometry.getAttribute('color');
 * console.assert(colors.itemSize === 3);
 * console.assert(colors.count === geometry.getAttribute('position').count);
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
 * Builds a camera-facing Three.js label group for tensor names and dimension guides.
 *
 * The label text is converted to shape geometry, centered, drawn with a
 * depth-independent material, and placed in a group whose `onBeforeRender`
 * callback copies the camera quaternion so the label billboards toward the user.
 * Overlong text is truncated to the viewer text limit before shape generation.
 *
 * @param text - Label string to render, such as a tensor name or an axis-size label; strings longer than the viewer limit are truncated with an ellipsis.
 * @param color - Three.js material color for the label glyphs, usually a CSS hex string matching the guide or tensor annotation.
 * @returns A non-culled Group containing the text mesh, with high render order and billboard behavior ready to position in the scene.
 * @noThrows For normal viewer label strings and Three.js-supported color values, label creation only allocates geometry/material objects and has no expected viewer-level throw path.
 * @example
 * const label = createTextLabel('batch: 32', '#00ff00');
 *
 * console.assert(label.children.length === 1);
 * console.assert(label.frustumCulled === false);
 * console.assert(label.renderOrder === 10000);
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
 * Chooses the guide color for one axis family in the viewer's world-axis mapping.
 *
 * World key `0` produces red shades, `1` produces green shades, and `2`
 * produces blue shades. Families later in the same world-axis group receive a
 * brighter channel value so adjacent dimension guides remain distinguishable.
 *
 * @param worldKey - Axis family channel: `0` for red/X-like guides, `1` for green/Y-like guides, or `2` for blue/Z-like guides.
 * @param familyIndex - Zero-based position of the family within all guides mapped to the same world axis.
 * @param familyCount - Number of families sharing that world axis; values below one are clamped for color calculation.
 * @returns CSS hex color string used for dimension guide lines and their matching text labels.
 * @noThrows The calculation clamps the family ratio and only uses Three.js `Color` channel assignment, so invalid family counts do not create an expected throw path.
 * @example
 * console.assert(axisFamilyColor(0, 0, 2) === '#800000');
 * console.assert(axisFamilyColor(1, 1, 2) === '#00ff00');
 * console.assert(axisFamilyColor(2, 0, 0) === '#0000ff');
 */
export function axisFamilyColor(worldKey: 0 | 1 | 2, familyIndex: number, familyCount: number): string {
    const t = Math.max(1, familyIndex + 1) / Math.max(1, familyCount);
    const color = new Color();
    if (worldKey === 1) color.setRGB(0, t, 0);
    else if (worldKey === 2) color.setRGB(0, 0, t);
    else color.setRGB(t, 0, 0);
    return `#${color.getHexString()}`;
}
