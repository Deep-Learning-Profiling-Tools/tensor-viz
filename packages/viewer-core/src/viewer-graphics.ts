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

const LABEL_FONT = new FontLoader().parse(helvetikerBoldFont as never);

export function createLine(points: Vector3[], color: string): Line {
    const geometry = new BufferGeometry().setFromPoints(points);
    const line = new Line(geometry, new LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
    line.renderOrder = 5_000;
    return line;
}

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

export function createTextLabel(text: string, color = '#334155'): Group {
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

export function axisFamilyColor(worldKey: 0 | 1 | 2, familyIndex: number, familyCount: number): string {
    const t = Math.max(1, familyIndex + 1) / Math.max(1, familyCount);
    const color = new Color();
    if (worldKey === 1) color.setRGB(0, t, 0);
    else if (worldKey === 2) color.setRGB(0, 0, t);
    else color.setRGB(t, 0, 0);
    return `#${color.getHexString()}`;
}
