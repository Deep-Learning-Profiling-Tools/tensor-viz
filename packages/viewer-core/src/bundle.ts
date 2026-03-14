import JSZip from 'jszip';
import { unravelIndex } from './layout.js';
import { createTypedArray } from './npy.js';
import { product } from './view.js';
import type { BundleManifest, ColorInstruction, CustomColor, NumericArray, TensorRecord, ViewerSnapshot } from './types.js';

function byteView(data: NumericArray): Uint8Array {
    return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

function sparseColorInstructions(tensor: TensorRecord): ColorInstruction[] | undefined {
    if (tensor.customColors.size === 0) return undefined;
    const coords: number[][] = [];
    const colors: number[][] = [];
    for (let index = 0; index < product(tensor.shape); index += 1) {
        const coord = unravelIndex(index, tensor.shape);
        const color = tensor.customColors.get(coord.join(','));
        if (!color) continue;
        coords.push(coord);
        colors.push([...color.value]);
    }
    if (coords.length === 0) return undefined;
    const groups = new Map<string, { mode: 'rgba' | 'hs'; coords: number[][]; color: number[] }>();
    coords.forEach((coord, index) => {
        const color = colors[index];
        if (!color) return;
        const mode = color.length === 2 ? 'hs' : 'rgba';
        const key = `${mode}:${color.join(',')}`;
        const group = groups.get(key) ?? { mode, coords: [], color };
        group.coords.push(coord);
        groups.set(key, group);
    });
    return Array.from(groups.values()).map((group) => ({
        mode: group.mode,
        kind: 'coords',
        coords: group.coords,
        color: group.color,
    }));
}

export async function saveBundle(snapshot: ViewerSnapshot, tensors: TensorRecord[]): Promise<Blob> {
    const zip = new JSZip();
    const manifest: BundleManifest = {
        version: 1,
        viewer: snapshot,
        tensors: tensors.map((tensor) => ({
            id: tensor.id,
            name: tensor.name,
            dtype: tensor.dtype,
            shape: tensor.shape.slice(),
            byteOrder: 'little',
            dataFile: `tensors/${tensor.id}.bin`,
            offset: tensor.offset,
            view: {
                visible: tensor.view.canonical,
                hiddenIndices: tensor.view.hiddenIndices.slice(),
            },
            colorInstructions: sparseColorInstructions(tensor),
        })),
    };

    zip.file('manifest.json', JSON.stringify(manifest, null, 2));
    tensors.forEach((tensor) => {
        zip.file(`tensors/${tensor.id}.bin`, byteView(tensor.data));
    });
    return zip.generateAsync({ type: 'blob' });
}

export async function loadBundle(file: Blob): Promise<{ manifest: BundleManifest; tensors: Map<string, NumericArray> }> {
    const zip = await JSZip.loadAsync(file);
    const manifestEntry = zip.file('manifest.json');
    if (!manifestEntry) throw new Error('Bundle is missing manifest.json.');
    const manifest = JSON.parse(await manifestEntry.async('string')) as BundleManifest;
    if (manifest.version !== 1) throw new Error(`Unsupported bundle version ${manifest.version}.`);

    const tensors = new Map<string, NumericArray>();
    for (const tensor of manifest.tensors) {
        const entry = zip.file(tensor.dataFile);
        if (!entry) throw new Error(`Bundle is missing ${tensor.dataFile}.`);
        const buffer = await entry.async('arraybuffer');
        tensors.set(tensor.id, createTypedArray(tensor.dtype, buffer));
    }
    return { manifest, tensors };
}
