/**
 * parsed form of one named linear-layout basis block.
 *
 * @example
 * const value: NamedLayoutSpec = {} as NamedLayoutSpec;
 */
export type NamedLayoutSpec = {
    name: string;
    inputs: string[];
    outputs: string[];
    bases: number[][][];
};

/**
 * Parses the layout-spec editor text into named linear-layout definitions.
 *
 * Blank lines and `#` comments are ignored. Each definition starts with a
 * `<name>: [inputs] -> [outputs]` signature followed by one JSON basis row for
 * each input label; rows may appear in any order and are reordered to match the
 * signature.
 *
 * @param text - Contents of the layout specs textarea, including optional blank lines and `#` comments.
 * @returns Parsed layout specs with unique names, input/output labels, and basis rows ordered by the signature input labels.
 * @throws Error when a signature is malformed, a basis row is missing, a row uses an unknown or duplicate input label, a basis vector is invalid JSON or has the wrong output length, or two layouts use the same name.
 * @example
 * const specs = parseLayoutSpecs(`
 * swizzle: [m, n] -> [row, col]
 * m: [[1, 0], [0, 1]]
 * n: [[0, 1]] # comment text is ignored
 * `);
 *
 * expect(specs).toEqual([
 *   {
 *     name: 'swizzle',
 *     inputs: ['m', 'n'],
 *     outputs: ['row', 'col'],
 *     bases: [
 *       [[1, 0], [0, 1]],
 *       [[0, 1]],
 *     ],
 *   },
 * ]);
 * @example
 * expect(() =>
 *   parseLayoutSpecs(`
 * dup: [m] -> [row]
 * m: [[1]]
 * dup: [n] -> [col]
 * n: [[1]]
 * `),
 * ).toThrow('Layout names must be unique; received duplicate dup.');
 */
export function parseLayoutSpecs(text: string): NamedLayoutSpec[] {
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    const specs: NamedLayoutSpec[] = [];
    let index = 0;
    while (index < lines.length) {
        while (index < lines.length && !stripLayoutComment(lines[index]!).trim()) index += 1;
        if (index >= lines.length) break;
        const signatureLine = stripLayoutComment(lines[index]!).trim();
        const signature = parseSignature(signatureLine);
        index += 1;
        const basisByLabel = new Map<string, number[][]>();
        // read rows by label instead of position so the canonical formatter can
        // preserve signature order while users keep spec rows grouped naturally.
        for (let axis = 0; axis < signature.inputs.length; axis += 1) {
            while (index < lines.length && !stripLayoutComment(lines[index]!).trim()) index += 1;
            const line = stripLayoutComment(lines[index] ?? '').trim();
            if (!line) {
                throw new Error(`Layout ${signature.name} is missing basis row for ${signature.inputs[axis]}.`);
            }
            const match = line.match(/^([A-Za-z][0-9]*)\s*:\s*(.+)$/);
            if (!match) {
                throw new Error(`Layout ${signature.name} basis rows must use "<label>: <json>" syntax.`);
            }
            const axisLabel = match[1]!;
            if (!signature.inputs.includes(axisLabel)) {
                throw new Error(`Layout ${signature.name} received basis row for unknown input label ${axisLabel}.`);
            }
            if (basisByLabel.has(axisLabel)) {
                throw new Error(`Layout ${signature.name} has duplicate basis row for ${axisLabel}.`);
            }
            basisByLabel.set(axisLabel, parseBasisRow(match[2]!, signature.outputs.length, axisLabel));
            index += 1;
        }
        const bases = signature.inputs.map((axisLabel) => {
            const basis = basisByLabel.get(axisLabel);
            if (!basis) throw new Error(`Layout ${signature.name} is missing basis row for ${axisLabel}.`);
            return basis;
        });
        specs.push({
            name: signature.name,
            inputs: signature.inputs,
            outputs: signature.outputs,
            bases,
        });
        while (index < lines.length && !stripLayoutComment(lines[index]!).trim()) index += 1;
    }
    const duplicate = duplicateValue(specs.map((spec) => spec.name));
    if (duplicate) throw new Error(`Layout names must be unique; received duplicate ${duplicate}.`);
    return specs;
}

/**
 * Removes the inline `#` comment portion from one layout-spec line before parsing.
 *
 * @param line - A single raw line from the layout specs textarea.
 * @returns The same line truncated before the first `#`, preserving any whitespace before the comment marker.
 * @noThrows Uses only `String.prototype.replace` with a fixed regular expression on the supplied string, with no parsing or validation branch.
 * @example
 * expect(stripLayoutComment('m: [[1, 0]] # row contribution')).toBe('m: [[1, 0]] ');
 */
export function stripLayoutComment(line: string): string {
    return line.replace(/#.*$/, '');
}

/**
 * Parses a layout signature line into its layout name and axis label lists.
 *
 * @param line - Trimmed signature text in `<name>: [inputLabels] -> [outputLabels]` form.
 * @returns The parsed layout name plus ordered input and output labels for later basis-row validation.
 * @throws Error when the line does not match the signature syntax, a label list contains an invalid label token, or an input or output label is repeated.
 * @example
 * expect(parseSignature('wmma: [m, n] -> [row, col]')).toEqual({
 *   name: 'wmma',
 *   inputs: ['m', 'n'],
 *   outputs: ['row', 'col'],
 * });
 * @example
 * expect(() => parseSignature('wmma: [m, m] -> [row]')).toThrow(
 *   'Layout wmma has duplicate input label m.',
 * );
 */
export function parseSignature(line: string): { name: string; inputs: string[]; outputs: string[] } {
    const match = line.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\[(.*)\]\s*->\s*\[(.*)\]\s*$/);
    if (!match) {
        throw new Error('Each specification must start with "<name>: [labels] -> [labels]".');
    }
    const name = match[1]!;
    const inputs = parseLabelList(match[2] ?? '', `${name} inputs`);
    const outputs = parseLabelList(match[3] ?? '', `${name} outputs`);
    const duplicateInput = duplicateValue(inputs);
    if (duplicateInput) throw new Error(`Layout ${name} has duplicate input label ${duplicateInput}.`);
    const duplicateOutput = duplicateValue(outputs);
    if (duplicateOutput) throw new Error(`Layout ${name} has duplicate output label ${duplicateOutput}.`);
    return { name, inputs, outputs };
}

/**
 * format parsed specs back into the canonical notation used for preset matching.
 *
 * @param specs - specs input used by this operation (NamedLayoutSpec[]).
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * formatSpecsText(specs);
 */
export function formatSpecsText(specs: NamedLayoutSpec[]): string {
    return specs.map((spec) => [
        `${spec.name}: [${spec.inputs.join(',')}] -> [${spec.outputs.join(',')}]`,
        ...spec.bases.map((row, axis) => `${spec.inputs[axis]}: ${JSON.stringify(row)}`),
    ].join('\n')).join('\n\n');
}

/**
 * parse label list for the current viewer state.
 *
 * @param source - source input used by this operation (string).
 * @param label - label input used by this operation (string).
 * @returns Text entries formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * parseLabelList(source, label);
 */
function parseLabelList(source: string, label: string): string[] {
    const trimmed = source.trim();
    if (!trimmed) return [];
    return trimmed.split(',').map((entry) => {
        const value = entry.trim();
        if (!/^[A-Za-z][0-9]*$/.test(value)) {
            throw new Error(`${label} may only contain labels like T, A0, or B12 (received ${JSON.stringify(value)}).`);
        }
        return value;
    });
}

/**
 * Parses the JSON basis-vector list for one input axis of a layout spec.
 *
 * Each nested vector describes the non-negative integer contribution of that
 * input axis to every output axis, so every vector must have exactly one entry
 * per output label.
 *
 * @param line - JSON text that appears after `<axisLabel>:` in a basis row.
 * @param outputCount - Number of output labels in the layout signature; each basis vector must have this length.
 * @param axisLabel - Input-axis label used to identify the row in validation errors.
 * @returns Basis vectors for the input axis as arrays of non-negative integers, ready for GF(2) matrix conversion.
 * @throws Error when the row is not valid JSON, the parsed value or a basis entry is not an array, a basis vector length differs from `outputCount`, or a vector entry is negative or fractional.
 * @example
 * expect(parseBasisRow('[[1, 0], [0, 1]]', 2, 'm')).toEqual([
 *   [1, 0],
 *   [0, 1],
 * ]);
 * @example
 * expect(() => parseBasisRow('[[1, -1]]', 2, 'm')).toThrow(
 *   'm basis 1[2] must be a non-negative integer.',
 * );
 */
function parseBasisRow(line: string, outputCount: number, axisLabel: string): number[][] {
    let parsed: unknown;
    try {
        parsed = JSON.parse(line);
    } catch {
        throw new Error(`${axisLabel} bases must be valid JSON.`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${axisLabel} bases must be a JSON array.`);
    return parsed.map((basis, basisIndex) => {
        if (!Array.isArray(basis)) {
            throw new Error(`${axisLabel} basis ${basisIndex + 1} must be an array.`);
        }
            if (basis.length !== outputCount) {
                throw new Error(`${axisLabel} basis ${basisIndex + 1} must have length ${outputCount}.`);
            }
            // bases are integer bit contributions; negative or fractional values
            // would break the gf(2) matrix conversion in linear-layout.ts.
            return basis.map((value, outputAxis) => {
            if (!Number.isInteger(value) || Number(value) < 0) {
                throw new Error(`${axisLabel} basis ${basisIndex + 1}[${outputAxis + 1}] must be a non-negative integer.`);
            }
            return Number(value);
        });
    });
}

/**
 * Finds the first string that appears more than once while scanning a label list.
 *
 * @param values - Ordered layout names or axis labels to check for repeated entries.
 * @returns The first repeated string encountered during the left-to-right scan, or `null` when every value is unique.
 * @noThrows Only records strings in an in-memory `Set` and performs no parsing, validation, or user-code callbacks.
 * @example
 * expect(duplicateValue(['m', 'n', 'm'])).toBe('m');
 * expect(duplicateValue(['m', 'n', 'k'])).toBeNull();
 */
function duplicateValue(values: string[]): string | null {
    const seen = new Set<string>();
    for (const value of values) {
        if (seen.has(value)) return value;
        seen.add(value);
    }
    return null;
}
