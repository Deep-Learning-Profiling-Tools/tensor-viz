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
 * parse the editor notation used in the layout specs textarea.
 *
 * keeping this as the only specs parser prevents notation changes from
 * spreading into preset matching, legacy migration, and runtime evaluation.
 *
 * @param text - Text supplied by the caller.
 * @returns Array of computed entries for the caller.
 * @throws Error when the requested input or state is invalid.
 * @example
 * parseLayoutSpecs(text);
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
 * remove layout comments before syntax parsing.
 *
 * @param line - line input used by this operation (string).
 * @returns Text formatted for the caller.
 * @noThrows This function has no direct throw path.
 * @example
 * stripLayoutComment(line);
 */
export function stripLayoutComment(line: string): string {
    return line.replace(/#.*$/, '');
}

/**
 * parse one `<name>: [inputs] -> [outputs]` signature line.
 *
 * @param line - line input used by this operation (string).
 * @returns Object containing computed state for the caller.
 * @throws Error when the requested input or state is invalid.
 * @example
 * parseSignature(line);
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
 * parse basis row for the current viewer state.
 *
 * @param line - line input used by this operation (string).
 * @param outputCount - output count input used by this operation (number).
 * @param axisLabel - axis label input used by this operation (string).
 * @returns Array of computed entries for the caller.
 * @throws Error when the requested input or state is invalid.
 * @example
 * parseBasisRow(line, outputCount, axisLabel);
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
 * return duplicate value for the current viewer state.
 *
 * @param values - values input used by this operation (string[]).
 * @returns Computed value, or null when no value is available.
 * @noThrows This function has no direct throw path.
 * @example
 * duplicateValue(values);
 */
function duplicateValue(values: string[]): string | null {
    const seen = new Set<string>();
    for (const value of values) {
        if (seen.has(value)) return value;
        seen.add(value);
    }
    return null;
}
