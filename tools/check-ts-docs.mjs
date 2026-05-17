import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { relative, resolve, sep } from 'node:path';
import ts from 'typescript';

const ROOT = resolve(import.meta.dirname, '..');
const SOURCE_ROOTS = [
    'packages/viewer-core/src',
    'packages/viewer-demo/src',
];
const DEFAULT_MIN_COMMENT_RATIO = 0.1;
const HELPER_MIN_REFERENCES = 3;
const SMALL_HELPER_MAX_LINES = 3;
const SMALL_HELPER_MAX_STATEMENTS = 1;
const INTERFACE_BOUNDARY_PATTERN = /@(interfaceBoundary|boundary|publicApi)\b|interface boundary/i;

const rawArgs = process.argv.slice(2);
const args = new Set(rawArgs.filter((arg) => arg.startsWith('--')));
const pathArgs = rawArgs.filter((arg) => !arg.startsWith('--'));
const stagedOnly = args.has('--staged');
const strictJsdoc = args.has('--strict-jsdoc');
const minCommentRatio = optionNumber('--min-comment-ratio', DEFAULT_MIN_COMMENT_RATIO);

/** return one numeric CLI option, or the fallback when the option is omitted. */
function optionNumber(name, fallback) {
    const value = process.argv.find((arg) => arg.startsWith(`${name}=`))?.split('=')[1];
    return value === undefined ? fallback : Number(value);
}

/** return whether a path is a first-party TypeScript source file worth auditing. */
function isSourceFile(path) {
    return (path.endsWith('.ts') || path.endsWith('.tsx'))
        && !path.endsWith('.d.ts')
        && !path.endsWith('.test.ts')
        && !path.endsWith('.spec.ts')
        && !path.split(sep).includes('dist')
        && !path.split(sep).includes('node_modules');
}

/** recursively collect source files below one directory. */
function walk(dir) {
    return readdirSync(dir).flatMap((entry) => {
        const path = resolve(dir, entry);
        if (statSync(path).isDirectory()) return walk(path);
        return isSourceFile(path) ? [path] : [];
    });
}

/** return staged TypeScript source paths for incremental pre-commit checks. */
function stagedFiles() {
    const output = execFileSync(
        'git',
        ['diff', '--cached', '--name-only', '--diff-filter=ACMR'],
        { cwd: ROOT, encoding: 'utf8' },
    );
    return output
        .split('\n')
        .filter(Boolean)
        .map((path) => resolve(ROOT, path))
        .filter(isSourceFile);
}

/** return audited source files from explicit file or directory arguments. */
function explicitFiles() {
    return pathArgs.flatMap((arg) => {
        const path = resolve(ROOT, arg);
        if (statSync(path).isDirectory()) return walk(path);
        return isSourceFile(path) ? [path] : [];
    });
}

/** return the files that should be audited for this invocation. */
function sourceFiles() {
    if (pathArgs.length > 0) return explicitFiles();
    if (stagedOnly) return stagedFiles();
    return SOURCE_ROOTS.flatMap((root) => walk(resolve(ROOT, root)));
}

/** return a stable file:line:column string for diagnostics. */
function lineAndColumn(sourceFile, position) {
    const { line, character } = sourceFile.getLineAndCharacterOfPosition(position);
    return `${relative(ROOT, sourceFile.fileName)}:${line + 1}:${character + 1}`;
}

/** return whether a declaration has an adjacent JSDoc block immediately above it. */
function hasJsdoc(node, sourceFile) {
    const text = sourceFile.getFullText();
    const ranges = ts.getLeadingCommentRanges(text, node.getFullStart()) ?? [];
    const last = ranges.at(-1);
    if (!last || !text.slice(last.pos, last.pos + 3).startsWith('/**')) return false;
    return text.slice(last.end, node.getStart(sourceFile)).trim() === '';
}

/** return the adjacent JSDoc text for one declaration. */
function jsdocText(node, sourceFile) {
    const text = sourceFile.getFullText();
    const ranges = ts.getLeadingCommentRanges(text, node.getFullStart()) ?? [];
    const last = ranges.at(-1);
    if (!last || !text.slice(last.pos, last.pos + 3).startsWith('/**')) return '';
    return text.slice(last.pos, last.end);
}

/** return a human-readable declaration name for diagnostics. */
function functionName(node) {
    if (node.name?.getText) return node.name.getText();
    if (ts.isConstructorDeclaration(node)) return 'constructor';
    return '<anonymous>';
}

/** return a human-readable declaration kind for diagnostics. */
function declarationLabel(node) {
    if (ts.isFunctionDeclaration(node)) return `function ${functionName(node)}`;
    if (ts.isMethodDeclaration(node)) return `method ${functionName(node)}`;
    if (ts.isGetAccessorDeclaration(node)) return `getter ${functionName(node)}`;
    if (ts.isSetAccessorDeclaration(node)) return `setter ${functionName(node)}`;
    if (ts.isConstructorDeclaration(node)) return 'constructor';
    if (ts.isClassDeclaration(node)) return `class ${functionName(node)}`;
    if (ts.isInterfaceDeclaration(node)) return `interface ${functionName(node)}`;
    if (ts.isTypeAliasDeclaration(node)) return `type ${functionName(node)}`;
    if (ts.isEnumDeclaration(node)) return `enum ${functionName(node)}`;
    if (ts.isVariableStatement(node)) return `function variable ${functionVariableDeclarations(node).map(({ name }) => name).join(', ')}`;
    return String(node.kind);
}

/** return whether a declaration is explicitly exported from its module. */
function hasExportModifier(node) {
    return node.modifiers?.some((modifier) => (
        modifier.kind === ts.SyntaxKind.ExportKeyword || modifier.kind === ts.SyntaxKind.DefaultKeyword
    )) ?? false;
}

/** return whether a function return type is explicitly empty. */
function returnsVoid(node) {
    if (!node.type) return false;
    const text = node.type.getText();
    return text === 'void' || text === 'undefined' || text === 'never';
}

/** return top-level const/let declarations initialized by function expressions. */
function functionVariableDeclarations(statement) {
    return statement.declarationList.declarations
        .filter((decl) => decl.initializer
            && (ts.isArrowFunction(decl.initializer) || ts.isFunctionExpression(decl.initializer)))
        .map((decl) => ({ name: decl.name.getText(), nameNode: decl.name, node: statement }));
}

/** return whether a variable statement contains at least one function initializer. */
function hasFunctionInitializer(statement) {
    return functionVariableDeclarations(statement).length > 0;
}

/** return whether a declaration should have a JSDoc block. */
function shouldCheckNode(node) {
    if (ts.isFunctionDeclaration(node)
        || ts.isMethodDeclaration(node)
        || ts.isGetAccessorDeclaration(node)
        || ts.isSetAccessorDeclaration(node)
        || ts.isConstructorDeclaration(node)
        || ts.isClassDeclaration(node)
        || ts.isInterfaceDeclaration(node)
        || ts.isTypeAliasDeclaration(node)
        || ts.isEnumDeclaration(node)) {
        return true;
    }
    return ts.isVariableStatement(node) && hasFunctionInitializer(node);
}

/** return strict JSDoc tag failures when strict mode is requested. */
function strictJsdocErrors(node, sourceFile) {
    const text = jsdocText(node, sourceFile);
    const errors = [];
    if (!strictJsdoc || text === '') return errors;
    if (ts.isFunctionDeclaration(node)
        || ts.isMethodDeclaration(node)
        || ts.isConstructorDeclaration(node)
        || ts.isGetAccessorDeclaration(node)
        || ts.isSetAccessorDeclaration(node)) {
        const params = node.parameters?.map((param) => param.name.getText()) ?? [];
        params.forEach((param) => {
            if (!new RegExp(`@param\\s+${param}\\b`).test(text)) errors.push(`missing @param ${param}`);
        });
        if (!ts.isConstructorDeclaration(node)
            && !ts.isSetAccessorDeclaration(node)
            && !returnsVoid(node)
            && !/@returns?\b/.test(text)) {
            errors.push('missing @returns');
        }
        if (!/@example\b/.test(text)) errors.push('missing @example');
    }
    if ((ts.isClassDeclaration(node) || ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node))
        && !/@example\b/.test(text)) {
        errors.push('missing @example');
    }
    return errors;
}

/** return JSDoc failures in one parsed source file. */
function docstringFailures(sourceFile) {
    const failures = [];
    const visit = (node) => {
        if (shouldCheckNode(node)) {
            if (!hasJsdoc(node, sourceFile)) {
                failures.push(`${lineAndColumn(sourceFile, node.getStart(sourceFile))} ${declarationLabel(node)} lacks a JSDoc block`);
            } else {
                strictJsdocErrors(node, sourceFile).forEach((error) => {
                    failures.push(`${lineAndColumn(sourceFile, node.getStart(sourceFile))} ${declarationLabel(node)} ${error}`);
                });
            }
        }
        ts.forEachChild(node, visit);
    };
    visit(sourceFile);
    return failures;
}

/** return whether a helper is documented as an interface boundary. */
function isInterfaceBoundary(node, sourceFile) {
    return INTERFACE_BOUNDARY_PATTERN.test(jsdocText(node, sourceFile));
}

/** return the implementation node for a top-level helper candidate. */
function helperImplementation(node) {
    if (ts.isFunctionDeclaration(node)) return node;
    if (!ts.isVariableStatement(node)) return null;
    return node.declarationList.declarations.find((decl) => (
        decl.initializer
        && (ts.isArrowFunction(decl.initializer) || ts.isFunctionExpression(decl.initializer))
    ))?.initializer ?? null;
}

/** return whether a top-level helper is small enough that it should usually be inlined. */
function isSmallHelper(node, sourceFile) {
    const implementation = helperImplementation(node);
    if (!implementation) return false;
    const start = sourceFile.getLineAndCharacterOfPosition(implementation.getStart(sourceFile)).line;
    const end = sourceFile.getLineAndCharacterOfPosition(implementation.getEnd()).line;
    if (!implementation.body || !ts.isBlock(implementation.body)) return true;
    return (end - start + 1) <= SMALL_HELPER_MAX_LINES
        && implementation.body.statements.length <= SMALL_HELPER_MAX_STATEMENTS;
}

/** return non-exported top-level helper candidates in one source file. */
function helperCandidates(sourceFile) {
    return sourceFile.statements.flatMap((statement) => {
        if (ts.isFunctionDeclaration(statement)
            && statement.name
            && !hasExportModifier(statement)
            && !isInterfaceBoundary(statement, sourceFile)
            && isSmallHelper(statement, sourceFile)) {
            return [{ name: statement.name.text, nameNode: statement.name, node: statement }];
        }
        if (ts.isVariableStatement(statement)
            && !hasExportModifier(statement)
            && !isInterfaceBoundary(statement, sourceFile)
            && isSmallHelper(statement, sourceFile)) {
            return functionVariableDeclarations(statement);
        }
        return [];
    });
}

/** return whether an identifier occurrence is the declaration itself or an object key. */
function isNonReferenceIdentifier(identifier, nameNode) {
    const parent = identifier.parent;
    return identifier === nameNode
        || (ts.isPropertyAccessExpression(parent) && parent.name === identifier)
        || (ts.isPropertyAssignment(parent) && parent.name === identifier)
        || (ts.isMethodDeclaration(parent) && parent.name === identifier)
        || (ts.isPropertyDeclaration(parent) && parent.name === identifier);
}

/** count local identifier references to one helper declaration. */
function referenceCount(sourceFile, name, nameNode) {
    let count = 0;
    const visit = (node) => {
        if (ts.isIdentifier(node) && node.text === name && !isNonReferenceIdentifier(node, nameNode)) count += 1;
        ts.forEachChild(node, visit);
    };
    visit(sourceFile);
    return count;
}

/** return helper-use-count failures in one parsed source file. */
function helperUsageFailures(sourceFile) {
    return helperCandidates(sourceFile)
        .map(({ name, nameNode, node }) => ({
            name,
            references: referenceCount(sourceFile, name, nameNode),
            node,
        }))
        .filter(({ references }) => references < HELPER_MIN_REFERENCES)
        .map(({ name, references, node }) => (
            `${lineAndColumn(sourceFile, node.getStart(sourceFile))} small helper ${name} has ${references} reference${references === 1 ? '' : 's'}; expected at least ${HELPER_MIN_REFERENCES}, inlining, or a JSDoc @interfaceBoundary tag`
        ));
}

/** return the set of source lines occupied by comments. */
function commentLines(text) {
    const lines = new Set();
    let inBlockComment = false;
    text.split('\n').forEach((line, index) => {
        const trimmed = line.trim();
        const startsComment = trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*');
        const inlineComment = line.includes('//') || line.includes('/*');
        if (inBlockComment || startsComment || inlineComment) lines.add(index + 1);
        if (inBlockComment && line.includes('*/')) inBlockComment = false;
        if (!inBlockComment && line.includes('/*') && !line.includes('*/')) inBlockComment = true;
    });
    return lines;
}

/** return comment-ratio failures for one source file. */
function commentRatioFailure(path) {
    const text = readFileSync(path, 'utf8');
    const comments = commentLines(text);
    const scopedLines = text
        .split('\n')
        .map((line, index) => ({ text: line, number: index + 1 }))
        .filter((line) => line.text.trim());
    if (scopedLines.length === 0) return null;
    const ratio = scopedLines.filter((line) => comments.has(line.number)).length / scopedLines.length;
    if (ratio >= minCommentRatio) return null;
    return `${relative(ROOT, path)} file comment ratio ${(ratio * 100).toFixed(1)}% is below ${(minCommentRatio * 100).toFixed(1)}%`;
}

/** parse one source file with full AST nodes for declaration audits. */
function parseSourceFile(path) {
    return ts.createSourceFile(
        path,
        readFileSync(path, 'utf8'),
        ts.ScriptTarget.Latest,
        true,
    );
}

const files = sourceFiles();
if (files.length === 0) {
    console.log(stagedOnly ? 'No staged TypeScript source files to check.' : 'No TypeScript source files matched.');
    process.exit(0);
}

const parsedFiles = files.map(parseSourceFile);
const failures = [
    ...parsedFiles.flatMap((sourceFile) => docstringFailures(sourceFile)),
    ...parsedFiles.flatMap((sourceFile) => helperUsageFailures(sourceFile)),
    ...files.map(commentRatioFailure),
].filter(Boolean);

if (failures.length > 0) {
    console.error(`TypeScript comment/doc audit failed (${failures.length} issue${failures.length === 1 ? '' : 's'}):`);
    failures.forEach((failure) => console.error(`  - ${failure}`));
    process.exit(1);
}

console.log(`Checked ${files.length} TypeScript source file${files.length === 1 ? '' : 's'} for JSDoc blocks, helper use counts, and comment density.`);
