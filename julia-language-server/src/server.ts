#!/usr/bin/env node
import {
  createConnection, ProposedFeatures, TextDocuments, TextDocumentSyncKind, InitializeResult,
  CompletionItemKind, SymbolKind, ResponseError, ErrorCodes, Location, TextEdit,
  DocumentSymbol, SignatureHelp, FileChangeType, WorkspaceFoldersChangeEvent
} from 'vscode-languageserver/node';
import { TextDocument } from 'vscode-languageserver-textdocument';
import { pathToFileURL, fileURLToPath } from 'node:url';
import * as path from 'node:path';
import { Analysis, Binding, Node, initializeParser, keywords, occurrenceAt, range } from './analysis';
import { Workspace } from './workspace';
import { builtins, types } from './builtins';

const connection = createConnection(ProposedFeatures.all);
const documents = new TextDocuments(TextDocument);
const workspace = new Workspace();
let ready: Promise<void> = Promise.resolve();
const position = (a: Analysis, offset: number) => a.document.positionAt(offset);
const location = (b: Binding) => Location.create(b.uri, range(workspace.files.get(b.uri)!.document, b.node));
function current(uri: string) { return workspace.files.get(uri); }
function at(a: Analysis, p: { line: number; character: number }) { return occurrenceAt(a, a.document.offsetAt(p)); }
function markdown(b: Binding) { return { kind: 'markdown' as const, value: `\`\`\`julia\n${b.signature}\n\`\`\`${b.doc ? `\n\n${b.doc}` : ''}` }; }
function kind(b: Binding): CompletionItemKind {
  return b.kind === SymbolKind.Function ? CompletionItemKind.Function : b.kind === SymbolKind.Module ? CompletionItemKind.Module : b.kind === SymbolKind.Struct ? CompletionItemKind.Struct : b.kind === SymbolKind.Field ? CompletionItemKind.Field : CompletionItemKind.Variable;
}
function publish(a: Analysis) {
  connection.sendDiagnostics({ uri: a.document.uri, version: a.document.version, diagnostics: a.diagnostics });
}
connection.onInitialize(async params => {
  await initializeParser();
  const roots = params.workspaceFolders?.map(f => f.uri) ?? (params.rootUri ? [params.rootUri] : []);
  ready = workspace.scan(roots);
  const result: InitializeResult = {
    serverInfo: { name: 'Julia Workbench', version: '0.1.0' },
    capabilities: {
      positionEncoding: 'utf-16', textDocumentSync: TextDocumentSyncKind.Incremental,
      completionProvider: { triggerCharacters: ['.'] }, hoverProvider: true,
      definitionProvider: true, referencesProvider: true, renameProvider: { prepareProvider: true },
      documentSymbolProvider: true, workspaceSymbolProvider: true,
      signatureHelpProvider: { triggerCharacters: ['(', ','], retriggerCharacters: [','] },
      foldingRangeProvider: true, documentLinkProvider: { resolveProvider: false },
      workspace: { workspaceFolders: { supported: true, changeNotifications: true } }
    }
  };
  return result;
});
connection.onInitialized(async () => {
  await ready;
  for (const warning of workspace.warnings) connection.console.warn(warning);
});
documents.onDidOpen(e => { workspace.open.add(e.document.uri); publish(workspace.set(e.document)); });
documents.onDidChangeContent(e => { publish(workspace.set(e.document)); });
documents.onDidClose(e => {
  workspace.open.delete(e.document.uri);
  ready = ready.then(async () => {
    if (e.document.uri.startsWith('file:')) await workspace.load(e.document.uri); else if (!workspace.open.has(e.document.uri)) workspace.remove(e.document.uri);
  });
  connection.sendDiagnostics({ uri: e.document.uri, diagnostics: [] });
});
connection.onDidChangeWatchedFiles(e => {
  ready = ready.then(async () => {
    if (e.changes.some(c => /\/(?:Project|Manifest)\.toml$/.test(c.uri))) {
      await workspace.scan(workspace.roots.map(root => pathToFileURL(root).href));
      for (const warning of workspace.warnings) connection.console.warn(warning);
      return;
    }
    for (const change of e.changes) if (!workspace.open.has(change.uri)) {
      if (change.type === FileChangeType.Deleted) workspace.remove(change.uri); else await workspace.load(change.uri);
    }
  });
});
connection.onNotification('workspace/didChangeWorkspaceFolders', ({ event: e }: { event: WorkspaceFoldersChangeEvent }) => {
  ready = ready.then(async () => {
    const removed = e.removed.map(f => f.uri.endsWith('/') ? f.uri : `${f.uri}/`);
    for (const uri of workspace.files.keys()) if (!workspace.open.has(uri) && removed.some(root => uri.startsWith(root))) workspace.remove(uri);
    const roots = workspace.roots.map(r => pathToFileURL(r).href).filter(r => !e.removed.some(f => f.uri === r));
    await workspace.scan([...roots, ...e.added.map(f => f.uri)]);
  });
});
connection.onHover(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a) return null;
  const o = at(a, p.position); if (!o) return null;
  const b = workspace.resolve(o)[0];
  if (b) return { contents: markdown(b), range: range(a.document, o.node) };
  const builtin = builtins[o.node.text];
  if (builtin) return { contents: { kind: 'markdown', value: `\`\`\`julia\n${builtin.signature}\n\`\`\`\n\n${builtin.doc}` }, range: range(a.document, o.node) };
  return null;
});
connection.onDefinition(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a) return [];
  const o = at(a, p.position); return o ? workspace.resolve(o).map(location) : [];
});
connection.onReferences(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a) return [];
  const o = at(a, p.position); if (!o) return [];
  return workspace.references(workspace.resolve(o), p.context.includeDeclaration).map(({ a, o }) => Location.create(a.document.uri, range(a.document, o.node)));
});
function renameTarget(uri: string, p: { line: number; character: number }) {
  const a = current(uri); const o = a && at(a, p); const bindings = o && workspace.resolve(o);
  if (!a || !o || !bindings?.length) throw new ResponseError(ErrorCodes.InvalidParams, 'This symbol cannot be resolved statically.');
  if (bindings.some(b => !workspace.isEditable(b.uri))) throw new ResponseError(ErrorCodes.InvalidParams, 'Dependency sources are read-only; rename is limited to workspace symbols.');
  if (bindings.some(b => workspace.unresolvedImport(b))) throw new ResponseError(ErrorCodes.InvalidParams, 'The original imported symbol is unavailable; rename would change an unresolved import.');
  return { a, o, bindings };
}
connection.onPrepareRename(async p => {
  await ready; const { a, o } = renameTarget(p.textDocument.uri, p.position);
  return { range: range(a.document, o.node), placeholder: o.node.text };
});
connection.onRenameRequest(async p => {
  await ready;
  const { o, bindings } = renameTarget(p.textDocument.uri, p.position);
  // Let the Julia grammar validate identifiers, including Unicode and ! suffixes.
  if (keywords.has(p.newName) || !/^[\p{L}_][\p{L}\p{N}\p{M}_!′]*$/u.test(p.newName)) throw new ResponseError(ErrorCodes.InvalidParams, 'Enter a Julia identifier (letters, digits, _, Unicode, optional !).');
  const refs = workspace.references(bindings, true);
  if (p.newName !== o.node.text && refs.some(({ o }) => workspace.visible(o.scope).some(b => b.name === p.newName))) throw new ResponseError(ErrorCodes.InvalidParams, 'Rename would conflict with an existing binding.');
  return { documentChanges: [...new Set(refs.map(r => r.a.document.uri))].map(uri => ({
    textDocument: { uri, version: workspace.open.has(uri) ? workspace.files.get(uri)!.document.version : null },
    edits: refs.filter(r => r.a.document.uri === uri).map(({ a, o }) => TextEdit.replace(range(a.document, o.node), p.newName))
  })) };
});
connection.onCompletion(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a) return [];
  const offset = a.document.offsetAt(p.position);
  const o = occurrenceAt(a, offset);
  let scope = o?.scope ?? a.root;
  if (!o) {
    // Pick the innermost lexical scope even on a blank line.
    for (const candidate of a.occurrences) if (candidate.scope.node.startIndex <= offset && offset <= candidate.scope.node.endIndex && candidate.scope.node.endIndex - candidate.scope.node.startIndex < scope.node.endIndex - scope.node.startIndex) scope = candidate.scope;
  }
  const prefix = a.document.getText().slice(0, offset).match(/@?[\p{L}\p{N}\p{M}_!′]*$/u)?.[0] ?? '';
  const start = offset - prefix.length;
  const replacement = { start: position(a, start), end: p.position };
  let bindings = workspace.visible(scope);
  const before = a.document.getText().slice(0, start);
  const member = before.endsWith('.');
  if (member) {
    const base = occurrenceAt(a, start - 1);
    bindings = base ? workspace.resolve(base).flatMap(b => b.memberScope ? workspace.memberBindings(b.memberScope) : []) : [];
  }
  const items = bindings.map(b => ({ label: b.name, kind: kind(b), detail: b.signature, documentation: markdown(b), textEdit: TextEdit.replace(replacement, b.name), sortText: `0${b.name}` }));
  if (!member) {
    for (const [name, b] of Object.entries(builtins)) if (!bindings.some(b => b.name === name)) items.push({ label: name, kind: CompletionItemKind.Function, detail: b.signature, documentation: { kind: 'markdown', value: b.doc }, textEdit: TextEdit.replace(replacement, name), sortText: `1${name}` });
    for (const name of [...types, ...keywords]) items.push({ label: name, kind: keywords.has(name) ? CompletionItemKind.Keyword : CompletionItemKind.Class, detail: '', documentation: { kind: 'markdown', value: '' }, textEdit: TextEdit.replace(replacement, name), sortText: `2${name}` });
  }
  return [...new Map(items.filter(i => i.label.startsWith(prefix)).map(i => [i.label, i])).values()];
});
connection.onDocumentSymbol(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a) return [];
  const selected = a.bindings.filter(b => b.kind !== SymbolKind.TypeParameter && (b.kind !== SymbolKind.Variable || b.scope === a.root || b.scope.module));
  const symbols = new Map<Binding, DocumentSymbol>();
  for (const b of selected) symbols.set(b, { name: b.name, detail: b.signature, kind: b.kind, range: range(a.document, b.container), selectionRange: range(a.document, b.node), children: [] });
  const roots: DocumentSymbol[] = [];
  for (const [b, symbol] of symbols) {
    const parent = selected.filter(other => other !== b && other.container.startIndex <= b.node.startIndex && b.node.endIndex <= other.container.endIndex && other.container.id !== b.container.id).sort((x, y) => (x.container.endIndex - x.container.startIndex) - (y.container.endIndex - y.container.startIndex))[0];
    if (parent) symbols.get(parent)!.children!.push(symbol); else roots.push(symbol);
  }
  return roots;
});
connection.onWorkspaceSymbol(async p => {
  await ready;
  return [...workspace.files.values()].flatMap(a => a.bindings.filter(b => b.scope === a.root || b.kind === SymbolKind.Function || b.kind === SymbolKind.Module || b.kind === SymbolKind.Struct)).filter(b => b.name.toLowerCase().includes(p.query.toLowerCase())).slice(0, 500).map(b => ({ name: b.name, kind: b.kind, location: location(b) }));
});
connection.onSignatureHelp(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a) return null;
  const offset = a.document.offsetAt(p.position);
  let node: Node | null = a.tree.rootNode.descendantForIndex(Math.max(0, offset - 1));
  while (node && !(node.type === 'argument_list' && node.parent?.type === 'call_expression')) node = node.parent;
  if (!node?.parent) return null;
  const callee = node.parent.namedChildren[0];
  const nameNode = callee.type === 'field_expression' ? callee.namedChildren.at(-1)! : callee;
  const o = a.occurrences.find(o => o.node.id === nameNode.id);
  const bindings = o ? workspace.resolve(o) : [];
  const builtin = builtins[nameNode.text];
  const signatures = bindings.length ? bindings.map(b => ({ label: b.signature, documentation: b.doc, parameters: b.parameters.map(label => ({ label })) })) : builtin ? [{ label: builtin.signature, documentation: builtin.doc, parameters: builtin.parameters.map(label => ({ label })) }] : [];
  if (!signatures.length) return null;
  const active = node.children.filter(c => c.type === ',' && c.startIndex < offset).length;
  return { signatures, activeSignature: 0, activeParameter: Math.min(active, Math.max(0, signatures[0].parameters.length - 1)) } satisfies SignatureHelp;
});
connection.onFoldingRanges(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a) return [];
  const result: { startLine: number; endLine: number; kind?: string }[] = [];
  const walk = (n: Node) => {
    if (n.endPosition.row > n.startPosition.row && (n.type.endsWith('_definition') || n.type.endsWith('_statement') || n.type === 'block_comment' || n.type === 'string_literal')) result.push({ startLine: n.startPosition.row, endLine: n.endPosition.row - 1, kind: n.type.includes('comment') ? 'comment' : 'region' });
    for (const child of n.namedChildren) walk(child);
  };
  walk(a.tree.rootNode); return result;
});
connection.onDocumentLinks(async p => {
  await ready; const a = current(p.textDocument.uri); if (!a || !a.document.uri.startsWith('file:')) return [];
  return a.occurrences.filter(o => o.node.text === 'include' && o.node.parent?.type === 'call_expression').flatMap(o => {
    const arg = o.node.parent!.namedChildren[1]?.namedChildren[0];
    if (!arg || !/^"[^"$\\]*"$/.test(arg.text)) return [];
    return [{ range: range(a.document, arg), target: pathToFileURL(path.resolve(path.dirname(fileURLToPath(a.document.uri)), arg.text.slice(1, -1))).href }];
  });
});
documents.listen(connection);
connection.listen();
