import Parser from 'web-tree-sitter';
import { TextDocument } from 'vscode-languageserver-textdocument';
import { Diagnostic, DiagnosticSeverity, Range, SymbolKind } from 'vscode-languageserver/node';
import * as path from 'node:path';

export type Node = Parser.SyntaxNode;
export interface Scope { parent?: Scope; node: Node; bindings: Map<string, Binding[]>; module?: string; globals?: Set<string>; }
export interface Binding {
  name: string; node: Node; scope: Scope; kind: SymbolKind; doc: string;
  signature: string; parameters: string[]; container: Node; uri: string;
  memberScope?: Scope;
}
export interface Occurrence { node: Node; scope: Scope; declaration?: Binding; }
export interface Analysis {
  document: TextDocument; tree: Parser.Tree; root: Scope;
  bindings: Binding[]; occurrences: Occurrence[]; diagnostics: Diagnostic[];
}
export const keywords = new Set('baremodule begin break catch const continue do else elseif end export false finally for function global if import in isa let local macro module mutable primitive public quote return struct true try using where while abstract type'.split(' '));
let parser: Parser;
export async function initializeParser() {
  await Parser.init();
  parser = new Parser();
  parser.setLanguage(await Parser.Language.load(path.join(__dirname, 'tree-sitter-julia.wasm')));
}
export function range(document: TextDocument, node: Node): Range {
  return { start: document.positionAt(node.startIndex), end: document.positionAt(node.endIndex) };
}
function firstIdentifier(node?: Node | null): Node | undefined {
  if (!node) return;
  if (node.type === 'identifier') return node;
  return firstIdentifier(node.namedChildren[0]);
}
function findCall(node: Node): Node | undefined {
  if (node.type === 'call_expression') return node;
  if (['signature', 'where_expression', 'typed_expression'].includes(node.type)) return node.namedChildren.map(findCall).find(Boolean);
}
export function analyze(document: TextDocument): Analysis {
  const tree = parser.parse(document.getText());
  const root: Scope = { node: tree.rootNode, bindings: new Map() };
  const analysis: Analysis = { document, tree, root, bindings: [], occurrences: [], diagnostics: [] };
  const declared = new Map<number, Binding>();
  const nodeScopes = new Map<number, Scope>();
  const assignments: Binding[] = [];
  function newScope(node: Node, parent: Scope, module?: string): Scope {
    return { node, parent, bindings: new Map(), module };
  }
  function bind(node: Node | undefined, scope: Scope, kind: SymbolKind, container: Node, signature = '', parameters: string[] = []) {
    if (!node || declared.has(node.id)) return declared.get(node?.id ?? -1);
    const prev = container.previousNamedSibling ?? (container.parent?.type === 'const_statement' ? container.parent.previousNamedSibling : null);
    const doc = prev?.type === 'string_literal' ? prev.text.replace(/^"""|"""$/g, '').replace(/^"|"$/g, '').trim() : '';
    const name = container.type === 'macro_definition' && kind === SymbolKind.Function ? `@${node.text}` : node.text;
    const binding: Binding = { name, node, scope, kind, container, doc, signature: signature || container.text.split('\n')[0], parameters, uri: document.uri };
    declared.set(node.id, binding);
    scope.bindings.set(name, [...(scope.bindings.get(name) ?? []), binding]);
    analysis.bindings.push(binding);
    return binding;
  }
  function pattern(node: Node | undefined, scope: Scope, kind: SymbolKind, container: Node) {
    if (!node) return;
    if (node.type === 'identifier') { bind(node, scope, kind, container); return; }
    if (['typed_expression', 'named_argument', 'optional_parameter', 'splat_expression', 'let_binding', 'for_binding', 'binary_expression', 'parametrized_type_expression'].includes(node.type)) {
      pattern(node.namedChildren[0], scope, kind, container);
    } else if (['tuple_expression', 'argument_list', 'open_tuple', 'curly_expression'].includes(node.type)) {
      for (const child of node.namedChildren) pattern(child, scope, kind, container);
    }
  }
  function checkSyntax(node: Node) {
    if (node.type === 'ERROR' || node.isMissing) {
      if (analysis.diagnostics.length < 100) analysis.diagnostics.push({
        range: range(document, node), severity: DiagnosticSeverity.Error, source: 'Julia Workbench',
        code: 'syntax', message: node.isMissing ? `Expected ${node.type}` : `Unexpected syntax: ${node.text.slice(0, 70).replace(/\n/g, ' ')}`
      });
    }
  }
  function visit(node: Node, scope: Scope) {
    nodeScopes.set(node.id, scope);
    checkSyntax(node);
    if (node.type === 'quote_expression' || node.type === 'quote_statement') {
      const diagnosticsOnly = (n: Node) => { checkSyntax(n); for (const child of n.children) diagnosticsOnly(child); };
      for (const child of node.children) diagnosticsOnly(child);
      return;
    }
    if (['module_definition', 'struct_definition', 'abstract_definition', 'primitive_definition'].includes(node.type)) {
      const isModule = node.type === 'module_definition';
      const name = node.childForFieldName('name') ?? firstIdentifier(node.namedChildren.find(c => c.type === 'type_head'));
      const binding = bind(name, scope, isModule ? SymbolKind.Module : SymbolKind.Struct, node);
      const inner = newScope(node, scope, isModule ? name?.text : undefined);
      if (binding) binding.memberScope = inner;
      const head = node.namedChildren.find(c => c.type === 'type_head');
      if (head) for (const params of head.descendantsOfType('curly_expression')) for (const param of params.namedChildren) pattern(param, inner, SymbolKind.TypeParameter, node);
      for (const child of node.children) visit(child, child.id === name?.id ? scope : inner);
      return;
    }
    const call = node.type === 'function_definition' || node.type === 'macro_definition'
      ? findCall(node.namedChildren[0]) : node.type === 'assignment' ? findCall(node.namedChildren[0]) : undefined;
    if (call) {
      const callee = call.namedChildren[0];
      // Qualified method extensions are references, not declarations of the module name.
      const name = callee.type === 'identifier' ? callee : undefined;
      const args = call.namedChildren.find(c => c.type === 'argument_list');
      const signature = node.namedChildren[0].text;
      bind(name, scope, node.type === 'macro_definition' ? SymbolKind.Function : SymbolKind.Function, node, signature, args?.namedChildren.map(c => c.text) ?? []);
      const inner = newScope(node, scope);
      if (args) for (const arg of args.namedChildren) pattern(arg, inner, SymbolKind.Variable, node);
      function whereParams(n: Node) {
        if (n.type === 'where_expression') for (const param of n.namedChildren.slice(1)) pattern(param, inner, SymbolKind.TypeParameter, node);
        for (const child of n.namedChildren) whereParams(child);
      }
      whereParams(node.namedChildren[0]);
      for (const child of node.children) visit(child, inner);
      return;
    }
    if (['let_statement', 'for_statement', 'while_statement', 'comprehension_expression', 'arrow_function_expression', 'do_clause', 'catch_clause'].includes(node.type)) {
      scope = newScope(node, scope);
      nodeScopes.set(node.id, scope);
      if (node.type === 'arrow_function_expression' || node.type === 'do_clause' || node.type === 'catch_clause') pattern(node.namedChildren[0], scope, SymbolKind.Variable, node);
    }
    if (['assignment', 'compound_assignment_expression', 'for_binding', 'let_binding'].includes(node.type)) {
      let target = scope;
      if (node.parent?.type === 'global_statement') while (target.parent && !target.module) target = target.parent;
      pattern(node.namedChildren[0], target, node.parent?.type === 'const_statement' ? SymbolKind.Constant : SymbolKind.Variable, node);
      if (['assignment', 'compound_assignment_expression'].includes(node.type) && node.parent?.type !== 'local_statement') {
        for (const b of analysis.bindings) if (b.container.id === node.id) assignments.push(b);
      }
    }
    if (node.type === 'typed_expression' && node.parent?.type === 'struct_definition') pattern(node, scope, SymbolKind.Field, node);
    if (['local_statement', 'global_statement'].includes(node.type)) {
      let target = scope;
      if (node.type === 'global_statement') {
        while (target.parent && !target.module) target = target.parent;
        for (const child of node.namedChildren) {
          const id = firstIdentifier(child);
          if (id) { scope.globals ??= new Set(); scope.globals.add(id.text); }
        }
      }
      for (const child of node.namedChildren) if (child.type === 'identifier') bind(child, target, SymbolKind.Variable, node);
    }
    for (const child of node.children) visit(child, scope);
  }
  visit(tree.rootNode, root);
  // Assignments in nested local scopes update existing outer locals, including
  // declarations appearing later in the enclosing function. Globals need `global`.
  for (const b of assignments) {
    if (!b.scope.parent || b.scope.module) continue;
    if (b.scope.globals?.has(b.name)) {
      let target = b.scope;
      while (target.parent && !target.module) target = target.parent;
      b.scope.bindings.set(b.name, (b.scope.bindings.get(b.name) ?? []).filter(x => x !== b));
      if (!b.scope.bindings.get(b.name)?.length) b.scope.bindings.delete(b.name);
      b.scope = target;
      target.bindings.set(b.name, [...(target.bindings.get(b.name) ?? []), b]);
      continue;
    }
    // Parameters and explicit local/let/iteration bindings establish a new local.
    if ((b.scope.bindings.get(b.name) ?? []).some(other => other !== b && !['assignment', 'compound_assignment_expression'].includes(other.container.type))) continue;
    let outer = b.scope.parent;
    while (outer.parent && !outer.module) {
      if (outer.bindings.has(b.name)) {
        b.scope.bindings.set(b.name, (b.scope.bindings.get(b.name) ?? []).filter(x => x !== b));
        if (!b.scope.bindings.get(b.name)?.length) b.scope.bindings.delete(b.name);
        b.scope = outer;
        outer.bindings.set(b.name, [...outer.bindings.get(b.name)!, b]);
        break;
      }
      outer = outer.parent;
    }
  }
  function collect(node: Node) {
    if (node.type === 'quote_expression' || node.type === 'quote_statement') return;
    if (node.type === 'identifier') analysis.occurrences.push({ node, scope: nodeScopes.get(node.id) ?? root, declaration: declared.get(node.id) });
    for (const child of node.namedChildren) collect(child);
  }
  collect(tree.rootNode);
  return analysis;
}

export function occurrenceAt(a: Analysis, offset: number): Occurrence | undefined {
  return a.occurrences.find(o => o.node.startIndex <= offset && offset < o.node.endIndex)
    ?? a.occurrences.find(o => o.node.endIndex === offset);
}
export function lexical(scope: Scope, name: string): Binding[] {
  for (let s: Scope | undefined = scope; s; s = s.parent) {
    const found = s.bindings.get(name);
    if (found?.length) return found;
  }
  return [];
}
