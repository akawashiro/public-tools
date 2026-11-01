import { promises as fs } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { TextDocument } from 'vscode-languageserver-textdocument';
import { Analysis, Binding, Node, Occurrence, Scope, analyze } from './analysis';
import { Projects } from './projects';
import { SymbolKind } from 'vscode-languageserver/node';

const excluded = new Set(['.git', 'node_modules', '.julia', 'dist', 'build', '.vscode-test']);
export class Workspace {
  files = new Map<string, Analysis>();
  open = new Set<string>();
  roots: string[] = [];
  warnings: string[] = [];
  readonly projects = new Projects();
  private aliases = new Map<Scope, Scope>();
  private groups = new Map<Scope, Scope[]>();
  private dirty = true;
  private imports = new Map<Scope, Node[]>();
  private lookupCache = new Map<Scope, Map<string, Binding[]>>();
  private resolutionCache = new Map<number, Binding[]>();
  private treeFiles = new Map<object, Analysis>();
  set(document: TextDocument) {
    const old = this.files.get(document.uri);
    const result = analyze(document);
    this.files.set(document.uri, result);
    old?.tree.delete();
    this.dirty = true;
    return result;
  }
  async load(uri: string) {
    if (this.open.has(uri) || !uri.startsWith('file:')) return;
    try {
      const file = fileURLToPath(uri);
      const stat = await fs.stat(file);
      if (stat.size > 2 * 1024 * 1024) { this.warnings.push(`Skipped file larger than 2 MiB: ${file}`); return; }
      const text = await fs.readFile(file, 'utf8');
      if (!this.open.has(uri)) this.set(TextDocument.create(uri, 'julia', 0, text));
    } catch { if (!this.open.has(uri)) this.remove(uri); }
  }
  remove(uri: string) { this.files.get(uri)?.tree.delete(); this.files.delete(uri); this.dirty = true; }
  async scan(uris: string[]) {
    this.warnings = [];
    this.roots = uris.filter(u => u.startsWith('file:')).map(u => fileURLToPath(u));
    for (const uri of this.files.keys()) if (!this.open.has(uri) && !this.isEditable(uri)) this.remove(uri);
    let count = 0;
    const projects: string[] = [];
    const walk = async (dir: string): Promise<void> => {
      let entries;
      try { entries = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }
      for (const entry of entries) {
        if (excluded.has(entry.name) || entry.isSymbolicLink()) continue;
        const file = path.join(dir, entry.name);
        if (entry.isFile() && entry.name === 'Project.toml') projects.push(dir);
        if (entry.isDirectory()) await walk(file);
        else if (entry.isFile() && file.endsWith('.jl')) {
          if (++count > 10000) return;
          await this.load(pathToFileURL(file).href);
        }
      }
    };
    for (const root of this.roots) await walk(root);
    if (count > 10000) this.warnings.push('Workspace indexing limited to 10000 Julia files.');
    await this.projects.discover(projects);
    this.warnings.push(...this.projects.warnings);
    const queue = this.projects.entrySources().map(s => ({ file: s.entry, directory: s.directory }));
    const visited = new Set<string>();
    for (let index = 0; index < queue.length && visited.size < 3000; index++) {
      const { file, directory } = queue[index];
      if (visited.has(file)) continue;
      visited.add(file);
      const uri = pathToFileURL(file).href;
      if (!this.files.has(uri)) await this.load(uri);
      const a = this.files.get(uri);
      if (!a) continue;
      for (const o of a.occurrences) {
        if (o.node.text !== 'include' || o.node.parent?.type !== 'call_expression') continue;
        const arg = o.node.parent.namedChildren[1]?.namedChildren[0];
        if (!arg || !/^"[^"$\\]*"$/.test(arg.text)) continue;
        const included = path.resolve(path.dirname(file), arg.text.slice(1, -1));
        if (included.startsWith(directory + path.sep) && included.endsWith('.jl')) queue.push({ file: included, directory });
      }
      for (const statement of a.tree.rootNode.descendantsOfType(['using_statement', 'import_statement'])) for (const item of statement.namedChildren) {
        const imported = item.type === 'selected_import' ? item.namedChildren[0] : item;
        if (imported.text.startsWith('.')) continue;
        const name = imported.descendantsOfType('identifier')[0]?.text;
        if (!name) continue;
        const source = this.projects.resolve(file, name);
        if (source) queue.push({ file: source.entry, directory: source.directory });
      }
    }
    if (queue.length > visited.size && visited.size >= 3000) this.warnings.push('Dependency indexing limited to 3000 files.');
    this.dirty = true;
  }
  isEditable(uri: string) { return uri.startsWith('untitled:') || (uri.startsWith('file:') && (this.roots.some(root => fileURLToPath(uri).startsWith(root + path.sep)) || (!this.roots.length && this.open.has(uri)))); }
  private canonical(scope: Scope): Scope {
    const seen = new Set<Scope>();
    while (this.aliases.has(scope) && !seen.has(scope)) { seen.add(scope); scope = this.aliases.get(scope)!; }
    return scope;
  }
  private relink() {
    if (!this.dirty) return;
    this.dirty = false;
    this.lookupCache.clear(); this.resolutionCache.clear(); this.treeFiles.clear();
    for (const a of this.files.values()) this.treeFiles.set(a.tree, a);
    this.aliases.clear(); this.groups.clear(); this.imports.clear();
    for (const a of this.files.values()) {
      if (!a.document.uri.startsWith('file:')) continue;
      for (const o of a.occurrences) {
        if (o.node.text !== 'include' || o.node.parent?.type !== 'call_expression') continue;
        const call = o.node.parent;
        const args = call.namedChildren[1]?.namedChildren;
        if (args?.length !== 1 || args[0].type !== 'string_literal') continue;
        const raw = args[0].text;
        // Only literal paths: never execute Julia or interpolate strings.
        if (!/^"[^"$\\]*"$/.test(raw)) continue;
        const uri = pathToFileURL(path.resolve(path.dirname(fileURLToPath(a.document.uri)), raw.slice(1, -1))).href;
        const included = this.files.get(uri);
        if (included && included !== a && !this.aliases.has(included.root)) this.aliases.set(included.root, o.scope);
      }
    }
    const add = (s: Scope) => {
      const canonical = this.canonical(s);
      const list = this.groups.get(canonical) ?? [];
      if (!list.includes(s)) list.push(s);
      this.groups.set(canonical, list);
    };
    for (const a of this.files.values()) { add(a.root); for (const b of a.bindings) { add(b.scope); if (b.memberScope) add(b.memberScope); } }
    for (const a of this.files.values()) for (const o of a.occurrences) {
      let n: Node | null = o.node.parent;
      while (n && n.type !== 'using_statement' && n.type !== 'import_statement') n = n.parent;
      if (!n) continue;
      const scope = this.canonical(o.scope);
      const nodes = this.imports.get(scope) ?? [];
      if (!nodes.some(other => other.id === n!.id)) nodes.push(n);
      this.imports.set(scope, nodes);
    }
  }
  private lookup(scope: Scope, name: string, resolving = new Map<Scope, Set<string>>()): Binding[] {
    this.relink();
    if (resolving.size) return this.lookupUncached(scope, name, resolving);
    const cached = this.lookupCache.get(scope)?.get(name);
    if (cached) return cached;
    const found = this.lookupUncached(scope, name, resolving);
    const cache = this.lookupCache.get(scope) ?? new Map<string, Binding[]>();
    cache.set(name, found); this.lookupCache.set(scope, cache);
    return found;
  }
  private lookupUncached(scope: Scope, name: string, resolving: Map<Scope, Set<string>>): Binding[] {
    const seen = new Set<Scope>();
    while (!seen.has(scope)) {
      scope = this.canonical(scope); seen.add(scope);
      if (scope.globals?.has(name)) while (scope.parent && !scope.module) scope = this.canonical(scope.parent);
      if (resolving.get(scope)?.has(name)) return [];
      resolving.set(scope, new Set([...(resolving.get(scope) ?? []), name]));
      const found = (this.groups.get(scope) ?? [scope]).flatMap(s => s.bindings.get(name) ?? []);
      if (found.length) return found.every(b => b.kind === SymbolKind.Function) ? [...new Set([...found, ...this.imported(scope, name, resolving)])] : found;
      const imported = this.imported(scope, name, resolving);
      if (imported.length) return imported;
      if (!scope.parent) break;
      scope = scope.parent;
    }
    return [];
  }
  private modulePath(node: Node, scope: Scope, resolving: Map<Scope, Set<string>>, endIndex = Infinity): Binding[] {
    const identifiers = node.descendantsOfType('identifier').filter(id => id.endIndex <= endIndex);
    if (node.type === 'identifier' && !identifiers.some(id => id.id === node.id)) identifiers.unshift(node);
    if (!identifiers.length) return [];
    const dots = node.text.match(/^\.+/)?.[0].length ?? 0;
    const uri = this.treeFiles.get(scope.node.tree)?.document.uri;
    const source = !dots && uri?.startsWith('file:') ? this.projects.resolve(fileURLToPath(uri), identifiers[0].text) : undefined;
    for (let i = 1; i < dots; i++) scope = this.canonical(scope).parent ?? scope;
    let bindings = source ? (this.files.get(pathToFileURL(source.entry).href)?.bindings.filter(b => b.kind === SymbolKind.Module && b.name === source.name) ?? []) : this.lookup(scope, identifiers[0].text, new Map(resolving));
    for (const id of identifiers.slice(1)) bindings = bindings.flatMap(b => b.memberScope ? this.members(b.memberScope, id.text, resolving) : []);
    return bindings;
  }
  private imported(scope: Scope, name: string, resolving: Map<Scope, Set<string>>): Binding[] {
    const result: Binding[] = [];
    for (const statement of this.imports.get(scope) ?? []) for (const item of statement.namedChildren) {
      if (item.type === 'selected_import') {
        if (!item.namedChildren.slice(1).some(n => n.text === name)) continue;
        const modules = this.modulePath(item.namedChildren[0], scope, resolving);
        for (const module of modules) if (module.memberScope) result.push(...this.members(module.memberScope, name, resolving));
      } else {
        const modules = this.modulePath(item, scope, resolving);
        for (const module of modules) {
          if (module.name === name) result.push(module);
          if (statement.type === 'using_statement' && module.memberScope) {
            const exported = this.exported(module.memberScope);
            if (exported.has(name)) result.push(...this.members(module.memberScope, name, resolving));
          }
        }
      }
    }
    return [...new Set(result)];
  }
  private exported(scope: Scope): Set<string> {
    const names = new Set<string>();
    for (const group of this.groups.get(this.canonical(scope)) ?? [scope]) for (const node of group.node.namedChildren) {
      if (node.type === 'export_statement') for (const id of node.descendantsOfType('identifier')) names.add(id.parent?.type === 'macro_identifier' ? `@${id.text}` : id.text);
    }
    return names;
  }
  resolve(o: Occurrence): Binding[] {
    this.relink();
    const cached = this.resolutionCache.get(o.node.id);
    if (cached) return cached;
    const found = this.resolveUncached(o);
    this.resolutionCache.set(o.node.id, found);
    return found;
  }
  private resolveUncached(o: Occurrence): Binding[] {
    if (o.declaration) return this.lookup(o.declaration.scope, o.declaration.name);
    const parent = o.node.parent;
    let importPath = o.node;
    while (importPath.parent && ['scoped_identifier', 'import_path'].includes(importPath.parent.type)) importPath = importPath.parent;
    const importParent = importPath.parent;
    if (importParent && (['using_statement', 'import_statement'].includes(importParent.type) || (importParent.type === 'selected_import' && importParent.namedChildren[0].id === importPath.id))) {
      return this.modulePath(importPath, o.scope, new Map(), o.node.endIndex);
    }
    if (parent?.type === 'field_expression' && parent.namedChildren.at(-1)?.id === o.node.id) {
      const base = parent.namedChildren[0];
      const a = this.files.get(this.uriFor(o));
      const name = base.type === 'field_expression' ? base.namedChildren.at(-1)! : base;
      const baseOccurrence = a?.occurrences.find(r => r.node.id === name.id);
      if (!baseOccurrence) return [];
      return this.resolve(baseOccurrence).flatMap(b => b.memberScope ? this.members(b.memberScope, o.node.text) : []);
    }
    // Keyword labels do not refer to local variables.
    if (parent?.type === 'named_argument' && parent.namedChildren[0].id === o.node.id) return [];
    return this.lookup(o.scope, parent?.type === 'macro_identifier' ? `@${o.node.text}` : o.node.text);
  }
  memberBindings(scope: Scope) {
    this.relink(); scope = this.canonical(scope);
    const bindings = (this.groups.get(scope) ?? [scope]).flatMap(s => [...s.bindings.values()].flat());
    for (const statement of this.imports.get(scope) ?? []) for (const item of statement.namedChildren) {
      const names = item.type === 'selected_import' ? item.namedChildren.slice(1).map(n => n.text) : this.modulePath(item, scope, new Map()).flatMap(b => [b.name, ...(b.memberScope ? this.exported(b.memberScope) : [])]);
      for (const name of names) bindings.push(...this.members(scope, name));
    }
    return [...new Set(bindings)];
  }
  private members(scope: Scope, name: string, resolving = new Map<Scope, Set<string>>()): Binding[] {
    this.relink(); scope = this.canonical(scope);
    if (resolving.get(scope)?.has(name)) return [];
    const found = (this.groups.get(scope) ?? [scope]).flatMap(s => s.bindings.get(name) ?? []);
    const next = new Map(resolving); next.set(scope, new Set([...(resolving.get(scope) ?? []), name]));
    if (found.length) return found.every(b => b.kind === SymbolKind.Function) ? [...new Set([...found, ...this.imported(scope, name, next)])] : found;
    return this.imported(scope, name, next);
  }
  unresolvedImport(binding: Binding) {
    this.relink();
    const scope = this.canonical(binding.scope);
    const selected = (this.imports.get(scope) ?? []).some(s => s.namedChildren.some(n => n.type === 'selected_import' && n.namedChildren.slice(1).some(id => id.text === binding.name)));
    return selected && !this.imported(scope, binding.name, new Map()).length;
  }
  uriFor(o: Occurrence) {
    return this.treeFiles.get(o.node.tree)?.document.uri ?? '';
  }
  visible(scope: Scope): Binding[] {
    this.relink();
    const seen = new Set<Scope>(); const names = new Set<string>(); const result: Binding[] = [];
    while (!seen.has(scope)) {
      scope = this.canonical(scope); seen.add(scope);
      for (const group of this.groups.get(scope) ?? [scope]) for (const [name, bindings] of group.bindings) {
        if (!names.has(name)) { result.push(...bindings); names.add(name); }
      }
      for (const statement of this.imports.get(scope) ?? []) for (const item of statement.namedChildren) {
        const candidates = item.type === 'selected_import' ? item.namedChildren.slice(1).map(n => n.text) : this.modulePath(item, scope, new Map()).flatMap(b => [b.name, ...(b.memberScope ? this.exported(b.memberScope) : [])]);
        for (const name of candidates) if (!names.has(name)) { result.push(...this.lookup(scope, name)); names.add(name); }
      }
      if (!scope.parent) break;
      scope = scope.parent;
    }
    return result;
  }
  references(bindings: Binding[], includeDeclaration: boolean) {
    const targets = new Set(bindings);
    const names = new Set(bindings.map(b => b.name.replace(/^@/, '')));
    const result: { a: Analysis; o: Occurrence }[] = [];
    for (const a of this.files.values()) for (const o of a.occurrences) {
      if (!names.has(o.node.text)) continue;
      if (!includeDeclaration && o.declaration) continue;
      if (this.resolve(o).some(b => targets.has(b))) result.push({ a, o });
    }
    return result;
  }
}
