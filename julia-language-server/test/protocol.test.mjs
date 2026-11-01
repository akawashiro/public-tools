import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, writeFile, rm, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createMessageConnection, StreamMessageReader, StreamMessageWriter } from 'vscode-jsonrpc/node.js';

async function client(t, files = {}) {
  const dir = await mkdtemp(join(tmpdir(), 'julia-workbench-'));
  for (const [name, text] of Object.entries(files)) await writeFile(join(dir, name), text);
  const child = spawn(process.execPath, ['dist/server.js', '--stdio']);
  let errors = ''; child.stderr.on('data', d => errors += d);
  const rpc = createMessageConnection(new StreamMessageReader(child.stdout), new StreamMessageWriter(child.stdin));
  rpc.listen();
  const diagnostics = new Map();
  rpc.onNotification('textDocument/publishDiagnostics', p => diagnostics.set(p.uri, p));
  t.after(async () => {
    if (child.exitCode === null) {
      try { await Promise.race([rpc.sendRequest('shutdown'), new Promise((_, reject) => setTimeout(() => reject(new Error('shutdown timeout')), 2000).unref())]); rpc.sendNotification('exit'); } catch {}
    }
    rpc.dispose(); child.kill(); await rm(dir, { recursive: true, force: true });
    assert.equal(errors, '', 'server stderr');
  });
  const initialized = await Promise.race([rpc.sendRequest('initialize', { processId: process.pid, capabilities: {}, workspaceFolders: [{ uri: pathToFileURL(dir).href, name: 'test' }] }), new Promise((_, reject) => setTimeout(() => reject(new Error(`initialize timeout: ${errors}`)), 5000).unref())]);
  assert.equal(initialized.capabilities.positionEncoding, 'utf-16');
  rpc.sendNotification('initialized', {});
  const uri = name => pathToFileURL(join(dir, name)).href;
  const request = (method, name, line = 0, character = 0, rest = {}) => rpc.sendRequest(method, { textDocument: { uri: uri(name) }, position: { line, character }, ...rest });
  const open = (name, text, version = 1) => rpc.sendNotification('textDocument/didOpen', { textDocument: { uri: uri(name), languageId: 'julia', version, text } });
  return { rpc, uri, request, open, diagnostics, dir };
}

test('lexical definitions, references, rename, docs, signature, symbols', { timeout: 15000 }, async t => {
  const c = await client(t);
  c.open('main.jl', '"""Add one."""\nfunction addone(x::Int)\n y = x + 1\n return y\nend\nfunction other(x)\n return x\nend\naddone(4)\n');
  const def = await c.request('textDocument/definition', 'main.jl', 2, 5);
  assert.equal(def[0].range.start.line, 1);
  assert.equal(def[0].range.start.character, 16);
  const refs = await c.request('textDocument/references', 'main.jl', 2, 5, { context: { includeDeclaration: true } });
  assert.deepEqual(refs.map(r => r.range.start.line), [1, 2]);
  const rename = await c.request('textDocument/rename', 'main.jl', 2, 5, { newName: 'value' });
  assert.equal(rename.documentChanges[0].textDocument.version, 1);
  assert.equal(rename.documentChanges[0].edits.length, 2);
  await assert.rejects(c.request('textDocument/rename', 'main.jl', 2, 5, { newName: 'y' }), /conflict/);
  await assert.rejects(c.request('textDocument/rename', 'main.jl', 2, 5, { newName: 'end' }), /identifier/);
  const hover = await c.request('textDocument/hover', 'main.jl', 8, 2);
  assert.match(hover.contents.value, /Add one/);
  const sig = await c.request('textDocument/signatureHelp', 'main.jl', 8, 8);
  assert.equal(sig.signatures[0].label, 'addone(x::Int)');
  const symbols = await c.request('textDocument/documentSymbol', 'main.jl');
  assert.deepEqual(symbols.map(s => s.name), ['addone', 'other']);
  const complete = await c.request('textDocument/completion', 'main.jl', 8, 3);
  assert(complete.some(i => i.label === 'addone'));
  const folds = await c.request('textDocument/foldingRange', 'main.jl');
  assert(folds.some(f => f.startLine === 1 && f.endLine === 3));
});

test('module include graph, isolated files, cross-file rename and document links', { timeout: 15000 }, async t => {
  const c = await client(t, {
    'main.jl': 'module Demo\ninclude("math.jl")\nanswer = twice(21)\nend\nDemo.twice(2)\n',
    'math.jl': '"""Double a value."""\ntwice(x) = x * 2\n',
    'unrelated.jl': 'twice(x) = x + 10\n'
  });
  const defs = await c.request('textDocument/definition', 'main.jl', 2, 10);
  assert.equal(defs.length, 1); assert.equal(defs[0].uri, c.uri('math.jl'));
  const qualified = await c.request('textDocument/definition', 'main.jl', 4, 7);
  assert.equal(qualified[0].uri, c.uri('math.jl'));
  const refs = await c.request('textDocument/references', 'math.jl', 1, 2, { context: { includeDeclaration: true } });
  assert.equal(refs.length, 3);
  const edits = await c.request('textDocument/rename', 'math.jl', 1, 2, { newName: 'double_value' });
  assert.equal(edits.documentChanges.length, 2);
  assert(edits.documentChanges.every(e => e.textDocument.version === null));
  const links = await c.request('textDocument/documentLink', 'main.jl');
  assert.equal(links[0].target, c.uri('math.jl'));
  const symbols = await c.rpc.sendRequest('workspace/symbol', { query: 'twice' });
  assert.equal(symbols.length, 2);
});

test('UTF-16 positions, incremental updates, diagnostics and close reload', { timeout: 15000 }, async t => {
  const c = await client(t, { 'unicode.jl': 'α(x) = x\nα(2)\n' });
  c.open('unicode.jl', 'α(x) = x\n"😀"; α(2)\n');
  const defs = await c.request('textDocument/definition', 'unicode.jl', 1, 6);
  assert.equal(defs[0].range.start.character, 0);
  c.rpc.sendNotification('textDocument/didChange', { textDocument: { uri: c.uri('unicode.jl'), version: 2 }, contentChanges: [{ range: { start: { line: 1, character: 6 }, end: { line: 1, character: 7 } }, text: 'β' }] });
  assert.deepEqual(await c.request('textDocument/definition', 'unicode.jl', 1, 6), []);
  c.rpc.sendNotification('textDocument/didChange', { textDocument: { uri: c.uri('unicode.jl'), version: 3 }, contentChanges: [{ text: 'function broken(\n' }] });
  await c.request('textDocument/documentSymbol', 'unicode.jl');
  assert(c.diagnostics.get(c.uri('unicode.jl')).diagnostics.length > 0);
  assert.equal(c.diagnostics.get(c.uri('unicode.jl')).version, 3);
  c.rpc.sendNotification('textDocument/didClose', { textDocument: { uri: c.uri('unicode.jl') } });
  const restored = await c.request('textDocument/definition', 'unicode.jl', 1, 0);
  assert.equal(restored[0]?.range.start.line, 0);
  assert.deepEqual(c.diagnostics.get(c.uri('unicode.jl')).diagnostics, []);
});

test('property names and keyword labels never rename unrelated locals', { timeout: 15000 }, async t => {
  const c = await client(t);
  c.open('safe.jl', 'x = 1\nobj.x\nf(x=2)\nprintln(x)\n# x\n"x"\n');
  const refs = await c.request('textDocument/references', 'safe.jl', 0, 0, { context: { includeDeclaration: true } });
  assert.deepEqual(refs.map(r => r.range.start.line), [0, 3]);
  await assert.rejects(c.request('textDocument/prepareRename', 'safe.jl', 1, 4), /cannot be resolved/);
});

test('exports and selected imports resolve local modules', { timeout: 15000 }, async t => {
  const c = await client(t, {
    'imports.jl': 'module A\nexport foo\nfoo(x) = x\nprivate(x) = x\nend\nusing .A\nfoo(2)\nprivate(2)\nmodule B\nusing ..A: private\nprivate(3)\nend\n'
  });
  const def = await c.request('textDocument/definition', 'imports.jl', 6, 1);
  assert.equal(def[0]?.range.start.line, 2);
  assert.deepEqual(await c.request('textDocument/definition', 'imports.jl', 7, 1), []);
  const selected = await c.request('textDocument/definition', 'imports.jl', 10, 1);
  assert.equal(selected[0]?.range.start.line, 3);
  const refs = await c.request('textDocument/references', 'imports.jl', 2, 1, { context: { includeDeclaration: true } });
  assert.deepEqual(refs.map(r => r.range.start.line), [1, 2, 6]);
  const completion = await c.request('textDocument/completion', 'imports.jl', 6, 1);
  assert(completion.some(i => i.label === 'foo'));
});

test('nested assignments capture outer locals, let and loop bindings stay separate', { timeout: 15000 }, async t => {
  const c = await client(t);
  c.open('scope.jl', 'function f(x)\n for i in 1:3\n x += i\n end\n let x = 10\n println(x)\n end\n return x\nend\n');
  const refs = await c.request('textDocument/references', 'scope.jl', 0, 11, { context: { includeDeclaration: true } });
  assert.deepEqual(refs.map(r => r.range.start.line), [0, 2, 7]);
  const inner = await c.request('textDocument/references', 'scope.jl', 4, 5, { context: { includeDeclaration: true } });
  assert.deepEqual(inner.map(r => r.range.start.line), [4, 5]);
});

test('macro namespace, nested module members, type parameters and quoted symbols', { timeout: 15000 }, async t => {
  const c = await client(t);
  c.open('syntax.jl', 'macro ident(x)\n x\nend\nident(x) = x\n@ident 1\nident(1)\nmodule A\nmodule B\nf(x) = x\nend\nend\nA.B.f(2)\nstruct Box{T}\n value::T\nend\nx=1\n:x\n:(x+1)\nquote\n x\nend\nprintln(x)\n');
  const macro = await c.request('textDocument/references', 'syntax.jl', 0, 7, { context: { includeDeclaration: true } });
  assert.deepEqual(macro.map(r => r.range.start.line), [0, 4]);
  const fn = await c.request('textDocument/references', 'syntax.jl', 3, 2, { context: { includeDeclaration: true } });
  assert.deepEqual(fn.map(r => r.range.start.line), [3, 5]);
  const nested = await c.request('textDocument/definition', 'syntax.jl', 11, 4);
  assert.equal(nested[0]?.range.start.line, 8);
  const generic = await c.request('textDocument/definition', 'syntax.jl', 13, 8);
  assert.equal(generic[0]?.range.start.line, 12);
  const quoted = await c.request('textDocument/references', 'syntax.jl', 15, 0, { context: { includeDeclaration: true } });
  assert.deepEqual(quoted.map(r => r.range.start.line), [15, 21]);
});

test('disk file changes and deletion update the index while unsaved buffers win', { timeout: 15000 }, async t => {
  const c = await client(t, { 'watch.jl': 'original(x)=x\n' });
  assert.equal((await c.rpc.sendRequest('workspace/symbol', { query: 'original' })).length, 1);
  await writeFile(join(c.dir, 'watch.jl'), 'changed(x)=x\n');
  c.rpc.sendNotification('workspace/didChangeWatchedFiles', { changes: [{ uri: c.uri('watch.jl'), type: 2 }] });
  assert.equal((await c.rpc.sendRequest('workspace/symbol', { query: 'original' })).length, 0);
  assert.equal((await c.rpc.sendRequest('workspace/symbol', { query: 'changed' })).length, 1);
  c.open('watch.jl', 'unsaved(x)=x\n');
  c.rpc.sendNotification('workspace/didChangeWatchedFiles', { changes: [{ uri: c.uri('watch.jl'), type: 2 }] });
  assert.equal((await c.rpc.sendRequest('workspace/symbol', { query: 'unsaved' })).length, 1);
  c.rpc.sendNotification('textDocument/didClose', { textDocument: { uri: c.uri('watch.jl') } });
  assert.equal((await c.rpc.sendRequest('workspace/symbol', { query: 'changed' })).length, 1);
  await rm(join(c.dir, 'watch.jl'));
  c.rpc.sendNotification('workspace/didChangeWatchedFiles', { changes: [{ uri: c.uri('watch.jl'), type: 3 }] });
  assert.equal((await c.rpc.sendRequest('workspace/symbol', { query: '' })).length, 0);
});

test('nested parameters and explicit globals do not capture outer locals', { timeout: 15000 }, async t => {
  const c = await client(t);
  c.open('bindings.jl', 'x=0\nfunction outer(x)\n function inner(x)\n x=2\n return x\n end\n function update()\n global x\n x=3\n return x\n end\n return x\nend\n');
  const inner = await c.request('textDocument/references', 'bindings.jl', 2, 16, { context: { includeDeclaration: true } });
  assert.deepEqual(inner.map(r => r.range.start.line), [2, 3, 4]);
  const outer = await c.request('textDocument/references', 'bindings.jl', 1, 15, { context: { includeDeclaration: true } });
  assert.deepEqual(outer.map(r => r.range.start.line), [1, 11]);
  const global = await c.request('textDocument/references', 'bindings.jl', 0, 0, { context: { includeDeclaration: true } });
  assert.deepEqual(global.map(r => r.range.start.line), [0, 7, 8, 9]);
});

test('Manifest path dependency navigation and completion without editing dependencies', { timeout: 15000 }, async t => {
  const dep = await mkdtemp(join(tmpdir(), 'julia-workbench-package-'));
  t.after(() => rm(dep, { recursive: true, force: true }));
  await mkdir(join(dep, 'src'));
  await writeFile(join(dep, 'Project.toml'), 'name="Fixture"\nuuid="11111111-1111-1111-1111-111111111111"\n');
  await writeFile(join(dep, 'src/Fixture.jl'), 'module Fixture\nexport double_value\ninclude("math.jl")\nend\n');
  await writeFile(join(dep, 'src/math.jl'), '"""Double the input."""\ndouble_value(x) = x * 2\n');
  const c = await client(t, {
    'Project.toml': '[deps]\nFixture="11111111-1111-1111-1111-111111111111"\n',
    'Manifest.toml': `manifest_format="2.0"\n[[deps.Fixture]]\nuuid="11111111-1111-1111-1111-111111111111"\npath=${JSON.stringify(dep)}\n`,
    'main.jl': 'using Fixture\ndouble_value(2)\nFixture.double_value(3)\n'
  });
  const def = await c.request('textDocument/definition', 'main.jl', 1, 3);
  const packageDef = await c.request('textDocument/definition', 'main.jl', 0, 8);
  assert.equal(packageDef[0]?.uri, pathToFileURL(join(dep, 'src/Fixture.jl')).href);
  assert.equal(def[0]?.uri, pathToFileURL(join(dep, 'src/math.jl')).href);
  assert.equal(def[0]?.range.start.line, 1);
  const qualified = await c.request('textDocument/definition', 'main.jl', 2, 12);
  assert.equal(qualified[0]?.uri, def[0].uri);
  const hover = await c.request('textDocument/hover', 'main.jl', 1, 3);
  assert.match(hover.contents.value, /Double the input/);
  const completion = await c.request('textDocument/completion', 'main.jl', 1, 4);
  assert(completion.some(i => i.label === 'double_value'));
  await assert.rejects(c.request('textDocument/rename', 'main.jl', 1, 3, { newName: 'twice' }), /read-only/);
  await writeFile(join(c.dir, 'Project.toml'), '[deps]\n');
  c.rpc.sendNotification('workspace/didChangeWatchedFiles', { changes: [{ uri: c.uri('Project.toml'), type: 2 }] });
  assert.deepEqual(await c.request('textDocument/definition', 'main.jl', 1, 3), []);
});

test('exported parametric aliases resolve through selected imports', { timeout: 15000 }, async t => {
  const c = await client(t, {
    'aliases.jl': 'module A\nexport Alias\nconst Alias{T} = Vector{T}\nend\nmodule B\nimport ..A: Alias\nexport Alias\nend\nusing .B\nAlias{Int}()\n'
  });
  const definitions = await c.request('textDocument/definition', 'aliases.jl', 9, 2);
  assert.equal(definitions[0]?.range.start.line, 2);
});

test('method extensions retain imported identity and unresolved imports cannot be renamed', { timeout: 15000 }, async t => {
  const c = await client(t);
  c.open('extensions.jl', 'module A\nf(x)=x\nend\nimport .A: f\nf(x::Int)=x+1\nf(2)\nimport Unavailable: g\ng(x::Int)=x\ng(3)\n');
  const def = await c.request('textDocument/definition', 'extensions.jl', 5, 0);
  assert.deepEqual(def.map(d => d.range.start.line).sort((a, b) => a-b), [1, 4]);
  const edits = await c.request('textDocument/rename', 'extensions.jl', 5, 0, { newName: 'renamed' });
  assert.deepEqual(edits.documentChanges[0].edits.map(e => e.range.start.line), [1, 3, 4, 5]);
  await assert.rejects(c.request('textDocument/rename', 'extensions.jl', 8, 0, { newName: 'newg' }), /unresolved import/);
});
