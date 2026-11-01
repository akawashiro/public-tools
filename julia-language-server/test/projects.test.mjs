import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import projects from '../dist/projects.js';
const { Projects, versionSlug } = projects;

test('Julia depot slugs match existing Julia-generated package paths', () => {
  assert.equal(versionSlug('90137ffa-7385-5640-81b9-e52037218182', 'eeafab08ae20c62c44c8399ccb9354a04b80db50'), 'MSJcA');
  assert.equal(versionSlug('6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf', 'f1dff6729bc61f4d49e140da1af55dcd1ac97b2f'), 'QNsku');
  assert.equal(versionSlug('../../invalid', 'invalid'), '');
});

test('exact Manifest hashes select matching depot sources, not another installed version', async t => {
  const root = await mkdtemp(join(tmpdir(), 'julia-project-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const depot = join(root, 'depot');
  const project = join(root, 'project');
  const uuid = '11111111-1111-1111-1111-111111111111';
  const hash = '0123456789abcdef0123456789abcdef01234567';
  const selected = join(depot, 'packages/Example', versionSlug(uuid, hash));
  await mkdir(join(selected, 'src'), { recursive: true });
  await mkdir(project);
  await writeFile(join(selected, 'Project.toml'), `name="Example"\nuuid="${uuid}"\n`);
  await writeFile(join(selected, 'src/Example.jl'), 'module Example\nend\n');
  await writeFile(join(project, 'Project.toml'), `[deps]\nExample="${uuid}"\n`);
  await writeFile(join(project, 'Manifest.toml'), `manifest_format="2.0"\n[[deps.Example]]\nuuid="${uuid}"\ngit-tree-sha1="${hash}"\n`);
  const resolver = new Projects([depot]);
  await resolver.discover([project]);
  assert.equal(resolver.resolve(join(project, 'main.jl'), 'Example').directory, selected);
  assert.equal(resolver.resolve(join(project, 'main.jl'), 'NotADependency'), undefined);
  await writeFile(join(project, 'Manifest.toml'), `manifest_format="2.0"\n[[deps.Example]]\nuuid="${uuid}"\ngit-tree-sha1="${'a'.repeat(40)}"\n`);
  await resolver.discover([project]);
  assert.equal(resolver.resolve(join(project, 'main.jl'), 'Example'), undefined);
});
