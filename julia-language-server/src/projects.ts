import { promises as fs } from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { parse } from 'smol-toml';

interface Metadata { name?: string; uuid?: string; deps?: Record<string, string>; sources?: Record<string, { path?: string }>; }
interface ManifestEntry { uuid?: string; path?: string; 'git-tree-sha1'?: string; }
export interface PackageSource { name: string; uuid: string; directory: string; entry: string; }
interface Environment { directory: string; project: Metadata; sources: Map<string, PackageSource>; }

function crc32c(bytes: Uint8Array, initial = 0): number {
  let crc = (initial ^ 0xffffffff) >>> 0;
  for (const byte of bytes) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (crc & 1 ? 0x82f63b78 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
/** Julia's version_slug: CRC32C(UUID UInt128 little endian, SHA1 bytes), base 62. */
export function versionSlug(uuid: string, sha: string) {
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(uuid) || !/^[0-9a-f]{40}$/i.test(sha)) return '';
  let value = crc32c(Buffer.from(sha, 'hex'), crc32c(Buffer.from(uuid.replaceAll('-', ''), 'hex').reverse()));
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let slug = '';
  for (let i = 0; i < 5; i++) { slug += alphabet[value % 62]; value = Math.floor(value / 62); }
  return slug;
}
const within = (file: string, directory: string) => file === directory || file.startsWith(directory + path.sep);
async function readToml(file: string): Promise<Record<string, unknown> | undefined> {
  try { return parse(await fs.readFile(file, 'utf8')); } catch { return undefined; }
}
export class Projects {
  private environments: Environment[] = [];
  private packageProjects = new Map<string, Metadata>();
  warnings: string[] = [];
  constructor(private depots = (process.env.JULIA_DEPOT_PATH?.split(path.delimiter).filter(Boolean) ?? [path.join(os.homedir(), '.julia')])) {}
  async discover(directories: string[]) {
    this.environments = []; this.packageProjects.clear(); this.warnings = [];
    for (const directory of [...new Set(directories)]) {
      const project = await readToml(path.join(directory, 'Project.toml')) as Metadata | undefined;
      if (!project) continue;
      const manifest = await readToml(path.join(directory, 'Manifest.toml'));
      const sources = new Map<string, PackageSource>();
      const environment = { directory, project, sources };
      this.environments.push(environment);
      // Manifest format 2 nests entries under deps; format 1 stores them at top level.
      const entries = (manifest?.deps ?? manifest ?? {}) as Record<string, ManifestEntry[]>;
      for (const [name, versions] of Object.entries(entries)) {
        if (!Array.isArray(versions)) continue;
        for (const entry of versions) {
          if (typeof entry.uuid !== 'string') continue;
          const dirs: string[] = [];
          if (typeof entry.path === 'string') dirs.push(path.resolve(directory, entry.path));
          else if (typeof entry['git-tree-sha1'] === 'string') {
            const slug = versionSlug(entry.uuid, entry['git-tree-sha1']);
            if (slug && /^[\p{L}\p{N}_]+$/u.test(name)) for (const depot of this.depots) dirs.push(path.join(depot, 'packages', name, slug));
          }
          for (const dir of dirs) {
            const source = await this.source(name, entry.uuid, dir);
            if (source) { sources.set(entry.uuid, source); break; }
          }
        }
      }
      for (const [name, uuid] of Object.entries(project.deps ?? {})) {
        if (sources.has(uuid)) continue;
        const sourcePath = project.sources?.[name]?.path;
        if (typeof sourcePath === 'string') {
          const source = await this.source(name, uuid, path.resolve(directory, sourcePath));
          if (source) sources.set(uuid, source);
        }
        if (!sources.has(uuid)) this.warnings.push(`Source not found for ${name} in ${directory}; instantiate the Julia environment to enable dependency navigation.`);
      }
      if (project.name && project.uuid) {
        const source = await this.source(project.name, project.uuid, directory);
        if (source) sources.set(project.uuid, source);
      }
    }
  }
  private async source(name: string, uuid: string, directory: string): Promise<PackageSource | undefined> {
    if (!/^[\p{L}\p{N}_]+$/u.test(name)) return;
    const metadata = await readToml(path.join(directory, 'Project.toml')) as Metadata | undefined;
    if (!metadata || metadata.uuid !== uuid || metadata.name !== name) return;
    const entry = path.join(directory, 'src', `${name}.jl`);
    try { if (!(await fs.stat(entry)).isFile()) return; } catch { return; }
    this.packageProjects.set(directory, metadata);
    return { name, uuid, directory, entry };
  }
  allSources() { return [...new Map(this.environments.flatMap(e => [...e.sources.values()]).map(s => [s.entry, s])).values()]; }
  entrySources() {
    return this.environments.flatMap(e => [...Object.values(e.project.deps ?? {}), ...(e.project.uuid ? [e.project.uuid] : [])].flatMap(uuid => e.sources.get(uuid) ?? []));
  }
  resolve(file: string, name: string): PackageSource | undefined {
    const project = this.environments.filter(e => within(file, e.directory)).sort((a, b) => b.directory.length - a.directory.length)[0];
    if (project) {
      const uuid = project.project.name === name ? project.project.uuid : project.project.deps?.[name];
      return uuid ? project.sources.get(uuid) : undefined;
    }
    for (const e of this.environments) {
      const owner = [...e.sources.values()].filter(s => within(file, s.directory)).sort((a, b) => b.directory.length - a.directory.length)[0];
      if (!owner) continue;
      const metadata = this.packageProjects.get(owner.directory);
      const uuid = owner.name === name ? owner.uuid : metadata?.deps?.[name];
      if (uuid) return e.sources.get(uuid);
    }
  }
}
