import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
await mkdir('dist', { recursive: true });
await copyFile('node_modules/tree-sitter-julia/tree-sitter-julia.wasm', 'dist/tree-sitter-julia.wasm');
await writeFile('THIRD_PARTY_NOTICES.md', '# Parser licenses\n\n## tree-sitter-julia 0.23.1\n\n' + await readFile('node_modules/tree-sitter-julia/LICENSE', 'utf8') + '\n## web-tree-sitter 0.24.7\n\n' + await readFile('node_modules/web-tree-sitter/LICENSE', 'utf8'));
