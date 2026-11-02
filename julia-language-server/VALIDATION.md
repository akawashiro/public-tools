# Validation — 2026-11-01

Delivered artifact: `julia-workbench-0.1.0.vsix` (1.36 MiB compressed).

SHA-256: `29d35a2ea6baeb5b9fb889fc1aad90e0a89e124b79828d3d1d9ac564ee62523c`

## Objective evidence

| Requirement | Evidence |
| --- | --- |
| Independent Julia LSP implementation | `src/analysis.ts`, `src/workspace.ts`, `src/projects.ts`, `src/server.ts`; no LanguageServer.jl dependency. Tree-sitter supplies the grammar, Microsoft libraries supply the LSP transport/client. |
| Useful source editing features | `npm test`: 14 passing tests, including actual server subprocess communication, definitions, references, safe rename, completion, hover, signatures, diagnostics, symbols, links, folding, Unicode positions, file changes and close/reload. |
| Project and dependency navigation | Tests cover Manifest path dependencies, exact depot versions, source availability, dependency rename protection, and changes to Project.toml. Manual test with installed StaticArrays 1.9.7 resolved `SVector` through StaticArrays to StaticArraysCore. |
| VSCode integration | VSIX successfully installed into `/tmp/julia-lsp-vscode-packaged-extensions`; VSCode 1.104.3 launched the installed extension using an isolated profile and Xvfb. `test/vscode.cjs` passed activation, definition, hover, completion, rename, diagnostics and restart checks. |
| Installable distribution | `npm run package` generated the VSIX, containing the JavaScript server/client, Julia grammar WASM, web-tree-sitter runtime, TOML parser, other runtime dependencies, syntax grammar and licenses. |
| Other LSP clients | Tests communicate with `node dist/server.js --stdio` using standard Content-Length JSON-RPC framing. README supplies the command for editor configuration. No separate Neovim GUI test was performed. |

The final installed-artifact VSCode run recorded:

```text
2026-11-01 14:08:10.555 [info] Julia Workbench VSCode integration passed: activation, definition, hover, completion, rename, diagnostics, restart.
```

Runtime dependency check: `npm audit --omit=dev` → `found 0 vulnerabilities`.

## Real package check

A temporary project used the installed Manifest and a direct StaticArrays dependency. The server indexed 55 files and resolved `SVector` to:

```julia
# StaticArraysCore/src/StaticArraysCore.jl
SVector{S, T} = SArray{Tuple{S}, T, 1, S}
```

On this machine, the measured first completion query returned 85 visible bindings in approximately 228 ms, and reference search returned 84 occurrences in approximately 292 ms. These are observational measurements, not guaranteed latency bounds.

## Limits of the evidence

Tests establish the supported static editing features, not full Julia compiler equivalence. Runtime type inference, macro expansion, dynamic include/eval, automatic standard-library source discovery, environment stacks, version-specific Manifest selection and import aliases remain outside this release. README documents the supported scope and limits. Marketplace publication was not performed; the local VSIX is ready to install.
