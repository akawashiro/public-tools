# Julia Workbench

Julia 向けに一から実装した静的解析 LSP サーバーと VSCode 拡張です。既存の Julia LanguageServer.jl は利用していません。LSP 通信には Microsoft のライブラリ、構文解析には Tree-sitter Julia の WASM 文法を使用しています。

Julia 本体のインストールやプロジェクトの実行なしで起動します。作業中のソースを実行せずに解析するため、未信頼ワークスペースでも利用できます。

## VSCode で使う

開発・ビルドには **Node.js 22 以上と npm** を使用してください。

```sh
npm ci --ignore-scripts
npm test
npm run package
code --install-extension julia-workbench-0.1.0.vsix
```

VSCode で Julia プロジェクトのフォルダーを開き、`.jl` ファイルを開くと起動します。配布済み VSIX から使う場合、npm や Julia 本体は不要です。別の Julia 拡張が言語機能を提供している場合は、そのワークスペースでは片方を無効にすると、補完や診断の重複を避けられます。

- 補完: `Ctrl+Space`
- 定義ジャンプ: `F12`
- 参照検索: `Shift+F12`
- リネーム: `F2`
- ホバー: 識別子にマウスを合わせる
- 引数ヒント: `Ctrl+Shift+Space`
- シンボル検索: `Ctrl+T` / ファイル内は `Ctrl+Shift+O`

Output パネルの **Julia Workbench** でサーバー出力を確認できます。再起動はコマンドパレットの **Julia Workbench: Restart Language Server** を使用してください。開発時は `npm run build` の後、F5 でも実行できます。

## 機能

- Julia 文法による構文診断。未保存ファイルと増分変更、UTF-16 位置に対応。
- 関数・引数・ローカル変数・定数・型・モジュールの定義ジャンプ、参照検索、リネーム。
- 関数、ループ、let、クロージャーなどのスコープを考慮した名前解決。
- リテラルの `include("relative/path.jl")` を介したファイル間解析とリンク。
- ワークスペース内モジュールの修飾参照、`using .Module` と export、`using ..Module: name` による名前解決。
- `Project.toml` と `Manifest.toml` に基づく依存パッケージの定義ジャンプ・ホバー・補完。Manifest の path、Project の sources.path、およびインストール済み depot の正確な git-tree-sha1 を使用。依存ソースは書き換えません。
- スコープ内の名前と基本的な Base 関数・型・キーワードの補完。
- ソース直前の docstring のホバー、関数定義の引数ヒント。
- ファイルとワークスペースのシンボル検索、折り畳み、構文色分け、コメント・括弧操作。
- ファイル監視、複数ルートのワークスペース、ファイルを閉じた際のディスク内容への復帰。

リネームはコメント・文字列・解決できないプロパティ名・呼び出し側のキーワードラベルを変更しません。既存の可視名と衝突する場合は拒否します。同名の複数メソッドはまとめて扱い、型によるディスパッチの選択は行いません。

## 解析範囲と制限

これは静的なソース解析に基づく言語サーバーです。Julia の実行時の型推論、マクロ展開、`eval`、動的 include、外部パッケージの実行は行いません。型エラーや未定義変数の診断は提供せず、診断は構文エラーが対象です。Base と標準ライブラリのソース環境は自動解析せず、Base の補完・ドキュメントは小規模な組み込み辞書です。

依存先は `Manifest.toml` で固定された版を `JULIA_DEPOT_PATH`（未指定時は `~/.julia`）から読み込みます。パッケージのダウンロードや `Pkg.instantiate` は実行しません。依存パッケージの入口から、リテラル include と import によって到達できるソースを解析します。Manifest の版がインストールされていない場合は、別の版に勝手に切り替えず、Output に記録します。現在は通常の `Project.toml` / `Manifest.toml` が対象で、バージョン別 Manifest、LOAD_PATH の環境スタック、標準ライブラリの解決は未対応です。

`using` / `import` の alias (`as`)、再 export、演算子のメソッド定義、複雑な生成コードなどは解析が限定的です。レシーバーの型が分からない `object.field` はリネーム対象にしません。引用コードやマクロを多用するコードには解析の曖昧さが残るため、リネーム時には VSCode の変更プレビューを確認してください。同一ソースを複数の異なるモジュールから include する構成は未対応です。

ワークスペースの `.jl` を最大 10,000 ファイル、依存ソースを最大 3,000 ファイル、ディスク上の各ファイルは最大 2 MiB まで読み込みます。`.git`、`node_modules`、`.julia`、`dist`、`build` は除外し、シンボリックリンクはたどりません。独立したファイルの同名シンボルは混同しないよう、ファイル間の名前解決は include とモジュール参照による関係を使用します。

## 他のエディター

Node.js 20 以上でビルド済みサーバーを標準入出力から起動できます。

```sh
node /absolute/path/julia-language-server/dist/server.js --stdio
```

LSP クライアントのファイルタイプを `julia`、プロジェクトのルートを `Project.toml` または `.git` のあるディレクトリに設定します。Neovim 0.11 では、たとえば次の設定を使用できます。

```lua
vim.lsp.config('julia_workbench', {
  cmd = { 'node', '/absolute/path/julia-language-server/dist/server.js', '--stdio' },
  filetypes = { 'julia' },
  root_markers = { 'Project.toml', '.git' },
})
vim.lsp.enable('julia_workbench')
```

## 検証と実装

`npm test` はビルド後、実際のサーバー子プロセスと JSON-RPC で通信します。名前解決、スコープ分離、ファイル間参照とリネーム、import/export、Unicode の位置、編集中の更新、構文診断を検証します。

VSCode の拡張ホスト向けテストは `test/vscode.cjs` にあります。隔離したプロファイルで次を実行できます（Linux のヘッドレス環境では Xvfb などが必要です）。

```sh
code --extensionDevelopmentPath="$PWD" \
  --extensionTestsPath="$PWD/test/vscode.cjs" \
  --user-data-dir=/tmp/julia-workbench-test-user \
  --extensions-dir=/tmp/julia-workbench-test-extensions \
  --disable-workspace-trust "$PWD/examples" --wait
```

- `src/analysis.ts`: 構文木、宣言、字句スコープ、構文診断。
- `src/workspace.ts`: ファイル管理、include グラフ、import/export、参照解決。
- `src/projects.ts`: Julia のプロジェクト・Manifest・depot から依存ソースを解決。
- `src/server.ts`: LSP の各機能。
- `src/extension.ts`: VSCode クライアントとファイル監視。

プロトコルは [LSP 3.17](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/)、クライアント構成は [VSCode Language Server Extension Guide](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide) を参照しています。

MIT License。パーサー文法とランタイムのライセンスは VSIX の `THIRD_PARTY_NOTICES.md` を参照してください。
