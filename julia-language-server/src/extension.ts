import * as vscode from 'vscode';
import { LanguageClient, TransportKind } from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
export async function activate(context: vscode.ExtensionContext) {
  const watcher = vscode.workspace.createFileSystemWatcher('**/*.{jl,toml}');
  context.subscriptions.push(watcher);
  client = new LanguageClient('juliaWorkbench', 'Julia Workbench', {
    module: context.asAbsolutePath('dist/server.js'), transport: TransportKind.stdio
  }, {
    documentSelector: [{ scheme: 'file', language: 'julia' }, { scheme: 'untitled', language: 'julia' }],
    synchronize: { fileEvents: watcher }
  });
  context.subscriptions.push(vscode.commands.registerCommand('juliaWorkbench.restart', async () => {
    await client?.restart();
  }));
  await client.start();
}
export async function deactivate() { await client?.stop(); }
