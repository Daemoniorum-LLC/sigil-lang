/**
 * Sigil Language Extension for VS Code
 *
 * Provides language support through the Oracle LSP server:
 * - Real-time diagnostics
 * - Hover information
 * - Code completion
 * - Go-to-definition
 *
 * Designed for both human developers and AI agents.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

// Morpheme quick-pick items
const MORPHEMES = [
    { label: 'τ', description: 'Transform/Map', detail: 'Applies a transformation to each element' },
    { label: 'φ', description: 'Filter', detail: 'Selects elements matching a predicate' },
    { label: 'σ', description: 'Sort', detail: 'Orders elements' },
    { label: 'ρ', description: 'Reduce', detail: 'Folds elements to a single value' },
    { label: 'λ', description: 'Lambda', detail: 'Anonymous function' },
    { label: 'Σ', description: 'Sum', detail: 'Sum all elements' },
    { label: 'Π', description: 'Product', detail: 'Multiply all elements' },
    { label: '·', description: 'Incorporation', detail: 'Noun-verb fusion' },
    { label: '⌛', description: 'Await', detail: 'Await async operation' },
    { label: '!', description: 'Known', detail: 'Evidentiality: directly computed' },
    { label: '?', description: 'Uncertain', detail: 'Evidentiality: may be absent' },
    { label: '~', description: 'Reported', detail: 'Evidentiality: external source' },
    { label: '‽', description: 'Paradox', detail: 'Evidentiality: trust boundary' },
];

export async function activate(context: vscode.ExtensionContext) {
    console.log('Sigil extension activating...');

    // Check if LSP is enabled
    const config = vscode.workspace.getConfiguration('sigil');
    const lspEnabled = config.get<boolean>('oracle.enabled', true);

    if (lspEnabled) {
        await startLanguageServer(context);
    }

    // Register the restart command
    context.subscriptions.push(
        vscode.commands.registerCommand('sigil.restartServer', async () => {
            if (client) {
                await client.stop();
            }
            await startLanguageServer(context);
            vscode.window.showInformationMessage('Sigil Oracle restarted');
        })
    );

    // Register the insert morpheme command
    context.subscriptions.push(
        vscode.commands.registerCommand('sigil.insertMorpheme', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                return;
            }

            const selection = await vscode.window.showQuickPick(MORPHEMES, {
                placeHolder: 'Select a morpheme to insert',
                matchOnDescription: true,
                matchOnDetail: true,
            });

            if (selection) {
                editor.edit((editBuilder) => {
                    editBuilder.insert(editor.selection.active, selection.label);
                });
            }
        })
    );

    // Watch for configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(async (e) => {
            if (e.affectsConfiguration('sigil.oracle.enabled')) {
                const enabled = config.get<boolean>('oracle.enabled', true);
                if (enabled && !client) {
                    await startLanguageServer(context);
                } else if (!enabled && client) {
                    await client.stop();
                    client = undefined;
                }
            }
        })
    );

    console.log('Sigil extension activated');
}

async function startLanguageServer(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('sigil');
    const serverPath = config.get<string>('oracle.path', 'sigil-oracle');
    const trace = config.get<string>('trace.server', 'off');

    // Server options - run the Oracle LSP
    const serverOptions: ServerOptions = {
        run: {
            command: serverPath,
            transport: TransportKind.stdio,
        },
        debug: {
            command: serverPath,
            transport: TransportKind.stdio,
            options: {
                env: {
                    ...process.env,
                    RUST_LOG: 'sigil_oracle=debug',
                },
            },
        },
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'sigil' },
            { scheme: 'untitled', language: 'sigil' },
        ],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.{sigil,sg}'),
        },
        outputChannelName: 'Sigil Oracle',
        traceOutputChannel: vscode.window.createOutputChannel('Sigil Oracle Trace'),
    };

    // Create and start the client
    client = new LanguageClient(
        'sigil-oracle',
        'Sigil Oracle',
        serverOptions,
        clientOptions
    );

    try {
        await client.start();
        console.log('Sigil Oracle LSP client started');
    } catch (error) {
        console.error('Failed to start Sigil Oracle:', error);
        vscode.window.showWarningMessage(
            `Failed to start Sigil Oracle language server. Make sure 'sigil-oracle' is installed and in your PATH. Error: ${error}`
        );
    }

    context.subscriptions.push({
        dispose: () => {
            if (client) {
                client.stop();
            }
        },
    });
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
