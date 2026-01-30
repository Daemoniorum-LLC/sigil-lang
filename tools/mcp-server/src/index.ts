#!/usr/bin/env node

/**
 * Sigil MCP Server
 *
 * Enables AI systems to write, run, type-check, and analyze Sigil code
 * through the Model Context Protocol.
 *
 * Tools:
 * - sigil_run: Execute Sigil code and return output
 * - sigil_check: Type-check code with evidentiality enforcement
 * - sigil_ir: Get AI-readable intermediate representation
 * - sigil_explain: Describe what Sigil code does
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { exec } from "child_process";
import { promisify } from "util";
import { writeFile, unlink, mkdtemp } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

const execAsync = promisify(exec);

// Path to the Sigil binary - configurable via environment variable
const SIGIL_BIN = process.env.SIGIL_BIN || "sigil";

/**
 * Execute a Sigil CLI command
 */
async function runSigilCommand(
  args: string[],
  input?: string
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  let tempDir: string | null = null;
  let tempFile: string | null = null;

  try {
    // If we have input code, write it to a temp file
    if (input !== undefined) {
      tempDir = await mkdtemp(join(tmpdir(), "sigil-"));
      tempFile = join(tempDir, "input.sigil");
      await writeFile(tempFile, input, "utf-8");
      args = args.map((arg) => (arg === "__INPUT__" ? tempFile! : arg));
    }

    const command = `${SIGIL_BIN} ${args.join(" ")}`;
    const { stdout, stderr } = await execAsync(command, {
      timeout: 30000, // 30 second timeout
      maxBuffer: 10 * 1024 * 1024, // 10MB buffer
    });

    return { stdout, stderr, exitCode: 0 };
  } catch (error: any) {
    return {
      stdout: error.stdout || "",
      stderr: error.stderr || error.message,
      exitCode: error.code || 1,
    };
  } finally {
    // Cleanup temp files
    if (tempFile) {
      try {
        await unlink(tempFile);
      } catch {}
    }
    if (tempDir) {
      try {
        await unlink(tempDir);
      } catch {}
    }
  }
}

/**
 * Strip ANSI color codes from output
 */
function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, "");
}

/**
 * Tool definitions
 */
const TOOLS: Tool[] = [
  {
    name: "sigil_run",
    description: `Execute Sigil code and return the output.

Sigil is a polysynthetic programming language designed for AI systems, featuring:
- Evidentiality types: ! (known), ? (uncertain), ~ (reported), ‽ (paradox)
- Morpheme operators: τ (transform/map), φ (filter), σ (sort), ρ (reduce)
- Pipeline syntax for data flow

Example:
\`\`\`sigil
fn main() {
    let nums = [1, 2, 3, 4, 5];
    let doubled = nums |τ{_ * 2};
    print(doubled);
}
\`\`\``,
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "The Sigil source code to execute",
        },
        backend: {
          type: "string",
          enum: ["interpreter", "jit", "llvm"],
          description: "Execution backend: interpreter (default), jit (fast), or llvm (fastest)",
        },
      },
      required: ["code"],
    },
  },
  {
    name: "sigil_check",
    description: `Type-check Sigil code with evidentiality enforcement.

Returns type errors including evidence mismatches. The evidentiality system ensures:
- Known (!) data can satisfy any requirement
- Uncertain (?) data cannot satisfy known requirements
- Reported (~) data requires validation before use as uncertain/known

This is crucial for AI safety - it tracks data provenance at the type level.`,
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "The Sigil source code to type-check",
        },
      },
      required: ["code"],
    },
  },
  {
    name: "sigil_ir",
    description: `Get the AI-readable intermediate representation (IR) of Sigil code.

Returns a JSON structure that represents the code in a format optimized for AI analysis.
This allows AI systems to reason about code structure without parsing source text.

The IR includes:
- Function definitions with types
- Expression trees
- Control flow structure
- Evidence annotations`,
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "The Sigil source code to analyze",
        },
        format: {
          type: "string",
          enum: ["json", "pretty"],
          description: "Output format (default: json)",
        },
      },
      required: ["code"],
    },
  },
  {
    name: "sigil_explain",
    description: `Explain what a piece of Sigil code does in natural language.

Useful for understanding unfamiliar Sigil code or for generating documentation.
Analyzes the code structure and describes its behavior, including:
- What the code computes
- How data flows through pipelines
- What evidence levels are used and why`,
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "The Sigil source code to explain",
        },
      },
      required: ["code"],
    },
  },
];

/**
 * Handle tool execution
 */
async function handleToolCall(
  name: string,
  args: Record<string, unknown>
): Promise<string> {
  switch (name) {
    case "sigil_run": {
      const code = args.code as string;
      const backend = (args.backend as string) || "interpreter";

      // "jit" and "llvm" are separate commands, not flags
      const command = backend === "jit" ? "jit" : backend === "llvm" ? "llvm" : "run";
      const result = await runSigilCommand(
        [command, "__INPUT__"],
        code
      );

      if (result.exitCode !== 0) {
        return `Execution failed:\n${stripAnsi(result.stderr || result.stdout)}`;
      }
      return result.stdout || "(no output)";
    }

    case "sigil_check": {
      const code = args.code as string;
      const result = await runSigilCommand(["check", "__INPUT__"], code);

      const output = stripAnsi(result.stdout + result.stderr);
      if (result.exitCode === 0) {
        return "✓ Type check passed - no errors";
      }
      return `Type errors found:\n${output}`;
    }

    case "sigil_ir": {
      const code = args.code as string;
      const format = (args.format as string) || "json";

      const result = await runSigilCommand(["dump-ir", "__INPUT__"], code);

      if (result.exitCode !== 0) {
        return `IR generation failed:\n${stripAnsi(result.stderr)}`;
      }

      if (format === "pretty") {
        try {
          const ir = JSON.parse(result.stdout);
          return JSON.stringify(ir, null, 2);
        } catch {
          return result.stdout;
        }
      }
      return result.stdout;
    }

    case "sigil_explain": {
      const code = args.code as string;

      // First, get the IR to understand the structure
      const irResult = await runSigilCommand(["dump-ir", "__INPUT__"], code);

      if (irResult.exitCode !== 0) {
        // If IR fails, try to explain based on parsing errors
        return `Unable to analyze code:\n${stripAnsi(irResult.stderr)}`;
      }

      try {
        const ir = JSON.parse(irResult.stdout);
        return generateExplanation(code, ir);
      } catch {
        return "Unable to parse IR for explanation";
      }
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

/**
 * Generate a natural language explanation of Sigil code
 */
function generateExplanation(code: string, ir: any): string {
  const lines: string[] = [];

  lines.push("## Code Analysis\n");

  // Check for evidentiality markers in the code
  const hasKnown = code.includes("!");
  const hasUncertain = code.includes("?");
  const hasReported = code.includes("~");

  if (hasKnown || hasUncertain || hasReported) {
    lines.push("### Evidentiality");
    if (hasKnown) lines.push("- Uses **known (!)** evidence - data that is certain");
    if (hasUncertain)
      lines.push("- Uses **uncertain (?)** evidence - data that may vary");
    if (hasReported)
      lines.push("- Uses **reported (~)** evidence - external data requiring validation");
    lines.push("");
  }

  // Check for morpheme operators
  const morphemes = [];
  if (code.includes("|τ") || code.includes("|τ")) morphemes.push("τ (transform/map)");
  if (code.includes("|φ")) morphemes.push("φ (filter)");
  if (code.includes("|σ")) morphemes.push("σ (sort)");
  if (code.includes("|ρ")) morphemes.push("ρ (reduce)");
  if (code.includes("|α")) morphemes.push("α (first)");
  if (code.includes("|ω")) morphemes.push("ω (last)");

  if (morphemes.length > 0) {
    lines.push("### Data Flow Operations");
    lines.push(`Uses morpheme operators: ${morphemes.join(", ")}`);
    lines.push("");
  }

  // Analyze functions from IR
  if (ir.functions && ir.functions.length > 0) {
    lines.push("### Functions");
    for (const func of ir.functions) {
      const params =
        func.params?.map((p: any) => `${p.name}: ${p.type || "any"}`).join(", ") ||
        "";
      const returnType = func.return_type || "unit";
      lines.push(`- \`${func.name}(${params}) -> ${returnType}\``);
    }
    lines.push("");
  }

  // Summary
  lines.push("### Summary");
  lines.push(
    "This Sigil code " +
      (morphemes.length > 0
        ? "processes data through pipelines using morpheme operators. "
        : "") +
      (hasReported
        ? "Handles external data with proper evidence tracking. "
        : "") +
      (hasKnown ? "Produces results with known certainty. " : "")
  );

  return lines.join("\n");
}

/**
 * Main server setup
 */
async function main() {
  const server = new Server(
    {
      name: "sigil-mcp",
      version: "0.1.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List available tools
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return { tools: TOOLS };
  });

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      const result = await handleToolCall(name, args || {});
      return {
        content: [{ type: "text", text: result }],
      };
    } catch (error: any) {
      return {
        content: [{ type: "text", text: `Error: ${error.message}` }],
        isError: true,
      };
    }
  });

  // Start the server
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("Sigil MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
