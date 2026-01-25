/**
 * Sigil Playground Backend Server
 *
 * Provides execution, type-checking, and IR generation for the Sigil Playground.
 * See docs/PLAYGROUND_BACKEND_SPEC.md for the full API specification.
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { execFile } from 'child_process';
import { writeFile, unlink, mkdir } from 'fs/promises';
import { randomUUID } from 'crypto';
import { join } from 'path';
import { tmpdir } from 'os';

// Configuration
const PORT = process.env.PORT || 8080;
const SIGIL_BIN = process.env.SIGIL_BIN || 'sigil';
const MAX_CODE_SIZE = 64 * 1024; // 64 KB
const DEFAULT_TIMEOUT = 5000; // 5 seconds
const MAX_TIMEOUT = 30000; // 30 seconds
const MAX_OUTPUT_SIZE = 1024 * 1024; // 1 MB

// Initialize Express app
const app = express();

// Security middleware
app.use(helmet({
  crossOriginResourcePolicy: { policy: 'cross-origin' },
}));

// CORS configuration
app.use(cors({
  origin: [
    'http://localhost:3000',
    'http://localhost:5173',
    'https://sigil-lang.org',
    'https://playground.sigil-lang.org',
  ],
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 60, // 60 requests per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: {
      type: 'rate_limit',
      message: 'Too many requests, please try again later',
    },
  },
});
app.use(limiter);

// Body parsing with size limit
app.use(express.json({ limit: '64kb' }));

// Request ID middleware
app.use((req, res, next) => {
  req.id = randomUUID();
  res.setHeader('X-Request-Id', req.id);
  next();
});

// Temp directory for code files
const TEMP_DIR = join(tmpdir(), 'sigil-playground');
await mkdir(TEMP_DIR, { recursive: true });

/**
 * Execute Sigil code in a temp file
 */
async function executeSigil(code, command, args = [], timeoutMs = DEFAULT_TIMEOUT) {
  const tempFile = join(TEMP_DIR, `${randomUUID()}.sigil`);

  try {
    await writeFile(tempFile, code, 'utf-8');

    const startTime = process.hrtime.bigint();

    return new Promise((resolve) => {
      const proc = execFile(
        SIGIL_BIN,
        [command, ...args, tempFile],
        {
          timeout: Math.min(timeoutMs, MAX_TIMEOUT),
          maxBuffer: MAX_OUTPUT_SIZE,
          env: { ...process.env, NO_COLOR: '1' },
        },
        (error, stdout, stderr) => {
          const endTime = process.hrtime.bigint();
          const executionTimeMs = Number(endTime - startTime) / 1_000_000;

          // Clean up temp file
          unlink(tempFile).catch(() => {});

          if (error?.killed) {
            resolve({
              success: false,
              error: {
                type: 'timeout',
                message: `Execution exceeded ${timeoutMs}ms limit`,
              },
              execution_time_ms: executionTimeMs,
            });
            return;
          }

          if (error) {
            resolve({
              success: false,
              output: stdout || undefined,
              error: {
                type: error.code === 1 ? 'runtime' : 'internal',
                message: stderr || error.message,
              },
              execution_time_ms: executionTimeMs,
            });
            return;
          }

          resolve({
            success: true,
            output: stdout,
            stderr: stderr || undefined,
            execution_time_ms: executionTimeMs,
          });
        }
      );
    });
  } catch (e) {
    // Clean up on error
    unlink(tempFile).catch(() => {});
    throw e;
  }
}

/**
 * Strip ANSI escape codes from output
 */
function stripAnsi(str) {
  return str.replace(/\x1b\[[0-9;]*m/g, '');
}

/**
 * Parse type errors from sigil check output
 */
function parseTypeErrors(output) {
  const errors = [];
  const lines = output.split('\n');

  for (const line of lines) {
    // Match error pattern: [E0003] Error: message
    const errorMatch = line.match(/\[E(\d+)\]\s*Error:\s*(.+)/);
    if (errorMatch) {
      errors.push({
        code: `E${errorMatch[1]}`,
        severity: 'error',
        message: errorMatch[2],
      });
    }
  }

  return errors;
}

// =============================================================================
// API Endpoints
// =============================================================================

/**
 * GET /health - Health check
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    version: '0.1.0',
    sigil_bin: SIGIL_BIN,
    uptime_seconds: Math.floor(process.uptime()),
  });
});

/**
 * POST /run - Execute Sigil code
 */
app.post('/run', async (req, res) => {
  const startTime = Date.now();

  try {
    const { code, backend = 'interpreter', timeout_ms = DEFAULT_TIMEOUT } = req.body;

    // Validate request
    if (!code || typeof code !== 'string') {
      return res.status(400).json({
        success: false,
        error: {
          type: 'invalid_request',
          message: 'Missing or invalid "code" field',
        },
      });
    }

    if (code.length > MAX_CODE_SIZE) {
      return res.status(400).json({
        success: false,
        error: {
          type: 'invalid_request',
          message: `Code exceeds maximum size of ${MAX_CODE_SIZE} bytes`,
        },
      });
    }

    // Map backend to command
    const command = backend === 'jit' ? 'jit' : backend === 'llvm' ? 'llvm' : 'run';

    const result = await executeSigil(code, command, [], timeout_ms);

    res.setHeader('X-Execution-Time-Ms', result.execution_time_ms.toFixed(2));
    res.json(result);

  } catch (e) {
    console.error(`[${req.id}] Run error:`, e);
    res.status(500).json({
      success: false,
      error: {
        type: 'internal',
        message: 'Internal server error',
      },
    });
  }
});

/**
 * POST /check - Type-check Sigil code
 */
app.post('/check', async (req, res) => {
  try {
    const { code, format = 'simple' } = req.body;

    // Validate request
    if (!code || typeof code !== 'string') {
      return res.status(400).json({
        success: false,
        error: {
          type: 'invalid_request',
          message: 'Missing or invalid "code" field',
        },
      });
    }

    if (code.length > MAX_CODE_SIZE) {
      return res.status(400).json({
        success: false,
        error: {
          type: 'invalid_request',
          message: `Code exceeds maximum size of ${MAX_CODE_SIZE} bytes`,
        },
      });
    }

    const result = await executeSigil(code, 'check', []);

    if (result.success) {
      res.json({
        success: true,
        errors: [],
        warnings: [],
        execution_time_ms: result.execution_time_ms,
      });
    } else {
      const errors = parseTypeErrors(stripAnsi(result.error?.message || ''));
      res.json({
        success: false,
        errors: errors.length > 0 ? errors : [{
          code: 'E0000',
          severity: 'error',
          message: stripAnsi(result.error?.message || 'Unknown error'),
        }],
        warnings: [],
        execution_time_ms: result.execution_time_ms,
      });
    }

  } catch (e) {
    console.error(`[${req.id}] Check error:`, e);
    res.status(500).json({
      success: false,
      error: {
        type: 'internal',
        message: 'Internal server error',
      },
    });
  }
});

/**
 * POST /ir - Generate AI IR
 */
app.post('/ir', async (req, res) => {
  try {
    const { code, format = 'json' } = req.body;

    // Validate request
    if (!code || typeof code !== 'string') {
      return res.status(400).json({
        success: false,
        error: {
          type: 'invalid_request',
          message: 'Missing or invalid "code" field',
        },
      });
    }

    if (code.length > MAX_CODE_SIZE) {
      return res.status(400).json({
        success: false,
        error: {
          type: 'invalid_request',
          message: `Code exceeds maximum size of ${MAX_CODE_SIZE} bytes`,
        },
      });
    }

    const args = format === 'compact' ? ['--compact'] : [];
    const result = await executeSigil(code, 'dump-ir', args);

    if (result.success) {
      try {
        const ir = JSON.parse(result.output);
        res.json({
          success: true,
          ir: ir,
          execution_time_ms: result.execution_time_ms,
        });
      } catch (e) {
        res.json({
          success: true,
          ir: result.output,
          execution_time_ms: result.execution_time_ms,
        });
      }
    } else {
      res.json({
        success: false,
        error: {
          type: 'parse_error',
          message: stripAnsi(result.error?.message || 'Failed to generate IR'),
        },
        execution_time_ms: result.execution_time_ms,
      });
    }

  } catch (e) {
    console.error(`[${req.id}] IR error:`, e);
    res.status(500).json({
      success: false,
      error: {
        type: 'internal',
        message: 'Internal server error',
      },
    });
  }
});

/**
 * POST /format - Format Sigil code (placeholder)
 */
app.post('/format', async (req, res) => {
  const { code } = req.body;

  // Format not yet implemented in Sigil CLI
  res.json({
    success: false,
    error: {
      type: 'not_implemented',
      message: 'Code formatting is not yet implemented',
    },
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: {
      type: 'not_found',
      message: `Endpoint ${req.method} ${req.path} not found`,
    },
  });
});

// Error handler
app.use((err, req, res, next) => {
  console.error(`[${req.id}] Unhandled error:`, err);
  res.status(500).json({
    success: false,
    error: {
      type: 'internal',
      message: 'Internal server error',
    },
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Sigil Playground Backend running on port ${PORT}`);
  console.log(`Using Sigil binary: ${SIGIL_BIN}`);
  console.log(`Temp directory: ${TEMP_DIR}`);
});
