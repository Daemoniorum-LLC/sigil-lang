#!/usr/bin/env node

/**
 * Simple test script for the Sigil MCP server
 * Tests each tool by invoking them directly
 */

import { spawn } from 'child_process';
import { createInterface } from 'readline';

const SIGIL_BIN = process.env.SIGIL_BIN || '/home/user/sigil-lang/parser/target/release/sigil';

async function testRun() {
  console.log('\n=== Testing sigil_run ===');
  const code = `fn main() {
    let nums = [1, 2, 3, 4, 5];
    let sum = nums |ρ{0, acc, x => acc + x};
    print(sum);
}`;

  const proc = spawn(SIGIL_BIN, ['run', '-'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env }
  });

  proc.stdin.write(code);
  proc.stdin.end();

  let stdout = '';
  let stderr = '';

  proc.stdout.on('data', (data) => { stdout += data; });
  proc.stderr.on('data', (data) => { stderr += data; });

  return new Promise((resolve) => {
    proc.on('close', (code) => {
      console.log('Exit code:', code);
      console.log('Output:', stdout || '(empty)');
      if (stderr) console.log('Stderr:', stderr);
      resolve(code === 0);
    });
  });
}

async function testCheck() {
  console.log('\n=== Testing sigil_check ===');

  // Test valid code
  const validCode = `fn main() {
    print("Hello, Sigil!");
}`;

  const proc = spawn(SIGIL_BIN, ['check', '-'], {
    stdio: ['pipe', 'pipe', 'pipe']
  });

  proc.stdin.write(validCode);
  proc.stdin.end();

  let stdout = '';
  let stderr = '';

  proc.stdout.on('data', (data) => { stdout += data; });
  proc.stderr.on('data', (data) => { stderr += data; });

  return new Promise((resolve) => {
    proc.on('close', (code) => {
      console.log('Valid code check - Exit code:', code);
      console.log('Output:', (stdout + stderr).replace(/\x1b\[[0-9;]*m/g, '') || '(empty)');
      resolve(code === 0);
    });
  });
}

async function testIR() {
  console.log('\n=== Testing sigil_ir ===');

  const code = `fn add(a: int, b: int) -> int {
    a + b
}`;

  const proc = spawn(SIGIL_BIN, ['ir', '-'], {
    stdio: ['pipe', 'pipe', 'pipe']
  });

  proc.stdin.write(code);
  proc.stdin.end();

  let stdout = '';
  let stderr = '';

  proc.stdout.on('data', (data) => { stdout += data; });
  proc.stderr.on('data', (data) => { stderr += data; });

  return new Promise((resolve) => {
    proc.on('close', (code) => {
      console.log('IR generation - Exit code:', code);
      if (stdout) {
        try {
          const ir = JSON.parse(stdout);
          console.log('IR parsed successfully, functions:', ir.functions?.length || 0);
        } catch (e) {
          console.log('IR output (first 200 chars):', stdout.slice(0, 200));
        }
      }
      if (stderr) console.log('Stderr:', stderr);
      resolve(code === 0);
    });
  });
}

async function main() {
  console.log('Sigil MCP Server - Tool Tests');
  console.log('Using binary:', SIGIL_BIN);

  const results = [];

  results.push(await testCheck());
  results.push(await testRun());
  results.push(await testIR());

  console.log('\n=== Results ===');
  console.log('check:', results[0] ? 'PASS' : 'FAIL');
  console.log('run:', results[1] ? 'PASS' : 'FAIL');
  console.log('ir:', results[2] ? 'PASS' : 'FAIL');

  const allPassed = results.every(r => r);
  console.log('\nOverall:', allPassed ? 'ALL TESTS PASSED' : 'SOME TESTS FAILED');
  process.exit(allPassed ? 0 : 1);
}

main();
