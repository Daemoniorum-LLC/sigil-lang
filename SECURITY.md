# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting Vulnerabilities

**Email:** security@daemoniorum.com

Tell us:
- What the vulnerability is
- How to reproduce it
- What the impact could be

**Don't** open a public issue for security problems.

## Response

- We'll acknowledge within 48 hours
- We'll assess within a week
- Fix timeline depends on severity

## What Sigil's Type System Does (and Doesn't) Do

Sigil's evidentiality markers help you *track* data trust:

- `~` (Reported) marks data as untrusted - but **doesn't sanitize it**
- `‽` (Paradox) marks trust boundaries - **you** decide what crosses them
- The compiler catches when you use `~` data where `!` is expected

This is a *compile-time* tool. It prevents you from accidentally trusting external data, but it doesn't make your code secure by magic.

## Runtime Security

- The interpreter executes code directly - don't run untrusted `.sg` files
- JIT/AOT compilation produces native binaries - standard binary security applies
- The REPL has no sandbox

## Scope

This policy covers:
- `sigil-parser` (the compiler)
- The standard library
- Official tools (LSP, MCP server, VSCode extension)

Third-party code is your problem.
