# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Sigil, please report it responsibly:

**Email:** security@daemoniorum.com

**Please include:**
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

**Do NOT:**
- Open a public issue for security vulnerabilities
- Disclose the vulnerability publicly before it's fixed

## Response Timeline

- **Acknowledgment:** Within 48 hours
- **Initial assessment:** Within 1 week
- **Fix timeline:** Depends on severity, typically 30-90 days

## Security Considerations

Sigil's evidentiality type system is designed to help track data trust at compile time. However:

- The `~` (Reported) marker indicates untrusted data but doesn't automatically sanitize it
- The `‽` (Paradox) marker is for explicit trust boundaries - use with care
- The interpreter executes arbitrary code - don't run untrusted Sigil programs
- LLVM/JIT compilation produces native code - same security model as any native binary

## Scope

This security policy covers:
- The Sigil compiler (`sigil-parser` crate)
- The official standard library
- Official tooling (LSP server, MCP server, VSCode extension)

Third-party packages and user code are outside this scope.
