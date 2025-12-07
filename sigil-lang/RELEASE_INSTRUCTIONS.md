# Creating the v0.1.0 Release

After merging the feature branch to main, run these commands:

## 1. Create and Push Tag

```bash
# Make sure you're on main with latest changes
git checkout main
git pull origin main

# Create annotated tag
git tag -a v0.1.0 -m "Sigil v0.1.0 - Initial Release

First public release of the Sigil programming language.

Features:
- Evidentiality type system (!, ?, ~, ‽ markers)
- Morpheme operators (τ, φ, σ, ρ, Σ, Π, α, ω)
- Cranelift JIT compilation
- LLVM native compilation (optional)
- Interactive REPL
- AI IR output for LLM integration
- Comprehensive stdlib (400+ functions)

Installation:
- cargo install sigil-parser
- brew tap daemoniorum/sigil https://github.com/Daemoniorum-LLC/sigil-lang && brew install sigil
- npm install -g @daemoniorum/sigil-mcp (MCP server)
"

# Push the tag to trigger release workflow
git push origin v0.1.0
```

## 2. What Happens Automatically

The GitHub Actions release workflow will:

1. Build binaries for:
   - Linux x86_64 (`sigil-linux-x86_64.tar.gz`)
   - Linux aarch64 (`sigil-linux-aarch64.tar.gz`)
   - macOS x86_64 (`sigil-macos-x86_64.tar.gz`)
   - macOS aarch64 (`sigil-macos-aarch64.tar.gz`)
   - Windows x86_64 (`sigil-windows-x86_64.zip`)

2. Create SHA256 checksums

3. Publish GitHub Release with:
   - Auto-generated release notes
   - All binary artifacts
   - Checksums file

4. Update Homebrew formula with new version

## 3. After Release

Verify the release:
- Check https://github.com/Daemoniorum-LLC/sigil-lang/releases
- Test installation: `cargo install sigil-parser`
- Test Homebrew: `brew upgrade sigil`

## Delete This File

This file can be deleted after the release is created:
```bash
rm RELEASE_INSTRUCTIONS.md
git add -A && git commit -m "chore: Remove release instructions"
git push
```
