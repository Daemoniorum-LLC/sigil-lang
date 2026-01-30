# Homebrew Tap for Sigil

Official Homebrew tap for installing the Sigil programming language.

## Installation

### Quick Install (from crates.io)

```bash
# If you have Rust installed
cargo install sigil-parser
```

### Via Homebrew Tap

First, tap the repository:

```bash
brew tap daemoniorum/sigil https://github.com/Daemoniorum-LLC/sigil-lang
```

Then install:

```bash
brew install sigil
```

### Install from HEAD (latest development)

```bash
brew install --HEAD sigil
```

## Updating

```bash
brew upgrade sigil
```

## Uninstalling

```bash
brew uninstall sigil
brew untap daemoniorum/sigil
```

## What's Installed

- `sigil` - The Sigil compiler and interpreter

## Usage

```bash
# Run a Sigil program
sigil run hello.sigil

# Type check a program
sigil check hello.sigil

# Interactive REPL
sigil repl

# Generate AI IR
sigil dump-ir hello.sigil

# JIT compile and run
sigil jit hello.sigil
```

## Example Program

```sigil
fn main() {
    // Evidence markers track data provenance
    let sensor_data~ = read_sensor();     // External (reported)
    let validated? = validate(sensor_data); // Uncertain
    let result! = compute(validated);      // Known (verified)

    print("Result: " + str(result));
}
```

## Documentation

- [Language Guide](https://github.com/Daemoniorum-LLC/sigil-lang/blob/main/docs/AI_TUTORIAL.md)
- [Evidence System](https://github.com/Daemoniorum-LLC/sigil-lang/blob/main/docs/EVIDENCE.md)
- [Full Documentation](https://github.com/Daemoniorum-LLC/sigil-lang/tree/main/docs)

## License

Daemoniorum Source-Available License - See [LICENSE](https://github.com/Daemoniorum-LLC/sigil-lang/blob/main/LICENSE)
