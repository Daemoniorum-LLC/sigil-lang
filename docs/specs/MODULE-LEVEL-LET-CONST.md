# Spec: Module-Level Let-Constants (≔)

## Status: Implementing

## Problem

The Sigil migration tool converts Rust `const` declarations to:
```sigil
☉ ≔ MIN_MATCH: usize = 4;
```

But the parser only accepts `const` keyword for module-level constants:
```sigil
☉ const MIN_MATCH: usize = 4;  // Works
☉ ≔ MIN_MATCH: usize = 4;      // Error: expected item, found Let
```

This inconsistency means migrated code doesn't compile.

## Solution

Extend the parser to accept `Token::Let` (≔) as an alternative syntax for module-level constant definitions, treating it identically to `Token::Const`.

## Syntax

Both forms should be equivalent:
```sigil
// Traditional (Rust-compatible)
const NAME: Type = value;
☉ const NAME: Type = value;

// Native Sigil (≔)
≔ NAME: Type = value;
☉ ≔ NAME: Type = value;
```

## Implementation

### Parser Change (parser.rs ~line 1333)

Add case for `Token::Let` in the item-parsing match:

```rust
Some(Token::Let) => {
    // Module-level ≔ is a const definition (native Sigil syntax)
    Item::Const(self.parse_let_const_with_doc_comments(visibility, doc_comments)?)
}
```

### New Parser Function

```rust
fn parse_let_const_with_doc_comments(
    &mut self,
    visibility: Visibility,
    doc_comments: Vec<DocComment>,
) -> ParseResult<ConstDef> {
    self.expect(Token::Let)?;  // consume ≔
    let name = self.parse_ident()?;
    self.expect(Token::Colon)?;
    let ty = self.parse_type()?;
    self.expect(Token::Eq)?;
    let value = self.parse_expr()?;
    self.expect_semi_or_item_start()?;

    Ok(ConstDef {
        doc_comments,
        visibility,
        name,
        ty,
        value,
    })
}
```

### Interpreter

No changes needed - already handles `Item::Const` correctly.

## Testing

```sigil
☉ ≔ X: i64 = 42;
≔ Y: i64 = X + 1;  // Private const

rite main() {
    println("X = ", X);
    println("Y = ", Y);
}
```

Expected output:
```
X = 42
Y = 43
```

## Migration Impact

This change enables all haagenti crate migrations to compile without manual edits.
