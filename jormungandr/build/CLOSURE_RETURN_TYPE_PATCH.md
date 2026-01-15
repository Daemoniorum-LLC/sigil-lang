# Closure Return Type Support - Implementation Plan

**Date:** 2026-01-14
**Goal:** Add support for `|x: T| -> R { }` closure syntax

---

## Required Changes

### 1. AST Modification (src/ast.sg:911-914)

**BEFORE:**
```sigil
    Closure {
        params: ![ClosureParam],
        body: !Box<Expr>,
    },
```

**AFTER:**
```sigil
    Closure {
        params: ![ClosureParam],
        return_type: ?TypeExpr,
        body: !Box<Expr>,
    },
```

### 2. Parser Modification (src/parser.sg:3079-3089)

**Location:** `parse_block_or_closure` function

**BEFORE:**
```sigil
let params = self.parse_closure_params()?;
// Accept either => or | as closure arrow
if !self.consume_if(&Token::FatArrow) {
    self.expect(Token::Pipe)?;
}
let body = self.parse_expr()?;
self.expect(Token::RBrace)?;
return Ok(Expr::Closure {
    params,
    body: Box::new(body),
});
```

**AFTER:**
```sigil
let params = self.parse_closure_params()?;
// Accept either => or | as closure arrow
if !self.consume_if(&Token::FatArrow) {
    self.expect(Token::Pipe)?;
}

// Parse optional return type: |x: T| -> R { }
let return_type = if self.consume_if(&Token::Arrow) {
    ?self.parse_type()?
} else {
    null
};

let body = self.parse_expr()?;
self.expect(Token::RBrace)?;
return Ok(Expr::Closure {
    params,
    return_type,
    body: Box::new(body),
});
```

---

## Additional Locations

### Other closure parsing sites (need same fix):

1. **Line 3718** - in `parse_primary_expr` for inline closures
2. **Line 3738** - another closure parsing location
3. **Line 3773** - morpheme closure context
4. **Line 3887** - another closure location

All need the return_type field added.

---

## Compilation Steps

```bash
cd /home/crook/dev2/workspace/sigil/sigil-lang/jormungandr

# 1. Backup current source
cp src/ast.sg src/ast.sg.bak
cp src/parser.sg src/parser.sg.bak

# 2. Apply patches (manual edits)
# Edit src/ast.sg line 911-914
# Edit src/parser.sg lines 3079-3089, 3718, 3738, 3773, 3887

# 3. Compile with working compiler
cd build
./sigil2 compile ../src/*.sg -o sigil3_with_closure_types.c 2>&1 | tee compile_log.txt

# 4. Fix any compile errors
sed -i '/duplicate sigil_add/d' sigil3_with_closure_types.c
sed -i '/^#endif \/\* SIGIL_BUILTINS_DEFINED \*\/$/d' sigil3_with_closure_types.c

# 5. Build binary
gcc -o sigil3_closure sigil3_with_closure_types.c -lm

# 6. Test
./sigil3_closure check /tmp/test_closure_return_type.sg
```

---

## Testing

### Test 1: Basic Return Type
```sigil
let add = |x: i32| -> i32 { x + 1 };
```

### Test 2: Evidential Return Type
```sigil
let parse = |s: &str| -> Result<i32>! { Ok(42) };
```

### Test 3: Complex Return Type
```sigil
let mapper = |x: i32| -> Vec<i32>! { vec![x, x * 2] };
```

### Test 4: secrets.sigil Pattern
```sigil
let parse_u64 = |key: &str| -> Result<u64>! {
    let pattern = format!("\"{}\":", key);
    // ... parsing logic
    Ok(42)
};
```

---

## Rollback

If something breaks:
```bash
cp src/ast.sg.bak src/ast.sg
cp src/parser.sg.bak src/parser.sg
```

---

## Success Criteria

- [ ] Test files with closure return types parse successfully
- [ ] secrets.sigil compiles without errors
- [ ] Existing code still works (no regressions)
- [ ] Can compile styx-core files

---

**This is the key enhancement needed for Styx compilation!** 🔑
