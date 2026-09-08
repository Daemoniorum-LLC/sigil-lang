#!/usr/bin/env python3
"""Rewrite legacy (Rust-shaped) Sigil into the current native syntax.

Sigil's keywords changed — `fn` became `rite`, `let` became `≔`, `::` became `·` —
and a large part of this repository never followed. The files this fixes do not
parse at all; they fail on their first token.

The one rule that matters: NEVER rewrite inside a string literal or a comment.
An earlier regex-based rewriter in this repo corrupted C templates, error
messages and identifier interiors precisely because it did not track them. This
walks the source character by character and only substitutes in code regions.

Usage:  migrate_legacy_syntax.py [--write] FILE...
        Without --write it reports what would change and touches nothing.
"""
import re
import sys

# Word-for-word replacements, applied only in code regions. Order matters: the
# `pub`-prefixed forms must be tried before the bare ones.
WORD_SUBS = [
    (r"\bpub\s+fn\b",     "☉ rite"),
    (r"\bpub\s+struct\b", "☉ Σ"),
    (r"\bpub\s+enum\b",   "☉ ᛈ"),
    (r"\bpub\s+trait\b",  "☉ aspect"),
    (r"\bpub\s+mod\b",    "☉ scroll"),
    (r"\bpub\s+const\b",  "☉ const"),
    (r"\bpub\s+use\b",    "☉ invoke"),
    (r"\bfn\b",           "rite"),
    (r"\bstruct\b",       "Σ"),
    (r"\benum\b",         "ᛈ"),
    (r"\btrait\b",        "aspect"),
    (r"\bmod\b",          "scroll"),
    (r"\buse\b",          "invoke"),
    (r"\blet\s+mut\b",    "≔ vary"),
    (r"\blet\b",          "≔"),
    (r"\bmatch\b",        "⌥"),
    (r"\breturn\b",       "⤺"),
    (r"\bif\b",           "⎇"),
    (r"\belse\b",         "⎉"),
    (r"\bwhile\b",        "⟳"),
    (r"&mut\b",           "&Δ"),
    (r"\bcrate·",         "tome·"),
    # Anything else `pub` qualifies — `pub type`, `pub const`, `pub static`, and
    # struct fields such as `pub children`. Runs last so the specific forms above
    # win first.
    (r"\bpub\s+",          "☉ "),
]

# `λ name(` is the older spelling of a function declaration, from before `rite`.
# A real lambda is `|x| …`, so requiring an identifier and an open paren after it
# distinguishes the two.
LAMBDA_FN = re.compile(r"λ\s+([A-Za-z_]\w*)\s*\(")

# `impl Foo {` and `impl Trait for Foo {` — the second form binds the trait.
# `impl` may carry its own generics — `impl<T> Vec<T>`, `impl[F: FnOnce() + 'static]
# Closure[F]` — which the first version did not match, so those items kept the word
# `impl` and never parsed. The generic list is preserved on the ⊢.
IMPL_GENERICS = r"(?:<[^>]*>|\[[^\]]*\])?"
IMPL_FOR = re.compile(
    r"\bimpl(" + IMPL_GENERICS + r")\s+([A-Za-z_][\w·:<>\[\], &']*?)\s+for\s+"
    r"([A-Za-z_][\w·:<>\[\], &']*?)\s*\{"
)
IMPL_PLAIN = re.compile(
    r"\bimpl(" + IMPL_GENERICS + r")\s+([A-Za-z_][\w·:<>\[\], &']*?)\s*\{"
)
# `impl` already rewritten to ⊢ by an earlier pass, but `for` left behind — Sigil
# spells `impl Trait for Type` as `⊢ Trait ∀ Type`.
IMPL_FOR_LEFTOVER = re.compile(
    r"⊢(" + IMPL_GENERICS + r")\s+([A-Za-z_][\w·:<>\[\], &']*?)\s+for\s+"
    r"([A-Za-z_][\w·:<>\[\], &']*?)\s*\{"
)
# `for x in xs {` — must run before the bare `in` of other constructs.
# `for x in xs` and `for (a, b) in xs` — the tuple pattern form is common in
# iteration over maps and has to be matched too.
FOR_IN = re.compile(r"\bfor\s+(\([^)]*\)|\w+)\s+in\s+")


def _is_char_literal(src, i):
    """True when src[i] opens a char literal rather than a lifetime tick."""
    rest = src[i + 1 : i + 6]
    if rest.startswith("\\"):
        return "'" in rest[1:4]          # '\n', '\t', '\\', '\''
    return len(rest) >= 2 and rest[1] == "'"  # 'x'


def split_regions(src):
    """Yield (text, is_code) spans, so substitutions never touch strings or comments."""
    spans, buf, i, n = [], [], 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "'" and not _is_char_literal(src, i):
            # A lifetime tick, not a quote. `'static` and `'a` are not string
            # literals, and treating them as one made the walker stop transforming
            # at the first lifetime and silently resume only at the next
            # apostrophe — so `☉ rite f[F: Fn() + 'static](…)` migrated its
            # signature and left the whole body untouched. It failed quietly,
            # which is the worst way for a rewriter to fail.
            buf.append(c); i += 1
        elif c == '"' or c == "'":
            spans.append(("".join(buf), True)); buf = []
            quote, j = c, i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2; continue
                if src[j] == quote:
                    j += 1; break
                j += 1
            spans.append((src[i:j], False)); i = j
        elif c == "/" and nxt == "/":
            spans.append(("".join(buf), True)); buf = []
            j = src.find("\n", i)
            j = n if j == -1 else j
            spans.append((src[i:j], False)); i = j
        elif c == "/" and nxt == "*":
            spans.append(("".join(buf), True)); buf = []
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            spans.append((src[i:j], False)); i = j
        else:
            buf.append(c); i += 1
    spans.append(("".join(buf), True))
    return spans


def migrate_code(code):
    code = IMPL_FOR_LEFTOVER.sub(lambda m: f"⊢{m.group(1)} {m.group(2)} ∀ {m.group(3)} {{", code)
    code = IMPL_FOR.sub(lambda m: f"⊢{m.group(1)} {m.group(2)} ∀ {m.group(3)} {{", code)
    code = IMPL_PLAIN.sub(lambda m: f"⊢{m.group(1)} {m.group(2)} {{", code)
    code = FOR_IN.sub(lambda m: f"∀ {m.group(1)} ∈ ", code)
    code = LAMBDA_FN.sub(lambda m: f"rite {m.group(1)}(", code)
    for pat, rep in WORD_SUBS:
        code = re.sub(pat, rep, code)
    code = code.replace("::", "·")
    return code


def migrate(src):
    return "".join(t if not is_code else migrate_code(t) for t, is_code in split_regions(src))


def main(argv):
    write = "--write" in argv
    files = [a for a in argv if not a.startswith("--")]
    changed = 0
    for path in files:
        with open(path, encoding="utf-8") as fh:
            before = fh.read()
        after = migrate(before)
        if after == before:
            continue
        changed += 1
        if write:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(after)
        print(("rewrote " if write else "would rewrite ") + path)
    print(f"{changed} file(s) {'changed' if write else 'would change'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


# ---------------------------------------------------------------------------
# Rust standard-library paths
# ---------------------------------------------------------------------------
# `std·collections·HashMap·new()` PARSES — it is just a path — and then dies at
# run time with "undefined variable: `std`". Because `sigil check` does not
# resolve names (S23), files carrying these check clean and crash on first use.
#
# Sigil spells these unqualified. Only names verified to resolve as a path root
# are rewritten; anything else is left exactly as it was, because swapping one
# undefined name for another is not a fix.
STD_LEAVES_THAT_RESOLVE = {
    "Add", "Any", "Arc", "AtomicBool", "AtomicU8", "AtomicU64", "AtomicUsize",
    "BTreeMap", "Debug", "Display", "Duration", "Formatter", "FromStr", "Future",
    "Hash", "HashMap", "HashSet", "Hasher", "Instant", "Ordering", "PhantomData",
    "Rc", "Read", "RefCell", "Result", "Start", "String", "TcpListener",
    "TcpStream", "UnexpectedEof", "VecDeque", "args", "max", "sleep", "take",
}

STD_PATH = re.compile(r"\bstd·(?:[a-z_]+·)+([A-Za-z_]\w*)")

# `invoke std·collections·HashMap;` imports a builtin that needs no import.
#
# Anchored to a SINGLE line, deliberately. A first version used `[^;]*;`, and
# because many of these invokes carry no semicolon at all, it ran past the end of
# its line and swallowed the next declaration whole — it ate an entire
# `☉ Θ StorageBackend` trait out of engram/src/storage/mod.sg. `[^\n;]*` cannot
# cross a newline.
STD_INVOKE = re.compile(r"^[ \t]*(?:☉ )?invoke\s+std·[^\n;]*;?[ \t]*\n", re.M)


def migrate_std_paths(src):
    """Rewrite resolvable std· paths to their Sigil spelling. String/comment safe."""
    def one(text):
        return STD_PATH.sub(
            lambda m: m.group(1) if m.group(1) in STD_LEAVES_THAT_RESOLVE else m.group(0),
            text,
        )
    out = "".join(one(t) if is_code else t for t, is_code in split_regions(src))
    return STD_INVOKE.sub("", out)


# ---------------------------------------------------------------------------
# Constructs that have a Sigil spelling, just not the Rust one
# ---------------------------------------------------------------------------
# Established by probing the compiler, not by assumption:
#
#   in("rdi") x       ->  ∈("rdi") x     asm input operands use ∈, like the rest
#                                        of Sigil's vocabulary (the parser matches
#                                        Token::ElementOf here)
#   vary x: T;        ->  ≔ vary x: T;   `vary` is a modifier on the ≔ binder, not
#                                        a statement starter of its own
#   c: impl Fn()      ->  c: ⊢ Fn()      `impl Trait` in argument position is
#                                        spelled with the same impl sigil
#
# Generic bounds needed no change at all: [T: Clone], [F: Fn() + 'static] and
# [T: Into<String>] already parse.

ASM_IN = re.compile(r"(?<![\w·])in\(")
# Rust's `lateout` says "written only at the end"; Sigil's asm has out, ∈, inout,
# clobber and options, and `out` is the operand it means.
ASM_LATEOUT = re.compile(r"(?<![\w·])lateout\(")

# An evidence marker on an ENUM VARIANT DECLARATION — `Tcp(TcpStream)!,`,
# `Unknown(i32)!,`, `A!,`. Evidence belongs to a value, not to the shape of a
# variant, and Sigil's parser rejects it. Dropping the marker is the only reading
# that keeps the declaration.
VARIANT_MARKER = re.compile(r"^(\s*[A-Z]\w*(?:\([^)]*\))?)[!?~◊](\s*,)", re.M)

# `aspect` is Sigil's `trait` keyword, so it cannot also be an identifier. Sources
# predating that keyword use it as a field name, a parameter and a local.
# Inline object types in a parameter — `props: { code: String, language: String }`.
# Sigil has no anonymous struct type, so this degrades to Any. Type information is
# lost, which is worse than a named struct would be; a named struct would also mean
# rewriting every call site, and these are hand-written sources with callers
# elsewhere. The React generator's normalize_prop_type makes the same trade.
INLINE_OBJ_PARAM = re.compile(r"(:\s*)\{[^{}]*:[^{}]*\}(?=\s*[,)])")

ASPECT_IDENT = re.compile(
    r"(?<![\w·])aspect(?=\s*[:,)=]|\s*$)|(?<=[(,]\s)aspect(?=\s*:)", re.M
)
BARE_VARY = re.compile(r"^([ \t]*)vary\s+(\w+\s*:)", re.M)
IMPL_ARG = re.compile(r"(?<![\w·])impl\s+(?=[A-Z])")


def migrate_constructs(src, in_asm_only=True):
    """Rewrite Rust-spelled constructs to their Sigil equivalents. Region-safe."""
    def one(text):
        text = BARE_VARY.sub(lambda m: f"{m.group(1)}≔ vary {m.group(2)}", text)
        text = IMPL_ARG.sub("⊢ ", text)
        text = VARIANT_MARKER.sub(lambda m: m.group(1) + m.group(2), text)
        text = ASPECT_IDENT.sub("aspect_", text)
        text = INLINE_OBJ_PARAM.sub(lambda m: m.group(1) + "Any", text)
        return text

    out = "".join(one(t) if is_code else t for t, is_code in split_regions(src))

    # `in(` is only an asm operand inside an asm! block; rewriting it everywhere
    # would hit ordinary calls to a function named `in`.
    if "asm!" in out:
        pieces, idx = [], 0
        for m in re.finditer(r"asm!\s*\(", out):
            start = m.end() - 1
            depth, j = 0, start
            while j < len(out):
                if out[j] == "(":
                    depth += 1
                elif out[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            pieces.append(out[idx:start])
            asm_body = ASM_IN.sub("∈(", out[start : j + 1])
            pieces.append(ASM_LATEOUT.sub("out(", asm_body))
            idx = j + 1
        pieces.append(out[idx:])
        out = "".join(pieces)
    return out
