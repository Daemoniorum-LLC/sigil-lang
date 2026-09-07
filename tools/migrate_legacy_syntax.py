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
IMPL_FOR = re.compile(r"\bimpl\s+([A-Za-z_][\w·:<>, ]*?)\s+for\s+([A-Za-z_][\w·:<>, ]*?)\s*\{")
IMPL_PLAIN = re.compile(r"\bimpl\s+([A-Za-z_][\w·:<>, ]*?)\s*\{")
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
    code = IMPL_FOR.sub(lambda m: f"⊢ {m.group(1)} ∀ {m.group(2)} {{", code)
    code = IMPL_PLAIN.sub(lambda m: f"⊢ {m.group(1)} {{", code)
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
