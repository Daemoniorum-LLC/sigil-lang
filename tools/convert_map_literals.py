#!/usr/bin/env python3
"""Convert `{ "k" => v, … }` map literals to HashMap·from([("k", v), …]).

Sigil has no map-literal syntax; HashMap·from is the constructor. Indentation and
the trailing evidentiality marker are preserved, and only a brace block whose
entries are all `expr => expr` is touched.
"""
import re, sys

OPEN = re.compile(r"(?<![\w·])\{\s*$")

def convert(src):
    lines = src.split("\n")
    out, i, n = [], 0, 0
    while i < len(lines):
        line = lines[i]
        m = OPEN.search(line)
        # A match block also has `=>` entries. Converting one turns
        # `⌥ (scene_id, reality) {` into `⌥ (scene_id, reality) HashMap·from([`,
        # which is how the first run mangled enemies.sigil.
        if not m or "⌥" in line or "⎇" in line or "⎉" in line:
            out.append(line); i += 1; continue
        # collect until the matching closing brace at the same indent
        depth = line.count("{") - line.count("}")
        j, body = i + 1, []
        while j < len(lines) and depth > 0:
            depth += lines[j].count("{") - lines[j].count("}")
            if depth == 0:
                break
            body.append(lines[j]); j += 1
        if j >= len(lines):
            out.append(line); i += 1; continue
        entries = [b for b in body if b.strip() and not b.strip().startswith("//")]
        if not entries or not all("=>" in e for e in entries):
            out.append(line); i += 1; continue
        head = line[: m.start()]
        close = lines[j]
        tail = close.strip()[1:]                       # e.g. "~," after the }
        indent = re.match(r"\s*", body[0]).group(0) if body else "    "
        pairs = []
        for b in body:
            st = b.strip()
            if not st or st.startswith("//"):
                pairs.append(b); continue
            k, _, v = st.partition("=>")
            # A trailing line comment must not be swept into the value:
            # `"MemoryDrain" => 0.4,  // Primary ability` became
            # `("MemoryDrain", 0.4,  // Primary ability),` on the first run.
            comment = ""
            if "//" in v:
                v, _, rest = v.partition("//")
                comment = "  //" + rest
            v = v.rstrip().rstrip(",")
            pairs.append(f"{indent}({k.strip()}, {v.strip()}),{comment}")
        out.append(f"{head}HashMap·from([")
        out.extend(pairs)
        out.append(re.match(r"\s*", close).group(0) + "])" + tail)
        n += 1
        i = j + 1
    return "\n".join(out), n

if __name__ == "__main__":
    total = 0
    for path in sys.argv[1:]:
        src = open(path, encoding="utf-8").read()
        new, k = convert(src)
        if k:
            open(path, "w", encoding="utf-8").write(new)
            print(f"  {path}: {k} map literal(s)")
            total += k
    print(f"{total} converted")
