#!/usr/bin/env python3
"""Convert map literals whose VALUES span multiple lines.

`layer_modifiers: { RealityLayer·Fractured => LightingMod { … }, … }` — the first
converter assumed one entry per line and could not see these. Splits the block on
top-level commas and each entry on its first top-level `=>`.
"""
import re, sys

def split_top(text, sep):
    """Split on `sep` only at bracket depth 0. Quote- and comment-aware."""
    out, buf, depth, i, n = [], [], 0, 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            buf.append(text[i : j + 1]); i = j + 1; continue
        if text.startswith("//", i):
            j = text.find("\n", i); j = n if j < 0 else j
            buf.append(text[i:j]); i = j; continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            out.append("".join(buf)); buf = []; i += len(sep); continue
        buf.append(c); i += 1
    out.append("".join(buf))
    return out


def convert(src):
    n = 0
    while True:
        m = re.search(r"(\n(\s*)([\w_]+): )\{\s*\n", src)
        if not m:
            break
        start = m.end() - 1                      # at the newline after `{`
        brace = src.rindex("{", m.start(), m.end())
        depth, j = 0, brace
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = src[brace + 1 : j]
        entries = [e for e in split_top(body, ",") if e.strip()]
        if not entries or not all("=>" in split_top(e, "=>")[0] + "=>" for e in entries):
            # not a map literal — skip past this brace and keep looking
            src = src[: m.start() + 1] + src[m.start() + 1 :].replace("{", "\x00", 1)
            continue
        pairs = []
        for e in entries:
            k, _, v = e.partition("=>")
            pairs.append(f"{m.group(2)}    ({k.strip()}, {v.strip()}),")
        joined = "\n".join(pairs)
        src = (
            src[: m.start()]
            + f"\n{m.group(2)}{m.group(3)}: HashMap·from([\n{joined}\n{m.group(2)}])"
            + src[j + 1 :]
        )
        n += 1
    return src.replace("\x00", "{"), n


if __name__ == "__main__":
    total = 0
    for path in sys.argv[1:]:
        src = open(path, encoding="utf-8").read()
        new, k = convert(src)
        if k:
            open(path, "w", encoding="utf-8").write(new)
            print(f"  {path}: {k}")
            total += k
    print(f"{total} converted")
