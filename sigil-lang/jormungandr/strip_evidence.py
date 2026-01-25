#!/usr/bin/env python3
"""
CG-111: Strip redundant sigil_with_evidence wrappers to reduce line lengths

The self-hosted compiler emits excessive evidence wrappers that bloat the
generated C code. This script removes redundant wrappers:

1. For constant values (int, char, bool, float), evidence is implied
2. Nested wrappers are redundant
"""

import re
import sys

def strip_evidence(content):
    """Strip redundant sigil_with_evidence wrappers."""

    # Stats tracking
    stats = {
        'int': 0,
        'char': 0,
        'bool': 0,
        'float': 0,
        'string': 0,
        'nested': 0,
    }

    # CG-111a: Strip evidence from constant int values (only simple integer literals)
    # sigil_with_evidence(sigil_int(N), SIGIL_KNOWN) -> sigil_int(N)
    # Where N is just a number like 0LL, 10LL, -5LL
    pattern = r'sigil_with_evidence\(sigil_int\((-?\d+LL)\), SIGIL_KNOWN\)'
    stats['int'] = len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_int(\1)', content)

    # CG-111b: Strip evidence from constant char values (only simple char literals)
    # sigil_with_evidence(sigil_char('x'), SIGIL_KNOWN) -> sigil_char('x')
    # Where 'x' is a simple char like 'a', '\n', '\x00'
    pattern = r"sigil_with_evidence\(sigil_char\(('(?:[^'\\]|\\.)')\), SIGIL_KNOWN\)"
    stats['char'] = len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_char(\1)', content)

    # CG-111c: Strip evidence from boolean literals only (true/false)
    # sigil_with_evidence(sigil_bool(true), SIGIL_KNOWN) -> sigil_bool(true)
    # sigil_with_evidence(sigil_bool(false), SIGIL_KNOWN) -> sigil_bool(false)
    pattern = r'sigil_with_evidence\(sigil_bool\((true|false)\), SIGIL_KNOWN\)'
    stats['bool'] = len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_bool(\1)', content)

    # CG-111d: Strip evidence from float literals only
    # sigil_with_evidence(sigil_float(N.M), SIGIL_KNOWN) -> sigil_float(N.M)
    pattern = r'sigil_with_evidence\(sigil_float\((-?\d+\.?\d*)\), SIGIL_KNOWN\)'
    stats['float'] = len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_float(\1)', content)

    # CG-111e: Strip evidence from string literals
    # sigil_with_evidence(sigil_str("x"), SIGIL_KNOWN) -> sigil_str("x")
    pattern = r'sigil_with_evidence\(sigil_str\(("(?:[^"\\]|\\.)*")\), SIGIL_KNOWN\)'
    stats['string'] = len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_str(\1)', content)

    # CG-111f: Flatten nested evidence wrappers
    # sigil_with_evidence(sigil_with_evidence(X, SIGIL_KNOWN), SIGIL_KNOWN) -> sigil_with_evidence(X, SIGIL_KNOWN)
    # This needs multiple passes since nesting can be deep
    prev_len = 0
    while len(content) != prev_len:
        prev_len = len(content)
        old_content = content
        # Match nested wrappers
        pattern = r'sigil_with_evidence\(sigil_with_evidence\(([^,]+), SIGIL_KNOWN\), SIGIL_KNOWN\)'
        content = re.sub(pattern, r'sigil_with_evidence(\1, SIGIL_KNOWN)', content)
        if content != old_content:
            stats['nested'] += 1

    return content, stats

def main():
    if len(sys.argv) < 2:
        print("Usage: strip_evidence.py <input.c> [output.c]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    with open(input_file, 'r') as f:
        content = f.read()

    original_len = len(content)
    content, stats = strip_evidence(content)
    new_len = len(content)

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"CG-111: Stripped redundant evidence wrappers:")
    print(f"  - int constants: {stats['int']}")
    print(f"  - char constants: {stats['char']}")
    print(f"  - bool values: {stats['bool']}")
    print(f"  - float values: {stats['float']}")
    print(f"  - string literals: {stats['string']}")
    print(f"  - nested wrappers: {stats['nested']} passes")
    print(f"  Size reduction: {original_len - new_len:,} bytes ({100*(original_len-new_len)/original_len:.1f}%)")

if __name__ == '__main__':
    main()
