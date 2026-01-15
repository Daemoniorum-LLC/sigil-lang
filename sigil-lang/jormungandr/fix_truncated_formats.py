#!/usr/bin/env python3
"""
Fix known truncated format strings in generated Sigil C code.

The issue: Some format! strings get truncated during compilation of large files.
This script restores the complete format strings for known patterns.
"""

import re
import sys

# Known truncated patterns and their complete versions
FIXES = {
    # emit_binary_add - truncated to ": {}.v.f)))"
    r'return sigil_format\(\s*:\s*\{\}\.v\.f\)\)\)\",\s*l,\s*r,\s*l,\s*r,\s*l,\s*r,\s*l,\s*r,\s*l,\s*l,\s*l,\s*r,\s*r,\s*r\);':
        'return sigil_format("({}.tag == TAG_INT && {}.tag == TAG_INT ? sigil_int({}.v.i + {}.v.i) : {}.tag == TAG_STRING || {}.tag == TAG_STRING ? sigil_concat({}, {}) : sigil_float(({}.tag == TAG_INT ? (double){}.v.i : {}.v.f) + ({}.tag == TAG_INT ? (double){}.v.i : {}.v.f)))",  l,  r,  l,  r,  l,  r,  l,  r,  l,  l,  l,  r,  r,  r);',
}

def fix_truncated_formats(content):
    fixed_count = 0

    for pattern, replacement in FIXES.items():
        matches = len(re.findall(pattern, content))
        if matches:
            content = re.sub(pattern, replacement, content)
            print(f"Fixed {matches} truncated format string(s)", file=sys.stderr)
            fixed_count += matches

    print(f"Total format string fixes: {fixed_count}", file=sys.stderr)
    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_truncated_formats.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r', encoding='latin-1') as f:
        content = f.read()

    fixed = fix_truncated_formats(content)

    with open(output_path, 'w', encoding='latin-1') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
