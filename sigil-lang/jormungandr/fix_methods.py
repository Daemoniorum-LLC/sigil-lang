#!/usr/bin/env python3
"""
CG-114: Fix method calls that should be function calls

The Sigil codegen emits .method() syntax which is invalid in C.
This script converts them to proper function calls.
"""

import sys
import re

def fix_methods(content):
    """Fix method calls that should be function calls."""
    stats = {
        'to_string': 0,
    }

    # CG-114a: .to_string() -> sigil_to_string(...)
    # Pattern: expr.to_string() -> sigil_to_string(expr)
    # This is tricky because expr can be complex like sigil_struct_field(...)

    # Match patterns like: sigil_struct_field(result, "value").to_string()
    pattern = r'(sigil_struct_field\([^)]+\))\.to_string\(\)'
    stats['to_string'] = len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_to_string(\1)', content)

    # Also match: pair.v.tup.fields[N].to_string()
    pattern = r'([a-z_][a-z0-9_]*\.v\.tup\.fields\[\d+\])\.to_string\(\)'
    stats['to_string'] += len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_to_string(\1)', content)

    # Also match simple variable.to_string()
    pattern = r'([a-z_][a-z0-9_]*)\.to_string\(\)'
    stats['to_string'] += len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_to_string(\1)', content)

    return content, stats

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_methods.py <input.c> [output.c]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    with open(input_file, 'r') as f:
        content = f.read()

    content, stats = fix_methods(content)

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"CG-114: Fixed method calls:")
    print(f"  - .to_string() → sigil_to_string(): {stats['to_string']}")

if __name__ == '__main__':
    main()
