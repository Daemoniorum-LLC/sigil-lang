#!/usr/bin/env python3
"""
CG-113: Fix C reserved keywords used as variable names

The Sigil codegen may emit reserved C keywords as variable names.
This script renames them to valid identifiers.
"""

import sys
import re

def fix_keywords(content):
    """Fix C reserved keywords used as variable names."""
    stats = {
        'default': 0,
    }

    # CG-113a: 'default' -> 'default_val'
    # Match 'default' as a variable name but not in switch contexts
    # SigilValue default = ...
    pattern = r'\bSigilValue default\b'
    stats['default'] += len(re.findall(pattern, content))
    content = re.sub(pattern, 'SigilValue default_val', content)

    # Also fix uses of the variable
    # = default; or , default) etc.
    # Be careful not to match 'default:' in switch
    # Match: default followed by ; , ) ] space etc but not :
    pattern = r'(\W)default(\s*[;,\)\]\s])'
    # Only replace if not part of 'default:'
    def replace_default(m):
        if m.group(2).strip().startswith(':'):
            return m.group(0)  # Don't replace default:
        return m.group(1) + 'default_val' + m.group(2)

    content = re.sub(pattern, replace_default, content)

    return content, stats

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_keywords.py <input.c> [output.c]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    with open(input_file, 'r') as f:
        content = f.read()

    content, stats = fix_keywords(content)

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"CG-113: Fixed C reserved keywords:")
    print(f"  - 'default' → 'default_val': {stats['default']}")

if __name__ == '__main__':
    main()
