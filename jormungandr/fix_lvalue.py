#!/usr/bin/env python3
"""
CG-116: Fix lvalue required as unary '&' operand errors

The code tries to take the address of temporaries (rvalues).
This script fixes these by restructuring the code.
"""

import sys
import re

def fix_lvalue(content):
    """Fix lvalue required errors."""
    stats = {
        'struct_set_field': 0,
    }

    # CG-116a: Fix sigil_struct_set_field(&sigil_with_evidence(...), ...)
    # Pattern: &sigil_with_evidence(_tN, SIGIL_KNOWN) -> &_tN
    # This works because we want to modify the original, not a wrapped copy
    pattern = r'sigil_struct_set_field\(&sigil_with_evidence\((_t\d+), SIGIL_KNOWN\),'
    stats['struct_set_field'] += len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_struct_set_field(&\1,', content)

    # CG-116b: Fix sigil_struct_set_field(&sigil_with_evidence(*(SigilValue*)..., SIGIL_KNOWN), ...)
    # This is in Rc::clone and Rc::drop - the pattern is:
    # &sigil_with_evidence(*(SigilValue*)sigil_with_evidence(_tN, SIGIL_KNOWN).v.ptr, SIGIL_KNOWN)
    # Should be: (SigilValue*)_tN.v.ptr
    pattern = r'sigil_struct_set_field\(&sigil_with_evidence\(\*\(SigilValue\*\)sigil_with_evidence\((_t\d+), SIGIL_KNOWN\)\.v\.ptr, SIGIL_KNOWN\),'
    stats['struct_set_field'] += len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_struct_set_field((SigilValue*)\1.v.ptr,', content)

    return content, stats

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_lvalue.py <input.c> [output.c]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    with open(input_file, 'r') as f:
        content = f.read()

    content, stats = fix_lvalue(content)

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"CG-116: Fixed lvalue errors:")
    print(f"  - sigil_struct_set_field: {stats['struct_set_field']}")

if __name__ == '__main__':
    main()
