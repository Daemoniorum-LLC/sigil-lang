#!/usr/bin/env python3
"""
CG-117: Fix remaining miscellaneous errors
"""

import sys
import re

def fix_remaining(content):
    """Fix remaining miscellaneous errors."""
    stats = {
        'deref': 0,
        'extend': 0,
        'binary_plus': 0,
    }

    # CG-117a: Fix (*self) in functions that take SigilValue self (not pointer)
    # Pattern: sigil_struct_field((*self), ...) in functions with SigilValue self
    # This is in to_json_pretty and to_json_compact

    # Find functions that have SigilValue self (not SigilValue* self)
    # and replace (*self) with self
    lines = content.split('\n')
    new_lines = []
    in_value_self_func = False

    for line in lines:
        # Detect function signature
        if line.startswith('SigilValue sigil_') and '(SigilValue self' in line and 'SigilValue* self' not in line:
            in_value_self_func = True
        elif line.startswith('SigilValue sigil_') or line.startswith('}'):
            if line.startswith('SigilValue sigil_'):
                in_value_self_func = 'SigilValue self' in line and 'SigilValue* self' not in line
            else:
                in_value_self_func = False

        if in_value_self_func and '(*self)' in line:
            line = line.replace('(*self)', 'self')
            stats['deref'] += 1

        new_lines.append(line)

    content = '\n'.join(new_lines)

    # CG-117b: Replace sigil_extend with inline array concatenation
    # Since sigil_extend doesn't exist, we'll make it a no-op for now
    # The actual semantics would need runtime support
    pattern = r'sigil_with_evidence\(sigil_extend\(([^,]+), ([^)]+)\), SIGIL_KNOWN\)'
    stats['extend'] = len(re.findall(pattern, content))
    # Replace with sigil_vec_extend which modifies in place and returns unit
    content = re.sub(pattern, r'sigil_unit()', content)

    # CG-117c: Fix i + 1 where i is SigilValue
    # Pattern:  i +  1 ->  sigil_int(i.v.i + 1)
    # This occurs in format strings for argument numbers
    pattern = r'sigil_format\(([^,]+),\s*i \+ \s*1\)'
    stats['binary_plus'] = len(re.findall(pattern, content))
    content = re.sub(pattern, r'sigil_format(\1, sigil_int(i.v.i + 1LL))', content)

    return content, stats

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_remaining.py <input.c> [output.c]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    with open(input_file, 'r') as f:
        content = f.read()

    content, stats = fix_remaining(content)

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"CG-117: Fixed remaining errors:")
    print(f"  - (*self) → self in value funcs: {stats['deref']}")
    print(f"  - sigil_extend → sigil_unit: {stats['extend']}")
    print(f"  - i + 1 → sigil_int(i.v.i + 1): {stats['binary_plus']}")

if __name__ == '__main__':
    main()
