#!/usr/bin/env python3
"""
Fix Result pattern matching access bug.
Generated code uses .v.e.data[0] but sigil_Ok/sigil_Err store in .v.ptr
"""

import re
import sys

def fix_result_access(content):
    lines = content.split('\n')
    result = []

    # Track which variables are being used in sigil_is_ok/sigil_is_err
    result_vars = set()

    for i, line in enumerate(lines):
        # Check for sigil_is_ok or sigil_is_err patterns
        match = re.search(r'sigil_is_(ok|err)\((\w+)\)', line)
        if match:
            var_name = match.group(2)
            result_vars.add(var_name)

        # Check if this line accesses .v.e.data[0] for a known result variable
        modified = False
        for var in list(result_vars):
            pattern = rf'\b{var}\.v\.e\.data\[0\]'
            if re.search(pattern, line):
                # Replace .v.e.data[0] with *(SigilValue*).v.ptr
                line = re.sub(pattern, f'*(SigilValue*){var}.v.ptr', line)
                modified = True

        result.append(line)

        # Clear result_vars at closing braces to avoid false positives
        if line.strip() == '}':
            result_vars.clear()

    return '\n'.join(result)

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_result_access.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r') as f:
        content = f.read()

    fixed = fix_result_access(content)

    with open(output_path, 'w') as f:
        f.write(fixed)

    print(f"Fixed Result access patterns in {output_path}")

if __name__ == '__main__':
    main()
