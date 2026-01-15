#!/usr/bin/env python3
"""
Fix missing writebacks for errors.push() calls.
"""

import re
import sys

def fix_errors_push(content):
    lines = content.split('\n')
    result = []

    # First pass: collect all temp vars that hold errors field
    errors_temps = {}
    for i, line in enumerate(lines):
        match = re.search(r'SigilValue (_t\d+) = sigil_struct_field\(([^,]+), "errors"\);', line)
        if match:
            temp_var = match.group(1)
            self_expr = match.group(2).strip()
            if '(*self)' in self_expr:
                errors_temps[temp_var] = '&(*self)'
            elif 'self' in self_expr:
                errors_temps[temp_var] = '&self'
            elif 'ctx' in self_expr:
                errors_temps[temp_var] = '&ctx'
            else:
                errors_temps[temp_var] = f'&{self_expr}'

    # Second pass: fix the push lines
    fixed_count = 0
    for line in lines:
        modified = False
        for temp_var, self_addr in errors_temps.items():
            # Match the push pattern
            pattern = rf'(\s*)sigil_with_evidence\(sigil_Vec____push\(sigil_with_evidence\({temp_var}, SIGIL_KNOWN\), ([^)]+)\), SIGIL_KNOWN\);'
            match = re.search(pattern, line)
            if match:
                indent = match.group(1)
                value_to_push = match.group(2)
                new_line = f'{indent}sigil_struct_set_field({self_addr}, "errors", sigil_Vec____push(sigil_with_evidence({temp_var}, SIGIL_KNOWN), {value_to_push}));'
                result.append(new_line)
                modified = True
                fixed_count += 1
                break

        if not modified:
            result.append(line)

    print(f"Fixed {fixed_count} errors push patterns")
    return '\n'.join(result)

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_errors_push.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r') as f:
        content = f.read()

    fixed = fix_errors_push(content)

    with open(output_path, 'w') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
