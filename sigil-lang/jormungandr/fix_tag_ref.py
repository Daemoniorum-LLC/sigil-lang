#!/usr/bin/env python3
"""
Fix unnecessary TAG_REF wrappers that break value comparisons.

The issue: Generated code wraps local variables in TAG_REF like:
  (SigilValue){ .tag = TAG_REF, .v.ptr = &token }

But sigil_eq has no handling for TAG_REF and always returns false when
comparing TAG_REF with TAG_ENUM (or other types).

This fix removes the TAG_REF wrapper for local variable references that
should just be passed by value.
"""

import re
import sys

def fix_tag_ref(content):
    lines = content.split('\n')
    result = []
    fixed_count = 0

    # Pattern matches: (SigilValue){ .tag = TAG_REF, .v.ptr = &VAR }
    # where VAR is a local variable (not Token____ constant)
    tag_ref_pattern = re.compile(
        r'\(SigilValue\)\{\s*\.tag\s*=\s*TAG_REF,\s*\.v\.ptr\s*=\s*&(\w+)\s*\}'
    )

    # All variables/constants should NOT be wrapped in TAG_REF for comparison
    # because sigil_eq does not handle TAG_REF and always returns false

    for i, line in enumerate(lines):
        modified_line = line

        # Find all TAG_REF patterns in this line
        matches = list(tag_ref_pattern.finditer(line))

        if matches:
            # Process from right to left to preserve offsets
            for match in reversed(matches):
                var_name = match.group(1)

                # Replace the TAG_REF wrapper with just the variable
                full_match = match.group(0)
                modified_line = modified_line[:match.start()] + var_name + modified_line[match.end():]
                fixed_count += 1

        result.append(modified_line)

    print(f"Fixed {fixed_count} TAG_REF wrappers", file=sys.stderr)
    return '\n'.join(result)

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_tag_ref.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r') as f:
        content = f.read()

    fixed = fix_tag_ref(content)

    with open(output_path, 'w') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
