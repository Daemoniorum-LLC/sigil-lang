#!/usr/bin/env python3
"""
CG-115: Fix self parameter type mismatches

Some methods take SigilValue self, but are called with SigilValue* self (or vice versa).
This script fixes these mismatches.
"""

import sys
import re

def fix_self_types(content):
    """Fix self parameter type mismatches."""
    stats = {
        'ptr_to_val': 0,  # (*self) needed
        'val_to_ptr': 0,  # self (remove &)
    }

    # Methods that take SigilValue self (by value)
    val_methods = [
        'sigil_TypeChecker____type_has_explicit_evidence',
        'sigil_TypeChecker____infer_literal',
        'sigil_Interpreter____eval_binary',
        'sigil_Interpreter____eval_unary',
    ]

    # Fix calls where we have SigilValue* self but method expects SigilValue
    # Pattern: method(self, ...) -> method((*self), ...)
    # But only in functions that have SigilValue* self

    lines = content.split('\n')
    new_lines = []
    in_ptr_function = False
    current_func_takes_ptr = False

    for i, line in enumerate(lines):
        # Detect function definitions
        if line.startswith('SigilValue sigil_'):
            # Check if this function takes SigilValue* self
            current_func_takes_ptr = 'SigilValue* self' in line

        # Fix closure &self when method expects SigilValue
        for method in val_methods:
            # Pattern: method(&self, ...) -> method(self, ...)
            pattern = rf'{method}\(&self,'
            if re.search(pattern, line):
                line = re.sub(pattern, f'{method}(self,', line)
                stats['val_to_ptr'] += 1

        # Fix calls in functions with SigilValue* self
        if current_func_takes_ptr:
            for method in val_methods:
                # Pattern: method(self, ...) -> method((*self), ...)
                # But not if already (*self)
                pattern = rf'{method}\(self,'
                if re.search(pattern, line) and '(*self)' not in line:
                    line = re.sub(pattern, f'{method}((*self),', line)
                    stats['ptr_to_val'] += 1

        new_lines.append(line)

    return '\n'.join(new_lines), stats

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_self_types.py <input.c> [output.c]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    with open(input_file, 'r') as f:
        content = f.read()

    content, stats = fix_self_types(content)

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"CG-115: Fixed self parameter type mismatches:")
    print(f"  - &self → self (for val methods): {stats['val_to_ptr']}")
    print(f"  - self → (*self) (for val methods in ptr funcs): {stats['ptr_to_val']}")

if __name__ == '__main__':
    main()
