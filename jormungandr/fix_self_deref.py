#!/usr/bin/env python3
"""
Fix self pointer/value mismatch in method calls.

The issue: When a method with SigilValue* self calls a method that takes SigilValue self
(by value), the caller passes `self` but should pass `*self`.

This script:
1. Finds all functions that take SigilValue self (by value)
2. Finds calls to those functions from functions that take SigilValue* self
3. Fixes the call to use *self instead of self
"""

import re
import sys

def count_braces_outside_strings(line):
    """Count { and } that are not inside string literals."""
    opens = 0
    closes = 0
    in_string = False
    escape = False
    for c in line:
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if not in_string:
            if c == '{':
                opens += 1
            elif c == '}':
                closes += 1
    return opens, closes

def fix_self_deref(content):
    lines = content.split('\n')

    # Find all functions that take SigilValue self (by value, not pointer)
    value_self_funcs = set()
    value_self_pattern = re.compile(r'^SigilValue (sigil_\w+)\(SigilValue self[,)]')

    for line in lines:
        match = value_self_pattern.match(line)
        if match:
            value_self_funcs.add(match.group(1))

    # Also add known functions that take SigilValue (not self) as first param
    value_first_funcs = {'sigil_struct_field', 'sigil_len', 'sigil_clone', 'sigil_truthy',
                         'sigil_contains', 'sigil_iter', 'sigil_to_string', 'sigil_eq',
                         'sigil_format'}

    # Find all functions that take SigilValue* self (pointer)
    ptr_self_funcs = set()
    ptr_self_pattern = re.compile(r'^SigilValue (sigil_\w+)\(SigilValue\* self[,)]')

    for line in lines:
        match = ptr_self_pattern.match(line)
        if match:
            ptr_self_funcs.add(match.group(1))

    print(f"Found {len(value_self_funcs)} functions taking SigilValue self (by value)", file=sys.stderr)
    print(f"Found {len(ptr_self_funcs)} functions taking SigilValue* self (pointer)", file=sys.stderr)

    # Now fix calls: in functions with SigilValue* self, calls to value_self_funcs
    # need to use *self instead of self

    result = []
    in_ptr_self_func = False
    current_func = None
    fixed_count = 0
    brace_depth = 0

    for line in lines:
        # Track which function we're in
        func_def = ptr_self_pattern.match(line)
        if func_def:
            in_ptr_self_func = True
            current_func = func_def.group(1)
            brace_depth = 0

        # Track brace depth to know when function ends (counting braces outside strings)
        if in_ptr_self_func:
            opens, closes = count_braces_outside_strings(line)
            brace_depth += opens - closes
            if brace_depth <= 0 and opens == 0 and closes > 0:
                in_ptr_self_func = False
                current_func = None

        # Fix calls to value_self_funcs when in ptr_self context
        if in_ptr_self_func:
            new_line = line
            for func_name in value_self_funcs:
                # Pattern: func_name(self, or func_name(self)
                # Replace with: func_name(*self, or func_name(*self)
                # But not if already *self
                pattern = rf'({re.escape(func_name)})\(self([,\)])'
                replacement = rf'\1(*self\2'
                matches = len(re.findall(pattern, new_line))
                if matches > 0:
                    new_line = re.sub(pattern, replacement, new_line)
                    fixed_count += matches
            # Also fix calls to value_first_funcs (like sigil_struct_field)
            for func_name in value_first_funcs:
                # Pattern: func_name(self, - note self followed by comma
                pattern = rf'({re.escape(func_name)})\(self,'
                replacement = rf'\1(*self,'
                matches = len(re.findall(pattern, new_line))
                if matches > 0:
                    new_line = re.sub(pattern, replacement, new_line)
                    fixed_count += matches
            result.append(new_line)
        else:
            result.append(line)

    print(f"Fixed {fixed_count} self -> *self dereferences", file=sys.stderr)

    # Now handle the reverse: in functions with SigilValue self (by value),
    # calls to ptr_self_funcs need to use &self instead of self
    content2 = '\n'.join(result)
    lines2 = content2.split('\n')
    result2 = []
    in_value_self_func = False
    brace_depth2 = 0
    fixed_count2 = 0

    for line in lines2:
        # Track which function we're in
        func_def = value_self_pattern.match(line)
        if func_def:
            in_value_self_func = True
            brace_depth2 = 0

        # Also check for closures which have SigilValue self
        closure_def = re.match(r'^SigilValue (sigil_closure_\d+)\(SigilValue self[,)]', line)
        if closure_def:
            in_value_self_func = True
            brace_depth2 = 0

        # Also detect closures that capture self from __closure_self
        closure_func = re.match(r'^static SigilValue sigil_closure_\d+\(', line)
        if closure_func:
            in_value_self_func = True
            brace_depth2 = 0

        # Track brace depth to know when function ends (counting braces outside strings)
        if in_value_self_func:
            opens, closes = count_braces_outside_strings(line)
            brace_depth2 += opens - closes
            if brace_depth2 <= 0 and opens == 0 and closes > 0:
                in_value_self_func = False

        # Fix calls to ptr_self_funcs when in value_self context
        if in_value_self_func:
            new_line = line
            for func_name in ptr_self_funcs:
                # Pattern: func_name(self, or func_name(self)
                # Replace with: func_name(&self, or func_name(&self)
                # But not if already &self or *self
                pattern = rf'({re.escape(func_name)})\(self([,\)])'
                replacement = rf'\1(&self\2'
                matches = len(re.findall(pattern, new_line))
                if matches > 0:
                    new_line = re.sub(pattern, replacement, new_line)
                    fixed_count2 += matches
            result2.append(new_line)
        else:
            result2.append(line)

    print(f"Fixed {fixed_count2} self -> &self references", file=sys.stderr)
    return '\n'.join(result2)

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_self_deref.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r', encoding='latin-1') as f:
        content = f.read()

    fixed = fix_self_deref(content)

    with open(output_path, 'w', encoding='latin-1') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
