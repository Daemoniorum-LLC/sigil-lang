#!/usr/bin/env python3
"""
Fix missing writebacks for ALL Vec::push() calls (generalized CG-121).

The issue: Generated code calls sigil_Vec____push() but doesn't store the result:
    sigil_with_evidence(sigil_Vec____push(name, value), SIGIL_KNOWN);

Should be:
    name = sigil_with_evidence(sigil_Vec____push(name, value), SIGIL_KNOWN);

This version handles nested parentheses properly.
"""

import re
import sys

def find_balanced_paren(s, start):
    """Find the matching closing paren for an open paren at position start.
    Returns the index of the closing paren, or -1 if not found."""
    if start >= len(s) or s[start] != '(':
        return -1
    depth = 1
    i = start + 1
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1

def extract_vec_push_parts(line):
    """Extract (indent, var_name, rest_of_line) from a Vec::push line.
    Returns None if this doesn't look like a Vec::push pattern."""

    # Check for standalone sigil_with_evidence(sigil_Vec____push(... pattern
    # (not preceded by assignment)
    match = re.match(r'^(\s*)sigil_with_evidence\(sigil_Vec____push\(', line)
    if match:
        indent = match.group(1)
        # Find where var_name starts
        var_start = match.end()
        # Extract var_name (identifier)
        var_match = re.match(r'([a-zA-Z_]\w*)', line[var_start:])
        if var_match:
            var_name = var_match.group(1)
            return (indent, var_name, line.rstrip())
    return None

def fix_vec_push(content):
    lines = content.split('\n')
    result = []
    fixed_count = 0

    # Pattern to detect lines that ALREADY have correct assignment (array = ...)
    # We need to check if the assignment target matches the first argument
    assignment_pattern = re.compile(r'^\s*\w+\s*=\s*sigil_with_evidence\(sigil_Vec____push\(')

    # Pattern for assignment to temp var that should go to array instead
    # _tNNN = sigil_Vec____push(array_name, ...) -> array_name = sigil_Vec____push(array_name, ...)
    temp_assign_pattern = re.compile(
        r'^(\s*)(_t\d+)\s*=\s*sigil_Vec____push\(([a-zA-Z_]\w*),\s*'
    )

    # Pattern for direct call without assignment (no sigil_with_evidence)
    # sigil_Vec____push(array_name, ...); -> array_name = sigil_Vec____push(array_name, ...);
    direct_pattern = re.compile(
        r'^(\s*)sigil_Vec____push\(([a-zA-Z_]\w*),\s*'
    )

    # Pattern for standalone Vec::push without assignment (with sigil_with_evidence)
    standalone_pattern = re.compile(
        r'^(\s*)sigil_with_evidence\(sigil_Vec____push\(([a-zA-Z_]\w*),\s*'
    )

    # Pattern for Vec::push where first arg is wrapped in sigil_with_evidence
    # sigil_with_evidence(sigil_Vec____push(sigil_with_evidence(_tNNNN, SIGIL_KNOWN), ...), SIGIL_KNOWN);
    wrapped_pattern = re.compile(
        r'^(\s*)sigil_with_evidence\(sigil_Vec____push\(sigil_with_evidence\(([a-zA-Z_]\w*),\s*SIGIL_KNOWN\),\s*'
    )

    for line in lines:
        # Skip if this line already has an assignment
        if assignment_pattern.match(line):
            result.append(line)
            continue

        # Check for temp var assignment that should go to array
        # _t123 = sigil_Vec____push(array, ...) -> array = sigil_Vec____push(array, ...)
        match = temp_assign_pattern.match(line)
        if match and line.rstrip().endswith(');'):
            indent = match.group(1)
            temp_var = match.group(2)
            array_name = match.group(3)
            if temp_var != array_name:  # Only fix if assigning to wrong var
                rest = line[match.end():]  # Get the rest after the pattern
                new_line = f'{indent}{array_name} = sigil_Vec____push({array_name}, {rest}'
                result.append(new_line)
                fixed_count += 1
                continue

        # Check for direct Vec::push without any assignment
        match = direct_pattern.match(line)
        if match and line.rstrip().endswith(');') and not line.strip().startswith('_t'):
            indent = match.group(1)
            var_name = match.group(2)
            # Prepend "var_name = " to the line
            new_line = f'{indent}{var_name} = {line.strip()}'
            result.append(new_line)
            fixed_count += 1
            continue

        # Check for standalone Vec::push pattern (with sigil_with_evidence)
        match = standalone_pattern.match(line)
        if match and line.rstrip().endswith(');'):
            indent = match.group(1)
            var_name = match.group(2)
            # Just prepend "var_name = " to the line
            new_line = f'{indent}{var_name} = {line.strip()}'
            result.append(new_line)
            fixed_count += 1
            continue

        # Check for wrapped pattern: Vec::push(sigil_with_evidence(_tNNNN, ...), ...)
        match = wrapped_pattern.match(line)
        if match and line.rstrip().endswith(');'):
            indent = match.group(1)
            var_name = match.group(2)
            # Prepend "var_name = " to the line
            new_line = f'{indent}{var_name} = {line.strip()}'
            result.append(new_line)
            fixed_count += 1
            continue

        result.append(line)

    print(f"Fixed {fixed_count} Vec::push patterns", file=sys.stderr)
    return '\n'.join(result)

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_vec_push.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r', encoding='latin-1') as f:
        content = f.read()

    fixed = fix_vec_push(content)

    with open(output_path, 'w', encoding='latin-1') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
