#!/usr/bin/env python3
"""
CG-112: Replace truncated functions with working bootstrap versions

The self-hosted compiler outputs truncated lines due to excessive evidence
wrappers. This script replaces truncated functions with their working
bootstrap counterparts.
"""

import sys
import re

def extract_function(content, func_name, start_pattern=None):
    """Extract a function body from C code."""
    if start_pattern is None:
        start_pattern = f'/* Function: {func_name} */'

    lines = content.split('\n')
    start_line = None
    brace_count = 0
    in_function = False
    function_lines = []

    for i, line in enumerate(lines):
        if start_pattern in line:
            start_line = i
            function_lines.append(line)
            continue

        if start_line is not None and not in_function:
            function_lines.append(line)
            if '{' in line:
                in_function = True
                brace_count = line.count('{') - line.count('}')
            continue

        if in_function:
            function_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                # End of function
                return start_line, i + 1, '\n'.join(function_lines)

    return None, None, None

def replace_function(content, func_name, new_body, start_pattern=None):
    """Replace a function in content with new body."""
    if start_pattern is None:
        start_pattern = f'/* Function: {func_name} */'

    lines = content.split('\n')
    start_line = None
    brace_count = 0
    in_function = False

    for i, line in enumerate(lines):
        if start_pattern in line:
            start_line = i
            continue

        if start_line is not None and not in_function:
            if '{' in line:
                in_function = True
                brace_count = line.count('{') - line.count('}')
            continue

        if in_function:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                # End of function
                end_line = i + 1
                new_lines = lines[:start_line] + new_body.split('\n') + lines[end_line:]
                return '\n'.join(new_lines), True

    return content, False

def main():
    if len(sys.argv) < 3:
        print("Usage: fix_truncated.py <target.c> <bootstrap.c> [output.c]")
        sys.exit(1)

    target_file = sys.argv[1]
    bootstrap_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else target_file

    with open(target_file, 'r') as f:
        target = f.read()

    with open(bootstrap_file, 'r') as f:
        bootstrap = f.read()

    # Functions known to be truncated
    truncated_functions = [
        'Lexer::lex_hex_escape',
        'Lexer::lex_unicode_escape',
    ]

    replaced = 0
    for func_name in truncated_functions:
        # Extract from bootstrap
        start, end, body = extract_function(bootstrap, func_name)
        if body is None:
            print(f"Warning: Could not find {func_name} in bootstrap")
            continue

        print(f"Found {func_name} in bootstrap: lines {start+1}-{end}")

        # Replace in target
        target, success = replace_function(target, func_name, body)
        if success:
            print(f"Replaced {func_name} in target")
            replaced += 1
        else:
            print(f"Warning: Could not find {func_name} in target")

    with open(output_file, 'w') as f:
        f.write(target)

    print(f"CG-112: Replaced {replaced} truncated functions")

if __name__ == '__main__':
    main()
