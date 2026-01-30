#!/usr/bin/env python3
"""
Fix missing writebacks for ALL mut self push() calls (Vec::push and String::push).

The issue: Generated code assigns push result to temp var instead of source:
    _t123 = sigil_Vec____push(array, value);
    _t456 = sigil_String____push(str, char);

Should be:
    array = sigil_Vec____push(array, value);
    str = sigil_String____push(str, char);

This is critical for String::push because when the string buffer reallocates,
the old pointer becomes stale, causing string truncation/corruption.
"""

import re
import sys

def fix_mut_push(content):
    lines = content.split('\n')
    result = []
    fixed_vec = 0
    fixed_string = 0

    # Patterns for Vec::push - temp var assignment (any ending)
    vec_temp_pattern = re.compile(
        r'^(\s*)(_t\d+)\s*=\s*sigil_Vec____push\(([a-zA-Z_]\w*),\s*(.+)\)([;]?)$'
    )
    vec_direct_pattern = re.compile(
        r'^(\s*)sigil_Vec____push\(([a-zA-Z_]\w*),\s*'
    )
    vec_standalone_pattern = re.compile(
        r'^(\s*)sigil_with_evidence\(sigil_Vec____push\(([a-zA-Z_]\w*),\s*'
    )
    vec_wrapped_pattern = re.compile(
        r'^(\s*)sigil_with_evidence\(sigil_Vec____push\(sigil_with_evidence\(([a-zA-Z_]\w*),\s*SIGIL_KNOWN\),\s*'
    )

    # Patterns for String::push - temp var assignment (any ending)
    string_temp_pattern = re.compile(
        r'^(\s*)(_t\d+)\s*=\s*sigil_String____push\(([a-zA-Z_]\w*),\s*(.+)\)([;]?)$'
    )
    string_direct_pattern = re.compile(
        r'^(\s*)sigil_String____push\(([a-zA-Z_]\w*),\s*'
    )

    # Patterns for String::push_str - temp var assignment (any ending)
    string_push_str_temp_pattern = re.compile(
        r'^(\s*)(_t\d+)\s*=\s*sigil_String____push_str\(([a-zA-Z_]\w*),\s*(.+)\)([;]?)$'
    )
    string_push_str_direct_pattern = re.compile(
        r'^(\s*)sigil_String____push_str\(([a-zA-Z_]\w*),\s*'
    )

    # Assignment check pattern - skip lines that already have correct assignment
    vec_assignment_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*.*sigil_Vec____push\(\1,')
    string_assignment_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*.*sigil_String____push\(\1,')
    string_push_str_assignment_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*.*sigil_String____push_str\(\1,')

    for line in lines:
        # Skip if already correctly assigned
        if vec_assignment_pattern.match(line) or string_assignment_pattern.match(line) or string_push_str_assignment_pattern.match(line):
            result.append(line)
            continue

        fixed = False

        # Vec::push with temp var assignment
        match = vec_temp_pattern.match(line)
        if match:
            indent = match.group(1)
            temp_var = match.group(2)
            array_name = match.group(3)
            rest_args = match.group(4)
            ending = match.group(5)
            if temp_var != array_name:
                new_line = f'{indent}{array_name} = sigil_Vec____push({array_name}, {rest_args}){ending}'
                result.append(new_line)
                fixed_vec += 1
                fixed = True

        # Vec::push direct call without assignment
        if not fixed:
            match = vec_direct_pattern.match(line)
            if match and line.rstrip().endswith(');') and not line.strip().startswith('_t'):
                indent = match.group(1)
                var_name = match.group(2)
                new_line = f'{indent}{var_name} = {line.strip()}'
                result.append(new_line)
                fixed_vec += 1
                fixed = True

        # Vec::push with sigil_with_evidence wrapper
        if not fixed:
            match = vec_standalone_pattern.match(line)
            if match and line.rstrip().endswith(');'):
                indent = match.group(1)
                var_name = match.group(2)
                new_line = f'{indent}{var_name} = {line.strip()}'
                result.append(new_line)
                fixed_vec += 1
                fixed = True

        # Vec::push with wrapped first arg
        if not fixed:
            match = vec_wrapped_pattern.match(line)
            if match and line.rstrip().endswith(');'):
                indent = match.group(1)
                var_name = match.group(2)
                new_line = f'{indent}{var_name} = {line.strip()}'
                result.append(new_line)
                fixed_vec += 1
                fixed = True

        # String::push with temp var assignment
        if not fixed:
            match = string_temp_pattern.match(line)
            if match:
                indent = match.group(1)
                temp_var = match.group(2)
                str_name = match.group(3)
                rest_args = match.group(4)
                ending = match.group(5)
                if temp_var != str_name:
                    new_line = f'{indent}{str_name} = sigil_String____push({str_name}, {rest_args}){ending}'
                    result.append(new_line)
                    fixed_string += 1
                    fixed = True

        # String::push direct call without assignment
        if not fixed:
            match = string_direct_pattern.match(line)
            if match and line.rstrip().endswith(');') and not line.strip().startswith('_t'):
                indent = match.group(1)
                var_name = match.group(2)
                new_line = f'{indent}{var_name} = {line.strip()}'
                result.append(new_line)
                fixed_string += 1
                fixed = True

        # String::push_str with temp var assignment
        if not fixed:
            match = string_push_str_temp_pattern.match(line)
            if match:
                indent = match.group(1)
                temp_var = match.group(2)
                str_name = match.group(3)
                rest_args = match.group(4)
                ending = match.group(5)
                if temp_var != str_name:
                    new_line = f'{indent}{str_name} = sigil_String____push_str({str_name}, {rest_args}){ending}'
                    result.append(new_line)
                    fixed_string += 1
                    fixed = True

        # String::push_str direct call without assignment
        if not fixed:
            match = string_push_str_direct_pattern.match(line)
            if match and line.rstrip().endswith(');') and not line.strip().startswith('_t'):
                indent = match.group(1)
                var_name = match.group(2)
                new_line = f'{indent}{var_name} = {line.strip()}'
                result.append(new_line)
                fixed_string += 1
                fixed = True

        if not fixed:
            result.append(line)

    print(f"Fixed {fixed_vec} Vec::push patterns", file=sys.stderr)
    print(f"Fixed {fixed_string} String::push/push_str patterns", file=sys.stderr)
    return '\n'.join(result)

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_mut_push.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r', encoding='latin-1') as f:
        content = f.read()

    fixed = fix_mut_push(content)

    with open(output_path, 'w', encoding='latin-1') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
