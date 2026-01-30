#!/usr/bin/env python3
"""
Fix various codegen bugs in generated Sigil C code.

Issues fixed:
1. String corruption: ssigil_struct_field(elf -> sigil_struct_field(self
2. Stray ? operators: expr?) -> expr)
3. Method calls in format strings: .to_uppercase() not properly converted
4. mut self pointer issues: self -> &self for pointer parameters
"""

import re
import sys

def fix_codegen_bugs(content):
    fixed_count = 0

    # Fix 1: String corruption - ssigil_struct_field(elf -> sigil_struct_field(self
    pattern1 = r'ssigil_struct_field\(elf,'
    replacement1 = 'sigil_struct_field(self,'
    count1 = len(re.findall(pattern1, content))
    content = re.sub(pattern1, replacement1, content)
    if count1:
        print(f"Fixed {count1} 'ssigil_struct_field(elf' corruptions", file=sys.stderr)
        fixed_count += count1

    # Fix 2: Remove stray ? operators before )
    # Pattern: expression?) -> expression)
    pattern2 = r'\?(\))'
    count2 = len(re.findall(pattern2, content))
    content = re.sub(pattern2, r'\1', content)
    if count2:
        print(f"Fixed {count2} stray '?' operators", file=sys.stderr)
        fixed_count += count2

    # Fix 3: Method calls like .to_uppercase() appearing in C code
    # These should have been converted to sigil_String____to_uppercase()
    # Pattern: identifier.to_uppercase()
    pattern3 = r'(\w+)\.to_uppercase\(\)'
    def replace_to_uppercase(match):
        var = match.group(1)
        return f'sigil_String____to_uppercase({var})'
    count3 = len(re.findall(pattern3, content))
    content = re.sub(pattern3, replace_to_uppercase, content)
    if count3:
        print(f"Fixed {count3} '.to_uppercase()' method calls", file=sys.stderr)
        fixed_count += count3

    # Fix 4: Similar for other common method calls
    method_fixes = [
        (r'(\w+)\.to_lowercase\(\)', r'sigil_String____to_lowercase(\1)'),
        (r'(\w+)\.trim\(\)', r'sigil_String____trim(\1)'),
        (r'(\w+)\.is_empty\(\)', r'sigil_String____is_empty(\1)'),
        (r'(\w+)\.to_string\(\)', r'sigil_to_string(\1)'),
        # Note: .len is tricky - don't fix v.arr.len or v.s which are C struct fields
        # Only fix standalone variable.len patterns
        # (r'(\w+)\.len\b', r'sigil_arr_len(\1)'),  # Too aggressive, disabled
    ]
    for pattern, replacement in method_fixes:
        count = len(re.findall(pattern, content))
        if count:
            content = re.sub(pattern, replacement, content)
            print(f"Fixed {count} method call patterns", file=sys.stderr)
            fixed_count += count

    # Fix 4a: .name field access should be sigil_struct_field(x, "name")
    pattern4a = r'(\w+)\.name\b(?!\s*=)'  # Not followed by assignment
    def replace_name_access(match):
        var = match.group(1)
        return f'sigil_struct_field({var}, "name")'
    count4a = len(re.findall(pattern4a, content))
    if count4a:
        content = re.sub(pattern4a, replace_name_access, content)
        print(f"Fixed {count4a} '.name' field accesses", file=sys.stderr)
        fixed_count += count4a

    # Fix 4b: Truncated function names (sigil_with_evide -> sigil_with_evidence)
    truncated_funcs = [
        (r'\bsigil_with_evide\b(?!nce)', 'sigil_with_evidence'),
    ]
    for pattern, replacement in truncated_funcs:
        count = len(re.findall(pattern, content))
        if count:
            content = re.sub(pattern, replacement, content)
            print(f"Fixed {count} truncated function names", file=sys.stderr)
            fixed_count += count

    # Fix 4c: 'default' is a C keyword, rename to 'default_val'
    # Match: = default; or = default, or (default, or default) or default =
    # But NOT "default" in strings
    # Pattern: default as a standalone word not in string context
    default_patterns = [
        (r'\bdefault([,;\)])', r'default_val\1'),  # default followed by , ; or )
        (r'=\s*default\b', r'= default_val'),      # = default
        (r'\bdefault\s*=', r'default_val ='),      # default =
    ]
    count4c = 0
    for pattern, replacement in default_patterns:
        count = len(re.findall(pattern, content))
        if count:
            content = re.sub(pattern, replacement, content)
            count4c += count
    if count4c:
        print(f"Fixed {count4c} 'default' C keyword conflicts", file=sys.stderr)
        fixed_count += count4c

    # Fix 5: Functions with SigilValue* self being called with self instead of &self
    # First, find all functions that take SigilValue* self
    ptr_self_funcs = set()
    func_pattern = re.compile(r'SigilValue (sigil_\w+)\(SigilValue\* self')
    for match in func_pattern.finditer(content):
        ptr_self_funcs.add(match.group(1))

    if ptr_self_funcs:
        # Now fix calls to these functions where self is passed without &
        # Pattern: func_name(self, ...) should be func_name(&self, ...)
        # But only for calls, not for declarations
        count5 = 0
        for func_name in ptr_self_funcs:
            # Match calls like: func_name(self, or func_name(self)
            # Need negative lookbehind to avoid matching declarations
            # Also need to handle calls that are not at the start of line
            call_pattern = rf'({re.escape(func_name)})\(self([,\)])'
            replacement = rf'\1(&self\2'
            matches = len(re.findall(call_pattern, content))
            if matches:
                content = re.sub(call_pattern, replacement, content)
                count5 += matches

        if count5:
            print(f"Fixed {count5} 'self' -> '&self' for mut self methods", file=sys.stderr)
            fixed_count += count5

    # Fix any remaining self that should be &self in mut self method calls
    # Look for patterns like: _____method(self, where method takes SigilValue*
    # This is a broader pattern to catch any we missed
    remaining_pattern = r'(sigil_\w+____\w+)\(self([,\)])'
    # Check if these functions are in our list and fix them
    for match in re.finditer(remaining_pattern, content):
        func_name = match.group(1)
        if func_name in ptr_self_funcs:
            # Already handled above, but let's count if any remain
            pass

    # Fix 6: For functions that take SigilValue self (by value), when called with
    # self in a context where self is SigilValue*, we need (*self)
    #
    # Context detection: If nearby lines use &self, then self is a pointer in this scope
    # This is complex - for now, we'll handle this case-by-case based on error patterns
    #
    # The key patterns we've seen:
    # 1. emit_pattern_condition(self, ...) where self is SigilValue* in context
    # 2. These calls appear near other calls that use &self
    #
    # Heuristic: Look for lines with func(self, that are in functions with SigilValue* self
    # by checking if the containing function definition has SigilValue* self

    print(f"Total fixes: {fixed_count}", file=sys.stderr)
    return content

def main():
    if len(sys.argv) < 2:
        print("Usage: fix_codegen_bugs.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r', encoding='latin-1') as f:
        content = f.read()

    fixed = fix_codegen_bugs(content)

    with open(output_path, 'w', encoding='latin-1') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
