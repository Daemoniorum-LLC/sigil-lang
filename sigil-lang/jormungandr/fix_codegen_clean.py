#!/usr/bin/env python3
"""
Fix codegen bugs WITHOUT the problematic self -> &self logic.
"""

import re
import sys

def fix_codegen_clean(content):
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
    pattern2 = r'\?(\))'
    count2 = len(re.findall(pattern2, content))
    content = re.sub(pattern2, r'\1', content)
    if count2:
        print(f"Fixed {count2} stray '?' operators", file=sys.stderr)
        fixed_count += count2

    # Fix 3: Method calls like .to_uppercase()
    method_fixes = [
        (r'(\w+)\.to_uppercase\(\)', r'sigil_String____to_uppercase(\1)'),
        (r'(\w+)\.to_lowercase\(\)', r'sigil_String____to_lowercase(\1)'),
        (r'(\w+)\.trim\(\)', r'sigil_String____trim(\1)'),
        (r'(\w+)\.is_empty\(\)', r'sigil_String____is_empty(\1)'),
        (r'(\w+)\.to_string\(\)', r'sigil_to_string(\1)'),
        # Note: .len() might be for arrays/strings - use sigil_len
        (r'(\w+)\.len\(\)', r'sigil_len(\1)'),
        # .name field access - use sigil_struct_field
        (r'(\w+)\.name(?!\s*=)', r'sigil_struct_field(\1, "name")'),
    ]
    for pattern, replacement in method_fixes:
        count = len(re.findall(pattern, content))
        if count:
            content = re.sub(pattern, replacement, content)
            print(f"Fixed {count} method call patterns", file=sys.stderr)
            fixed_count += count

    # Fix 4: Truncated function names
    truncated_funcs = [
        (r'\bsigil_with_evide\b(?!nce)', 'sigil_with_evidence'),
    ]
    for pattern, replacement in truncated_funcs:
        count = len(re.findall(pattern, content))
        if count:
            content = re.sub(pattern, replacement, content)
            print(f"Fixed {count} truncated function names", file=sys.stderr)
            fixed_count += count

    # Fix 5: 'default' is a C keyword
    default_patterns = [
        (r'\bdefault([,;\)])', r'default_val\1'),
        (r'=\s*default\b', r'= default_val'),
        (r'\bdefault\s*=', r'default_val ='),
    ]
    count5 = 0
    for pattern, replacement in default_patterns:
        count = len(re.findall(pattern, content))
        if count:
            content = re.sub(pattern, replacement, content)
            count5 += count
    if count5:
        print(f"Fixed {count5} 'default' C keyword conflicts", file=sys.stderr)
        fixed_count += count5

    # Fix 6: Calls with &_tNNN where function expects SigilValue (not pointer)
    # Pattern: sigil_CodeGen____emit_pattern_condition(&_t..., should be emit_pattern_condition(_t...,
    # when emit_pattern_condition takes SigilValue (not SigilValue*)
    pattern6a = r'sigil_CodeGen____emit_pattern_condition\(&(_t\d+),'
    count6a = len(re.findall(pattern6a, content))
    content = re.sub(pattern6a, r'sigil_CodeGen____emit_pattern_condition(\1,', content)
    if count6a:
        print(f"Fixed {count6a} '&_tN' to '_tN' for value-self methods", file=sys.stderr)
        fixed_count += count6a

    print(f"Total fixes: {fixed_count}", file=sys.stderr)
    return content

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: fix_codegen_clean.py <file.c> [output.c]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, 'r', encoding='latin-1') as f:
        content = f.read()

    fixed = fix_codegen_clean(content)

    with open(output_path, 'w', encoding='latin-1') as f:
        f.write(fixed)

    print(f"Output written to {output_path}")
