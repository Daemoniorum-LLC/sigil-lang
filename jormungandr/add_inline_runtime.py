#!/usr/bin/env python3
"""
CG-118: Add inline runtime implementations to sigil2_fixed.c

The self-compiled sigil2_fixed.c only has forward declarations for runtime
functions like sigil_Vec____new, sigil_String____new, etc. This script extracts
the inline implementations from sigil_combined.c and adds them.
"""

import sys
import re

def extract_inline_runtime(combined_path):
    """Extract the inline runtime section from sigil_combined.c"""
    with open(combined_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find the forward declarations for functions used by runtime (lines 208-211)
    fwd_decl_start = None
    inline_start = None
    end_idx = None

    for i, line in enumerate(lines):
        if '/* Forward declarations for functions used by runtime */' in line:
            fwd_decl_start = i
        if '/* === Inline Runtime Implementations === */' in line:
            inline_start = i
        # End at the IrEvidence method declarations
        if inline_start is not None and line.startswith('/* IrEvidence method declarations'):
            end_idx = i
            break

    # If we found both markers, use them
    if fwd_decl_start is not None and inline_start is not None:
        start_idx = fwd_decl_start
    elif inline_start is not None:
        start_idx = inline_start
    else:
        # Fallback: look for sigil_array_with_header definition
        for i, line in enumerate(lines):
            if 'static inline SigilValue sigil_array_with_header' in line:
                start_idx = i - 1  # Include comment
                break
        else:
            return None

        # Find end: look for "/* Built-in functions */" or "/* IrEvidence"
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith('/* IrEvidence') or lines[i].startswith('/* Built-in functions'):
                end_idx = i
                break

    if end_idx is None:
        end_idx = start_idx + 300  # Reasonable default

    return '\n'.join(lines[start_idx:end_idx])

def add_inline_runtime(sigil2_path, inline_runtime, output_path=None):
    """Add inline runtime to sigil2_fixed.c after value constructors"""
    with open(sigil2_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Find where to insert: after sigil_is_err definition, before forward declarations
    insert_idx = None
    for i, line in enumerate(lines):
        # Look for the end of basic runtime (after sigil_is_err, before I/O section)
        if 'static inline bool sigil_is_err' in line:
            # Find the next blank line or section header
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == '' or lines[j].startswith('/*'):
                    insert_idx = j
                    break
            break

    if insert_idx is None:
        # Fallback: insert after #endif /* SIGIL_RUNTIME_H */
        for i, line in enumerate(lines):
            if '#endif' in line and 'SIGIL_RUNTIME_H' in line:
                insert_idx = i + 1
                break

    if insert_idx is None:
        # Last resort: insert at line 200
        insert_idx = 200

    # Insert the inline runtime
    new_lines = lines[:insert_idx] + ['', inline_runtime, ''] + lines[insert_idx:]

    output_path = output_path or sigil2_path
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_lines))

    print(f"CG-118: Inserted inline runtime at line {insert_idx}")
    return insert_idx

def main():
    if len(sys.argv) < 2:
        print("Usage: add_inline_runtime.py <sigil2_fixed.c> [sigil_combined.c] [output.c]")
        sys.exit(1)

    sigil2_path = sys.argv[1]
    combined_path = sys.argv[2] if len(sys.argv) > 2 else 'build/sigil_combined.c'
    output_path = sys.argv[3] if len(sys.argv) > 3 else sigil2_path

    inline_runtime = extract_inline_runtime(combined_path)
    if inline_runtime is None:
        print("Error: Could not extract inline runtime")
        sys.exit(1)

    print(f"Extracted {len(inline_runtime.split(chr(10)))} lines of inline runtime")

    add_inline_runtime(sigil2_path, inline_runtime, output_path)
    print(f"Output written to {output_path}")

if __name__ == '__main__':
    main()
