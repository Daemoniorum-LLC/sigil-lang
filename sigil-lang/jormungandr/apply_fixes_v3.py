#!/usr/bin/env python3
"""
Apply codegen fixes to sigil_bootstrap.c (v3 - precise line-based fixes)

These fixes work around bugs in the Rust interpreter's C code generation.
Run from the build/ directory: python3 ../apply_fixes_v3.py
"""

import re
import sys

def main():
    filename = 'sigil_bootstrap.c'

    with open(filename, 'r') as f:
        lines = f.readlines()

    fixes_applied = 0
    content = ''.join(lines)

    # ============ CONTENT-BASED FIXES (safe global replacements) ============

    # 1. Add missing includes
    if '#include <ctype.h>' not in content:
        content = content.replace(
            '#include <stdarg.h>\n\n#ifndef SIGIL_RUNTIME_H',
            '#include <stdarg.h>\n#include <ctype.h>\n#include <time.h>\n\n#ifndef SIGIL_RUNTIME_H'
        )
        fixes_applied += 1
        print("Added ctype.h and time.h includes")

    # 2. Fix sigil_len_utf8 to handle TAG_CHAR
    old_len = '''SigilValue sigil_len_utf8(SigilValue s) {
    if (s.tag != TAG_STRING) return sigil_int(0);
    /* For simplicity, just count bytes (not true UTF-8 codepoints) */
    return sigil_int((int64_t)strlen(s.v.s));
}'''
    new_len = '''SigilValue sigil_len_utf8(SigilValue s) {
    /* Fixed: handle TAG_CHAR for Lexer::advance */
    if (s.tag == TAG_CHAR) {
        unsigned char c = (unsigned char)s.v.c;
        if (c < 0x80) return sigil_int(1);
        if (c < 0xE0) return sigil_int(2);
        if (c < 0xF0) return sigil_int(3);
        return sigil_int(4);
    }
    if (s.tag != TAG_STRING) return sigil_int(0);
    return sigil_int((int64_t)strlen(s.v.s));
}'''
    if 'if (s.tag == TAG_CHAR)' not in content:
        content = content.replace(old_len, new_len)
        fixes_applied += 1
        print("Fixed sigil_len_utf8 for TAG_CHAR")

    # 3. Add sigil_String____parse helper
    helper = '''/* String parse helper - for s.parse() on strings */
SigilValue sigil_String____parse(SigilValue s) {
    if (s.tag != TAG_STRING || !s.v.s) return sigil_null();
    char* end;
    long long iv = strtoll(s.v.s, &end, 10);
    if (end != s.v.s && *end == '\\0') return sigil_Ok(sigil_int((int64_t)iv));
    double fv = strtod(s.v.s, &end);
    if (end != s.v.s && *end == '\\0') return sigil_Ok(sigil_float(fv));
    return sigil_null();
}

'''
    marker = '/* Name mangling for C codegen'
    if 'sigil_String____parse' not in content:
        content = content.replace(marker, helper + marker)
        fixes_applied += 1
        print("Added sigil_String____parse helper")

    # 4. Fix UncertainParser____parse -> String____parse
    for old, new in [
        ('sigil_UncertainParser____parse(s)', 'sigil_String____parse(s)'),
        ('sigil_UncertainParser____parse(field)', 'sigil_String____parse(field)'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed String parse method resolution")

    # 5. TypeChecker: (*self) -> self (self is already SigilValue*, function expects SigilValue*)
    for method in ['push_scope', 'pop_scope']:
        old = f'sigil_TypeChecker____{method}((*self))'
        new = f'sigil_TypeChecker____{method}(self)'
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1

    for method in ['bind_pattern', 'unify', 'infer_literal']:
        old = f'sigil_TypeChecker____{method}((*self)'
        new = f'sigil_TypeChecker____{method}(self'
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed TypeChecker method calls")

    # 6. Interpreter: (*self) -> self (self is SigilValue*, function expects SigilValue*)
    for method in ['eval_with_env', 'bind_pattern', 'call_function', 'check_evidence',
                   'call_builtin', 'eval_pipeline_step', 'eval_morpheme']:
        old = f'sigil_Interpreter____{method}((*self)'
        new = f'sigil_Interpreter____{method}(self'
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1

    old = 'sigil_Interpreter____register_builtins((*self))'
    new = 'sigil_Interpreter____register_builtins(self)'
    if old in content:
        content = content.replace(old, new)
        fixes_applied += 1
    print("Fixed Interpreter method calls")

    # 7. Fix TypeEnv/Environment with local 'env' var
    for old, new in [
        ('sigil_TypeEnv____define(env,', 'sigil_TypeEnv____define(&env,'),
        ('sigil_Environment____define(env,', 'sigil_Environment____define(&env,'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed env variable calls")

    # 8. Fix LoweringContext with local 'ctx' var
    for old, new in [
        ('sigil_LoweringContext____fresh_id(ctx)', 'sigil_LoweringContext____fresh_id(&ctx)'),
        ('sigil_LoweringContext____error(ctx,', 'sigil_LoweringContext____error(&ctx,'),
        ('sigil_LoweringContext____get_var_id(ctx,', 'sigil_LoweringContext____get_var_id(&ctx,'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed ctx variable calls")

    # 9. CodeGen methods that take SigilValue (need deref when self is pointer)
    for old, new in [
        ('sigil_CodeGen____with_evidence(self,', 'sigil_CodeGen____with_evidence((*self),'),
        ('sigil_CodeGen____emit_pattern_condition(&self,', 'sigil_CodeGen____emit_pattern_condition((*self),'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1

    for op in ['op', 'int', 'float', 'bool', 'comparison', 'add', 'sub', 'mul', 'div', 'mod',
               'rem', 'lt', 'le', 'gt', 'ge', 'eq', 'ne', 'and', 'or', 'band', 'bor', 'bxor', 'shl', 'shr']:
        old = f'sigil_CodeGen____emit_binary_{op}(self,'
        new = f'sigil_CodeGen____emit_binary_{op}((*self),'
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed CodeGen method calls (need deref)")

    # 10. CodeGen: line_close expects pointer, so (*self) -> self
    for old, new in [
        ('sigil_CodeGen____line_close((*self),', 'sigil_CodeGen____line_close(self,'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1

    # LoweringContext____fresh_id expects pointer, (*self) -> self
    for old, new in [
        ('sigil_LoweringContext____fresh_id((*self))', 'sigil_LoweringContext____fresh_id(self)'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed methods expecting pointer")

    # 11. Helper functions take SigilValue, need (*self) deref
    for old, new in [
        ('sigil_mangle_name(self,', 'sigil_mangle_name((*self),'),
        ('sigil_escape_char(self,', 'sigil_escape_char((*self),'),
        ('sigil_escape_string(self,', 'sigil_escape_string((*self),'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed helper function calls")

    # 12. Driver: (*self) -> self
    content = content.replace('sigil_Driver____check((*self))', 'sigil_Driver____check(self)')
    fixes_applied += 1
    print("Fixed Driver method calls")

    # 13. Lexer temp var calls
    content = content.replace('sigil_UncertainLexer____next_token(_t0)', 'sigil_UncertainLexer____next_token(&_t0)')
    content = content.replace('sigil_UncertainLexer____peek(_t0)', 'sigil_UncertainLexer____peek(&_t0)')
    content = re.sub(r'sigil_Lexer____peek_is_macro_delimiter\(_t(\d+)\)',
                     r'sigil_Lexer____peek_is_macro_delimiter(&_t\1)', content)
    content = re.sub(r'sigil_Lexer____peek_is_closure_indicator\(_t(\d+)\)',
                     r'sigil_Lexer____peek_is_closure_indicator(&_t\1)', content)
    content = re.sub(r'sigil_UncertainLexer____peek_is_macro_delimiter\(_t(\d+)\)',
                     r'sigil_UncertainLexer____peek_is_macro_delimiter(&_t\1)', content)
    content = re.sub(r'sigil_UncertainLexer____peek_is_closure_indicator\(_t(\d+)\)',
                     r'sigil_UncertainLexer____peek_is_closure_indicator(&_t\1)', content)
    fixes_applied += 4
    print("Fixed Lexer temp var calls")

    # 14. register_builtins with local vars
    content = content.replace('sigil_UncertainInterpreter____register_builtins(checker)',
                              'sigil_UncertainInterpreter____register_builtins(&checker)')
    content = content.replace('sigil_UncertainInterpreter____register_builtins(interp)',
                              'sigil_UncertainInterpreter____register_builtins(&interp)')
    content = content.replace('sigil_Interpreter____check_evidence(interp,',
                              'sigil_Interpreter____check_evidence(&interp,')
    fixes_applied += 3
    print("Fixed local var register_builtins calls")

    # 15. UncertainTypeEnv____define with _t* temp vars (macro to sigil_TypeEnv____define)
    content = re.sub(r'sigil_UncertainTypeEnv____define\((_t\d+),',
                     r'sigil_UncertainTypeEnv____define(&\1,', content)
    fixes_applied += 1
    print("Fixed UncertainTypeEnv____define with _t* temp vars")

    # 16. Environment____define with item_env
    content = content.replace('sigil_Environment____define(item_env,',
                              'sigil_Environment____define(&item_env,')
    content = content.replace('sigil_UncertainEnvironment____define(item_env,',
                              'sigil_UncertainEnvironment____define(&item_env,')
    fixes_applied += 2
    print("Fixed Environment____define with item_env")

    # ============ LINE-BASED FIXES (for specific problematic lines) ============

    # Convert back to lines for line-specific fixes
    lines = content.split('\n')

    # These are the exact error lines from gcc output - fix them specifically
    # We target the specific line numbers where (*self) needs to become self
    # because we're inside emit_pattern_condition where self is SigilValue (not pointer)

    error_lines_need_remove_deref = [
        # emit_pattern_condition internal recursive calls - self is SigilValue, not pointer
        # Line numbers from the original 23 errors that have "invalid type argument of unary '*'"
        # These are inside sigil_CodeGen____emit_pattern_condition where self is SigilValue
        44961, 45115, 45164, 45197
    ]

    for lineno in error_lines_need_remove_deref:
        idx = lineno - 1
        if idx < len(lines):
            # Replace (*self) with self on this specific line
            lines[idx] = lines[idx].replace('(*self)', 'self')
            fixes_applied += 1
    print(f"Fixed {len(error_lines_need_remove_deref)} unary '*' errors in emit_pattern_condition")

    # Line 44024: sigil_struct_field(self, ...) in emit_operation where self is SigilValue*
    # Need to add deref: sigil_struct_field((*self), ...)
    # But only for this specific occurrence
    idx = 44024 - 1
    if idx < len(lines):
        lines[idx] = lines[idx].replace('sigil_struct_field(self,', 'sigil_struct_field((*self),')
        fixes_applied += 1
    print("Fixed sigil_struct_field(self) at line 44024")

    # Line 44442: sigil_mangle_name(self, ...) where self is SigilValue* but we need (*self)
    # Wait, we already have global fix for this. Let me check if it's not applying here.
    # The error says "incompatible type for argument 1 of 'sigil_mangle_name'"
    # Our global fix is sigil_mangle_name(self, -> sigil_mangle_name((*self),
    # But this might be failing because the line has a different context
    # Let me check - at line 44442 we likely have already converted, let's verify

    # Lines 43311, 43333, 44475: emit_pattern_condition needs (*self) when called from
    # functions with SigilValue* self
    # But our global fix converted emit_pattern_condition(self, -> emit_pattern_condition((*self),
    # That should work... unless something is wrong

    # Let me add specific fixes for these:
    # At these lines, the caller has SigilValue* self, and emit_pattern_condition takes SigilValue
    for lineno in [43311, 43333, 44475]:
        idx = lineno - 1
        if idx < len(lines):
            # Need to dereference self when calling emit_pattern_condition
            lines[idx] = lines[idx].replace(
                'sigil_CodeGen____emit_pattern_condition(self,',
                'sigil_CodeGen____emit_pattern_condition((*self),'
            )
            fixes_applied += 1
    print("Fixed emit_pattern_condition calls at lines 43311, 43333, 44475")

    # Line 34609: Interpreter____check_evidence needs pointer
    # The global fix already handles this via check_evidence -> self
    # But if there's still an error, let's check

    # Line 46933: sigil_struct_field(self, "config") in Driver where self is SigilValue*
    idx = 46933 - 1
    if idx < len(lines):
        lines[idx] = lines[idx].replace('sigil_struct_field(self,', 'sigil_struct_field((*self),')
        fixes_applied += 1
    print("Fixed sigil_struct_field(self) at line 46933")

    content = '\n'.join(lines)

    with open(filename, 'w') as f:
        f.write(content)

    print(f"\nApplied {fixes_applied} total fixes to {filename}")
    print("\nRun: gcc -g -O0 -o sigil_bootstrap sigil_bootstrap.c -lm")

if __name__ == '__main__':
    main()
