#!/usr/bin/env python3
"""
Apply codegen fixes to sigil_bootstrap.c (v2 - more comprehensive)

These fixes work around bugs in the Rust interpreter's C code generation.
Run from the build/ directory: python3 ../apply_fixes_v2.py
"""

import re
import sys

def main():
    filename = 'sigil_bootstrap.c'

    with open(filename, 'r') as f:
        content = f.read()

    fixes_applied = 0

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

    # 5. Fix TypeChecker methods: (*self) -> self (they expect pointer, self is already pointer)
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
    print("Fixed TypeChecker method calls (deref to pointer)")

    # 6. Fix Interpreter methods: (*self) -> self (they expect pointer)
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

    # 7. Fix Environment/TypeEnv with local 'env' var (need &env)
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

    # 9. Fix CodeGen methods that take SigilValue (not pointer)
    # These functions have SigilValue* self, but call functions expecting SigilValue
    # So we need (*self) to dereference the pointer
    for old, new in [
        ('sigil_CodeGen____with_evidence(self,', 'sigil_CodeGen____with_evidence((*self),'),
        ('sigil_CodeGen____emit_pattern_condition(&self,', 'sigil_CodeGen____emit_pattern_condition((*self),'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1

    # emit_binary_* methods take SigilValue, need (*self) when called with SigilValue* self
    for op in ['op', 'int', 'float', 'bool', 'comparison', 'add', 'sub', 'mul', 'div', 'mod',
               'rem', 'lt', 'le', 'gt', 'ge', 'eq', 'ne', 'and', 'or', 'band', 'bor', 'bxor', 'shl', 'shr']:
        old = f'sigil_CodeGen____emit_binary_{op}(self,'
        new = f'sigil_CodeGen____emit_binary_{op}((*self),'
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed CodeGen method calls (need deref)")

    # 10. Fix CodeGen methods that take SigilValue* (need self, not (*self))
    # line_close expects SigilValue*, so when called with (*self) it gets SigilValue - wrong
    for old, new in [
        ('sigil_CodeGen____line_close((*self),', 'sigil_CodeGen____line_close(self,'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1

    # LoweringContext____fresh_id expects SigilValue*, (*self) is wrong
    for old, new in [
        ('sigil_LoweringContext____fresh_id((*self))', 'sigil_LoweringContext____fresh_id(self)'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed methods expecting pointer (remove deref)")

    # 11. Fix helper functions: self -> (*self) (they expect SigilValue)
    for old, new in [
        ('sigil_mangle_name(self,', 'sigil_mangle_name((*self),'),
        ('sigil_escape_char(self,', 'sigil_escape_char((*self),'),
        ('sigil_escape_string(self,', 'sigil_escape_string((*self),'),
    ]:
        if old in content:
            content = content.replace(old, new)
            fixes_applied += 1
    print("Fixed helper function calls")

    # 12. Fix Driver: (*self) -> self
    content = content.replace('sigil_Driver____check((*self))', 'sigil_Driver____check(self)')
    fixes_applied += 1
    print("Fixed Driver method calls")

    # 13. Fix Lexer temp var calls
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

    # 14. Fix register_builtins with local vars
    content = content.replace('sigil_UncertainInterpreter____register_builtins(checker)',
                              'sigil_UncertainInterpreter____register_builtins(&checker)')
    content = content.replace('sigil_UncertainInterpreter____register_builtins(interp)',
                              'sigil_UncertainInterpreter____register_builtins(&interp)')
    content = content.replace('sigil_Interpreter____check_evidence(interp,',
                              'sigil_Interpreter____check_evidence(&interp,')
    fixes_applied += 3
    print("Fixed local var register_builtins calls")

    # 15. NEW: Fix UncertainTypeEnv____define with _t* temp vars
    # These pass a SigilValue to a macro that calls sigil_TypeEnv____define expecting SigilValue*
    content = re.sub(r'sigil_UncertainTypeEnv____define\((_t\d+),',
                     r'sigil_UncertainTypeEnv____define(&\1,', content)
    fixes_applied += 1
    print("Fixed UncertainTypeEnv____define with _t* temp vars")

    # 16. NEW: Fix Environment____define with local vars (item_env, etc.)
    content = content.replace('sigil_Environment____define(item_env,',
                              'sigil_Environment____define(&item_env,')
    content = content.replace('sigil_UncertainEnvironment____define(item_env,',
                              'sigil_UncertainEnvironment____define(&item_env,')
    fixes_applied += 2
    print("Fixed Environment____define with item_env")

    # 17. NEW: Fix emit_pattern_condition calls inside emit_pattern_condition
    # Inside sigil_CodeGen____emit_pattern_condition, self is SigilValue (not pointer)
    # So (*self) is invalid - need just self
    # These are recursive calls within the function
    content = re.sub(
        r'sigil_CodeGen____emit_pattern_condition\(\(\*self\),',
        r'sigil_CodeGen____emit_pattern_condition(self,',
        content
    )
    fixes_applied += 1
    print("Fixed recursive emit_pattern_condition calls")

    # 18. NEW: Fix sigil_struct_field(self, ...) in Driver functions
    # In sigil_Driver____compile, self is SigilValue*, need (*self)
    # But we need to be careful - only fix in Driver functions
    # Actually let's fix the specific pattern at line 46933
    content = content.replace(
        'sigil_struct_field(sigil_struct_field(self, "config"), "input_files")',
        'sigil_struct_field(sigil_struct_field((*self), "config"), "input_files")'
    )
    fixes_applied += 1
    print("Fixed sigil_struct_field(self) in Driver")

    with open(filename, 'w') as f:
        f.write(content)

    print(f"\nApplied {fixes_applied} fixes to {filename}")
    print("\nRun: gcc -g -O0 -o sigil_bootstrap sigil_bootstrap.c -lm")

if __name__ == '__main__':
    main()
