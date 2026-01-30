#!/usr/bin/env python3
"""
Apply ALL codegen fixes to sigil_bootstrap.c

These fixes work around bugs in the Rust interpreter's C code generation.
Run from the build/ directory: python3 ../apply_all_fixes.py
"""

import re
import sys

def main():
    filename = 'sigil_bootstrap.c'

    with open(filename, 'r') as f:
        content = f.read()

    fixes_applied = 0

    # ============ PHASE 1: Add missing runtime support ============

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

    # ============ PHASE 2: Fix Lexer calls ============
    # Lexer methods expect SigilValue* but are called with SigilValue

    # Fix Lexer with _t temp vars
    content = re.sub(r'sigil_Lexer____next_token\((_t\d+)\)', r'sigil_Lexer____next_token(&\1)', content)
    content = re.sub(r'sigil_UncertainLexer____next_token\((_t\d+)\)', r'sigil_UncertainLexer____next_token(&\1)', content)
    content = re.sub(r'sigil_Lexer____peek\((_t\d+)\)', r'sigil_Lexer____peek(&\1)', content)
    content = re.sub(r'sigil_UncertainLexer____peek\((_t\d+)\)', r'sigil_UncertainLexer____peek(&\1)', content)
    content = re.sub(r'sigil_Lexer____peek_is_macro_delimiter\((_t\d+)\)', r'sigil_Lexer____peek_is_macro_delimiter(&\1)', content)
    content = re.sub(r'sigil_UncertainLexer____peek_is_macro_delimiter\((_t\d+)\)', r'sigil_UncertainLexer____peek_is_macro_delimiter(&\1)', content)
    content = re.sub(r'sigil_Lexer____peek_is_closure_indicator\((_t\d+)\)', r'sigil_Lexer____peek_is_closure_indicator(&\1)', content)
    content = re.sub(r'sigil_UncertainLexer____peek_is_closure_indicator\((_t\d+)\)', r'sigil_UncertainLexer____peek_is_closure_indicator(&\1)', content)
    fixes_applied += 8
    print("Fixed Lexer method calls")

    # ============ PHASE 3: Fix Parser calls ============
    # Parser::parse expects SigilValue*, called with SigilValue
    content = re.sub(r'sigil_Parser____parse\((_t\d+)\)', r'sigil_Parser____parse(&\1)', content)
    content = re.sub(r'sigil_UncertainParser____parse\((_t\d+)\)', r'sigil_UncertainParser____parse(&\1)', content)
    fixes_applied += 2
    print("Fixed Parser method calls")

    # ============ PHASE 4: Fix TypeChecker calls ============
    # TypeChecker methods expect SigilValue* but generated code passes (*self) or self incorrectly

    # In functions with SigilValue* self, calls like func((*self)) should be func(self)
    for method in ['push_scope', 'pop_scope']:
        content = content.replace(f'sigil_TypeChecker____{method}((*self))', f'sigil_TypeChecker____{method}(self)')
        content = content.replace(f'sigil_UncertainTypeChecker____{method}((*self))', f'sigil_UncertainTypeChecker____{method}(self)')

    # bind_pattern and unify expect SigilValue*, so (*self) -> self
    for method in ['bind_pattern', 'unify']:
        content = content.replace(f'sigil_TypeChecker____{method}((*self)', f'sigil_TypeChecker____{method}(self')
        content = content.replace(f'sigil_UncertainTypeChecker____{method}((*self)', f'sigil_UncertainTypeChecker____{method}(self')

    # infer_literal expects SigilValue (not pointer), so self -> (*self) when self is pointer
    content = content.replace('sigil_TypeChecker____infer_literal(self,', 'sigil_TypeChecker____infer_literal((*self),')

    # Fix closures where self is SigilValue (not pointer) - need &self
    # These are static closures like sigil_typeck_closure_*
    content = re.sub(
        r'(sigil_typeck_closure_\d+\(SigilValue self,.*?\n.*?)sigil_TypeChecker____unify\(self,',
        r'\1sigil_TypeChecker____unify(&self,',
        content,
        flags=re.DOTALL
    )
    fixes_applied += 5
    print("Fixed TypeChecker method calls")

    # ============ PHASE 5: Fix Interpreter calls ============
    # Interpreter methods expect SigilValue* but are called with (*self)

    for method in ['register_builtins']:
        content = content.replace(f'sigil_Interpreter____{method}((*self))', f'sigil_Interpreter____{method}(self)')
        content = content.replace(f'sigil_UncertainInterpreter____{method}((*self))', f'sigil_UncertainInterpreter____{method}(self)')

    # Most Interpreter methods expect SigilValue*, so (*self) -> self
    for method in ['eval_with_env', 'bind_pattern', 'call_function',
                   'call_builtin', 'eval_pipeline_step', 'eval_morpheme']:
        content = content.replace(f'sigil_Interpreter____{method}((*self)', f'sigil_Interpreter____{method}(self')
        content = content.replace(f'sigil_UncertainInterpreter____{method}((*self)', f'sigil_UncertainInterpreter____{method}(self')

    # BUT: Interpreter::check_evidence expects SigilValue (not pointer), so self -> (*self)
    content = content.replace('sigil_Interpreter____check_evidence(self,', 'sigil_Interpreter____check_evidence((*self),')

    # Fix local var calls (checker, interp are SigilValue, need &)
    content = content.replace('sigil_Interpreter____register_builtins(checker)', 'sigil_Interpreter____register_builtins(&checker)')
    content = content.replace('sigil_UncertainInterpreter____register_builtins(checker)', 'sigil_UncertainInterpreter____register_builtins(&checker)')
    content = content.replace('sigil_Interpreter____register_builtins(interp)', 'sigil_Interpreter____register_builtins(&interp)')
    content = content.replace('sigil_UncertainInterpreter____register_builtins(interp)', 'sigil_UncertainInterpreter____register_builtins(&interp)')
    content = content.replace('sigil_Interpreter____check_evidence(interp,', 'sigil_Interpreter____check_evidence(&interp,')
    fixes_applied += 8
    print("Fixed Interpreter method calls")

    # ============ PHASE 6: Fix TypeEnv/Environment calls ============
    # These expect SigilValue* but are called with SigilValue

    # Fix local 'env' var
    content = content.replace('sigil_TypeEnv____define(env,', 'sigil_TypeEnv____define(&env,')
    content = content.replace('sigil_Environment____define(env,', 'sigil_Environment____define(&env,')
    content = content.replace('sigil_UncertainTypeEnv____define(env,', 'sigil_UncertainTypeEnv____define(&env,')
    content = content.replace('sigil_UncertainEnvironment____define(env,', 'sigil_UncertainEnvironment____define(&env,')

    # Fix local 'item_env' var
    content = content.replace('sigil_Environment____define(item_env,', 'sigil_Environment____define(&item_env,')
    content = content.replace('sigil_UncertainEnvironment____define(item_env,', 'sigil_UncertainEnvironment____define(&item_env,')

    # Fix _t* temp vars passed to UncertainTypeEnv/Environment
    # BUT: Only fix 4-argument calls, not 3-argument calls
    # 4-arg pattern: define(_t*, name, ty, evidence) - needs &_t*
    # 3-arg pattern: define(_t*, string, sigil_EvidentialValue____known(...)) - no change needed
    # The 3-arg macro passes self to sigil_struct_field which expects SigilValue
    # The 4-arg macro passes self to sigil_TypeEnv____define which expects SigilValue*

    # Only match 4-argument calls (where 4th arg is identifier or simple call, not sigil_EvidentialValue____known)
    # Pattern: define(_t*, ..., ..., evidence) or define(_t*, ..., ..., EvidenceLevel____Known())
    content = re.sub(
        r'sigil_UncertainTypeEnv____define\((_t\d+), ([^,]+), ([^,]+), ([^)]+(?:evidence|Evidence[^)]*\(\)))\)',
        r'sigil_UncertainTypeEnv____define(&\1, \2, \3, \4)',
        content
    )
    # Also fix direct TypeEnv____define calls with _t* args (4-arg version)
    content = re.sub(
        r'sigil_TypeEnv____define\((_t\d+), ([^,]+), ([^,]+), ([^)]+(?:evidence|Evidence[^)]*\(\)))\)',
        r'sigil_TypeEnv____define(&\1, \2, \3, \4)',
        content
    )
    # Environment define always expects pointer
    content = re.sub(r'sigil_UncertainEnvironment____define\((_t\d+),', r'sigil_UncertainEnvironment____define(&\1,', content)
    content = re.sub(r'sigil_Environment____define\((_t\d+),', r'sigil_Environment____define(&\1,', content)
    fixes_applied += 10
    print("Fixed TypeEnv/Environment method calls")

    # ============ PHASE 7: Fix LoweringContext calls ============
    # These expect SigilValue* but are called with (*self) or local vars

    content = content.replace('sigil_LoweringContext____fresh_id((*self))', 'sigil_LoweringContext____fresh_id(self)')
    content = content.replace('sigil_LoweringContext____error((*self),', 'sigil_LoweringContext____error(self,')
    content = content.replace('sigil_LoweringContext____get_var_id((*self),', 'sigil_LoweringContext____get_var_id(self,')

    # Fix local 'ctx' var
    content = content.replace('sigil_LoweringContext____fresh_id(ctx)', 'sigil_LoweringContext____fresh_id(&ctx)')
    content = content.replace('sigil_LoweringContext____error(ctx,', 'sigil_LoweringContext____error(&ctx,')
    content = content.replace('sigil_LoweringContext____get_var_id(ctx,', 'sigil_LoweringContext____get_var_id(&ctx,')
    fixes_applied += 6
    print("Fixed LoweringContext method calls")

    # ============ PHASE 8: Fix CodeGen calls ============
    # CodeGen methods have mixed conventions

    # line_close expects SigilValue*, so (*self) -> self
    content = content.replace('sigil_CodeGen____line_close((*self),', 'sigil_CodeGen____line_close(self,')

    # These methods expect SigilValue, so self -> (*self) when self is pointer
    for old, new in [
        ('sigil_CodeGen____with_evidence(self,', 'sigil_CodeGen____with_evidence((*self),'),
        ('sigil_CodeGen____emit_pattern_condition(self,', 'sigil_CodeGen____emit_pattern_condition((*self),'),
    ]:
        content = content.replace(old, new)

    # emit_binary_* methods take SigilValue
    for op in ['op', 'int', 'float', 'bool', 'comparison', 'add', 'sub', 'mul', 'div', 'mod',
               'rem', 'lt', 'le', 'gt', 'ge', 'eq', 'ne', 'and', 'or', 'band', 'bor', 'bxor', 'shl', 'shr']:
        content = content.replace(f'sigil_CodeGen____emit_binary_{op}(self,', f'sigil_CodeGen____emit_binary_{op}((*self),')
    fixes_applied += 20
    print("Fixed CodeGen method calls")

    # ============ PHASE 9: Fix helper functions ============
    # These expect SigilValue, so self -> (*self) when self is pointer
    # Handle optional whitespace after (

    content = re.sub(r'sigil_mangle_name\(\s*self,', r'sigil_mangle_name((*self),', content)
    content = re.sub(r'sigil_escape_char\(\s*self,', r'sigil_escape_char((*self),', content)
    content = re.sub(r'sigil_escape_string\(\s*self,', r'sigil_escape_string((*self),', content)
    fixes_applied += 3
    print("Fixed helper function calls")

    # ============ PHASE 10: Fix Driver calls ============
    content = content.replace('sigil_Driver____check((*self))', 'sigil_Driver____check(self)')
    fixes_applied += 1
    print("Fixed Driver method calls")

    # ============ PHASE 11: Fix struct_field calls ============
    # In CodeGen emit_operation, sigil_struct_field(self,...) needs (*self)
    # But we need to be careful not to break other uses
    # The specific error is at line 44004 in emit_operation
    # Let's target this specifically

    # Fix sigil_struct_field(self, "temp_counter") pattern in emit_operation
    # This appears when generating closure code
    content = content.replace(
        'sigil_struct_field(self, "temp_counter")',
        'sigil_struct_field((*self), "temp_counter")'
    )

    # Fix sigil_struct_field(self, "config") in Driver
    content = content.replace(
        'sigil_struct_field(sigil_struct_field(self, "config"), "input_files")',
        'sigil_struct_field(sigil_struct_field((*self), "config"), "input_files")'
    )
    fixes_applied += 2
    print("Fixed struct_field calls")

    # ============ PHASE 12: Fix recursive emit_pattern_condition calls ============
    # Inside sigil_CodeGen____emit_pattern_condition, self is SigilValue (not pointer)
    # So internal calls should use self, not (*self)
    # But we just converted self -> (*self) globally above
    # We need to revert these inside the function

    # Find the function and fix internal calls
    # The function starts with "SigilValue sigil_CodeGen____emit_pattern_condition(SigilValue self"
    # Look for the pattern and fix

    # Instead of complex regex, let's handle this by line
    lines = content.split('\n')
    in_emit_pattern_condition = False
    brace_count = 0

    for i, line in enumerate(lines):
        if 'SigilValue sigil_CodeGen____emit_pattern_condition(SigilValue self' in line:
            in_emit_pattern_condition = True
            brace_count = 0

        if in_emit_pattern_condition:
            # Count braces to track function scope
            brace_count += line.count('{') - line.count('}')

            # Fix (*self) back to self inside this function (self is SigilValue, not pointer)
            if 'emit_pattern_condition((*self),' in line:
                lines[i] = line.replace('emit_pattern_condition((*self),', 'emit_pattern_condition(self,')
                fixes_applied += 1
            # Also fix escape_char and escape_string which were incorrectly converted
            if 'escape_char((*self),' in line:
                lines[i] = lines[i].replace('escape_char((*self),', 'escape_char(self,')
                fixes_applied += 1
            if 'escape_string((*self),' in line:
                lines[i] = lines[i].replace('escape_string((*self),', 'escape_string(self,')
                fixes_applied += 1
            if 'mangle_name((*self),' in line:
                lines[i] = lines[i].replace('mangle_name((*self),', 'mangle_name(self,')
                fixes_applied += 1

            # End of function
            if brace_count == 0 and '{' in ''.join(lines[max(0,i-10):i+1]):
                in_emit_pattern_condition = False

    content = '\n'.join(lines)
    print("Fixed recursive emit_pattern_condition calls")

    # ============ PHASE 13: Additional fixes for remaining errors ============

    # Fix fold_env local variable in Environment____define
    content = content.replace('sigil_Environment____define(fold_env,', 'sigil_Environment____define(&fold_env,')
    fixes_applied += 1
    print("Fixed fold_env variable calls")

    # Fix remaining UncertainTypeEnv____define calls with any pattern of 4 args
    # More permissive regex: match _t* followed by any 3 comma-separated args
    content = re.sub(
        r'sigil_UncertainTypeEnv____define\((_t\d+), ([^,]+), ([^,]+), ((?:evidence|final_ev|EvidenceLevel____[^)]+|ev))\)',
        r'sigil_UncertainTypeEnv____define(&\1, \2, \3, \4)',
        content
    )
    fixes_applied += 1
    print("Fixed additional UncertainTypeEnv____define calls")

    # Fix check_evidence where self is pointer (in Interpreter____call_function)
    # The original has check_evidence(self,...) where self is SigilValue* - correct
    # But we converted check_evidence((*self),...) to check_evidence(self,...)
    # Need to check if there's a different issue...
    # Actually the issue might be a different self - let me check if there's self that's not a pointer

    # Fix emit_pattern_condition calls with &self (inside emit_pattern_condition, self is value not pointer)
    # These recursive calls should use self, not &self or (*self)
    content = content.replace('sigil_CodeGen____emit_pattern_condition(&self,', 'sigil_CodeGen____emit_pattern_condition(self,')
    # Also fix any (*self) remaining inside the function
    # But we need to be careful - only inside emit_pattern_condition where self is SigilValue

    # Actually, let's handle this by line - find emit_pattern_condition function and fix internal calls
    lines = content.split('\n')
    in_emit_pattern_condition = False
    brace_depth = 0
    func_start_line = -1

    for i, line in enumerate(lines):
        # Detect start of emit_pattern_condition function (takes SigilValue self, not pointer)
        if 'SigilValue sigil_CodeGen____emit_pattern_condition(SigilValue self,' in line:
            in_emit_pattern_condition = True
            brace_depth = 0
            func_start_line = i
            continue

        if in_emit_pattern_condition:
            brace_depth += line.count('{') - line.count('}')

            # Fix (*self) or &self back to self inside this function
            if 'emit_pattern_condition((*self),' in line or 'emit_pattern_condition(&self,' in line:
                lines[i] = line.replace('emit_pattern_condition((*self),', 'emit_pattern_condition(self,')
                lines[i] = lines[i].replace('emit_pattern_condition(&self,', 'emit_pattern_condition(self,')
                fixes_applied += 1

            # End of function (brace_depth back to 0 after we've seen at least one opening brace)
            if brace_depth == 0 and i > func_start_line + 2:
                in_emit_pattern_condition = False

    content = '\n'.join(lines)
    print("Fixed emit_pattern_condition recursive calls")

    # ============ SAVE ============
    with open(filename, 'w') as f:
        f.write(content)

    print(f"\nApplied {fixes_applied} total fixes to {filename}")
    print("\nRun: gcc -g -O0 -o sigil_bootstrap sigil_bootstrap.c -lm")

if __name__ == '__main__':
    main()
