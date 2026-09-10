

/* ====================================================================
 * Bootstrap completion block.
 *
 * bootstrap_fixed4.c compiles but has never linked: it declares and calls
 * 46 stdlib symbols whose definitions jormungandr only produces when it
 * RUNS (emit_builtin_impls), so its own bootstrap never contained them.
 * This block supplies exactly those, taken from what the current
 * emit_builtin_impls emits, plus prelude items an older codegen had not
 * yet started emitting.
 *
 * Tag values are verbatim from codegen.sg (TAG_STRINGBUILDER 17, TAG_MAP 18).
 * ==================================================================== */

#ifndef TAG_STRINGBUILDER
#define TAG_STRINGBUILDER 17
#endif
#ifndef TAG_MAP
#define TAG_MAP 18
#endif
#ifndef SB_INITIAL_CAP
#define SB_INITIAL_CAP 256
#endif
#ifndef SB_HEADER_SIZE
#define SB_HEADER_SIZE (sizeof(size_t) * 2)
#endif

/* Two distinct builder types; the emitter defines both. */
typedef struct StringBuilder {
    char* data;
    size_t len;
    size_t cap;
} StringBuilder;

typedef struct { size_t len; size_t cap; char data[]; } SigilStringBuilder;

/* SigilStringBuilder helpers the emitter defines as static inline; they sit
 * outside the function slices taken above, so they are reproduced verbatim. */
static inline SigilStringBuilder* get_builder(char* s) {
    return (SigilStringBuilder*)((char*)s - SB_HEADER_SIZE);
}

static inline SigilStringBuilder* ensure_capacity(SigilStringBuilder* sb, size_t needed) {
    if (sb->len + needed < sb->cap) return sb;
    size_t new_cap = sb->cap * 2;
    while (sb->len + needed >= new_cap) new_cap *= 2;
    SigilStringBuilder* new_sb = (SigilStringBuilder*)realloc(sb, SB_HEADER_SIZE + new_cap);
    new_sb->cap = new_cap;
    return new_sb;
}

static inline int is_string_builder(char* s) {
    if (!s) return 0;
    SigilStringBuilder* sb = get_builder(s);
    return (sb->cap >= SB_INITIAL_CAP && sb->cap <= 1024*1024*100 && sb->len <= sb->cap);
}



/* codegen.sg emits these, but its string literals were corrupted by a keyword
 * migration and now produce "typedef Σ StringBuilder", "⤺ sb;", "⎇ (...)" —
 * none of which is C. Written here as the C that was intended. */
static inline StringBuilder* sb_new(void) {
    StringBuilder* sb = (StringBuilder*)malloc(sizeof(StringBuilder));
    sb->cap = 256;
    sb->data = (char*)malloc(sb->cap);
    sb->data[0] = 0;
    sb->len = 0;
    return sb;
}

static inline void sb_ensure_cap(StringBuilder* sb, size_t needed) {
    if (sb->len + needed >= sb->cap) {
        sb->cap = (sb->len + needed + 1) * 2;
        sb->data = (char*)realloc(sb->data, sb->cap);
    }
}

static inline char* sb_to_string(StringBuilder* sb) {
    char* result = (char*)malloc(sb->len + 1);
    memcpy(result, sb->data, sb->len + 1);
    return result;
}

/* builtin impls jormungandr emits at runtime but its own bootstrap lacks */

SigilValue sigil_Option____unwrap_or(SigilValue opt, SigilValue def) {
if (opt.tag == TAG_OPTION_SOME && opt.v.ptr) return *((SigilValue*)opt.v.ptr);
if (opt.tag == TAG_NULL || opt.tag == TAG_UNIT) return def;
return opt;
}

SigilValue sigil_Option____is_some(SigilValue v) { return sigil_bool(v.tag == TAG_OPTION_SOME || (v.tag != TAG_NULL && v.tag != TAG_UNIT)); }

SigilValue sigil_Vec____with_capacity(SigilValue cap) { return sigil_array((size_t)cap.v.i); }

SigilValue sigil_Vec____len(SigilValue v) { return sigil_len(v); }

SigilValue sigil_max(SigilValue a, SigilValue b) { return a.v.i > b.v.i ? a : b; }

SigilValue sigil_as_mut_ptr(SigilValue v) { return v; }

SigilValue sigil_Box____from_raw(SigilValue ptr) { return ptr; }

SigilValue sigil_Box____into_raw(SigilValue b) { return b; }

SigilValue sigil_String____as_str(SigilValue s) { return s; }

SigilValue sigil_String____clone(SigilValue s) {
    if (s.tag == TAG_STRING && s.v.s) return sigil_string(s.v.s);
    return s;
}

SigilValue sigil_String____is_empty(SigilValue s) {
    if (s.tag != TAG_STRING) return sigil_bool(true);
    if (!s.v.s) return sigil_bool(true);
    return sigil_bool(s.v.s[0] == 0);
}

SigilValue sigil_rank(SigilValue v) { return sigil_int(v.evidence); }

SigilValue sigil_Result____Ok(SigilValue v) { SigilValue* inner = (SigilValue*)malloc(sizeof(SigilValue)); *inner = v; return (SigilValue){ .tag = TAG_RESULT_OK, .evidence = SIGIL_KNOWN, .v.ptr = inner }; }

SigilValue sigil_Result____Err(SigilValue v) { SigilValue* inner = (SigilValue*)malloc(sizeof(SigilValue)); *inner = v; return (SigilValue){ .tag = TAG_RESULT_ERR, .evidence = SIGIL_KNOWN, .v.ptr = inner }; }

SigilValue sigil_Result____ok(SigilValue r) { return sigil_bool(r.tag == TAG_RESULT_OK); }

SigilValue sigil_skip(SigilValue arr, SigilValue n) {
if (arr.tag != TAG_ARRAY || !arr.v.arr.data) return sigil_array(0);
size_t skip = (size_t)n.v.i;
if (skip >= arr.v.arr.len) return sigil_array(0);
SigilValue result = sigil_array(arr.v.arr.len - skip);
for (size_t i = 0; i < arr.v.arr.len - skip; i++) result.v.arr.data[i] = arr.v.arr.data[i + skip];
return result;
}

SigilValue sigil_with_note(SigilValue v, SigilValue note) { (void)note; return v; }

SigilValue sigil_char_at(SigilValue s, SigilValue idx) { return (s.tag == TAG_STRING && s.v.s && (size_t)idx.v.i < strlen(s.v.s)) ? sigil_char(s.v.s[idx.v.i]) : sigil_null(); }

SigilValue sigil_String____chars(SigilValue s) {
if (s.tag != TAG_STRING || !s.v.s) return sigil_array(0);
size_t len = strlen(s.v.s);
SigilValue arr = sigil_array(len);
for (size_t i = 0; i < len; i++) arr.v.arr.data[i] = sigil_char(s.v.s[i]);
return arr;
}

SigilValue sigil_Vec____pop(SigilValue v) {
if (v.tag != TAG_ARRAY || !v.v.arr.data || v.v.arr.len == 0) return sigil_null();
return v.v.arr.data[--v.v.arr.len];
}

SigilValue sigil_Vec____extend(SigilValue dst, SigilValue src) {
if (src.tag != TAG_ARRAY || !src.v.arr.data) return dst;
for (size_t i = 0; i < src.v.arr.len; i++) dst = sigil_Vec____push(dst, src.v.arr.data[i]);
return dst;
}

SigilValue sigil_String____contains(SigilValue s, SigilValue sub) {
if (s.tag != TAG_STRING || sub.tag != TAG_STRING) return sigil_bool(false);
if (!s.v.s || !sub.v.s) return sigil_bool(false);
return sigil_bool(strstr(s.v.s, sub.v.s) != NULL);
}

SigilValue sigil_String____ends_with(SigilValue s, SigilValue suffix) {
if (s.tag != TAG_STRING || suffix.tag != TAG_STRING) return sigil_bool(false);
if (!s.v.s || !suffix.v.s) return sigil_bool(false);
size_t slen = strlen(s.v.s), plen = strlen(suffix.v.s);
if (plen > slen) return sigil_bool(false);
return sigil_bool(strcmp(s.v.s + slen - plen, suffix.v.s) == 0);
}

SigilValue sigil_String____strip_prefix(SigilValue s, SigilValue prefix) {
if (s.tag != TAG_STRING || prefix.tag != TAG_STRING) return sigil_null();
if (!s.v.s || !prefix.v.s) return sigil_null();
size_t plen = strlen(prefix.v.s);
if (strncmp(s.v.s, prefix.v.s, plen) != 0) return sigil_null();
return sigil_Option____Some(sigil_string(s.v.s + plen));
}

SigilValue sigil_Map____entries(SigilValue map) {
if (map.tag != TAG_MAP) return sigil_array(0);
return sigil_array(0); /* TODO: implement properly */
}

SigilValue sigil_CrateConfig____default(void) { return sigil_struct("CrateConfig", NULL, NULL, 0); }

SigilValue sigil_FunctionAttrs____default(void) { return sigil_struct("FunctionAttrs", NULL, NULL, 0); }

SigilValue sigil_StructAttrs____default(void) { return sigil_struct("StructAttrs", NULL, NULL, 0); }

SigilValue sigil_String____new(void) {
SigilStringBuilder* sb = (SigilStringBuilder*)malloc(SB_HEADER_SIZE + SB_INITIAL_CAP);
sb->len = 0;
sb->cap = SB_INITIAL_CAP;
sb->data[0] = '\0';
return (SigilValue){ .tag = TAG_STRING, .evidence = SIGIL_KNOWN, .v.s = sb->data };
}

SigilValue sigil_String____push(SigilValue s, SigilValue c) {
if (s.tag != TAG_STRING || !s.v.s) return s;
char ch = '?';
if (c.tag == TAG_CHAR) ch = c.v.c;
else if (c.tag == TAG_INT) ch = (char)c.v.i;
if (is_string_builder(s.v.s)) {
SigilStringBuilder* sb = get_builder(s.v.s);
sb = ensure_capacity(sb, 1);
sb->data[sb->len++] = ch;
sb->data[sb->len] = '\0';
s.v.s = sb->data;
} else {
size_t len = strlen(s.v.s);
size_t new_cap = (len + 2 < SB_INITIAL_CAP) ? SB_INITIAL_CAP : (len + 2) * 2;
SigilStringBuilder* sb = (SigilStringBuilder*)malloc(SB_HEADER_SIZE + new_cap);
sb->len = len + 1;
sb->cap = new_cap;
memcpy(sb->data, s.v.s, len);
sb->data[len] = ch;
sb->data[len + 1] = '\0';
s.v.s = sb->data;
}
return s;
}

SigilValue sigil_String____split(SigilValue s, SigilValue delim) {
SigilValue result = sigil_Vec____new();
if (s.tag != TAG_STRING || delim.tag != TAG_STRING || !s.v.s || !delim.v.s) return result;
char* str = strdup(s.v.s);
char* token = strtok(str, delim.v.s);
while (token) {
result = sigil_Vec____push(result, sigil_string(strdup(token)));
token = strtok(NULL, delim.v.s);
}
free(str);
return result;
}

SigilValue sigil_eprint(SigilValue v) {
SigilValue s = sigil_display(v);
if (s.tag == TAG_STRING && s.v.s) fprintf(stderr, "%s", s.v.s);
return sigil_unit();
}

SigilValue sigil_error(SigilValue ctx, SigilValue err) {
fprintf(stderr, "Error: ");
SigilValue s = sigil_display(err);
if (s.tag == TAG_STRING && s.v.s) fprintf(stderr, "%s\n", s.v.s);
return sigil_unit();
}

SigilValue sigil_cloned(SigilValue v) {
switch (v.tag) {
case TAG_STRING: if (v.v.s) return sigil_string(strdup(v.v.s)); return v;
case TAG_ARRAY: {
SigilValue result = sigil_Vec____new();
for (size_t i = 0; i < v.v.arr.len; i++) result = sigil_Vec____push(result, sigil_cloned(v.v.arr.data[i]));
return result;
}
case TAG_STRUCT: {
SigilStruct* src = (SigilStruct*)v.v.ptr;
if (!src) return v;
SigilStruct* dst = (SigilStruct*)malloc(sizeof(SigilStruct));
dst->name = src->name ? strdup(src->name) : NULL;
dst->num_fields = src->num_fields;
dst->field_names = (const char**)malloc(src->num_fields * sizeof(char*));
dst->field_values = (SigilValue*)malloc(src->num_fields * sizeof(SigilValue));
for (size_t i = 0; i < src->num_fields; i++) {
dst->field_names[i] = src->field_names[i] ? strdup(src->field_names[i]) : NULL;
dst->field_values[i] = sigil_cloned(src->field_values[i]);
}
return (SigilValue){ .tag = TAG_STRUCT, .evidence = v.evidence, .v.ptr = dst };
}
default: return v;
}
}

SigilValue sigil_nth(SigilValue v, SigilValue n) {
if (v.tag != TAG_ARRAY || n.tag != TAG_INT) return sigil_unit();
int64_t idx = n.v.i;
if (idx < 0 || (size_t)idx >= v.v.arr.len) return sigil_unit();
return v.v.arr.data[idx];
}

SigilValue sigil_next(SigilValue iter) {
/* Iterator next - return Option::None for stub */
return Option____None;
}

SigilValue sigil_is_uppercase(SigilValue c) {
if (c.tag != TAG_CHAR) return sigil_bool(false);
return sigil_bool(c.v.c >= 'A' && c.v.c <= 'Z');
}

SigilValue sigil_is_duodecimal_digit(SigilValue c) {
if (c.tag != TAG_CHAR) return sigil_bool(false);
char ch = c.v.c;
return sigil_bool((ch >= '0' && ch <= '9') || ch == 'X' || ch == 'E' || ch == 'x' || ch == 'e');
}

SigilValue sigil_is_vigesimal_digit(SigilValue c) {
if (c.tag != TAG_CHAR) return sigil_bool(false);
char ch = c.v.c;
return sigil_bool((ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'J') || (ch >= 'a' && ch <= 'j'));
}

SigilValue sigil_len_utf8(SigilValue c) {
if (c.tag != TAG_CHAR) return sigil_int(0);
unsigned char ch = (unsigned char)c.v.c;
if (ch < 0x80) return sigil_int(1);
if ((ch & 0xE0) == 0xC0) return sigil_int(2);
if ((ch & 0xF0) == 0xE0) return sigil_int(3);
if ((ch & 0xF8) == 0xF0) return sigil_int(4);
return sigil_int(1);
}

SigilValue sigil_parse(SigilValue s) {
/* Parse stub - returns int if numeric, else string */
if (s.tag != TAG_STRING || !s.v.s) return s;
char* endptr;
long val = strtol(s.v.s, &endptr, 10);
if (*endptr == '\0') return sigil_int(val);
return s;
}

SigilValue sigil_map_err(SigilValue result, SigilValue mapper) {
/* Map error in Result - if Err, apply mapper */
if (sigil_is_struct_variant(result, "Result::Err") || sigil_is_struct_variant(result, "Err")) {
if (result.v.ptr && mapper.tag == TAG_CLOSURE) {
SigilValue err = *(SigilValue*)result.v.ptr;
SigilValue (*fn)(SigilValue) = (SigilValue (*)(SigilValue))mapper.v.ptr;
return sigil_Err(fn(err));
}
}
return result;
}

SigilValue sigil_to_json(SigilValue v, SigilValue opts) {
(void)opts;  /* unused for now */
return sigil_display(v);  /* Simplified: just display */
}

SigilValue sigil_kind_name(SigilValue v) {
switch (v.tag) {
case TAG_INT: return sigil_string("int");
case TAG_FLOAT: return sigil_string("float");
case TAG_STRING: return sigil_string("string");
case TAG_BOOL: return sigil_string("bool");
case TAG_CHAR: return sigil_string("char");
case TAG_ARRAY: return sigil_string("array");
case TAG_STRUCT: return sigil_string("struct");
case TAG_ENUM: return sigil_string("enum");
default: return sigil_string("unknown");
}
}

SigilValue sigil_collect_results(SigilValue arr) {
if (arr.tag != TAG_ARRAY) return sigil_Err(sigil_string("not an array"));
SigilValue result = sigil_Vec____new();
for (size_t i = 0; i < arr.v.arr.len; i++) {
SigilValue item = arr.v.arr.data[i];
if (sigil_is_struct_variant(item, "Result::Err") || sigil_is_struct_variant(item, "Err")) return item;
if (sigil_is_struct_variant(item, "Result::Ok") || sigil_is_struct_variant(item, "Ok")) {
if (item.v.ptr) result = sigil_Vec____push(result, *(SigilValue*)item.v.ptr);
} else {
result = sigil_Vec____push(result, item);
}
}
return sigil_Ok(result);
}


/* sigil_any — jormungandr's codegen emits a declaration for this (and call sites
 * for `.any(pred)`) but never emits a definition, so nothing ever provided it.
 * Written here as the exact dual of sigil_all, which the emitter does provide,
 * matching its closure-calling convention and array guards. */
SigilValue sigil_any(SigilValue arr, SigilValue closure) {
    if (arr.tag != TAG_ARRAY) return sigil_bool(false);
    typedef SigilValue (*ClosureFn)(SigilValue);
    ClosureFn fn = (ClosureFn)closure.v.ptr;
    for (size_t i = 0; i < arr.v.arr.len; i++) {
        SigilValue result = fn(arr.v.arr.data[i]);
        if (sigil_truthy(result)) return sigil_bool(true);
    }
    return sigil_bool(false);
}

