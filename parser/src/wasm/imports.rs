//! JS runtime import registration.
//!
//! Registers all imported functions from the JS runtime (sigil_runtime.js).

use std::collections::HashMap;
use wasm_encoder::ValType;

use super::types::ImportFn;

/// Import registry for WASM modules.
#[derive(Debug, Default)]
pub struct ImportRegistry {
    /// Registered imports
    imports: Vec<ImportFn>,
    /// Type signatures: (params, results) -> type_idx
    types: Vec<(Vec<ValType>, Vec<ValType>)>,
    /// Type deduplication map
    type_map: HashMap<(Vec<ValType>, Vec<ValType>), u32>,
    /// Function name -> function index
    func_map: HashMap<String, u32>,
}

impl ImportRegistry {
    /// Create a new import registry with standard imports.
    pub fn new() -> Self {
        let mut registry = Self::default();
        registry.register_standard_imports();
        registry
    }

    /// Create an empty registry (for testing).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Get or create a type index for the given signature.
    pub fn get_or_create_type(&mut self, params: Vec<ValType>, results: Vec<ValType>) -> u32 {
        let key = (params.clone(), results.clone());
        if let Some(&idx) = self.type_map.get(&key) {
            return idx;
        }
        let idx = self.types.len() as u32;
        self.types.push((params, results));
        self.type_map.insert(key, idx);
        idx
    }

    /// Add an import function.
    pub fn add_import(
        &mut self,
        module: &str,
        name: &str,
        params: Vec<ValType>,
        results: Vec<ValType>,
    ) -> u32 {
        let type_idx = self.get_or_create_type(params, results);
        let func_idx = self.imports.len() as u32;

        let import = ImportFn::new(module, name, type_idx);
        self.func_map.insert(import.qualified_name(), func_idx);
        self.imports.push(import);

        func_idx
    }

    /// Add an import function with an alias for direct lookup (for builtins like `print`).
    pub fn add_import_with_alias(
        &mut self,
        module: &str,
        name: &str,
        alias: &str,
        params: Vec<ValType>,
        results: Vec<ValType>,
    ) -> u32 {
        let func_idx = self.add_import(module, name, params, results);
        // Also register the alias for direct lookup
        self.func_map.insert(alias.to_string(), func_idx);
        func_idx
    }

    /// Look up a function index by qualified name.
    pub fn get_func(&self, qualified_name: &str) -> Option<u32> {
        self.func_map.get(qualified_name).copied()
    }

    /// Get all imports.
    pub fn imports(&self) -> &[ImportFn] {
        &self.imports
    }

    /// Get all types.
    pub fn types(&self) -> &[(Vec<ValType>, Vec<ValType>)] {
        &self.types
    }

    /// Get the number of imports.
    pub fn import_count(&self) -> u32 {
        self.imports.len() as u32
    }

    /// Get the return type for an import function by index.
    /// Returns None if no return value, Some(ValType) if single return.
    pub fn get_return_type(&self, func_idx: u32) -> Option<ValType> {
        let import = self.imports.get(func_idx as usize)?;
        let (_, results) = self.types.get(import.type_idx as usize)?;
        results.first().copied()
    }

    /// Get the parameter types for an import function by index.
    pub fn get_param_types(&self, func_idx: u32) -> Option<&[ValType]> {
        let import = self.imports.get(func_idx as usize)?;
        let (params, _) = self.types.get(import.type_idx as usize)?;
        Some(params.as_slice())
    }

    /// Get the type index for an import function by function index.
    pub fn get_func_type(&self, func_idx: u32) -> Option<u32> {
        let import = self.imports.get(func_idx as usize)?;
        Some(import.type_idx)
    }

    /// Register standard JS runtime imports.
    fn register_standard_imports(&mut self) {
        self.register_console_imports();
        self.register_string_imports();
        self.register_dom_imports();
        self.register_event_imports();
        self.register_timing_imports();
        self.register_fetch_imports();
        self.register_storage_imports();
        self.register_router_imports();
        self.register_memory_imports();
        self.register_morpheme_imports();
        self.register_map_imports();
        self.register_math_imports();
        self.register_vdom_imports();
        self.register_signal_imports();
        self.register_async_imports();
        self.register_browser_imports();
        self.register_json_imports();
    }

    fn register_browser_imports(&mut self) {
        use ValType::*;
        // Window and document access - add aliases for direct lookup
        self.add_import_with_alias("browser", "window", "window", vec![], vec![I32]);
        self.add_import_with_alias("browser", "document", "document", vec![], vec![I32]);
        // Window properties
        self.add_import("browser", "inner_width", vec![I32], vec![I32]);
        self.add_import("browser", "inner_height", vec![I32], vec![I32]);
        // Event listeners on window
        self.add_import("browser", "add_event_listener", vec![I32, I32, I32, I32], vec![I32]);
        self.add_import("browser", "remove_event_listener", vec![I32, I32], vec![]);
        // Media query
        self.add_import("browser", "match_media", vec![I32, I32], vec![I32]);
        self.add_import("browser", "mql_matches", vec![I32], vec![I32]);
        self.add_import("browser", "mql_add_listener", vec![I32, I32], vec![I32]);
        self.add_import("browser", "mql_remove_listener", vec![I32, I32], vec![]);
        // Dialogs. `confirm(msg)` guards every destructive action in the Lares
        // client; with no import behind it the call was an undefined function
        // and the whole module failed to build.
        self.add_import("browser", "confirm", vec![I32], vec![I32]);
        self.add_import("browser", "alert", vec![I32], vec![]);
        self.add_import("browser", "prompt", vec![I32, I32], vec![I32]);
    }

    fn register_console_imports(&mut self) {
        use ValType::*;
        self.add_import("console", "log_i64", vec![I64], vec![]);
        self.add_import("console", "log_f64", vec![F64], vec![]);
        self.add_import("console", "log_str", vec![I32, I32], vec![]);
        // Register 'print' as a builtin with direct lookup alias
        self.add_import_with_alias("console", "print", "print", vec![I64], vec![]);
        // Register console_log/warn/error aliases for string output (used by macros)
        // These take a string pointer (i64 with ptr in low bits) and log it
        self.add_import_with_alias("console", "log", "console_log", vec![I64], vec![]);
        self.add_import_with_alias("console", "warn", "console_warn", vec![I64], vec![]);
        self.add_import_with_alias("console", "error", "console_error", vec![I64], vec![]);
    }

    fn register_string_imports(&mut self) {
        use ValType::*;
        // String operations - all strings are ptr (i32) to length-prefixed UTF-8
        self.add_import("string", "concat", vec![I32, I32], vec![I32]); // (str1, str2) -> new_str
        self.add_import("string", "length", vec![I32], vec![I32]); // (str) -> length
        self.add_import("string", "slice", vec![I32, I32, I32], vec![I32]); // (str, start, end) -> new_str
        self.add_import("string", "eq", vec![I32, I32], vec![I32]); // (str1, str2) -> bool
        self.add_import("string", "from_int", vec![I64], vec![I32]);
        // `x·to_string()` where the compiler cannot prove which it is. The host
        // knows which addresses it wrote strings to; `from_int` would print the
        // decimal of the address.
        self.add_import_with_alias(
            "string",
            "from_value",
            "string_from_value",
            vec![I64],
            vec![I32],
        );
        // `x·to_string()` where the compiler cannot prove which it is. The host
        // knows which addresses it wrote strings to; `from_int` would print the
        // decimal of the address.
        self.add_import_with_alias(
            "string",
            "from_value",
            "string_from_value",
            vec![I64],
            vec![I32],
        ); // (int) -> str
        self.add_import("string", "from_float", vec![F64], vec![I32]); // (float) -> str
        // `String·from_utf8(bytes)`. A Sigil `Vec[u8]` is a host array handle,
        // so the host does the decoding; there is no byte buffer in linear
        // memory to walk.
        self.add_import("string", "from_utf8", vec![I32], vec![I32]); // (bytes) -> str
        // Aliased so the bare names resolve: `Number(x)` lowers to
        // `parse_float(x)`, and the qualified `string_parse_float` is not what
        // anyone writes.
        self.add_import_with_alias("string", "parse_int", "parse_int", vec![I32], vec![I64]);
        self.add_import_with_alias("string", "parse_float", "parse_float", vec![I32], vec![F64]);
        // `encodeURIComponent(s)` — every generated URL that carries a ticket
        // key or a query builds one, and there was no import behind it.
        self.add_import_with_alias(
            "string",
            "encode_uri_component",
            "encode_uri_component",
            vec![I32],
            vec![I32],
        );
        self.add_import_with_alias(
            "string",
            "decode_uri_component",
            "decode_uri_component",
            vec![I32],
            vec![I32],
        );
        // Additional string methods
        self.add_import("string", "lines", vec![I32], vec![I32]); // (str) -> array of strings
        self.add_import("string", "split_whitespace", vec![I32], vec![I32]); // (str) -> array of strings
        self.add_import("string", "split", vec![I32, I32], vec![I32]); // (str, delimiter) -> array
        self.add_import("string", "trim", vec![I32], vec![I32]); // (str) -> trimmed str
        self.add_import("string", "trim_start", vec![I32], vec![I32]); // (str) -> trimmed str
        self.add_import("string", "trim_end", vec![I32], vec![I32]); // (str) -> trimmed str
        self.add_import("string", "to_uppercase", vec![I32], vec![I32]); // (str) -> uppercase str
        self.add_import("string", "to_lowercase", vec![I32], vec![I32]); // (str) -> lowercase str
        self.add_import("string", "contains", vec![I32, I32], vec![I32]); // (str, substr) -> bool
        self.add_import("string", "starts_with", vec![I32, I32], vec![I32]); // (str, prefix) -> bool
        self.add_import("string", "ends_with", vec![I32, I32], vec![I32]); // (str, suffix) -> bool
        // `a·locale_compare(b)` — JavaScript's `localeCompare`, which every
        // sort comparator in the migrated client uses. -1, 0 or 1.
        self.add_import("string", "locale_compare", vec![I32, I32], vec![I64]);
        self.add_import("string", "replace", vec![I32, I32, I32], vec![I32]); // (str, from, to) -> new str
        self.add_import("string", "chars", vec![I32], vec![I32]); // (str) -> array of chars
        // `s.padStart(n, pad)` / `padEnd`. Migrated code formats timestamps and
        // ids with these, and they reached the backend as undefined functions.
        // `s.indexOf(x)` / `lastIndexOf(x)`, -1 when absent. `splitPath` finds
        // the last `/` with one; neither existed, so it reached the backend as
        // an undefined function.
        self.add_import("string", "index_of", vec![I32, I32], vec![I64]);
        self.add_import("string", "last_index_of", vec![I32, I32], vec![I64]);
        self.add_import("string", "pad_start", vec![I32, I64, I32], vec![I32]);
        self.add_import("string", "pad_end", vec![I32, I64, I32], vec![I32]);
    }

    fn register_dom_imports(&mut self) {
        use ValType::*;
        self.add_import("dom", "create_element", vec![I32, I32], vec![I32]);
        self.add_import("dom", "create_text", vec![I32, I32], vec![I32]);
        self.add_import(
            "dom",
            "set_attribute",
            vec![I32, I32, I32, I32, I32],
            vec![],
        );
        self.add_import("dom", "remove_attribute", vec![I32, I32, I32], vec![]);
        self.add_import("dom", "set_property", vec![I32, I32, I32, I64], vec![]);
        self.add_import("dom", "append_child", vec![I32, I32], vec![]);
        self.add_import("dom", "insert_before", vec![I32, I32, I32], vec![]);
        self.add_import("dom", "remove_child", vec![I32, I32], vec![]);
        self.add_import("dom", "replace_child", vec![I32, I32, I32], vec![]);
        self.add_import("dom", "set_text_content", vec![I32, I32, I32], vec![]);
        self.add_import("dom", "get_element_by_id", vec![I32, I32], vec![I32]);
        self.add_import("dom", "query_selector", vec![I32, I32], vec![I32]);
        self.add_import("dom", "clone_node", vec![I32, I32], vec![I32]);
    }

    fn register_event_imports(&mut self) {
        use ValType::*;
        self.add_import("events", "add_listener", vec![I32, I32, I32, I32], vec![I32]);
        self.add_import("events", "remove_listener", vec![I32], vec![]);
        self.add_import("events", "prevent_default", vec![I32], vec![]);
        self.add_import("events", "stop_propagation", vec![I32], vec![]);
        self.add_import("events", "get_target", vec![I32], vec![I32]);
        self.add_import("events", "get_value", vec![I32, I32], vec![I32]);
    }

    fn register_timing_imports(&mut self) {
        use ValType::*;
        self.add_import("timing", "now", vec![], vec![F64]);
        self.add_import("timing", "set_timeout", vec![I32, I32], vec![I32]);
        self.add_import("timing", "clear_timeout", vec![I32], vec![]);
        self.add_import("timing", "set_interval", vec![I32, I32], vec![I32]);
        self.add_import("timing", "clear_interval", vec![I32], vec![]);
        self.add_import("timing", "request_animation_frame", vec![I32], vec![I32]);
    }

    fn register_fetch_imports(&mut self) {
        use ValType::*;
        self.add_import("fetch", "start", vec![I32, I32, I32], vec![I32]);
        self.add_import("fetch", "poll", vec![I32], vec![I32]);
        self.add_import("fetch", "get_status", vec![I32], vec![I32]);
        // One parameter: the request handle. The runtime's `fetchGetBody(id)`
        // has always taken one, so the two-parameter declaration made every
        // call a signature mismatch.
        self.add_import("fetch", "get_body", vec![I32], vec![I32]);
        self.add_import("fetch", "abort", vec![I32], vec![]);
        // One request, as a promise.
        //
        // `start`/`poll`/`get_body` is a polling protocol, and a Sigil program
        // has no loop to poll from — `.await` is the only thing it says. This
        // returns a promise id that `async.await_promise` suspends on, so
        // `fetch_request(url, method, body).await` is the whole of it.
        self.add_import_with_alias(
            "fetch",
            "request",
            "fetch_request",
            vec![I32, I32, I32],
            vec![I32],
        );
        // The HTTP status the request came back with, so a program can tell a
        // 404 from a body it could not parse.
        self.add_import_with_alias("fetch", "status", "fetch_status", vec![I32], vec![I32]);
        self.add_import_with_alias("fetch", "ok", "fetch_ok", vec![I32], vec![I32]);
    }

    fn register_storage_imports(&mut self) {
        use ValType::*;
        self.add_import("storage", "local_get", vec![I32, I32, I32], vec![I32]);
        self.add_import("storage", "local_set", vec![I32, I32, I32, I32], vec![]);
        self.add_import("storage", "local_remove", vec![I32, I32], vec![]);
    }

    fn register_router_imports(&mut self) {
        use ValType::*;
        self.add_import("router", "push_state", vec![I32, I32], vec![]);
        self.add_import("router", "replace_state", vec![I32, I32], vec![]);
        self.add_import("router", "get_pathname", vec![I32], vec![I32]);
    }

    fn register_memory_imports(&mut self) {
        use ValType::*;
        self.add_import("memory", "alloc", vec![I32], vec![I32]);
        self.add_import("memory", "realloc", vec![I32, I32], vec![I32]);
        self.add_import("memory", "free", vec![I32], vec![]);
        // heap_alloc takes i64 size and returns i64 pointer (used for closures/structs)
        self.add_import_with_alias("memory", "heap_alloc", "heap_alloc", vec![I64], vec![I64]);
    }

    /// `HashMap` / `HashSet`. A Sigil map is a host object, like an array:
    /// there is no map in linear memory to walk. Without these, `HashMap·new()`
    /// was stubbed to a constant `0` and every `VElement.attrs` was a null
    /// pointer the renderer then dereferenced.
    fn register_map_imports(&mut self) {
        use ValType::*;
        self.add_import_with_alias("map", "new", "map_new", vec![], vec![I32]);
        self.add_import_with_alias("map", "set", "map_set", vec![I32, I64, I64], vec![]);
        self.add_import_with_alias("map", "get", "map_get", vec![I32, I64], vec![I64]);
        self.add_import_with_alias("map", "has", "map_has", vec![I32, I64], vec![I32]);
        self.add_import_with_alias("map", "remove", "map_remove", vec![I32, I64], vec![]);
        self.add_import_with_alias("map", "len", "map_len", vec![I32], vec![I32]);
        self.add_import_with_alias("map", "is_empty", "map_is_empty", vec![I32], vec![I32]);
        // Iteration: the host returns an array, which the array imports walk.
        self.add_import_with_alias("map", "keys", "map_keys", vec![I32], vec![I32]);
        self.add_import_with_alias("map", "values", "map_values", vec![I32], vec![I32]);
        self.add_import_with_alias("map", "entries", "map_entries", vec![I32], vec![I32]);
        // What `∀` iterates: a map becomes its entries, an array stays itself.
        self.add_import_with_alias("map", "iter_of", "iter_of", vec![I32], vec![I32]);
        // `HashMap·from(entries)` — an array of two-element `[k, v]` arrays,
        // which is what `Object.entries(x)` and `new Map(pairs)` both give.
        self.add_import_with_alias("map", "from", "map_from", vec![I32], vec![I32]);

        // HashSet. Its own group, not a map with dummy values: `∀ x ∈ set`
        // must yield the elements, and a map yields `[k, v]` pairs. Before
        // this, `HashSet·new()` produced a map and `set·add(x)` produced an
        // undefined `add` — the whole set surface was unimplemented.
        self.add_import_with_alias("set", "new", "set_new", vec![], vec![I32]);
        self.add_import_with_alias("set", "from", "set_from", vec![I32], vec![I32]);
        self.add_import_with_alias("set", "add", "set_add", vec![I32, I64], vec![]);
        self.add_import_with_alias("set", "has", "set_has", vec![I32, I64], vec![I32]);
        self.add_import_with_alias("set", "remove", "set_remove", vec![I32, I64], vec![]);
        self.add_import_with_alias("set", "len", "set_len", vec![I32], vec![I32]);
        self.add_import_with_alias("set", "values", "set_values", vec![I32], vec![I32]);
    }

    fn register_morpheme_imports(&mut self) {
        use ValType::*;
        // Core array operations with aliases for direct lookup
        self.add_import_with_alias("morpheme", "array_new", "array_new", vec![], vec![I32]);
        self.add_import_with_alias("morpheme", "array_push", "array_push", vec![I32, I64], vec![]);
        self.add_import_with_alias("morpheme", "array_get", "array_get", vec![I32, I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_set", "array_set", vec![I32, I32, I64], vec![]);
        self.add_import_with_alias("morpheme", "array_len", "array_len", vec![I32], vec![I32]);
        self.add_import_with_alias("morpheme", "array_map", "array_map", vec![I32, I32], vec![I32]);
        self.add_import_with_alias("morpheme", "array_filter", "array_filter", vec![I32, I32], vec![I32]);
        // Parallel morphemes - use Web Workers or SharedArrayBuffer for parallelism
        self.add_import("morpheme", "array_parallel_map", vec![I32, I32], vec![I32]);
        self.add_import("morpheme", "array_parallel_filter", vec![I32, I32], vec![I32]);
        self.add_import("morpheme", "array_parallel_reduce", vec![I32, I32, I64], vec![I64]);
        // `(array, init, closure)` - Sigil's `fold` takes the initial value
        // first, and the interpreter is the oracle for that.
        self.add_import_with_alias("morpheme", "array_reduce", "array_reduce", vec![I32, I64, I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_sort", "array_sort", vec![I32], vec![I32]);
        // `xs·sort(|a, b| …)`. Sigil's own `sort` takes no comparator, so a
        // comparator sort had nowhere to go — and dropping the comparator
        // silently sorts by something else.
        self.add_import_with_alias(
            "morpheme",
            "array_sort_by",
            "array_sort_by",
            vec![I32, I64],
            vec![I32],
        );
        self.add_import_with_alias("morpheme", "array_first", "array_first", vec![I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_last", "array_last", vec![I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_nth", "array_nth", vec![I32, I32], vec![I64]);
        // Additional reduce operations for ρ+, ρ*, etc.
        self.add_import_with_alias("morpheme", "array_sum", "array_sum", vec![I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_product", "array_product", vec![I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_min", "array_min", vec![I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_max", "array_max", vec![I32], vec![I64]);
        self.add_import_with_alias("morpheme", "array_all", "array_all", vec![I32], vec![I32]);
        self.add_import_with_alias("morpheme", "array_any", "array_any", vec![I32], vec![I32]);

        // The higher-order morphemes that take a closure. `array_all` and
        // `array_any` above take no predicate at all, so `xs·any(|x| …)` was
        // compiled as filter-then-count against a filter that did nothing.
        self.add_import_with_alias("morpheme", "array_find", "array_find", vec![I32, I32], vec![I64]);
        self.add_import_with_alias(
            "morpheme",
            "array_position",
            "array_position",
            vec![I32, I32],
            vec![I32],
        );
        self.add_import_with_alias("morpheme", "array_any_by", "array_any_by", vec![I32, I32], vec![I32]);
        self.add_import_with_alias("morpheme", "array_all_by", "array_all_by", vec![I32, I32], vec![I32]);
        self.add_import_with_alias(
            "morpheme",
            "array_flat_map",
            "array_flat_map",
            vec![I32, I32],
            vec![I32],
        );
        self.add_import_with_alias("morpheme", "array_flatten", "array_flatten", vec![I32], vec![I32]);
        self.add_import_with_alias("morpheme", "array_reverse", "array_reverse", vec![I32], vec![I32]);
        self.add_import_with_alias("morpheme", "array_random_element", "array_random_element", vec![I32], vec![I64]);
        // Vec::join - concatenate elements with separator
        self.add_import_with_alias("morpheme", "vec_join", "vec_join", vec![I32, I32], vec![I32]);
    }

    /// JSON, which the WASM backend had none of.
    ///
    /// Sigil's JSON lives on the interpreter as `json_parse` / `json_stringify`
    /// / `json_get` / `json_set` / `json_pretty`. Compiled to WASM those names
    /// resolved to nothing and, under the unresolved-call stub, quietly became
    /// `0` — so a web target could not read or write JSON at all, and would not
    /// say so. Qliphoth worked around it by calling `serde_json·`, a Rust crate
    /// Sigil does not have, in its storage, websocket, router-guard and history
    /// modules.
    ///
    /// Values are opaque host handles, like vnodes and arrays. Strings are i64
    /// pointers to length-prefixed data, matching the vdom convention.
    fn register_json_imports(&mut self) {
        use ValType::*;
        self.add_import_with_alias("json", "parse", "json_parse", vec![I64], vec![I64]);
        self.add_import_with_alias("json", "stringify", "json_stringify", vec![I64], vec![I64]);
        self.add_import_with_alias("json", "pretty", "json_pretty", vec![I64], vec![I64]);
        self.add_import_with_alias("json", "get", "json_get", vec![I64, I64], vec![I64]);
        self.add_import_with_alias("json", "set", "json_set", vec![I64, I64, I64], vec![I64]);

        // JavaScript truthiness, which the React migrator emits as `x·to_bool()`
        // for every `&&`, `||` and ternary over a non-boolean — 548 times across
        // the generated Lares client. `to_bool` existed only as an interpreter
        // free function, so the method form worked in neither backend. It is a
        // host call rather than an `i64.ne 0` because only the host can tell a
        // string pointer from a small integer, and `""` is falsy while a pointer
        // to it is not zero.
        self.add_import_with_alias("value", "to_bool", "to_bool", vec![I64], vec![I64]);

        // JS statics the React migrator emits that have no Sigil equivalent.
        // Each one is a host call for the same reason `to_bool` is: only the
        // host knows whether a value is a string, an array or an object.
        self.add_import_with_alias("value", "object_values", "object_values", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "object_keys", "object_keys", vec![I64], vec![I64]);
        self.add_import_with_alias(
            "value",
            "object_entries",
            "object_entries",
            vec![I64],
            vec![I64],
        );
        self.add_import_with_alias("value", "is_finite", "is_finite", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "is_nan", "is_nan", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "is_integer", "is_integer", vec![I64], vec![I64]);

        // `new Date(s).getTime()` — parsing a timestamp string to epoch millis.
        // `timing.now` covers the clock; nothing covered the parse, so every
        // `format_relative_time(new Date(x).getTime())` in the Lares client was
        // an undefined call.
        self.add_import_with_alias("timing", "parse", "timing_parse", vec![I64], vec![I64]);

        // `x.toFixed(2)` — a formatted string, not a rounded number, so
        // `math.round` is not it.
        self.add_import_with_alias("value", "to_fixed", "to_fixed", vec![I64, I64], vec![I64]);
        // `Array.isArray(x)` — only the host can tell an array handle from any
        // other i64.
        self.add_import_with_alias("value", "is_array", "is_array", vec![I64], vec![I64]);
        // `typeof x` — "string", "number", "object", … Only the host can say.
        self.add_import_with_alias("value", "type_of", "type_of", vec![I64], vec![I64]);

        // `Math.floor` and friends over the uniform value domain.
        //
        // `math.floor` next door takes and returns an F64, and a migrated
        // JavaScript number is an i64 here — so nothing could call it, and
        // `formatRelativeTime`, `formatFileSize`, `kbToSize` and everything
        // like them stayed untranslated on the name alone. These take the
        // value, whatever it is, and hand it to the host, which is the only
        // thing that can tell an integer from a handle to a float.
        self.add_import_with_alias("value", "math_floor", "math_floor", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "math_ceil", "math_ceil", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "math_round", "math_round", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "math_trunc", "math_trunc", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "math_abs", "math_abs", vec![I64], vec![I64]);
        self.add_import_with_alias("value", "math_min", "math_min", vec![I64, I64], vec![I64]);
        self.add_import_with_alias("value", "math_max", "math_max", vec![I64, I64], vec![I64]);

        // `Date.now()` — epoch milliseconds, as an i64.
        //
        // `timing.now` is the monotonic clock and is declared F64. Migrated
        // code was routed there, so it compared epoch timestamps against
        // milliseconds-since-page-load, as a float, in a value domain that is
        // otherwise entirely i64.
        self.add_import_with_alias("value", "date_now", "date_now", vec![], vec![I64]);

        // `new Date(x).toLocaleDateString()` and its siblings. Without it these
        // collapsed to `to_string` on the millisecond count, and the number
        // itself was rendered into the document.
        self.add_import_with_alias(
            "value",
            "date_format",
            "date_format",
            vec![I64, I64, I64],
            vec![I32],
        );
    }

    fn register_math_imports(&mut self) {
        use ValType::*;
        self.add_import("math", "sqrt", vec![F64], vec![F64]);
        self.add_import("math", "sin", vec![F64], vec![F64]);
        self.add_import("math", "cos", vec![F64], vec![F64]);
        self.add_import("math", "tan", vec![F64], vec![F64]);
        self.add_import("math", "pow", vec![F64, F64], vec![F64]);
        self.add_import("math", "exp", vec![F64], vec![F64]);
        self.add_import("math", "log", vec![F64], vec![F64]);
        self.add_import("math", "floor", vec![F64], vec![F64]);
        self.add_import("math", "ceil", vec![F64], vec![F64]);
        self.add_import("math", "round", vec![F64], vec![F64]);
        self.add_import("math", "abs", vec![F64], vec![F64]);
        self.add_import("math", "abs_int", vec![I64], vec![I64]); // integer abs
        self.add_import("math", "random", vec![], vec![F64]);
        // Additional math functions
        self.add_import("math", "clamp", vec![F64, F64, F64], vec![F64]); // (value, min, max) -> clamped
        self.add_import("math", "clamp_int", vec![I64, I64, I64], vec![I64]); // (value, min, max) -> clamped
        self.add_import("math", "min", vec![F64, F64], vec![F64]);
        self.add_import("math", "max", vec![F64, F64], vec![F64]);
        self.add_import("math", "min_int", vec![I64, I64], vec![I64]);
        self.add_import("math", "max_int", vec![I64, I64], vec![I64]);
        self.add_import("math", "signum", vec![F64], vec![F64]);
        self.add_import("math", "signum_int", vec![I64], vec![I64]);
    }

    fn register_vdom_imports(&mut self) {
        use ValType::*;
        // VDOM uses I64 for strings (pointer to length-prefixed data in memory)
        // The JS runtime reads the string from WASM memory
        self.add_import("vdom", "create_vnode", vec![I64], vec![I32]); // tagStrRef -> vnodeId
        self.add_import("vdom", "create_text_vnode", vec![I64], vec![I32]); // textStrRef -> vnodeId
        self.add_import("vdom", "create_fragment", vec![], vec![I32]);
        self.add_import("vdom", "set_vnode_prop", vec![I32, I64, I64], vec![]); // vnodeId, nameStrRef, value
        self.add_import("vdom", "set_vnode_str_prop", vec![I32, I64, I64], vec![]); // vnodeId, nameStrRef, valueStrRef
        self.add_import("vdom", "set_vnode_style", vec![I32, I64, I64], vec![]); // vnodeId, propStrRef, valueStrRef
        self.add_import("vdom", "append_vnode_child", vec![I32, I32], vec![]);
        self.add_import("vdom", "diff_and_patch", vec![I32, I32, I32], vec![]);
        self.add_import("vdom", "mount_vnode", vec![I32, I64], vec![I32]); // vnodeId, selectorStrRef -> domId
        self.add_import("vdom", "dispose", vec![I32], vec![]);
    }

    fn register_signal_imports(&mut self) {
        use ValType::*;
        self.add_import("signal", "create", vec![I64], vec![I32]);
        self.add_import("signal", "get", vec![I32], vec![I64]);
        self.add_import("signal", "set", vec![I32, I64], vec![]);
        self.add_import("signal", "subscribe", vec![I32, I32], vec![I32]);
        self.add_import("signal", "unsubscribe", vec![I32], vec![]);
        self.add_import("signal", "batch_start", vec![], vec![]);
        self.add_import("signal", "batch_end", vec![], vec![]);
        self.add_import("signal", "computed", vec![I32], vec![I32]);
        self.add_import("signal", "effect", vec![I32], vec![I32]);
    }

    fn register_async_imports(&mut self) {
        use ValType::*;
        // Promise creation and manipulation
        self.add_import("async", "promise_new", vec![], vec![I32]);
        self.add_import("async", "promise_resolve", vec![I32, I64], vec![]);
        self.add_import("async", "promise_reject", vec![I32, I32, I32], vec![]);
        self.add_import("async", "promise_then", vec![I32, I32, I32], vec![I32]);
        self.add_import("async", "promise_catch", vec![I32, I32], vec![I32]);
        self.add_import("async", "promise_all", vec![I32], vec![I32]);
        self.add_import("async", "promise_race", vec![I32], vec![I32]);
        self.add_import("async", "spawn", vec![I32], vec![I32]);
        self.add_import("async", "yield_now", vec![], vec![]);

        // Await support - suspends current execution and resumes with promise result
        // (promise_ptr) -> result_value
        // The JS runtime handles suspension/resumption via Asyncify or similar
        self.add_import("async", "await_promise", vec![I32], vec![I64]);

        // Create a continuation for state machine style async
        // (state_machine_ptr, next_state) -> continuation_ptr
        self.add_import("async", "create_continuation", vec![I32, I32], vec![I32]);

        // Resume execution from a suspended state
        // (state_machine_ptr, value) -> ()
        self.add_import("async", "resume", vec![I32, I64], vec![]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_registry() {
        let registry = ImportRegistry::empty();
        assert_eq!(registry.import_count(), 0);
    }

    #[test]
    fn test_standard_imports_registered() {
        let registry = ImportRegistry::new();
        // Should have many imports registered
        assert!(registry.import_count() > 50);
    }

    #[test]
    fn test_get_func_by_name() {
        let registry = ImportRegistry::new();

        // Console imports should be registered
        assert!(registry.get_func("console_log_i64").is_some());
        assert!(registry.get_func("console_log_f64").is_some());

        // Morpheme imports should be registered
        assert!(registry.get_func("morpheme_array_new").is_some());
        assert!(registry.get_func("morpheme_array_map").is_some());

        // Unknown function should return None
        assert!(registry.get_func("unknown_function").is_none());
    }

    #[test]
    fn test_type_deduplication() {
        let mut registry = ImportRegistry::empty();

        // Same signature should return same type index
        let idx1 = registry.get_or_create_type(vec![ValType::I64], vec![ValType::I64]);
        let idx2 = registry.get_or_create_type(vec![ValType::I64], vec![ValType::I64]);
        assert_eq!(idx1, idx2);

        // Different signature should return different type index
        let idx3 = registry.get_or_create_type(vec![ValType::I32], vec![ValType::I64]);
        assert_ne!(idx1, idx3);
    }

    #[test]
    fn test_add_custom_import() {
        let mut registry = ImportRegistry::empty();

        let func_idx = registry.add_import("custom", "my_func", vec![ValType::I64], vec![]);
        assert_eq!(func_idx, 0);
        assert_eq!(registry.get_func("custom_my_func"), Some(0));
    }

    #[test]
    fn test_import_categories() {
        let registry = ImportRegistry::new();

        // Check each category has at least one import
        let categories = [
            "console_log",
            "dom_create",
            "events_add",
            "timing_now",
            "fetch_start",
            "storage_local",
            "router_push",
            "memory_alloc",
            "morpheme_array",
            "math_sqrt",
            "vdom_create",
            "signal_create",
            "async_promise",
        ];

        for prefix in categories {
            let found = registry
                .imports()
                .iter()
                .any(|i| i.qualified_name().starts_with(prefix));
            assert!(found, "Missing import category: {}", prefix);
        }
    }

    #[test]
    fn test_imports_have_valid_types() {
        let registry = ImportRegistry::new();

        for import in registry.imports() {
            let type_idx = import.type_idx as usize;
            assert!(
                type_idx < registry.types().len(),
                "Import {} has invalid type index {}",
                import.qualified_name(),
                type_idx
            );
        }
    }
}
