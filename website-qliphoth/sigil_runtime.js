/**
 * Sigil WASM Runtime
 *
 * Provides JS implementations of all WASM imports required by Sigil programs.
 * This is the bridge between WASM and browser APIs.
 */

// =============================================================================
// Memory Management
// =============================================================================

let wasmMemory = null;
let wasmExports = null;
// The bump pointer, shared with the module.
//
// The module allocates too — enum construction bumps its own `__heap_ptr`
// global inline — so a separate pointer on this side is a second allocator over
// the same memory, and the two hand out the same addresses. When the module
// exports its global, that IS the pointer; `localHeapPtr` is the fallback for a
// module compiled before the export existed.
let localHeapPtr = 1024 * 64; // Start heap after 64KB stack
let heapGlobal = null;

function getHeapPtr() {
    return heapGlobal ? heapGlobal.value : localHeapPtr;
}
function setHeapPtr(v) {
    if (heapGlobal) heapGlobal.value = v;
    else localHeapPtr = v;
}

function setWasmExports(exports) {
    wasmExports = exports;
    wasmMemory = exports.memory;
    // Start the host's bump allocator above the module's own data.
    //
    // This side and the module's side are two allocators over one linear
    // memory, and the host used to assume the literals ended by 64 KB. The
    // Lares client's reach 71740, so every host allocation in the first 6 KB
    // overwrote a string the module would later read — a VNode's tag came back
    // empty and `createElement('')` threw, from the largest component only,
    // because the small ones never allocated enough to reach the overlap.
    // `__heap_base` is where the compiler put its own heap; anything at or
    // below it belongs to the module.
    heapGlobal = exports.__heap_ptr instanceof WebAssembly.Global
        ? exports.__heap_ptr
        : null;
    // An actor compiles to a `<Actor>_dispatch(msg_id, payload)` export. Their
    // presence is what tells this runtime that an `on*` prop carries a message
    // id rather than an indirect-function-table index — a module with no actors
    // has no dispatcher and keeps the older behaviour exactly.
    messageDispatchers = Object.entries(exports)
        .filter(([name, fn]) => name.endsWith('_dispatch') && typeof fn === 'function')
        .map(([name, fn]) => [name.slice(0, -'_dispatch'.length), fn]);
}

// Actor dispatchers, as [actorName, fn] pairs.
let messageDispatchers = [];
// Optional page hook, run after a message is delivered — where a re-render goes.
let afterMessage = null;

/// Register a callback to run after each dispatched message.
///
/// An actor handler mutates state in WASM globals; nothing re-renders on its
/// own. This is where a page puts its `render()` call.
export function onMessageDispatched(fn) {
    afterMessage = fn;
}

/// Deliver a message id (and optional payload) to every actor in the module.
///
/// Returns false when the module has no actors, so the caller can fall back to
/// the function-pointer convention.
export function dispatchMessage(msgId, payload = 0) {
    if (messageDispatchers.length === 0) {
        return false;
    }
    for (const [name, fn] of messageDispatchers) {
        try {
            fn(BigInt(msgId), BigInt(payload));
        } catch (e) {
            console.error(`[dispatch] ${name} failed for message ${msgId}:`, e);
        }
    }
    if (afterMessage) {
        afterMessage(Number(msgId), Number(payload));
    }
    return true;
}

function getMemory() {
    return new Uint8Array(wasmMemory.buffer);
}

function readString(ptr, len) {
    const p = Number(ptr);
    const l = Number(len);
    const mem = getMemory();
    const bytes = mem.slice(p, p + l);
    return new TextDecoder().decode(bytes);
}

// Read a length-prefixed string (Sigil's format: 4-byte len + bytes)
function readLengthPrefixedString(ptr) {
    const p = Number(ptr); // Convert BigInt from WASM to Number
    const mem = getMemory();
    const view = new DataView(wasmMemory.buffer);
    // A pointer that is not a string reads a garbage 32-bit length — up to 4 GB
    // — and this used to slice all of it, so one type confusion in the compiler
    // came out as a multi-megabyte allocation from a component that renders a
    // button. A length that does not fit in memory is not a string.
    if (!Number.isFinite(p) || p < 0 || p + 4 > view.byteLength) return '';
    const len = view.getUint32(p, true); // little-endian
    if (p + 4 + len > view.byteLength) return '';
    const bytes = mem.slice(p + 4, p + 4 + len);
    return new TextDecoder().decode(bytes);
}

function writeString(str) {
    const bytes = new TextEncoder().encode(str);
    heapReserve(bytes.length + 1);
    const ptr = getHeapPtr();
    const mem = getMemory();
    mem.set(bytes, ptr);
    setHeapPtr(ptr + bytes.length + 1); // +1 for null terminator
    return { ptr, len: bytes.length };
}

// Write a length-prefixed string (Sigil's format: 4-byte len + bytes)
// Make sure `size` more bytes fit above `heapPtr`, growing linear memory if
// they do not.
//
// Nothing grew memory on the string path: the module declares 16 pages and the
// host wrote past the end of them the moment it produced enough strings, with
// "offset is out of bounds" from a DataView write and no mention of memory. It
// took until `map` and `filter` actually ran — each one allocating a string per
// element — for a page to be exhausted at all.
function heapReserve(size) {
    const need = getHeapPtr() + size;
    if (!wasmMemory || wasmMemory.buffer.byteLength >= need) return;
    const have = wasmMemory.buffer.byteLength / 65536;
    const want = Math.ceil(need / 65536);
    try {
        wasmMemory.grow(want - have);
    } catch (e) {
        throw new RangeError(
            `Sigil heap exhausted: needed ${need} bytes, memory caps out at ` +
            `${wasmMemory.buffer.byteLength} (${e.message})`,
        );
    }
}

function writeLengthPrefixedString(str) {
    const bytes = new TextEncoder().encode(str);
    heapReserve(4 + bytes.length + 8);
    const ptr = getHeapPtr();
    const view = new DataView(wasmMemory.buffer);
    // Write 4-byte length
    view.setUint32(ptr, bytes.length, true); // little-endian
    // Write string bytes
    const mem = getMemory();
    mem.set(bytes, ptr + 4);
    // Align to 8 bytes
    setHeapPtr((ptr + 4 + bytes.length + 7) & ~7);
    return ptr;
}

// =============================================================================
// Signal System - Fine-grained Reactivity
// =============================================================================

const signals = new Map();          // id -> value
const signalSubscribers = new Map(); // id -> Set of effect runners
let nextSignalId = 1;
let nextEffectId = 1;

// Dependency tracking
let currentEffect = null;           // Currently executing effect (for auto-tracking)
let batchDepth = 0;
const pendingEffects = new Set();

// Effect registry
const effects = new Map();          // effectId -> { run, deps, cleanup }
// =============================================================================
// JSON
// =============================================================================
//
// Sigil's JSON is `json_parse` / `json_stringify` / `json_get` / `json_set` /
// `json_pretty`, and the WASM backend had no binding for any of them — under the
// unresolved-call stub they compiled to a constant 0, so a web target could not
// read or write JSON and would not say so. Values are opaque handles here, the
// same way vnodes and arrays are; strings are length-prefixed pointers.

const jsonValues = new Map();

// Everything in WASM is a uniform i64, so a JSON argument arrives as a bare
// number and could be a value handle, a pointer to a length-prefixed string, or
// an actual number. Handles are issued from a range no string pointer and no
// plausible integer can reach, and string data starts at HEAP_START (0x4000) —
// so the three are told apart by magnitude, then confirmed.
const JSON_HANDLE_BASE = 0x7000_0000;
const WASM_DATA_START = 0x4000;
let nextJsonId = JSON_HANDLE_BASE;

function jsonHandle(value) {
    const id = nextJsonId++;
    jsonValues.set(id, value);
    return BigInt(id);
}

/// Is this plausibly a pointer to a length-prefixed UTF-8 string?
function looksLikeStringPointer(n) {
    if (!Number.isInteger(n) || n < WASM_DATA_START) return false;
    const mem = getMemory();
    if (n + 4 > mem.byteLength) return false;
    const len = new DataView(wasmMemory.buffer).getUint32(n, true);
    return len <= 1 << 20 && n + 4 + len <= mem.byteLength;
}

/// Resolve an argument to a JS value: a handle we issued, a string we can read,
/// or the number itself.
function jsonResolve(ref) {
    const n = Number(ref);
    if (jsonValues.has(n)) {
        return jsonValues.get(n);
    }
    if (looksLikeStringPointer(n)) {
        try {
            return readLengthPrefixedString(n);
        } catch {
            return n;
        }
    }
    return n;
}

// JavaScript truthiness. The compiler emits `x·to_bool()` wherever React relied
// on `&&`, `||` or a ternary over a non-boolean. It is a host call because only
// the host can tell a string pointer from a small integer — `""` is falsy, and a
// pointer to it is not zero.
function valueToBool(ref) {
    const v = jsonResolve(ref);
    if (typeof v === 'string') return BigInt(v.length > 0 ? 1 : 0);
    if (Array.isArray(v)) return BigInt(v.length > 0 ? 1 : 0);
    if (v === null || v === undefined) return 0n;
    if (typeof v === 'number') return BigInt(v !== 0 ? 1 : 0);
    if (typeof v === 'boolean') return BigInt(v ? 1 : 0);
    return 1n;
}

function jsonParse(strRef) {
    const text = readLengthPrefixedString(strRef);
    try {
        return BigInt(jsonHandle(JSON.parse(text)));
    } catch (e) {
        console.error('[json.parse]', e.message);
        return BigInt(jsonHandle(null));
    }
}

function jsonStringify(ref) {
    return BigInt(writeLengthPrefixedString(JSON.stringify(jsonResolve(ref)) ?? 'null'));
}

function jsonPretty(ref) {
    return BigInt(writeLengthPrefixedString(JSON.stringify(jsonResolve(ref), null, 2) ?? 'null'));
}

/// Dotted path lookup, matching the interpreter's `json_get`: each segment is an
/// object key, or an array index when the segment parses as a number.
function jsonGet(ref, pathRef) {
    const path = readLengthPrefixedString(pathRef);
    let current = jsonResolve(ref);
    for (const key of path.split('.')) {
        if (current == null) break;
        if (Array.isArray(current)) {
            const i = Number.parseInt(key, 10);
            current = Number.isNaN(i) ? null : (current[i] ?? null);
        } else if (typeof current === 'object') {
            current = current[key] ?? null;
        } else {
            current = null;
        }
    }
    return BigInt(jsonHandle(current ?? null));
}

function jsonSet(ref, pathRef, valueRef) {
    const path = readLengthPrefixedString(pathRef).split('.');
    const root = structuredClone(jsonResolve(ref) ?? {});
    let current = root;
    for (const key of path.slice(0, -1)) {
        if (current[key] == null || typeof current[key] !== 'object') {
            current[key] = {};
        }
        current = current[key];
    }
    current[path[path.length - 1]] = jsonResolve(valueRef);
    return jsonHandle(root);
}


function signalCreate(initialValue) {
    const id = nextSignalId++;
    signals.set(id, initialValue);
    signalSubscribers.set(id, new Set());
    console.log(`[signal.create] id=${id} value=${initialValue}`);
    return id;
}

function signalGet(id) {
    const sigId = Number(id);
    // Auto-track dependency if we're inside an effect
    if (currentEffect !== null) {
        const subs = signalSubscribers.get(sigId);
        if (subs && !subs.has(currentEffect)) {
            subs.add(currentEffect);
            currentEffect.deps.add(sigId);
        }
    }
    const value = signals.get(sigId) ?? 0n;
    console.log(`[signal.get] id=${sigId} value=${value}`);
    return value;
}

function signalSet(id, value) {
    const sigId = Number(id);
    const old = signals.get(sigId);
    console.log(`[signal.set] id=${sigId} old=${old} new=${value}`);
    if (old !== value) {
        signals.set(sigId, value);
        // Notify subscribers
        const subs = signalSubscribers.get(sigId);
        if (subs) {
            if (batchDepth > 0) {
                subs.forEach(effect => pendingEffects.add(effect));
            } else {
                // Run effects immediately, but avoid infinite loops
                const toRun = [...subs];
                toRun.forEach(effect => {
                    if (effect !== currentEffect) {
                        effect.run();
                    }
                });
            }
        }
    }
}

function signalSubscribe(id, callbackPtr) {
    const sigId = Number(id);
    const cbIdx = Number(callbackPtr);
    const callback = () => {
        if (wasmExports && wasmExports.__indirect_function_table) {
            wasmExports.__indirect_function_table.get(cbIdx)();
        }
    };
    const subs = signalSubscribers.get(sigId) || new Set();
    const handle = { run: callback, deps: new Set() };
    subs.add(handle);
    signalSubscribers.set(sigId, subs);
    return callbackPtr;
}

function signalUnsubscribe(handle) {
    // Would need to track handles properly in production
}

function signalBatchStart() {
    batchDepth++;
    console.log(`[signal.batch_start] depth=${batchDepth}`);
}

function signalBatchEnd() {
    batchDepth--;
    console.log(`[signal.batch_end] depth=${batchDepth} pending=${pendingEffects.size}`);
    if (batchDepth === 0 && pendingEffects.size > 0) {
        const toRun = [...pendingEffects];
        pendingEffects.clear();
        toRun.forEach(effect => effect.run());
    }
}

function signalComputed(computePtr) {
    const cbIdx = Number(computePtr);
    // Create a computed signal - derives value from other signals
    const id = nextSignalId++;
    signals.set(id, 0n);
    signalSubscribers.set(id, new Set());

    const compute = () => {
        if (wasmExports && wasmExports.__indirect_function_table) {
            const result = wasmExports.__indirect_function_table.get(cbIdx)();
            const old = signals.get(id);
            if (result !== old) {
                signals.set(id, result);
                // Notify our subscribers
                const subs = signalSubscribers.get(id);
                if (subs && subs.size > 0) {
                    subs.forEach(effect => {
                        if (effect !== currentEffect) {
                            effect.run();
                        }
                    });
                }
            }
            return result;
        }
        return 0n;
    };

    // Create an effect that recomputes when dependencies change
    const effectRunner = {
        deps: new Set(),
        run: () => {
            const prev = currentEffect;
            currentEffect = effectRunner;
            try {
                compute();
            } finally {
                currentEffect = prev;
            }
        }
    };

    // Run once to establish dependencies and get initial value
    effectRunner.run();

    console.log(`[signal.computed] id=${id} initial=${signals.get(id)}`);
    return id;
}

function signalEffect(effectPtr) {
    const cbIdx = Number(effectPtr);
    // Create and run a side effect that auto-tracks dependencies
    const effectId = nextEffectId++;

    const effectRunner = {
        id: effectId,
        deps: new Set(),
        cleanup: null,
        run: () => {
            // Clean up previous run
            if (effectRunner.cleanup) {
                effectRunner.cleanup();
                effectRunner.cleanup = null;
            }

            // Clear old subscriptions
            for (const sigId of effectRunner.deps) {
                const subs = signalSubscribers.get(sigId);
                if (subs) {
                    subs.delete(effectRunner);
                }
            }
            effectRunner.deps.clear();

            // Run effect with dependency tracking
            const prev = currentEffect;
            currentEffect = effectRunner;
            try {
                if (wasmExports && wasmExports.__indirect_function_table) {
                    wasmExports.__indirect_function_table.get(cbIdx)();
                }
            } finally {
                currentEffect = prev;
            }
        }
    };

    effects.set(effectId, effectRunner);

    // Run immediately to establish dependencies
    effectRunner.run();

    console.log(`[signal.effect] id=${effectId} deps=${[...effectRunner.deps].join(',')}`);
    return effectId;
}

// =============================================================================
// Console
// =============================================================================

function consoleLogI64(value) {
    console.log('[sigil]', String(value));
}

function consoleLogF64(value) {
    console.log('[sigil]', value);
}

function consoleLogStr(ptr, len) {
    console.log('[sigil]', readString(ptr, len));
}

// console.log / warn / error take a single length-prefixed string pointer, not
// the (ptr, len) pair `log_str` takes. The compiler has emitted all three since
// the string macros landed; this runtime implemented none of them, and a WASM
// module that imports a function the runtime does not provide does not degrade
// — `WebAssembly.instantiate` throws and nothing runs at all.
function consoleLog(strRef) {
    console.log('[sigil]', readLengthPrefixedString(strRef));
}

function consoleWarn(strRef) {
    console.warn('[sigil]', readLengthPrefixedString(strRef));
}

function consoleError(strRef) {
    console.error('[sigil]', readLengthPrefixedString(strRef));
}

function consolePrint(value) {
    console.log(value);
}

// =============================================================================
// String Operations - All strings are length-prefixed (4-byte len + bytes)
// =============================================================================

function stringConcat(ptr1, ptr2) {
    // Read both strings and concatenate
    const str1 = readLengthPrefixedString(ptr1);
    const str2 = readLengthPrefixedString(ptr2);
    const result = str1 + str2;
    console.log('[string.concat]', JSON.stringify(str1), '+', JSON.stringify(str2), '=', JSON.stringify(result));
    return writeLengthPrefixedString(result);
}

// `x·len()` compiles to THIS import for every receiver — the compiler tries
// `string_length` first and it is always registered, so the array branch below
// it was dead and `xs·len()` on an array read a length prefix out of a
// collection id. The host is the only side that knows which kind a handle is,
// so it answers for all four.
function stringLength(ptr) {
    const n = Number(ptr);
    const arr = arrays.get(n);
    if (arr) return arr.length;
    const m = maps.get(n);
    if (m) return m.size;
    const st = sets.get(n);
    if (st) return st.size;
    const str = readLengthPrefixedString(ptr);
    return str.length;
}

function stringSlice(ptr, start, end) {
    const str = readLengthPrefixedString(ptr);
    const result = str.slice(Number(start), Number(end));
    return writeLengthPrefixedString(result);
}

function stringEq(ptr1, ptr2) {
    const str1 = readLengthPrefixedString(ptr1);
    const str2 = readLengthPrefixedString(ptr2);
    return str1 === str2 ? 1 : 0;
}

function stringFromInt(value) {
    const str = String(value);
    console.log('[string.from_int]', value, '=>', JSON.stringify(str));
    return writeLengthPrefixedString(str);
}

function stringFromFloat(value) {
    const str = String(value);
    console.log('[string.from_float]', value, '=>', JSON.stringify(str));
    return writeLengthPrefixedString(str);
}

// ---------------------------------------------------------------------------
// HashMap / HashSet. A Sigil map is a host object, the same way an array is:
// the module holds an id, not a buffer. Keys arrive as string handles, so they
// are read to text — two equal strings at different addresses are one key.
// ---------------------------------------------------------------------------
const maps = new Map();

// Arrays and maps share one id space. `xs·len()` and `∀ x ∈ xs` dispatch on the
// handle alone — they cannot tell which kind it is — so two separate counters
// meant array 1 and map 1 both existed and the array always won: a map with
// entries in it reported a length of 0.
let nextCollectionId = 1;

// The lowest address a string handle can have. Below it, a value is a number:
// string literals and the heap both live well above this, and `getUint32` at a
// small address happily reads a plausible-looking length out of whatever is
// there — so `xs·contains(9)` matched a "string" at address 9 and answered
// true for an array of 1, 2, 3.
const MIN_STRING_HANDLE = 1024;

function mapKey(k) {
    // A key is a string handle when it points at a readable length-prefixed
    // string, and a plain number otherwise.
    const n = Number(k);
    if (n < MIN_STRING_HANDLE) return n;
    try {
        const view = new DataView(getMemory().buffer);
        const len = view.getUint32(n, true);
        if (len < 4096 && n + 4 + len <= view.byteLength) {
            return readLengthPrefixedString(n);
        }
    } catch { /* not a string */ }
    return n;
}

function mapNew() {
    const id = nextCollectionId++;
    maps.set(id, new Map());
    return id;
}
function mapSet(mapId, k, v) {
    const m = maps.get(Number(mapId));
    if (m) m.set(mapKey(k), v);
}
function mapGet(mapId, k) {
    const m = maps.get(Number(mapId));
    const v = m ? m.get(mapKey(k)) : undefined;
    return v === undefined ? 0n : BigInt(v);
}
function mapHas(mapId, k) {
    const n = Number(mapId);
    if (sets.has(n)) return setHas(n, k);
    const m = maps.get(n);
    return m && m.has(mapKey(k)) ? 1 : 0;
}
function mapRemove(mapId, k) {
    const n = Number(mapId);
    if (sets.has(n)) { setRemove(n, k); return; }
    const m = maps.get(n);
    if (m) m.delete(mapKey(k));
}
function mapLen(mapId) {
    const n = Number(mapId);
    if (sets.has(n)) return setLen(n);
    const m = maps.get(n);
    return m ? m.size : 0;
}
function mapIsEmpty(mapId) {
    return mapLen(mapId) === 0 ? 1 : 0;
}
function arrayOf(values) {
    const id = arrayNew();
    const arr = arrays.get(id);
    for (const v of values) arr.push(v);
    return id;
}
function mapKeys(mapId) {
    const n = Number(mapId);
    if (sets.has(n)) return setValues(n);
    const m = maps.get(n);
    if (!m) return arrayOf([]);
    return arrayOf([...m.keys()].map((k) =>
        typeof k === 'string' ? BigInt(writeLengthPrefixedString(k)) : BigInt(k)));
}
function mapValues(mapId) {
    const n = Number(mapId);
    if (sets.has(n)) return setValues(n);
    const m = maps.get(n);
    return arrayOf(m ? [...m.values()].map(toI64) : []);
}
function mapEntries(mapId) {
    const n = Number(mapId);
    if (sets.has(n)) return setValues(n);
    const m = maps.get(n);
    if (!m) return arrayOf([]);
    // Each entry is a two-element array: [key, value].
    return arrayOf([...m.entries()].map(([k, v]) => BigInt(arrayOf([
        typeof k === 'string' ? BigInt(writeLengthPrefixedString(k)) : BigInt(k),
        BigInt(v),
    ]))));
}

// `HashMap·from(entries)` — an array of two-element [k, v] arrays, which is
// what `Object.entries(x)` and `new Map(pairs)` both hand over.
function mapFrom(srcId) {
    const id = mapNew();
    const m = maps.get(id);
    const arr = arrays.get(Number(srcId));
    if (arr) {
        for (const pair of arr) {
            const entry = arrays.get(Number(pair));
            if (entry && entry.length >= 2) m.set(mapKey(entry[0]), entry[1]);
        }
        return id;
    }
    const other = maps.get(Number(srcId));
    if (other) for (const [k, v] of other) m.set(k, v);
    return id;
}

function encodeUriComponent(ptr) {
    return writeLengthPrefixedString(encodeURIComponent(readLengthPrefixedString(ptr)));
}
function decodeUriComponent(ptr) {
    const s = readLengthPrefixedString(ptr);
    try {
        return writeLengthPrefixedString(decodeURIComponent(s));
    } catch {
        // A malformed escape is not a reason to trap the whole module.
        return writeLengthPrefixedString(s);
    }
}

// `a·locale_compare(b)` — JavaScript's `localeCompare`, which every migrated
// sort comparator uses.
function stringLocaleCompare(aRef, bRef) {
    const a = readLengthPrefixedString(aRef);
    const b = readLengthPrefixedString(bRef);
    return BigInt(a.localeCompare(b));
}

function stringParseInt(ptr) {
    const str = readLengthPrefixedString(ptr);
    return BigInt(parseInt(str, 10) || 0);
}

function stringParseFloat(ptr) {
    const str = readLengthPrefixedString(ptr);
    return parseFloat(str) || 0.0;
}

function stringLines(ptr) {
    const str = readLengthPrefixedString(ptr);
    const lines = str.split('\n');
    // Return as array handle
    const arr = lines.map(line => writeLengthPrefixedString(line));
    return createArrayFromValues(arr);
}

function stringSplitWhitespace(ptr) {
    const str = readLengthPrefixedString(ptr);
    const parts = str.trim().split(/\s+/).filter(s => s.length > 0);
    const arr = parts.map(part => writeLengthPrefixedString(part));
    return createArrayFromValues(arr);
}

function stringSplit(ptr, delimPtr) {
    const str = readLengthPrefixedString(ptr);
    const delim = readLengthPrefixedString(delimPtr);
    const parts = str.split(delim);
    const arr = parts.map(part => writeLengthPrefixedString(part));
    return createArrayFromValues(arr);
}

function stringTrim(ptr) {
    const str = readLengthPrefixedString(ptr);
    return writeLengthPrefixedString(str.trim());
}

function stringTrimStart(ptr) {
    const str = readLengthPrefixedString(ptr);
    return writeLengthPrefixedString(str.trimStart());
}

function stringTrimEnd(ptr) {
    const str = readLengthPrefixedString(ptr);
    return writeLengthPrefixedString(str.trimEnd());
}

function stringToUppercase(ptr) {
    const str = readLengthPrefixedString(ptr);
    return writeLengthPrefixedString(str.toUpperCase());
}

function stringToLowercase(ptr) {
    const str = readLengthPrefixedString(ptr);
    return writeLengthPrefixedString(str.toLowerCase());
}

function stringContains(ptr, searchPtr) {
    const n = Number(ptr);
    if (sets.has(n)) return setHas(n, searchPtr) ? 1 : 0;
    const arr = arrays.get(n);
    if (arr) {
        const needle = mapKey(searchPtr);
        return arr.some((v) => mapKey(v) === needle) ? 1 : 0;
    }
    const str = readLengthPrefixedString(ptr);
    const search = readLengthPrefixedString(searchPtr);
    return str.includes(search) ? 1 : 0;
}

function stringStartsWith(ptr, prefixPtr) {
    const str = readLengthPrefixedString(ptr);
    const prefix = readLengthPrefixedString(prefixPtr);
    return str.startsWith(prefix) ? 1 : 0;
}

function stringEndsWith(ptr, suffixPtr) {
    const str = readLengthPrefixedString(ptr);
    const suffix = readLengthPrefixedString(suffixPtr);
    return str.endsWith(suffix) ? 1 : 0;
}

function stringReplace(ptr, fromPtr, toPtr) {
    const str = readLengthPrefixedString(ptr);
    const from = readLengthPrefixedString(fromPtr);
    const to = readLengthPrefixedString(toPtr);
    return writeLengthPrefixedString(str.replaceAll(from, to));
}

function stringChars(ptr) {
    const str = readLengthPrefixedString(ptr);
    const chars = [...str].map(ch => writeLengthPrefixedString(ch));
    return createArrayFromValues(chars);
}

// Helper for string functions that return arrays
function createArrayFromValues(values) {
    const arrId = arrayNew();
    for (const val of values) {
        arrayPush(arrId, val);
    }
    return arrId;
}

// =============================================================================
// DOM Operations - All strings are length-prefixed (4-byte len + bytes)
// =============================================================================

const domElements = new Map();
let nextDomId = 1;

const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';
const SVG_TAGS = new Set(['svg', 'path', 'circle', 'rect', 'line', 'polyline', 'polygon', 'ellipse', 'g', 'defs', 'use', 'text', 'tspan', 'image', 'clipPath', 'mask', 'pattern', 'linearGradient', 'radialGradient', 'stop', 'symbol', 'marker', 'foreignObject']);

function domCreateElement(tagPtr) {
    const tag = readLengthPrefixedString(tagPtr);
    console.log('[dom.create_element]', tag);
    const el = SVG_TAGS.has(tag.toLowerCase())
        ? document.createElementNS(SVG_NAMESPACE, tag)
        : document.createElement(tag);
    const id = nextDomId++;
    domElements.set(id, el);
    return id;
}

function domCreateText(textPtr) {
    const text = readLengthPrefixedString(textPtr);
    console.log('[dom.create_text]', text);
    const node = document.createTextNode(text);
    const id = nextDomId++;
    domElements.set(id, node);
    return id;
}

function domSetAttribute(elId, namePtr, valuePtr) {
    const el = domElements.get(Number(elId));
    if (el) {
        const name = readLengthPrefixedString(namePtr);
        const value = readLengthPrefixedString(valuePtr);
        console.log('[dom.set_attribute]', elId, name, '=', value);
        el.setAttribute(name, value);
    }
}

function domRemoveAttribute(elId, namePtr) {
    const el = domElements.get(Number(elId));
    if (el) {
        const name = readLengthPrefixedString(namePtr);
        console.log('[dom.remove_attribute]', elId, name);
        el.removeAttribute(name);
    }
}

function domSetProperty(elId, namePtr, value) {
    const el = domElements.get(Number(elId));
    if (el) {
        const name = readLengthPrefixedString(namePtr);
        console.log('[dom.set_property]', elId, name, '=', value);
        el[name] = value;
    }
}

function domSetInnerHTML(elId, htmlPtr) {
    const el = domElements.get(Number(elId));
    if (el) {
        const html = readLengthPrefixedString(htmlPtr);
        console.log('[dom.set_inner_html]', elId, html.substring(0, 50) + '...');
        el.innerHTML = html;
    }
}

function domAppendChild(parentId, childId) {
    const parent = domElements.get(Number(parentId));
    const child = domElements.get(Number(childId));
    if (parent && child) {
        parent.appendChild(child);
    }
}

function domInsertBefore(parentId, newId, refId) {
    const parent = domElements.get(Number(parentId));
    const newNode = domElements.get(Number(newId));
    const ref = domElements.get(Number(refId));
    if (parent && newNode) {
        parent.insertBefore(newNode, ref);
    }
}

function domRemoveChild(parentId, childId) {
    const parent = domElements.get(Number(parentId));
    const child = domElements.get(Number(childId));
    if (parent && child) {
        parent.removeChild(child);
    }
}

function domReplaceChild(parentId, newId, oldId) {
    const parent = domElements.get(Number(parentId));
    const newNode = domElements.get(Number(newId));
    const oldNode = domElements.get(Number(oldId));
    if (parent && newNode && oldNode) {
        parent.replaceChild(newNode, oldNode);
    }
}

function domSetTextContent(elId, textPtr) {
    const el = domElements.get(Number(elId));
    if (el) {
        const text = readLengthPrefixedString(textPtr);
        console.log('[dom.set_text_content]', elId, text);
        el.textContent = text;
    }
}

function domGetElementById(idPtr) {
    const id = readLengthPrefixedString(idPtr);
    console.log('[dom.get_element_by_id]', id);
    const el = document.getElementById(id);
    if (el) {
        const domId = nextDomId++;
        domElements.set(domId, el);
        return domId;
    }
    return 0;
}

function domQuerySelector(selectorPtr) {
    const selector = readLengthPrefixedString(selectorPtr);
    console.log('[dom.query_selector]', selector);
    const el = document.querySelector(selector);
    if (el) {
        const id = nextDomId++;
        domElements.set(id, el);
        return id;
    }
    return 0;
}

function domCloneNode(elId, deep) {
    const el = domElements.get(Number(elId));
    if (el) {
        const clone = el.cloneNode(!!deep);
        const id = nextDomId++;
        domElements.set(id, clone);
        return id;
    }
    return 0;
}

function domGetValue(elId) {
    const el = domElements.get(Number(elId));
    if (el && 'value' in el) {
        return writeLengthPrefixedString(el.value);
    }
    return 0;
}

function domSetValue(elId, valuePtr) {
    const el = domElements.get(Number(elId));
    if (el && 'value' in el) {
        const value = readLengthPrefixedString(valuePtr);
        el.value = value;
    }
}

function domFocus(elId) {
    const el = domElements.get(Number(elId));
    if (el && el.focus) el.focus();
}

function domScrollTo(elId, x, y) {
    const el = domElements.get(Number(elId));
    if (el) el.scrollTo(Number(x), Number(y));
}

// =============================================================================
// Events
// =============================================================================

const eventListeners = new Map();
let nextListenerId = 1;

function eventsAddListener(elId, typePtr, callbackPtr, flags) {
    const el = domElements.get(Number(elId));
    if (!el) {
        console.warn('[events.add_listener] Element not found:', elId);
        // `-> i32`, so a Number. The success path below already returns one;
        // this early return did not, and a listener on an element that is not
        // there is the common case during a first render.
        return 0;
    }

    const type = readLengthPrefixedString(typePtr);
    const fnIdx = Number(callbackPtr);

    const listener = (event) => {
        if (wasmExports && wasmExports.__indirect_function_table) {
            try {
                const fn = wasmExports.__indirect_function_table.get(fnIdx);
                if (fn) fn();
            } catch (e) {
                console.error('[events] callback error:', e);
            }
        }
    };

    el.addEventListener(type, listener);
    const id = nextListenerId++;
    eventListeners.set(id, { el, type, listener });
    return id; // Return I32 (Number), not BigInt
}

function eventsRemoveListener(listenerId) {
    const id = Number(listenerId);
    const info = eventListeners.get(id);
    if (info) {
        info.el.removeEventListener(info.type, info.listener);
        eventListeners.delete(id);
    }
}

function eventsPreventDefault(eventId) {
    // Simplified
}

function eventsStopPropagation(eventId) {
    // Simplified
}

function eventsGetTarget(eventId) {
    return 0;
}

function eventsGetValue(eventId, resultPtr) {
    return 0;
}

// =============================================================================
// Timing
// =============================================================================

function timingNow() {
    return performance.now();
}

function timingSetTimeout(callbackPtr, ms) {
    const cb = Number(callbackPtr);
    const delay = Number(ms);
    return setTimeout(() => {
        if (wasmExports && wasmExports.__indirect_function_table) {
            wasmExports.__indirect_function_table.get(cb)();
        }
    }, delay);
}

function timingClearTimeout(id) {
    clearTimeout(Number(id));
}

function timingSetInterval(callbackPtr, ms) {
    const cb = Number(callbackPtr);
    const interval = Number(ms);
    return setInterval(() => {
        if (wasmExports && wasmExports.__indirect_function_table) {
            wasmExports.__indirect_function_table.get(cb)();
        }
    }, interval);
}

function timingClearInterval(id) {
    clearInterval(Number(id));
}

function timingRequestAnimationFrame(callbackPtr) {
    const cb = Number(callbackPtr);
    return requestAnimationFrame((time) => {
        if (wasmExports && wasmExports.__indirect_function_table) {
            wasmExports.__indirect_function_table.get(cb)(time);
        }
    });
}

// =============================================================================
// Fetch
// =============================================================================

const fetchRequests = new Map();
let nextFetchId = 1;

function fetchStart(urlPtr, urlLen, method) {
    const url = readString(urlPtr, urlLen);
    const id = nextFetchId++;

    fetch(url)
        .then(res => {
            fetchRequests.set(id, { status: res.status, body: null, done: false, response: res });
            return res.text();
        })
        .then(body => {
            const req = fetchRequests.get(id);
            if (req) {
                req.body = body;
                req.done = true;
            }
        })
        .catch(err => {
            fetchRequests.set(id, { status: 0, body: null, done: true, error: err });
        });

    fetchRequests.set(id, { status: 0, body: null, done: false });
    return id;
}

function fetchPoll(id) {
    const req = fetchRequests.get(Number(id));
    return req?.done ? 1 : 0;
}

function fetchGetStatus(id) {
    const req = fetchRequests.get(Number(id));
    return req?.status ?? 0;
}

function fetchGetBody(id) {
    const req = fetchRequests.get(Number(id));
    if (req?.body) {
        return writeLengthPrefixedString(req.body);
    }
    return 0;
}

function fetchGetHeaders(id) {
    const req = fetchRequests.get(Number(id));
    if (req?.response) {
        const headers = {};
        req.response.headers.forEach((value, key) => {
            headers[key] = value;
        });
        return writeLengthPrefixedString(JSON.stringify(headers));
    }
    return 0;
}

function fetchAbort(id) {
    fetchRequests.delete(Number(id));
}

// =============================================================================
// Storage
// =============================================================================

function storageLocalGet(keyPtr) {
    const key = readLengthPrefixedString(keyPtr);
    const value = localStorage.getItem(key);
    console.log('[storage.local_get]', key, '->', value || '(not found)');
    if (value) {
        return writeLengthPrefixedString(value);
    }
    return 0;
}

function storageLocalSet(keyPtr, valuePtr) {
    const key = readLengthPrefixedString(keyPtr);
    const value = readLengthPrefixedString(valuePtr);
    localStorage.setItem(key, value);
    console.log('[storage.local_set]', key, '=', value);
}

function storageLocalRemove(keyPtr) {
    const key = readLengthPrefixedString(keyPtr);
    localStorage.removeItem(key);
    console.log('[storage.local_remove]', key);
}

function storageLocalClear() {
    localStorage.clear();
    console.log('[storage.local_clear]');
}

function storageLocalKeys() {
    const keys = Object.keys(localStorage);
    console.log('[storage.local_keys]', keys);
    // Return as morpheme array handle
    const arrId = arrayNew();
    keys.forEach(k => arrayPush(arrId, writeLengthPrefixedString(k)));
    return arrId;
}

// =============================================================================
// Router
// =============================================================================

function routerPushState(urlPtr) {
    const url = readLengthPrefixedString(urlPtr);
    console.log('[router.push_state]', url);
    history.pushState(null, '', url);
}

function routerReplaceState(urlPtr) {
    const url = readLengthPrefixedString(urlPtr);
    console.log('[router.replace_state]', url);
    history.replaceState(null, '', url);
}

function routerGetPathname() {
    console.log('[router.get_pathname]', location.pathname);
    return writeLengthPrefixedString(location.pathname);
}

function routerGo(delta) {
    console.log('[router.go]', delta);
    history.go(delta);
}

function routerBack() {
    console.log('[router.back]');
    history.back();
}

function routerForward() {
    console.log('[router.forward]');
    history.forward();
}

// =============================================================================
// Memory
// =============================================================================

function memoryAlloc(size) {
    heapReserve(Number(size));
    const ptr = getHeapPtr();
    setHeapPtr(ptr + Number(size));
    return ptr;
}

function memoryRealloc(ptr, newSize) {
    // Simplified - just allocate new
    return memoryAlloc(newSize);
}

function memoryFree(ptr) {
    // No-op for bump allocator
}

function heapAlloc(size) {
    return BigInt(memoryAlloc(Number(size)));
}

// =============================================================================
// Math
// =============================================================================

const mathImports = {
    sqrt: (x) => Math.sqrt(Number(x)),
    sin: (x) => Math.sin(Number(x)),
    cos: (x) => Math.cos(Number(x)),
    tan: (x) => Math.tan(Number(x)),
    pow: (x, y) => Math.pow(Number(x), Number(y)),
    exp: (x) => Math.exp(Number(x)),
    log: (x) => Math.log(Number(x)),
    floor: (x) => Math.floor(Number(x)),
    ceil: (x) => Math.ceil(Number(x)),
    round: (x) => Math.round(Number(x)),
    abs: (x) => Math.abs(Number(x)),
    abs_int: (x) => x < 0n ? -x : x,
    random: Math.random,
    clamp: (x, min, max) => Math.min(Math.max(Number(x), Number(min)), Number(max)),
    clamp_int: (x, min, max) => x < min ? min : (x > max ? max : x),
    min: (a, b) => Math.min(Number(a), Number(b)),
    max: (a, b) => Math.max(Number(a), Number(b)),
    min_int: (a, b) => a < b ? a : b,
    max_int: (a, b) => a > b ? a : b,
    signum: (x) => x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0),
    signum_int: (x) => x > 0n ? 1n : (x < 0n ? -1n : 0n),
};

// =============================================================================
// Morpheme (Array) Operations
// =============================================================================

const arrays = new Map();

// ---------------------------------------------------------------------------
// HashSet. A set is its own host collection, not a map with dummy values: `∀ x
// ∈ set` has to yield the elements, and a map yields `[k, v]` pairs. Ids come
// from the one shared counter, so every dispatcher below can tell the three
// kinds apart by handle alone.
// ---------------------------------------------------------------------------
const sets = new Map();

function materializeKey(k) {
    return typeof k === 'string' ? BigInt(writeLengthPrefixedString(k)) : BigInt(k);
}

function setNew() {
    const id = nextCollectionId++;
    sets.set(id, new Set());
    return id;
}
function setAdd(setId, v) {
    const s = sets.get(Number(setId));
    if (s) s.add(mapKey(v));
}
function setHas(setId, v) {
    const s = sets.get(Number(setId));
    return s && s.has(mapKey(v)) ? 1 : 0;
}
function setRemove(setId, v) {
    const s = sets.get(Number(setId));
    if (s) s.delete(mapKey(v));
}
function setLen(setId) {
    const s = sets.get(Number(setId));
    return s ? s.size : 0;
}
function setValues(setId) {
    const s = sets.get(Number(setId));
    return arrayOf(s ? [...s].map(materializeKey) : []);
}
// `HashSet·from(xs)` — an array, another set, or a map's keys.
function setFrom(srcId) {
    const id = setNew();
    const s = sets.get(id);
    const n = Number(srcId);
    const arr = arrays.get(n);
    if (arr) {
        for (const v of arr) s.add(mapKey(v));
        return id;
    }
    const other = sets.get(n);
    if (other) {
        for (const k of other) s.add(k);
        return id;
    }
    const m = maps.get(n);
    if (m) {
        for (const k of m.keys()) s.add(k);
    }
    return id;
}


function arrayNew() {
    const id = nextCollectionId++;
    arrays.set(id, []);
    return id;
}

function arrayPush(arrId, value) {
    const arr = arrays.get(Number(arrId));
    if (arr) arr.push(value);
}

function arrayGet(arrId, index) {
    const n = Number(arrId);
    // `x·get(k)` compiles to THIS import whatever the receiver is, so a map
    // reached it with a key where an index belongs and read past the end of an
    // array it does not have. The host is the only side that knows the kind.
    const m = maps.get(n);
    if (m) {
        const v = m.get(mapKey(index));
        return v === undefined ? 0n : toI64(v);
    }
    const arr = arrays.get(n);
    if (!arr) return 0n;
    // The module declares an i64 result, so the value has to BE a BigInt.
    // An element that arrived as a plain number — anything the host pushed
    // itself — threw "Cannot convert N to a BigInt" at the call boundary,
    // which reads as a compiler bug and is a host one.
    return toI64(arr[Number(index)]);
}

/// Whatever a collection holds, as the i64 the module expects.
function toI64(v) {
    if (typeof v === 'bigint') return v;
    if (typeof v === 'number') return BigInt(Math.trunc(v));
    if (typeof v === 'boolean') return v ? 1n : 0n;
    if (typeof v === 'string') return BigInt(writeLengthPrefixedString(v));
    return 0n;
}

function arraySet(arrId, index, value) {
    const arr = arrays.get(Number(arrId));
    if (arr) arr[Number(index)] = value;
}

function arrayLen(arrId) {
    // `xs·len()` dispatches here for any collection, so a map has to answer
    // too — otherwise `attrs·len()` was 0 for a map with entries in it.
    const n = Number(arrId);
    const arr = arrays.get(n);
    if (arr) return arr.length;
    const m = maps.get(n);
    if (m) return m.size;
    const st = sets.get(n);
    return st ? st.size : 0;
}

// `∀` iterates this. A map becomes its entries — an array of two-element arrays
// — and an array is already itself, so `∀ x ∈ xs` and `∀ (k, v) ∈ m` are one
// loop over different contents. See S57: before this the loop read a 4-byte
// length out of linear memory, which is not where a Vec lives.
function iterOf(id) {
    const n = Number(id);
    if (arrays.has(n)) return n;
    if (maps.has(n)) return mapEntries(n);
    if (sets.has(n)) return setValues(n);
    // Not a collection this runtime knows: iterate nothing rather than trap.
    return arrayNew();
}

// Vec::join(separator). The compiler wraps both pointers to i32 before the
// call and extends the result back to i64.
function vecJoin(arrId, sepStrRef) {
    const arr = arrays.get(Number(arrId));
    const sep = readLengthPrefixedString(sepStrRef);
    if (!arr) {
        return writeLengthPrefixedString('');
    }
    // Elements are either string handles or numbers; a handle points at a
    // length-prefixed string inside the same linear memory.
    const parts = arr.map((v) => {
        const n = Number(v);
        try {
            return readLengthPrefixedString(n);
        } catch {
            return String(n);
        }
    });
    return writeLengthPrefixedString(parts.join(sep));
}

// A Sigil closure, callable from the host.
//
// The value is the pair `[table_idx, env_ptr]` in linear memory, and the
// calling convention is `(env, args…)` — the same one `call_indirect` uses.
// `null` when the pointer is not a closure, so a caller can fall back rather
// than trap.
function sigilClosure(closurePtr) {
    const ptr = Number(closurePtr);
    const table = wasmExports && wasmExports.__indirect_function_table;
    if (!ptr || !table) return null;
    let tableIdx, env;
    try {
        const view = new DataView(getMemory().buffer);
        tableIdx = Number(view.getBigInt64(ptr, true));
        env = view.getBigInt64(ptr + 8, true);
    } catch {
        return null;
    }
    let fn;
    try {
        fn = table.get(tableIdx);
    } catch {
        return null;
    }
    if (typeof fn !== 'function') return null;
    // Every Sigil closure has one signature, `(env, a0 … aN) -> i64`, so that a
    // table entry and a call site cannot disagree on arity. A caller with fewer
    // arguments pads: an omitted parameter arrives as `undefined`, which the
    // boundary rejects with "Cannot convert undefined to a BigInt".
    return (...args) => {
        const padded = args.slice(0, Math.max(0, fn.length - 1));
        while (padded.length < fn.length - 1) padded.push(0n);
        return fn(env, ...padded);
    };
}

// The higher-order array morphemes. Every one of these used to ignore its
// closure and hand the receiver straight back — `xs·map(f)` returned `xs`,
// `xs·filter(p)` returned `xs`, `xs·fold(f, init)` returned `init`. They
// compiled, validated and reported success, so a view rendered a list of the
// wrong things rather than failing.
//
// A handle that is not an array is returned unchanged: `Option·map` compiles
// to the same import, and that IS the identity on a value the host does not
// hold.
function arrayMap(arrId, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return Number(arrId);
    return arrayOf(arr.map((x) => toI64(call(x))));
}

function arrayFilter(arrId, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return Number(arrId);
    return arrayOf(arr.filter((x) => truthy(call(x))));
}

// `xs·fold(init, f)`. The INITIAL VALUE comes first — that is the order the
// interpreter accepts, and it is the oracle. The closure itself takes the
// accumulator first, matching Sigil's `|sum, v|`.
function arrayReduce(arrId, initial, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return toI64(initial);
    let acc = toI64(initial);
    for (const x of arr) acc = toI64(call(acc, x));
    return acc;
}

function arrayFind(arrId, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return 0n;
    const hit = arr.find((x) => truthy(call(x)));
    return hit === undefined ? 0n : toI64(hit);
}

// The index, or -1. Not an Option: the compiler's callers compare against -1.
function arrayPosition(arrId, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return -1;
    return arr.findIndex((x) => truthy(call(x)));
}

function arrayAnyBy(arrId, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return 0;
    return arr.some((x) => truthy(call(x))) ? 1 : 0;
}

function arrayAllBy(arrId, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return 0;
    return arr.every((x) => truthy(call(x))) ? 1 : 0;
}

// `xs·flat_map(f)`: each result that is itself an array is spliced in.
function arrayFlatMap(arrId, fnPtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(fnPtr);
    if (!arr || !call) return Number(arrId);
    const out = [];
    for (const x of arr) {
        const r = call(x);
        const inner = arrays.get(Number(r));
        if (inner) out.push(...inner);
        else out.push(toI64(r));
    }
    return arrayOf(out);
}

function arrayFlatten(arrId) {
    const arr = arrays.get(Number(arrId));
    if (!arr) return Number(arrId);
    const out = [];
    for (const x of arr) {
        const inner = arrays.get(Number(x));
        if (inner) out.push(...inner);
        else out.push(toI64(x));
    }
    return arrayOf(out);
}

function arrayReverse(arrId) {
    const arr = arrays.get(Number(arrId));
    if (!arr) return Number(arrId);
    return arrayOf([...arr].reverse());
}

// What a Sigil closure returning a bool hands back: 0/1 as i64, or a value the
// host is asked to judge. `0n` and `0` are the only falsehoods a predicate can
// return — a string handle is a pointer, and every pointer is truthy.
function truthy(v) {
    if (typeof v === 'bigint') return v !== 0n;
    if (typeof v === 'number') return v !== 0;
    return Boolean(v);
}

// `xs·sort(|a, b| …)`. The comparator is a Sigil closure — the pair
// `[table_idx, env_ptr]` in linear memory — and the uniform calling convention
// is `(env, args…)`, so the host can call it like any other.
function arraySortBy(arrId, closurePtr) {
    const arr = arrays.get(Number(arrId));
    const call = sigilClosure(closurePtr);
    if (!arr || !call) return Number(arrId);
    arr.sort((a, b) => Number(call(a, b)));
    return Number(arrId);
}

function arraySort(arrId) {
    const arr = arrays.get(Number(arrId));
    if (arr) arr.sort((a, b) => Number(a - b));
    return arrId;
}

function arrayFirst(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr && arr.length > 0 ? arr[0] : 0n;
}

function arrayLast(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr && arr.length > 0 ? arr[arr.length - 1] : 0n;
}

function arrayNth(arrId, n) {
    const arr = arrays.get(Number(arrId));
    return arr ? (arr[Number(n)] ?? 0n) : 0n;
}

function arraySum(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr ? arr.reduce((a, b) => a + b, 0n) : 0n;
}

function arrayProduct(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr && arr.length > 0 ? arr.reduce((a, b) => a * b, 1n) : 0n;
}

function arrayMin(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr && arr.length > 0 ? arr.reduce((a, b) => a < b ? a : b) : 0n;
}

function arrayMax(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr && arr.length > 0 ? arr.reduce((a, b) => a > b ? a : b) : 0n;
}

function arrayAll(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr && arr.every(x => x) ? 1 : 0;
}

function arrayAny(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr && arr.some(x => x) ? 1 : 0;
}

function arrayRandomElement(arrId) {
    const arr = arrays.get(Number(arrId));
    if (arr && arr.length > 0) {
        return arr[Math.floor(Math.random() * arr.length)];
    }
    return 0n;
}

// Parallel morphemes (simplified - just run sequentially)
function arrayParallelMap(arrId, fnPtr) { return arrayMap(arrId, fnPtr); }
function arrayParallelFilter(arrId, fnPtr) { return arrayFilter(arrId, fnPtr); }
function arrayParallelReduce(arrId, fnPtr, initial) { return arrayReduce(arrId, fnPtr, initial); }

// =============================================================================
// VDOM - Virtual DOM with real DOM rendering
// =============================================================================

const vnodes = new Map();
let nextVnodeId = 1;

// Map vnode IDs to their rendered DOM elements
const vnodeToDom = new Map();

function vdomCreateVnode(tagStrRef) {
    const id = nextVnodeId++;
    const tag = readLengthPrefixedString(tagStrRef);
    vnodes.set(id, { tag, props: {}, children: [], isText: false });
    console.log(`[vdom.create_vnode] id=${id} tag=${tag}`);
    return id; // Return I32 (Number), not BigInt
}

function vdomCreateTextVnode(textStrRef) {
    const id = nextVnodeId++;
    const text = readLengthPrefixedString(textStrRef);
    vnodes.set(id, { text, isText: true });
    console.log(`[vdom.create_text_vnode] id=${id} text=${JSON.stringify(text)}`);
    return id; // Return I32 (Number), not BigInt
}

function vdomCreateFragment() {
    const id = nextVnodeId++;
    vnodes.set(id, { isFragment: true, children: [] });
    console.log(`[vdom.create_fragment] id=${id}`);
    return id; // Return I32 (Number), not BigInt
}

function vdomSetVnodeProp(vnodeId, nameStrRef, value) {
    const id = Number(vnodeId);
    const vnode = vnodes.get(id);
    if (vnode && !vnode.isText) {
        const name = readLengthPrefixedString(nameStrRef);
        vnode.props[name] = value;
        console.log(`[vdom.set_prop] id=${id} ${name}=${value}`);
    }
}

function vdomSetVnodeStrProp(vnodeId, nameStrRef, valueStrRef) {
    const id = Number(vnodeId);
    const vnode = vnodes.get(id);
    if (vnode && !vnode.isText) {
        const name = readLengthPrefixedString(nameStrRef);
        const value = readLengthPrefixedString(valueStrRef);
        vnode.props[name] = value;
        console.log(`[vdom.set_str_prop] id=${id} ${name}=${JSON.stringify(value)}`);
    }
}

function vdomSetVnodeStyle(vnodeId, propStrRef, valueStrRef) {
    const id = Number(vnodeId);
    const vnode = vnodes.get(id);
    if (vnode && !vnode.isText) {
        const prop = readLengthPrefixedString(propStrRef);
        const value = readLengthPrefixedString(valueStrRef);
        vnode.style ??= {};
        vnode.style[prop] = value;
        console.log(`[vdom.set_style] id=${id} ${prop}=${JSON.stringify(value)}`);
    }
}

function vdomAppendVnodeChild(parentId, childId) {
    const pId = Number(parentId);
    const cId = Number(childId);
    const parent = vnodes.get(pId);
    if (parent && !parent.isText) {
        parent.children = parent.children || [];
        parent.children.push(cId);
        console.log(`[vdom.append_child] parent=${pId} child=${cId}`);
    }
}

// Render a vnode to an actual DOM element
function renderVnodeToDom(vnodeId) {
    const vnode = vnodes.get(vnodeId);
    if (!vnode) return null;

    if (vnode.isText) {
        return document.createTextNode(vnode.text);
    }

    if (vnode.isFragment) {
        const frag = document.createDocumentFragment();
        for (const childId of (vnode.children || [])) {
            const childDom = renderVnodeToDom(childId);
            if (childDom) frag.appendChild(childDom);
        }
        return frag;
    }

    // Regular element (with SVG namespace support)
    const el = SVG_TAGS.has(vnode.tag.toLowerCase())
        ? document.createElementNS(SVG_NAMESPACE, vnode.tag)
        : document.createElement(vnode.tag);

    // Set properties/attributes
    for (const [name, value] of Object.entries(vnode.props || {})) {
        if (name.endsWith('_payload')) {
            continue; // read by the event handler above, never an attribute
        }
        if (name.startsWith('on')) {
            // Event handler. In an actor module the value is a message id, which
            // is what `VNode·on_click(message_id: u64)` is declared to take; in a
            // flat FFI module it is an indirect-function-table index. Try the
            // dispatcher first — it reports false when there are no actors.
            const eventName = name.slice(2).toLowerCase();
            // A message id alone cannot say *which* row was clicked, and
            // `VNode·on_click(message_id: u64)` has no slot for it. A sibling
            // `<event>_payload` prop carries the argument.
            const payload = vnode.props?.[`${name}_payload`] ?? 0;
            el.addEventListener(eventName, () => {
                if (typeof value !== 'bigint' && typeof value !== 'number') {
                    return;
                }
                if (dispatchMessage(value, payload)) {
                    return;
                }
                const table = wasmExports?.__indirect_function_table;
                if (table) {
                    try { table.get(Number(value))(); } catch (e) { console.error(e); }
                }
            });
        } else if (name === 'style' && typeof value === 'string') {
            el.setAttribute('style', value);
        } else if (name === 'class' || name === 'className') {
            // `setAttribute`, not `className`: on an SVG element `className` is
            // a read-only SVGAnimatedString, so assigning it threw and took the
            // whole mount down. Qliphoth's `bell_glyph` is an `<svg>`.
            el.setAttribute('class', String(value));
        } else if (name === 'id') {
            el.id = String(value);
        } else if (name === 'innerHTML') {
            el.innerHTML = String(value);
        } else if (typeof value === 'string') {
            el.setAttribute(name, value);
        } else if (typeof value === 'boolean' || value === 1n || value === 1) {
            el.setAttribute(name, '');
        }
    }

    // Append children
    for (const childId of (vnode.children || [])) {
        const childDom = renderVnodeToDom(childId);
        if (childDom) el.appendChild(childDom);
    }

    return el;
}

function vdomDiffAndPatch(oldId, newId, domId) {
    // For now, just replace - full diffing is future work
    const oId = Number(oldId);
    const nId = Number(newId);
    const dId = Number(domId);
    const oldDom = vnodeToDom.get(oId) || domElements.get(dId);
    if (oldDom && oldDom.parentNode) {
        const newDom = renderVnodeToDom(nId);
        if (newDom) {
            oldDom.parentNode.replaceChild(newDom, oldDom);
            vnodeToDom.set(nId, newDom);
        }
    }
}

function vdomMountVnode(vnodeId, selectorStrRef) {
    const id = Number(vnodeId);
    const selector = readLengthPrefixedString(selectorStrRef);
    console.log(`[vdom.mount] id=${id} selector=${selector}`);

    const container = document.querySelector(selector) || document.getElementById(selector.replace('#', ''));
    if (!container) {
        console.error(`[vdom.mount] Container not found: ${selector}`);
        return 0; // Return I32 (Number), not BigInt
    }

    const dom = renderVnodeToDom(id);
    if (dom) {
        container.innerHTML = ''; // Clear existing content
        container.appendChild(dom);
        vnodeToDom.set(id, dom);
        console.log(`[vdom.mount] Mounted vnode ${id} to ${selector}`);
        return id; // Return I32 (Number), not BigInt
    }
    return 0; // Return I32 (Number), not BigInt
}

function vdomDispose(vnodeId) {
    const id = Number(vnodeId);
    const dom = vnodeToDom.get(id);
    if (dom && dom.parentNode) {
        dom.parentNode.removeChild(dom);
    }
    vnodeToDom.delete(id);
    vnodes.delete(id);
}

// =============================================================================
// Promise - Async/Await Support
// =============================================================================

const promises = new Map();
let nextPromiseId = 1;

// Promise states
const PROMISE_PENDING = 0, PROMISE_RESOLVED = 1, PROMISE_REJECTED = 2;

function promiseNew() {
    const id = nextPromiseId++;
    promises.set(id, {
        id,
        state: PROMISE_PENDING,
        value: 0n,
        error: null,
        thenCallbacks: [],
        catchCallbacks: [],
    });
    console.log('[promise.new] ->', id);
    return id;
}

function promiseResolve(id, value) {
    const p = promises.get(Number(id));
    if (!p || p.state !== PROMISE_PENDING) return;
    p.state = PROMISE_RESOLVED;
    p.value = value;
    console.log('[promise.resolve]', id, value);
    // Execute then callbacks
    for (const cb of p.thenCallbacks) {
        try {
            if (typeof cb === 'function') cb(value);
        } catch (e) {
            console.error('[promise] then callback error:', e);
        }
    }
}

function promiseReject(id, errorPtr, errorLen) {
    const p = promises.get(Number(id));
    if (!p || p.state !== PROMISE_PENDING) return;
    p.state = PROMISE_REJECTED;
    p.error = errorPtr ? readLengthPrefixedString(errorPtr) : 'Unknown error';
    console.log('[promise.reject]', id, p.error);
    // Execute catch callbacks
    for (const cb of p.catchCallbacks) {
        try {
            if (typeof cb === 'function') cb(p.error);
        } catch (e) {
            console.error('[promise] catch callback error:', e);
        }
    }
}

function promiseThen(id, callbackTableIdx, envPtr) {
    const p = promises.get(Number(id));
    if (!p) return 0;
    const newPromiseId = promiseNew();
    console.log('[promise.then]', id, '-> new promise', newPromiseId);

    const callback = (value) => {
        console.log('[promise] then callback triggered with', value);
        promiseResolve(newPromiseId, value);
    };

    if (p.state === PROMISE_RESOLVED) {
        setTimeout(() => callback(p.value), 0);
    } else if (p.state === PROMISE_PENDING) {
        p.thenCallbacks.push(callback);
    }
    return newPromiseId;
}

function promiseCatch(id, callbackTableIdx) {
    const p = promises.get(Number(id));
    if (!p) return 0;
    const newPromiseId = promiseNew();
    console.log('[promise.catch]', id, '-> new promise', newPromiseId);

    const callback = (error) => {
        console.log('[promise] catch callback triggered:', error);
        promiseResolve(newPromiseId, 0n);
    };

    if (p.state === PROMISE_REJECTED) {
        setTimeout(() => callback(p.error), 0);
    } else if (p.state === PROMISE_PENDING) {
        p.catchCallbacks.push(callback);
    }
    return newPromiseId;
}

function promiseAll(arrayId) {
    const id = promiseNew();
    console.log('[promise.all] -> promise', id);
    // Simplified: resolve immediately (full impl would wait for all)
    setTimeout(() => promiseResolve(id, 0n), 0);
    return id;
}

function promiseRace(arrayId) {
    const id = promiseNew();
    console.log('[promise.race] -> promise', id);
    // Simplified: resolve immediately (full impl would wait for first)
    setTimeout(() => promiseResolve(id, 0n), 0);
    return id;
}

function promiseSpawn(funcTableIdx) {
    const taskId = nextPromiseId++;
    console.log('[promise.spawn] task', taskId);
    return taskId;
}

function promiseYieldNow() {
    console.log('[promise.yield_now]');
}

function promiseAwait(id) {
    const p = promises.get(Number(id));
    if (!p) return 0n;
    console.log('[promise.await]', id, 'state=', p.state);
    if (p.state === PROMISE_RESOLVED) {
        return p.value;
    }
    console.log('[promise] WARNING: await on pending promise');
    return 0n;
}

function promiseContinuation(stateMachinePtr, nextState) {
    const contId = nextPromiseId++;
    console.log('[promise.continuation]', contId, 'state:', nextState);
    return contId;
}

function promiseResume(stateMachinePtr, value) {
    console.log('[promise.resume] ptr:', stateMachinePtr, 'value:', value);
}

// =============================================================================
// Export Runtime
// =============================================================================

// Debug wrapper to catch BigInt errors
function wrapImports(imports, moduleName) {
    const wrapped = {};
    for (const [name, fn] of Object.entries(imports)) {
        if (typeof fn === 'function') {
            wrapped[name] = (...args) => {
                try {
                    return fn(...args);
                } catch (e) {
                    if (e.message?.includes('BigInt')) {
                        console.error(`[BIGINT ERROR] ${moduleName}.${name}`, args, e);
                    }
                    throw e;
                }
            };
        } else {
            wrapped[name] = fn;
        }
    }
    return wrapped;
}

export function createImports() {
    return {
        console: wrapImports({
            log_i64: consoleLogI64,
            log_f64: consoleLogF64,
            log_str: consoleLogStr,
            print: consolePrint,
            println_i64: consoleLogI64,
            println_f64: consoleLogF64,
            println_str: consoleLogStr,
            println: consolePrint,
            log: consoleLog,
            warn: consoleWarn,
            error: consoleError,
        }, 'console'),
        map: wrapImports({
            new: mapNew,
            set: mapSet,
            get: mapGet,
            has: mapHas,
            remove: mapRemove,
            len: mapLen,
            is_empty: mapIsEmpty,
            keys: mapKeys,
            values: mapValues,
            entries: mapEntries,
            iter_of: iterOf,
            from: mapFrom,
        }, 'map'),
        set: wrapImports({
            new: setNew,
            from: setFrom,
            add: setAdd,
            has: setHas,
            remove: setRemove,
            len: setLen,
            values: setValues,
        }, 'set'),
        string: wrapImports({
            concat: stringConcat,
            length: stringLength,
            slice: stringSlice,
            eq: stringEq,
            from_int: stringFromInt,
            from_float: stringFromFloat,
            parse_int: stringParseInt,
            parse_float: stringParseFloat,
            lines: stringLines,
            split_whitespace: stringSplitWhitespace,
            split: stringSplit,
            trim: stringTrim,
            trim_start: stringTrimStart,
            trim_end: stringTrimEnd,
            to_uppercase: stringToUppercase,
            to_lowercase: stringToLowercase,
            contains: stringContains,
            starts_with: stringStartsWith,
            ends_with: stringEndsWith,
            locale_compare: stringLocaleCompare,
            encode_uri_component: encodeUriComponent,
            decode_uri_component: decodeUriComponent,
            replace: stringReplace,
            chars: stringChars,
        }, 'string'),
        dom: {
            create_element: domCreateElement,
            create_text: domCreateText,
            set_attribute: domSetAttribute,
            remove_attribute: domRemoveAttribute,
            set_property: domSetProperty,
            set_inner_html: domSetInnerHTML,
            append_child: domAppendChild,
            insert_before: domInsertBefore,
            remove_child: domRemoveChild,
            replace_child: domReplaceChild,
            set_text_content: domSetTextContent,
            get_element_by_id: domGetElementById,
            query_selector: domQuerySelector,
            clone_node: domCloneNode,
            get_value: domGetValue,
            set_value: domSetValue,
            focus: domFocus,
            scroll_to: domScrollTo,
        },
        events: {
            add_listener: eventsAddListener,
            remove_listener: eventsRemoveListener,
            prevent_default: eventsPreventDefault,
            stop_propagation: eventsStopPropagation,
            get_target: eventsGetTarget,
            get_value: eventsGetValue,
        },
        timing: {
            now: timingNow,
            set_timeout: timingSetTimeout,
            clear_timeout: timingClearTimeout,
            set_interval: timingSetInterval,
            clear_interval: timingClearInterval,
            request_animation_frame: timingRequestAnimationFrame,
        },
        fetch: {
            start: fetchStart,
            poll: fetchPoll,
            get_status: fetchGetStatus,
            get_body: fetchGetBody,
            get_headers: fetchGetHeaders,
            abort: fetchAbort,
        },
        storage: {
            local_get: storageLocalGet,
            local_set: storageLocalSet,
            local_remove: storageLocalRemove,
            local_clear: storageLocalClear,
            local_keys: storageLocalKeys,
        },
        router: {
            push_state: routerPushState,
            replace_state: routerReplaceState,
            get_pathname: routerGetPathname,
            go: routerGo,
            back: routerBack,
            forward: routerForward,
        },
        memory: {
            alloc: memoryAlloc,
            realloc: memoryRealloc,
            free: memoryFree,
            heap_alloc: heapAlloc,
        },
        math: mathImports,
        morpheme: {
            array_new: arrayNew,
            vec_join: vecJoin,
            array_push: arrayPush,
            array_get: arrayGet,
            array_set: arraySet,
            array_len: arrayLen,
            array_map: arrayMap,
            array_filter: arrayFilter,
            array_reduce: arrayReduce,
            array_sort: arraySort,
            array_sort_by: arraySortBy,
            array_first: arrayFirst,
            array_last: arrayLast,
            array_nth: arrayNth,
            array_sum: arraySum,
            array_product: arrayProduct,
            array_min: arrayMin,
            array_max: arrayMax,
            array_all: arrayAll,
            array_any: arrayAny,
            array_find: arrayFind,
            array_position: arrayPosition,
            array_any_by: arrayAnyBy,
            array_all_by: arrayAllBy,
            array_flat_map: arrayFlatMap,
            array_flatten: arrayFlatten,
            array_reverse: arrayReverse,
            array_random_element: arrayRandomElement,
            array_parallel_map: arrayParallelMap,
            array_parallel_filter: arrayParallelFilter,
            array_parallel_reduce: arrayParallelReduce,
        },
        vdom: {
            create_vnode: vdomCreateVnode,
            create_text_vnode: vdomCreateTextVnode,
            create_fragment: vdomCreateFragment,
            set_vnode_prop: vdomSetVnodeProp,
            set_vnode_str_prop: vdomSetVnodeStrProp,
            set_vnode_style: vdomSetVnodeStyle,
            append_vnode_child: vdomAppendVnodeChild,
            diff_and_patch: vdomDiffAndPatch,
            mount_vnode: vdomMountVnode,
            dispose: vdomDispose,
        },
        value: {
            to_bool: valueToBool,
        },
        json: {
            parse: jsonParse,
            stringify: jsonStringify,
            pretty: jsonPretty,
            get: jsonGet,
            set: jsonSet,
        },
        signal: {
            create: signalCreate,
            get: signalGet,
            set: signalSet,
            subscribe: signalSubscribe,
            unsubscribe: signalUnsubscribe,
            batch_start: signalBatchStart,
            batch_end: signalBatchEnd,
            computed: signalComputed,
            effect: signalEffect,
        },
        promise: {
            new: promiseNew,
            resolve: promiseResolve,
            reject: promiseReject,
            then: promiseThen,
            catch: promiseCatch,
            all: promiseAll,
            race: promiseRace,
            spawn: promiseSpawn,
            yield_now: promiseYieldNow,
            await: promiseAwait,
            continuation: promiseContinuation,
            resume: promiseResume,
        },
        // async module - alias for promise functions with async naming convention
        async: {
            promise_new: promiseNew,
            promise_resolve: promiseResolve,
            promise_reject: promiseReject,
            promise_then: promiseThen,
            promise_catch: promiseCatch,
            promise_all: promiseAll,
            promise_race: promiseRace,
            spawn: promiseSpawn,
            yield_now: promiseYieldNow,
            await_promise: promiseAwait,
            create_continuation: promiseContinuation,
            resume: promiseResume,
        },
        // browser module - window/document access
        browser: {
            // Every one of these is declared `-> i32`. They used to return
            // BigInt, so the call itself threw "Cannot convert a BigInt value
            // to a number" — a whole group that could never be called at all.
            window: () => 0,  // Return handle to window
            document: () => 0,  // Return handle to document
            inner_width: () => (typeof window !== 'undefined' ? window.innerWidth : 1920),
            inner_height: () => (typeof window !== 'undefined' ? window.innerHeight : 1080),
            add_event_listener: (target, event, callback, capture) => {
                // Stub - would need callback registry
                return 0;
            },
            remove_event_listener: (target, listenerId) => {},
            match_media: (query) => 0,
            mql_matches: (mql) => 0,
            mql_add_listener: (mql, callback) => 0,
            mql_remove_listener: (mql, listenerId) => {},
            // Dialogs. Outside a browser they answer the way a dismissed
            // dialog does, so a headless render does not trap.
            confirm: (msgPtr) => {
                const msg = readLengthPrefixedString(msgPtr);
                if (typeof window === 'undefined' || !window.confirm) return 0;
                return window.confirm(msg) ? 1 : 0;
            },
            alert: (msgPtr) => {
                const msg = readLengthPrefixedString(msgPtr);
                if (typeof window !== 'undefined' && window.alert) window.alert(msg);
            },
            prompt: (msgPtr, defPtr) => {
                const msg = readLengthPrefixedString(msgPtr);
                const def = defPtr ? readLengthPrefixedString(defPtr) : '';
                if (typeof window === 'undefined' || !window.prompt) {
                    return writeLengthPrefixedString('');
                }
                return writeLengthPrefixedString(window.prompt(msg, def) ?? '');
            },
        },
    };
}

export async function loadWasm(wasmPath, additionalImports = {}) {
    const imports = createImports();

    // Merge additional imports
    Object.assign(imports, additionalImports);

    const response = await fetch(wasmPath);
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, imports);

    setWasmExports(instance.exports);

    return instance;
}

// Convenience function to mount a vnode to a selector from JS
export function mountVnode(vnodeId, selector) {
    const container = document.querySelector(selector) || document.getElementById(selector.replace('#', ''));
    if (!container) {
        console.error(`[mount] Container not found: ${selector}`);
        return 0;
    }
    const dom = renderVnodeToDom(vnodeId);
    if (dom) {
        container.innerHTML = '';
        container.appendChild(dom);
        vnodeToDom.set(vnodeId, dom);
        console.log(`[mount] Mounted vnode ${vnodeId} to ${selector}`);
        return vnodeId;
    }
    return 0;
}

// SigilRuntime class for convenient usage
export class SigilRuntime {
    constructor() {
        this.instance = null;
    }

    getImports() {
        return createImports();
    }

    init(instance) {
        this.instance = instance;
        setWasmExports(instance.exports);
    }

    mount(vnodeId, selector) {
        return mountVnode(vnodeId, selector);
    }
}

export default { createImports, loadWasm, mountVnode, SigilRuntime };
