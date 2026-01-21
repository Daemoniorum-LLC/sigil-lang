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
let heapPtr = 1024 * 64; // Start heap after 64KB stack

function setWasmExports(exports) {
    wasmExports = exports;
    wasmMemory = exports.memory;
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
    const len = view.getUint32(p, true); // little-endian
    const bytes = mem.slice(p + 4, p + 4 + len);
    return new TextDecoder().decode(bytes);
}

function writeString(str) {
    const bytes = new TextEncoder().encode(str);
    const ptr = heapPtr;
    const mem = getMemory();
    mem.set(bytes, ptr);
    heapPtr += bytes.length + 1; // +1 for null terminator
    return { ptr, len: bytes.length };
}

// Write a length-prefixed string (Sigil's format: 4-byte len + bytes)
function writeLengthPrefixedString(str) {
    const bytes = new TextEncoder().encode(str);
    const ptr = heapPtr;
    const view = new DataView(wasmMemory.buffer);
    // Write 4-byte length
    view.setUint32(ptr, bytes.length, true); // little-endian
    // Write string bytes
    const mem = getMemory();
    mem.set(bytes, ptr + 4);
    // Align to 8 bytes
    heapPtr += 4 + bytes.length;
    heapPtr = (heapPtr + 7) & ~7;
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

function stringLength(ptr) {
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
    const str = readLengthPrefixedString(ptr);
    const search = readLengthPrefixedString(searchPtr);
    return str.includes(search) ? 1n : 0n;
}

function stringStartsWith(ptr, prefixPtr) {
    const str = readLengthPrefixedString(ptr);
    const prefix = readLengthPrefixedString(prefixPtr);
    return str.startsWith(prefix) ? 1n : 0n;
}

function stringEndsWith(ptr, suffixPtr) {
    const str = readLengthPrefixedString(ptr);
    const suffix = readLengthPrefixedString(suffixPtr);
    return str.endsWith(suffix) ? 1n : 0n;
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

// =============================================================================
// Events
// =============================================================================

const eventListeners = new Map();
let nextListenerId = 1;

function eventsAddListener(elId, typePtr, callbackPtr, flags) {
    const el = domElements.get(Number(elId));
    if (!el) {
        console.warn('[events.add_listener] Element not found:', elId);
        return 0n;
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
    const ptr = heapPtr;
    heapPtr += size;
    // Grow memory if needed
    const pages = Math.ceil(heapPtr / 65536);
    if (wasmMemory && wasmMemory.buffer.byteLength < pages * 65536) {
        wasmMemory.grow(pages - wasmMemory.buffer.byteLength / 65536);
    }
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
let nextArrayId = 1;

function arrayNew() {
    const id = nextArrayId++;
    arrays.set(id, []);
    return id;
}

function arrayPush(arrId, value) {
    const arr = arrays.get(Number(arrId));
    if (arr) arr.push(value);
}

function arrayGet(arrId, index) {
    const arr = arrays.get(Number(arrId));
    return arr ? (arr[Number(index)] ?? 0n) : 0n;
}

function arraySet(arrId, index, value) {
    const arr = arrays.get(Number(arrId));
    if (arr) arr[Number(index)] = value;
}

function arrayLen(arrId) {
    const arr = arrays.get(Number(arrId));
    return arr ? arr.length : 0;
}

function arrayMap(arrId, fnPtr) {
    // Simplified
    return Number(arrId);
}

function arrayFilter(arrId, fnPtr) {
    return Number(arrId);
}

function arrayReduce(arrId, fnPtr, initial) {
    return initial;
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
        if (name.startsWith('on')) {
            // Event handler - value is a function pointer
            const eventName = name.slice(2).toLowerCase();
            el.addEventListener(eventName, () => {
                // Call WASM function if it's a function index
                if (typeof value === 'bigint' || typeof value === 'number') {
                    const table = wasmExports?.__indirect_function_table;
                    if (table) {
                        try { table.get(Number(value))(); } catch (e) { console.error(e); }
                    }
                }
            });
        } else if (name === 'style' && typeof value === 'string') {
            el.setAttribute('style', value);
        } else if (name === 'class' || name === 'className') {
            el.className = String(value);
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
        }, 'console'),
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
            array_push: arrayPush,
            array_get: arrayGet,
            array_set: arraySet,
            array_len: arrayLen,
            array_map: arrayMap,
            array_filter: arrayFilter,
            array_reduce: arrayReduce,
            array_sort: arraySort,
            array_first: arrayFirst,
            array_last: arrayLast,
            array_nth: arrayNth,
            array_sum: arraySum,
            array_product: arrayProduct,
            array_min: arrayMin,
            array_max: arrayMax,
            array_all: arrayAll,
            array_any: arrayAny,
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
            append_vnode_child: vdomAppendVnodeChild,
            diff_and_patch: vdomDiffAndPatch,
            mount_vnode: vdomMountVnode,
            dispose: vdomDispose,
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
            window: () => 0n,  // Return handle to window
            document: () => 0n,  // Return handle to document
            inner_width: () => BigInt(typeof window !== 'undefined' ? window.innerWidth : 1920),
            inner_height: () => BigInt(typeof window !== 'undefined' ? window.innerHeight : 1080),
            add_event_listener: (target, event, callback, capture) => {
                // Stub - would need callback registry
                return 0n;
            },
            remove_event_listener: (target, listenerId) => {},
            match_media: (query) => 0n,
            mql_matches: (mql) => 0n,
            mql_add_listener: (mql, callback) => 0n,
            mql_remove_listener: (mql, listenerId) => {},
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
