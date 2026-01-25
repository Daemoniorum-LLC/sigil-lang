/**
 * Node.js test for Sigil WASM runtime
 * Tests that demo.wasm runs correctly with sigil_runtime.js imports
 */

import { readFileSync } from 'fs';

// Evidentiality tag helpers
const EVIDENCE = {
    TAG_MASK: 0xF000_0000_0000_0000n,
    VALUE_MASK: 0x0FFF_FFFF_FFFF_FFFFn,
};

function getValue(tagged) {
    return BigInt(tagged) & EVIDENCE.VALUE_MASK;
}

// Memory tracking
let memory;
let heapPointer = 1024;

// String utilities
function readString(ptr, len) {
    if (len === undefined) {
        const view = new DataView(memory.buffer);
        len = view.getUint32(ptr, true);
        ptr += 4;
    }
    const bytes = new Uint8Array(memory.buffer, ptr, len);
    return new TextDecoder().decode(bytes);
}

function writeString(str) {
    const bytes = new TextEncoder().encode(str);
    const ptr = heapPointer;
    const view = new DataView(memory.buffer);
    view.setUint32(ptr, bytes.length, true);
    const dest = new Uint8Array(memory.buffer, ptr + 4, bytes.length);
    dest.set(bytes);
    heapPointer += 4 + bytes.length;
    heapPointer = (heapPointer + 7) & ~7;
    return ptr;
}

// Mock allocations (morpheme arrays)
const allocations = new Map();
let nextAllocId = 1;

// Mock VDOM
const vnodes = new Map();
let nextVnodeId = 1;

// Mock signals
const signals = new Map();
let signalIdCounter = 0;

// Create import object
const imports = {
    console: {
        log_i64: (v) => console.log('  [i64]', Number(getValue(v))),
        log_f64: (v) => console.log('  [f64]', v),
        log_str: (p, l) => console.log('  [str]', readString(p, l)),
        print: (v) => console.log('  [print]', Number(getValue(v))),
    },

    string: {
        concat: (a, b) => writeString(readString(a) + readString(b)),
        length: (p) => readString(p).length,
        slice: (p, s, e) => writeString(readString(p).slice(Number(s), Number(e))),
        eq: (a, b) => readString(a) === readString(b) ? 1n : 0n,
        from_int: (v) => writeString(getValue(v).toString()),
        from_float: (v) => writeString(v.toString()),
        parse_int: (p) => { const v = parseInt(readString(p)); return isNaN(v) ? 0n : BigInt(v); },
        parse_float: (p) => parseFloat(readString(p)) || 0.0,
    },

    dom: {
        create_element: () => 1,
        create_text: () => 1,
        set_attribute: () => {},
        remove_attribute: () => {},
        set_property: () => {},
        append_child: () => {},
        insert_before: () => {},
        remove_child: () => {},
        replace_child: () => {},
        set_text_content: () => {},
        get_element_by_id: () => 1,
        query_selector: () => 1,
        clone_node: () => 1,
    },

    events: {
        add_listener: () => {},
        remove_listener: () => {},
        prevent_default: () => {},
        stop_propagation: () => {},
        get_target: () => 0,
        get_value: () => 0,
    },

    timing: {
        now: () => Date.now(),
        set_timeout: () => 0,
        clear_timeout: () => {},
        set_interval: () => 0,
        clear_interval: () => {},
        request_animation_frame: () => 0,
    },

    fetch: {
        start: () => 0,
        poll: () => 0,
        get_status: () => 0,
        get_body: () => 0,
        abort: () => {},
    },

    storage: {
        local_get: () => 0,
        local_set: () => {},
        local_remove: () => {},
    },

    router: {
        push_state: () => {},
        replace_state: () => {},
        get_pathname: () => 0,
    },

    memory: {
        alloc: (size) => {
            const ptr = heapPointer;
            heapPointer += Number(size);
            heapPointer = (heapPointer + 7) & ~7;
            return ptr;
        },
        realloc: () => 0,
        free: () => {},
        heap_alloc: (size) => {
            const ptr = heapPointer;
            heapPointer += Number(size);
            heapPointer = (heapPointer + 7) & ~7;
            return BigInt(ptr);
        },
    },

    morpheme: {
        array_new: () => {
            const id = nextAllocId++;
            allocations.set(id, []);
            return id;
        },
        array_push: (id, val) => {
            const arr = allocations.get(id);
            if (arr) arr.push(val);
        },
        array_get: (id, idx) => allocations.get(id)?.[Number(idx)] ?? 0n,
        array_set: (id, idx, val) => {
            const arr = allocations.get(id);
            if (arr) arr[Number(idx)] = val;
        },
        array_len: (id) => allocations.get(id)?.length ?? 0,
        array_map: (id) => {
            const newId = nextAllocId++;
            allocations.set(newId, [...(allocations.get(id) || [])]);
            return newId;
        },
        array_filter: (id) => {
            const newId = nextAllocId++;
            allocations.set(newId, [...(allocations.get(id) || [])]);
            return newId;
        },
        array_parallel_map: (id) => id,
        array_parallel_filter: (id) => id,
        array_parallel_reduce: (id, cb, init) => init,
        array_reduce: (id, cb, init) => init,
        array_sort: (id) => {
            const arr = allocations.get(id);
            if (arr) arr.sort((a, b) => a < b ? -1 : a > b ? 1 : 0);
            return id;
        },
        array_first: (id) => allocations.get(id)?.[0] ?? 0n,
        array_last: (id) => {
            const arr = allocations.get(id);
            return arr && arr.length > 0 ? arr[arr.length - 1] : 0n;
        },
        array_nth: (id, n) => allocations.get(id)?.[Number(n)] ?? 0n,
        array_sum: (id) => {
            const arr = allocations.get(id);
            return arr ? arr.reduce((a, b) => BigInt(a) + BigInt(b), 0n) : 0n;
        },
        array_product: (id) => {
            const arr = allocations.get(id);
            return arr ? arr.reduce((a, b) => BigInt(a) * BigInt(b), 1n) : 1n;
        },
        array_min: (id) => {
            const arr = allocations.get(id);
            return arr && arr.length > 0 ? arr.reduce((a, b) => a < b ? a : b) : 0n;
        },
        array_max: (id) => {
            const arr = allocations.get(id);
            return arr && arr.length > 0 ? arr.reduce((a, b) => a > b ? a : b) : 0n;
        },
        array_all: (id) => {
            const arr = allocations.get(id);
            return arr && arr.every(x => x !== 0n && x !== 0) ? 1 : 0;
        },
        array_any: (id) => {
            const arr = allocations.get(id);
            return arr && arr.some(x => x !== 0n && x !== 0) ? 1 : 0;
        },
        array_random_element: (id) => {
            const arr = allocations.get(id);
            return arr && arr.length > 0 ? arr[Math.floor(Math.random() * arr.length)] : 0n;
        },
    },

    math: {
        sqrt: Math.sqrt,
        sin: Math.sin,
        cos: Math.cos,
        tan: Math.tan,
        pow: (a, b) => a ** b,
        exp: Math.exp,
        log: Math.log,
        floor: Math.floor,
        ceil: Math.ceil,
        round: Math.round,
        abs: Math.abs,
        random: Math.random,
    },

    vdom: {
        _readStrRef: (ref) => {
            const ptr = Number(BigInt.asUintN(32, ref));
            return readString(ptr);
        },
        create_vnode: function(tagRef) {
            const tag = this._readStrRef(tagRef);
            const id = nextVnodeId++;
            vnodes.set(id, { tag, props: {}, children: [] });
            console.log(`  [vdom] create_vnode("${tag}") -> ${id}`);
            return id;
        },
        create_text_vnode: function(textRef) {
            const text = this._readStrRef(textRef);
            const id = nextVnodeId++;
            vnodes.set(id, { tag: '#text', text, props: {}, children: [] });
            console.log(`  [vdom] create_text_vnode("${text}") -> ${id}`);
            return id;
        },
        create_fragment: () => {
            const id = nextVnodeId++;
            vnodes.set(id, { tag: '#fragment', props: {}, children: [] });
            return id;
        },
        set_vnode_prop: function(vnodeId, nameRef, value) {
            const vnode = vnodes.get(vnodeId);
            if (!vnode) return;
            const name = this._readStrRef(nameRef);
            vnode.props[name] = value;
            console.log(`  [vdom] set_vnode_prop(${vnodeId}, "${name}", ${value})`);
        },
        set_vnode_str_prop: function(vnodeId, nameRef, valueRef) {
            const vnode = vnodes.get(vnodeId);
            if (!vnode) return;
            const name = this._readStrRef(nameRef);
            const value = this._readStrRef(valueRef);
            vnode.props[name] = value;
            console.log(`  [vdom] set_vnode_str_prop(${vnodeId}, "${name}", "${value}")`);
        },
        append_vnode_child: (parentId, childId) => {
            const parent = vnodes.get(parentId);
            if (parent) parent.children.push(childId);
            console.log(`  [vdom] append_vnode_child(${parentId}, ${childId})`);
        },
        diff_and_patch: () => {},
        mount_vnode: function(vnodeId, selectorRef) {
            const selector = this._readStrRef(selectorRef);
            console.log(`  [vdom] mount_vnode(${vnodeId}, "${selector}")`);
            return 1;
        },
        dispose: () => {},
    },

    signal: {
        create: (initial) => {
            const id = signalIdCounter++;
            signals.set(id, BigInt(initial));
            console.log(`  [signal] create(${initial}) -> ${id}`);
            return id;
        },
        get: (id) => {
            const val = signals.get(id) ?? 0n;
            console.log(`  [signal] get(${id}) -> ${val}`);
            return val;
        },
        set: (id, value) => {
            signals.set(id, BigInt(value));
            console.log(`  [signal] set(${id}, ${value})`);
        },
        subscribe: () => 0,
        unsubscribe: () => {},
        batch_start: () => {},
        batch_end: () => {},
        computed: () => 0,
        effect: () => 0,
    },

    async: {
        promise_new: () => 0,
        promise_resolve: () => {},
        promise_reject: () => {},
        promise_then: () => 0,
        promise_catch: () => 0,
        promise_all: () => 0,
        promise_race: () => 0,
        spawn: () => 0,
        yield_now: () => {},
        await_promise: () => 0n,
        create_continuation: () => 0,
        resume: () => {},
    },
};

// Bind vdom methods to itself
const vdomMethods = imports.vdom;
for (const key of Object.keys(vdomMethods)) {
    if (typeof vdomMethods[key] === 'function') {
        vdomMethods[key] = vdomMethods[key].bind(vdomMethods);
    }
}

// Load and run the WASM module
async function runTests() {
    console.log('=== Sigil WASM Demo Test ===\n');

    const bytes = readFileSync('demo.wasm');
    const { instance } = await WebAssembly.instantiate(bytes, imports);

    memory = instance.exports.memory;
    console.log('WASM module loaded successfully!\n');

    const tests = [
        { name: 'fibonacci', export: 'fibonacci', args: [10n], expected: 55n },
        { name: 'is_prime', export: 'is_prime', args: [7n], expected: 1n },
        { name: 'is_prime', export: 'is_prime', args: [9n], expected: 0n },
    ];

    console.log('Running unit tests:');
    for (const test of tests) {
        const fn = instance.exports[test.export];
        if (!fn) {
            console.log(`  [SKIP] ${test.name}: export not found`);
            continue;
        }
        const result = fn(...test.args);
        const passed = result === test.expected;
        console.log(`  [${passed ? 'PASS' : 'FAIL'}] ${test.name}(${test.args.join(', ')}) = ${result} (expected ${test.expected})`);
    }

    console.log('\nRunning demo functions:');

    console.log('\n--- demo_counter ---');
    try {
        instance.exports.demo_counter();
        console.log('  OK');
    } catch (e) {
        console.log('  ERROR:', e.message);
    }

    console.log('\n--- demo_pipeline ---');
    try {
        instance.exports.demo_pipeline();
        console.log('  OK');
    } catch (e) {
        console.log('  ERROR:', e.message);
    }

    console.log('\n--- demo_fibonacci ---');
    try {
        instance.exports.demo_fibonacci();
        console.log('  OK');
    } catch (e) {
        console.log('  ERROR:', e.message);
    }

    console.log('\n--- demo_factorial ---');
    try {
        instance.exports.demo_factorial();
        console.log('  OK');
    } catch (e) {
        console.log('  ERROR:', e.message);
    }

    console.log('\n--- demo_primes ---');
    try {
        instance.exports.demo_primes();
        console.log('  OK');
    } catch (e) {
        console.log('  ERROR:', e.message);
    }

    console.log('\n--- demo_vdom ---');
    try {
        instance.exports.demo_vdom();
        console.log('  OK');
        console.log('  VNodes created:', vnodes.size);
    } catch (e) {
        console.log('  ERROR:', e.message);
    }

    console.log('\n--- main ---');
    try {
        instance.exports.main();
        console.log('  OK');
    } catch (e) {
        console.log('  ERROR:', e.message);
    }

    console.log('\n=== All tests completed ===');
}

runTests().catch(console.error);
