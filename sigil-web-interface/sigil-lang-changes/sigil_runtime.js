/**
 * Sigil Web Runtime
 *
 * JavaScript runtime for Sigil WASM modules.
 * Provides DOM bindings, event handling, and evidentiality tracking.
 *
 * Built for AI-native web development with sigil-web-interface.
 */

// Evidentiality tag constants (must match wasm_codegen.rs)
const EVIDENCE = {
    KNOWN: 0x0000_0000_0000_0000n,     // !
    UNCERTAIN: 0x1000_0000_0000_0000n, // ?
    REPORTED: 0x2000_0000_0000_0000n,  // ~
    PARADOX: 0x3000_0000_0000_0000n,   // ‽
    TAG_MASK: 0xF000_0000_0000_0000n,
    VALUE_MASK: 0x0FFF_FFFF_FFFF_FFFFn,
};

/**
 * Extract evidence tag from a tagged value
 */
function getEvidence(tagged) {
    const tag = BigInt(tagged) & EVIDENCE.TAG_MASK;
    switch (tag) {
        case EVIDENCE.KNOWN: return 'known';
        case EVIDENCE.UNCERTAIN: return 'uncertain';
        case EVIDENCE.REPORTED: return 'reported';
        case EVIDENCE.PARADOX: return 'paradox';
        default: return 'unknown';
    }
}

/**
 * Extract value from a tagged value
 */
function getValue(tagged) {
    return BigInt(tagged) & EVIDENCE.VALUE_MASK;
}

/**
 * Tag a value with evidentiality
 */
function tagValue(value, evidence) {
    const tag = EVIDENCE[evidence.toUpperCase()] || EVIDENCE.KNOWN;
    return (BigInt(value) & EVIDENCE.VALUE_MASK) | tag;
}

/**
 * Initialize Sigil WASM module
 * @param {string} wasmPath - Path to the .wasm file
 * @param {object} options - Configuration options
 * @returns {Promise<object>} - Sigil instance with exports
 */
export async function initSigil(wasmPath, options = {}) {
    const {
        container = document.body,
        debug = false,
    } = options;

    // Memory management
    let memory;
    const allocations = new Map();
    let nextAllocId = 1;
    let heapPointer = 1024; // Start after reserved space

    // Element tracking (WASM uses integer handles)
    const elements = new Map();
    let nextElementId = 1;

    // Event callback tracking
    const eventCallbacks = new Map();
    let nextCallbackId = 1;

    // Pending fetch requests
    const fetchRequests = new Map();
    let nextFetchId = 1;

    /**
     * Read a string from WASM memory
     */
    function readString(ptr, len) {
        if (len === undefined) {
            // Length-prefixed string
            const view = new DataView(memory.buffer);
            len = view.getUint32(ptr, true);
            ptr += 4;
        }
        const bytes = new Uint8Array(memory.buffer, ptr, len);
        return new TextDecoder().decode(bytes);
    }

    /**
     * Write a string to WASM memory
     */
    function writeString(str) {
        const bytes = new TextEncoder().encode(str);
        const ptr = heapPointer;
        const view = new DataView(memory.buffer);

        // Write length prefix
        view.setUint32(ptr, bytes.length, true);

        // Write string data
        const dest = new Uint8Array(memory.buffer, ptr + 4, bytes.length);
        dest.set(bytes);

        heapPointer += 4 + bytes.length;
        // Align to 8 bytes
        heapPointer = (heapPointer + 7) & ~7;

        return ptr;
    }

    /**
     * Get or create element handle
     */
    function getElementHandle(element) {
        for (const [id, el] of elements) {
            if (el === element) return id;
        }
        const id = nextElementId++;
        elements.set(id, element);
        return id;
    }

    /**
     * Get element from handle
     */
    function getElement(handle) {
        return elements.get(handle);
    }

    // Import functions for WASM
    const imports = {
        // Console logging
        console: {
            log_i32(value) {
                if (debug) console.log('[sigil:i32]', value);
                else console.log(value);
            },
            log_i64(value) {
                const evidence = getEvidence(value);
                const val = getValue(value);
                if (debug) console.log(`[sigil:i64:${evidence}]`, val.toString());
                else console.log(Number(val));
            },
            log_f64(value) {
                if (debug) console.log('[sigil:f64]', value);
                else console.log(value);
            },
            log_str(ptr, len) {
                const str = readString(ptr, len);
                if (debug) console.log('[sigil:str]', str);
                else console.log(str);
            },
            // Alias for print builtin
            print(value) {
                const val = getValue(value);
                if (debug) console.log('[sigil:print]', val.toString());
                else console.log(Number(val));
            },
        },

        // String operations
        string: {
            concat(aPtr, bPtr) {
                const a = readString(aPtr);
                const b = readString(bPtr);
                return writeString(a + b);
            },

            length(strPtr) {
                const str = readString(strPtr);
                return str.length;
            },

            slice(strPtr, start, end) {
                const str = readString(strPtr);
                const sliced = str.slice(Number(start), Number(end));
                return writeString(sliced);
            },

            eq(aPtr, bPtr) {
                const a = readString(aPtr);
                const b = readString(bPtr);
                return a === b ? 1n : 0n;
            },

            from_int(value) {
                const str = getValue(value).toString();
                return writeString(str);
            },

            from_float(value) {
                const str = value.toString();
                return writeString(str);
            },

            parse_int(strPtr) {
                const str = readString(strPtr);
                const val = parseInt(str, 10);
                return isNaN(val) ? 0n : BigInt(val);
            },

            parse_float(strPtr) {
                const str = readString(strPtr);
                return parseFloat(str) || 0.0;
            },
        },

        // DOM operations
        dom: {
            create_element(tagPtr, tagLen) {
                const tag = readString(tagPtr, tagLen);
                const element = document.createElement(tag);
                return getElementHandle(element);
            },

            create_text(textPtr, textLen) {
                const text = readString(textPtr, textLen);
                const node = document.createTextNode(text);
                return getElementHandle(node);
            },

            set_attribute(elemHandle, namePtr, nameLen, valuePtr, valueLen) {
                const elem = getElement(elemHandle);
                if (!elem) return;
                const name = readString(namePtr, nameLen);
                const value = readString(valuePtr, valueLen);
                elem.setAttribute(name, value);
            },

            remove_attribute(elemHandle, namePtr, nameLen) {
                const elem = getElement(elemHandle);
                if (!elem) return;
                const name = readString(namePtr, nameLen);
                elem.removeAttribute(name);
            },

            set_property(elemHandle, namePtr, nameLen, value) {
                const elem = getElement(elemHandle);
                if (!elem) return;
                const name = readString(namePtr, nameLen);
                elem[name] = Number(getValue(value));
            },

            append_child(parentHandle, childHandle) {
                const parent = getElement(parentHandle);
                const child = getElement(childHandle);
                if (parent && child) {
                    parent.appendChild(child);
                }
            },

            remove_child(parentHandle, childHandle) {
                const parent = getElement(parentHandle);
                const child = getElement(childHandle);
                if (parent && child) {
                    parent.removeChild(child);
                }
            },

            insert_before(parentHandle, newChildHandle, refChildHandle) {
                const parent = getElement(parentHandle);
                const newChild = getElement(newChildHandle);
                const refChild = getElement(refChildHandle);
                if (parent && newChild) {
                    parent.insertBefore(newChild, refChild || null);
                }
            },

            replace_child(parentHandle, newChildHandle, oldChildHandle) {
                const parent = getElement(parentHandle);
                const newChild = getElement(newChildHandle);
                const oldChild = getElement(oldChildHandle);
                if (parent && newChild && oldChild) {
                    parent.replaceChild(newChild, oldChild);
                }
            },

            set_text_content(elemHandle, textPtr, textLen) {
                const elem = getElement(elemHandle);
                if (!elem) return;
                const text = readString(textPtr, textLen);
                elem.textContent = text;
            },

            get_element_by_id(idPtr, idLen) {
                const id = readString(idPtr, idLen);
                const elem = document.getElementById(id);
                return elem ? getElementHandle(elem) : 0;
            },

            query_selector(selectorPtr, selectorLen) {
                const selector = readString(selectorPtr, selectorLen);
                const elem = document.querySelector(selector);
                return elem ? getElementHandle(elem) : 0;
            },

            clone_node(elemHandle, deep) {
                const elem = getElement(elemHandle);
                if (!elem) return 0;
                const cloned = elem.cloneNode(Boolean(deep));
                return getElementHandle(cloned);
            },
        },

        // Event handling
        events: {
            add_listener(elemHandle, eventPtr, eventLen, callbackId) {
                const elem = getElement(elemHandle);
                if (!elem) return;

                const eventType = readString(eventPtr, eventLen);

                const handler = (e) => {
                    // Call WASM callback
                    if (instance.exports[`__callback_${callbackId}`]) {
                        instance.exports[`__callback_${callbackId}`]();
                    }
                };

                elem.addEventListener(eventType, handler);
                eventCallbacks.set(callbackId, { elem, eventType, handler });
            },

            remove_listener(callbackId) {
                const callback = eventCallbacks.get(callbackId);
                if (callback) {
                    callback.elem.removeEventListener(callback.eventType, callback.handler);
                    eventCallbacks.delete(callbackId);
                }
            },
        },

        // Timing
        timing: {
            now() {
                return performance.now();
            },

            set_timeout(callbackId, ms) {
                return setTimeout(() => {
                    if (instance.exports[`__callback_${callbackId}`]) {
                        instance.exports[`__callback_${callbackId}`]();
                    }
                }, ms);
            },

            clear_timeout(timeoutId) {
                clearTimeout(timeoutId);
            },

            request_animation_frame(callbackId) {
                return requestAnimationFrame((timestamp) => {
                    if (instance.exports[`__callback_${callbackId}`]) {
                        instance.exports[`__callback_${callbackId}`](timestamp);
                    }
                });
            },
        },

        // Fetch API
        fetch: {
            fetch_start(urlPtr, urlLen) {
                const url = readString(urlPtr, urlLen);
                const id = nextFetchId++;

                fetchRequests.set(id, {
                    status: 'pending',
                    response: null,
                    error: null,
                });

                fetch(url)
                    .then(async (response) => {
                        const body = await response.text();
                        fetchRequests.set(id, {
                            status: 'done',
                            response: body,
                            error: null,
                        });
                    })
                    .catch((err) => {
                        fetchRequests.set(id, {
                            status: 'error',
                            response: null,
                            error: err.message,
                        });
                    });

                return id;
            },

            fetch_poll(fetchId) {
                const req = fetchRequests.get(fetchId);
                if (!req) return -1; // Invalid
                if (req.status === 'pending') return 0;
                if (req.status === 'error') return -2;
                return 1; // Done
            },

            fetch_get_body(fetchId, bufPtr) {
                const req = fetchRequests.get(fetchId);
                if (!req || req.status !== 'done') return 0;

                // Write response body to WASM memory
                const bytes = new TextEncoder().encode(req.response);
                const dest = new Uint8Array(memory.buffer, bufPtr, bytes.length);
                dest.set(bytes);

                return bytes.length;
            },
        },

        // Memory management
        memory: {
            alloc(size) {
                const ptr = heapPointer;
                heapPointer += size;
                // Align to 8 bytes
                heapPointer = (heapPointer + 7) & ~7;

                allocations.set(ptr, size);
                return ptr;
            },

            free(ptr) {
                allocations.delete(ptr);
                // Note: Simple bump allocator doesn't actually free
            },

            // heap_alloc for i64 size/pointer (used by closures/structs)
            heap_alloc(size) {
                const sizeNum = Number(size);
                const ptr = heapPointer;
                heapPointer += sizeNum;
                // Align to 8 bytes
                heapPointer = (heapPointer + 7) & ~7;

                allocations.set(ptr, sizeNum);
                return BigInt(ptr);
            },
        },

        // Math functions
        math: {
            sqrt: Math.sqrt,
            sin: Math.sin,
            cos: Math.cos,
            pow: Math.pow,
            random: Math.random,
        },

        // Morpheme array operations
        morpheme: {
            // Create a new array with given capacity
            array_new(capacity) {
                const id = nextAllocId++;
                const arr = new Array(capacity);
                arr.length = 0;  // Empty but with capacity hint
                allocations.set(id, arr);
                return id;
            },

            // Push element to array
            array_push(arrId, value) {
                const arr = allocations.get(arrId);
                if (arr) arr.push(value);
            },

            // Get element at index
            array_get(arrId, index) {
                const arr = allocations.get(arrId);
                return arr ? (arr[index] ?? 0n) : 0n;
            },

            // Set element at index
            array_set(arrId, index, value) {
                const arr = allocations.get(arrId);
                if (arr) arr[index] = value;
            },

            // Get array length
            array_len(arrId) {
                const arr = allocations.get(arrId);
                return arr ? arr.length : 0;
            },

            // Map: apply callback to each element (callback is table index)
            array_map(arrId, callbackTableIdx) {
                const arr = allocations.get(arrId);
                if (!arr) return 0;

                const resultId = nextAllocId++;
                const result = [];

                for (const elem of arr) {
                    // Call the callback via indirect call
                    const callbackFn = instance.exports.__indirect_function_table?.get(callbackTableIdx);
                    if (callbackFn) {
                        result.push(callbackFn(elem));
                    } else {
                        result.push(elem);
                    }
                }

                allocations.set(resultId, result);
                return resultId;
            },

            // Filter: keep elements where callback returns non-zero
            array_filter(arrId, callbackTableIdx) {
                const arr = allocations.get(arrId);
                if (!arr) return 0;

                const resultId = nextAllocId++;
                const result = [];

                for (const elem of arr) {
                    const callbackFn = instance.exports.__indirect_function_table?.get(callbackTableIdx);
                    if (callbackFn) {
                        const keep = callbackFn(elem);
                        if (keep !== 0n && keep !== 0) {
                            result.push(elem);
                        }
                    }
                }

                allocations.set(resultId, result);
                return resultId;
            },

            // Reduce: fold array with callback(acc, elem) -> acc
            array_reduce(arrId, callbackTableIdx, initial) {
                const arr = allocations.get(arrId);
                if (!arr) return initial;

                let acc = initial;
                for (const elem of arr) {
                    const callbackFn = instance.exports.__indirect_function_table?.get(callbackTableIdx);
                    if (callbackFn) {
                        acc = callbackFn(acc, elem);
                    }
                }
                return acc;
            },

            // Sort array (in place, returns same id)
            array_sort(arrId) {
                const arr = allocations.get(arrId);
                if (arr) {
                    arr.sort((a, b) => {
                        // BigInt comparison
                        if (a < b) return -1;
                        if (a > b) return 1;
                        return 0;
                    });
                }
                return arrId;
            },

            // Get first element
            array_first(arrId) {
                const arr = allocations.get(arrId);
                return arr && arr.length > 0 ? arr[0] : 0n;
            },

            // Get last element
            array_last(arrId) {
                const arr = allocations.get(arrId);
                return arr && arr.length > 0 ? arr[arr.length - 1] : 0n;
            },

            // Get nth element
            array_nth(arrId, n) {
                const arr = allocations.get(arrId);
                return arr && n >= 0 && n < arr.length ? arr[n] : 0n;
            },

            // Sum all elements
            array_sum(arrId) {
                const arr = allocations.get(arrId);
                if (!arr || arr.length === 0) return 0n;
                return arr.reduce((a, b) => BigInt(a) + BigInt(b), 0n);
            },

            // Product of all elements
            array_product(arrId) {
                const arr = allocations.get(arrId);
                if (!arr || arr.length === 0) return 1n;
                return arr.reduce((a, b) => BigInt(a) * BigInt(b), 1n);
            },

            // Minimum element
            array_min(arrId) {
                const arr = allocations.get(arrId);
                if (!arr || arr.length === 0) return 0n;
                return arr.reduce((a, b) => BigInt(a) < BigInt(b) ? BigInt(a) : BigInt(b));
            },

            // Maximum element
            array_max(arrId) {
                const arr = allocations.get(arrId);
                if (!arr || arr.length === 0) return 0n;
                return arr.reduce((a, b) => BigInt(a) > BigInt(b) ? BigInt(a) : BigInt(b));
            },

            // All elements truthy (non-zero)
            array_all(arrId) {
                const arr = allocations.get(arrId);
                if (!arr || arr.length === 0) return 1;  // Empty is vacuously true
                return arr.every(x => x !== 0n && x !== 0) ? 1 : 0;
            },

            // Any element truthy (non-zero)
            array_any(arrId) {
                const arr = allocations.get(arrId);
                if (!arr || arr.length === 0) return 0;
                return arr.some(x => x !== 0n && x !== 0) ? 1 : 0;
            },

            // Get random element
            array_random_element(arrId) {
                const arr = allocations.get(arrId);
                if (!arr || arr.length === 0) return 0n;
                const idx = Math.floor(Math.random() * arr.length);
                return arr[idx];
            },
        },

        // Router operations
        router: {
            push_state(urlPtr, urlLen) {
                const url = readString(urlPtr, urlLen);
                history.pushState({}, '', url);
            },

            replace_state(urlPtr, urlLen) {
                const url = readString(urlPtr, urlLen);
                history.replaceState({}, '', url);
            },

            get_pathname(bufPtr) {
                const pathname = location.pathname;
                const bytes = new TextEncoder().encode(pathname);
                const dest = new Uint8Array(memory.buffer, bufPtr, bytes.length);
                dest.set(bytes);
                return bytes.length;
            },
        },

        // Storage operations
        storage: {
            local_get(keyPtr, keyLen, bufPtr) {
                const key = readString(keyPtr, keyLen);
                const value = localStorage.getItem(key);
                if (!value) return 0;
                const bytes = new TextEncoder().encode(value);
                const dest = new Uint8Array(memory.buffer, bufPtr, bytes.length);
                dest.set(bytes);
                return bytes.length;
            },

            local_set(keyPtr, keyLen, valuePtr, valueLen) {
                const key = readString(keyPtr, keyLen);
                const value = readString(valuePtr, valueLen);
                localStorage.setItem(key, value);
            },

            local_remove(keyPtr, keyLen) {
                const key = readString(keyPtr, keyLen);
                localStorage.removeItem(key);
            },
        },

        // VDOM operations
        vdom: {
            // VNode storage
            _vnodes: new Map(),
            _nextVnodeId: 1,
            _mountedRoots: new Map(),  // vnodeId -> realDOMElement

            // VNode types
            VNODE_ELEMENT: 0,
            VNODE_TEXT: 1,
            VNODE_COMPONENT: 2,
            VNODE_FRAGMENT: 3,

            /**
             * Read string from I64 reference (pointer to length-prefixed data)
             */
            _readStrRef(strRef) {
                const ptr = Number(BigInt.asUintN(32, strRef));
                return readString(ptr);  // readString handles length prefix
            },

            /**
             * Create a VNode
             * @param {bigint} tagStrRef - I64 pointer to tag string in WASM memory
             * @returns {number} VNode ID
             */
            create_vnode(tagStrRef) {
                const tag = this._readStrRef(tagStrRef);
                const id = this._nextVnodeId++;
                this._vnodes.set(id, {
                    id,
                    type: this.VNODE_ELEMENT,
                    tag,
                    props: {},
                    children: [],
                    key: null,
                    ref: null,
                });
                return id;
            },

            /**
             * Create a text VNode
             * @param {bigint} textStrRef - I64 pointer to text string in WASM memory
             */
            create_text_vnode(textStrRef) {
                const text = this._readStrRef(textStrRef);
                const id = this._nextVnodeId++;
                this._vnodes.set(id, {
                    id,
                    type: this.VNODE_TEXT,
                    tag: '#text',
                    text,
                    props: {},
                    children: [],
                });
                return id;
            },

            /**
             * Create a fragment VNode (multiple children, no wrapper)
             */
            create_fragment() {
                const id = this._nextVnodeId++;
                this._vnodes.set(id, {
                    id,
                    type: this.VNODE_FRAGMENT,
                    tag: '#fragment',
                    props: {},
                    children: [],
                });
                return id;
            },

            /**
             * Set property on VNode
             * @param {number} vnodeId - VNode ID
             * @param {bigint} nameStrRef - I64 pointer to property name
             * @param {bigint} value - Property value (I64)
             */
            set_vnode_prop(vnodeId, nameStrRef, value) {
                const vnode = this._vnodes.get(vnodeId);
                if (!vnode) return;

                const name = this._readStrRef(nameStrRef);

                // Handle special props
                if (name === 'key') {
                    vnode.key = Number(getValue(value));
                } else if (name.startsWith('on')) {
                    // Event handler - value is callback table index
                    vnode.props[name] = {
                        type: 'event',
                        handler: Number(getValue(value)),
                    };
                } else {
                    // Regular prop - extract value from tagged i64
                    vnode.props[name] = getValue(value);
                }
            },

            /**
             * Set string property on VNode
             * @param {number} vnodeId - VNode ID
             * @param {bigint} nameStrRef - I64 pointer to property name
             * @param {bigint} valueStrRef - I64 pointer to property value
             */
            set_vnode_str_prop(vnodeId, nameStrRef, valueStrRef) {
                const vnode = this._vnodes.get(vnodeId);
                if (!vnode) return;

                const name = this._readStrRef(nameStrRef);
                const value = this._readStrRef(valueStrRef);
                vnode.props[name] = value;
            },

            /**
             * Append child to VNode
             */
            append_vnode_child(parentId, childId) {
                const parent = this._vnodes.get(parentId);
                const child = this._vnodes.get(childId);
                if (parent && child) {
                    parent.children.push(childId);
                }
            },

            /**
             * Mount VNode to real DOM
             * @param {number} vnodeId - VNode to mount
             * @param {bigint} selectorStrRef - I64 pointer to CSS selector string
             * @returns {number} Real DOM element handle
             */
            mount_vnode(vnodeId, selectorStrRef) {
                const vnode = this._vnodes.get(vnodeId);
                if (!vnode) return 0;

                const selector = this._readStrRef(selectorStrRef);
                const container = document.querySelector(selector);
                if (!container) return 0;

                const realElement = this._createRealElement(vnode);
                container.appendChild(realElement);

                this._mountedRoots.set(vnodeId, realElement);
                return getElementHandle(realElement);
            },

            /**
             * Create real DOM element from VNode
             */
            _createRealElement(vnode) {
                if (vnode.type === this.VNODE_TEXT) {
                    return document.createTextNode(vnode.text || '');
                }

                if (vnode.type === this.VNODE_FRAGMENT) {
                    const fragment = document.createDocumentFragment();
                    for (const childId of vnode.children) {
                        const child = this._vnodes.get(childId);
                        if (child) {
                            fragment.appendChild(this._createRealElement(child));
                        }
                    }
                    return fragment;
                }

                // Element node
                const element = document.createElement(vnode.tag);

                // Set properties
                for (const [name, value] of Object.entries(vnode.props)) {
                    if (name.startsWith('on') && value?.type === 'event') {
                        // Event handler
                        const eventType = name.slice(2).toLowerCase();
                        const callbackIdx = value.handler;
                        element.addEventListener(eventType, (e) => {
                            const callback = instance.exports.__indirect_function_table?.get(callbackIdx);
                            if (callback) {
                                callback();
                            }
                        });
                    } else if (name === 'class' || name === 'className') {
                        element.className = String(value);
                    } else if (name === 'style' && typeof value === 'object') {
                        Object.assign(element.style, value);
                    } else if (name === 'value') {
                        element.value = String(value);
                    } else if (name === 'checked') {
                        element.checked = Boolean(value);
                    } else if (typeof value === 'bigint') {
                        element.setAttribute(name, String(value));
                    } else {
                        element.setAttribute(name, String(value));
                    }
                }

                // Add children
                for (const childId of vnode.children) {
                    const child = this._vnodes.get(childId);
                    if (child) {
                        element.appendChild(this._createRealElement(child));
                    }
                }

                // Store reference for diffing
                vnode._realElement = element;
                return element;
            },

            /**
             * Diff and patch: compare old and new VNodes, apply minimal changes
             * @param {number} oldVnodeId - Previous VNode tree
             * @param {number} newVnodeId - New VNode tree
             * @param {number} containerHandle - Container DOM element
             */
            diff_and_patch(oldVnodeId, newVnodeId, containerHandle) {
                const oldVnode = this._vnodes.get(oldVnodeId);
                const newVnode = this._vnodes.get(newVnodeId);
                const container = getElement(containerHandle);

                if (!container) return;

                if (!oldVnode) {
                    // First render - just mount
                    if (newVnode) {
                        const element = this._createRealElement(newVnode);
                        container.appendChild(element);
                    }
                    return;
                }

                const realElement = oldVnode._realElement || this._mountedRoots.get(oldVnodeId);
                if (!realElement) return;

                this._patch(oldVnode, newVnode, realElement.parentNode, realElement);
            },

            /**
             * Internal patch algorithm
             */
            _patch(oldVnode, newVnode, parent, oldElement) {
                // Same node - no change
                if (oldVnode === newVnode) return;

                // New node is null - remove
                if (!newVnode) {
                    if (oldElement && parent) {
                        parent.removeChild(oldElement);
                    }
                    return;
                }

                // Old node is null - create new
                if (!oldVnode) {
                    const element = this._createRealElement(newVnode);
                    parent.appendChild(element);
                    return;
                }

                // Different types or tags - replace entire node
                if (oldVnode.type !== newVnode.type || oldVnode.tag !== newVnode.tag) {
                    const newElement = this._createRealElement(newVnode);
                    parent.replaceChild(newElement, oldElement);
                    return;
                }

                // Same type - update in place
                if (newVnode.type === this.VNODE_TEXT) {
                    // Text node - update text content
                    if (oldVnode.text !== newVnode.text) {
                        oldElement.textContent = newVnode.text;
                    }
                    newVnode._realElement = oldElement;
                    return;
                }

                // Element node - diff props
                this._patchProps(oldVnode.props, newVnode.props, oldElement);

                // Diff children
                this._patchChildren(oldVnode.children, newVnode.children, oldElement);

                newVnode._realElement = oldElement;
            },

            /**
             * Patch props between old and new
             */
            _patchProps(oldProps, newProps, element) {
                // Remove old props not in new
                for (const name of Object.keys(oldProps)) {
                    if (!(name in newProps)) {
                        if (name.startsWith('on')) {
                            // Remove event listener (simplified - in real impl we'd track handlers)
                        } else {
                            element.removeAttribute(name);
                        }
                    }
                }

                // Set new/changed props
                for (const [name, value] of Object.entries(newProps)) {
                    const oldValue = oldProps[name];
                    if (oldValue !== value) {
                        if (name.startsWith('on') && value?.type === 'event') {
                            // Event handler changed - would need proper cleanup in production
                            const eventType = name.slice(2).toLowerCase();
                            element.addEventListener(eventType, () => {
                                const callback = instance.exports.__indirect_function_table?.get(value.handler);
                                if (callback) callback();
                            });
                        } else if (name === 'class' || name === 'className') {
                            element.className = String(value);
                        } else if (name === 'value') {
                            element.value = String(value);
                        } else if (name === 'checked') {
                            element.checked = Boolean(value);
                        } else if (typeof value === 'bigint') {
                            element.setAttribute(name, String(value));
                        } else {
                            element.setAttribute(name, String(value));
                        }
                    }
                }
            },

            /**
             * Patch children using keyed reconciliation
             */
            _patchChildren(oldChildIds, newChildIds, parent) {
                const oldChildren = oldChildIds.map(id => this._vnodes.get(id)).filter(Boolean);
                const newChildren = newChildIds.map(id => this._vnodes.get(id)).filter(Boolean);

                // Build key -> old child map for keyed reconciliation
                const oldKeyMap = new Map();
                oldChildren.forEach((child, i) => {
                    const key = child.key ?? i;
                    oldKeyMap.set(key, { vnode: child, index: i });
                });

                const oldElements = Array.from(parent.childNodes);

                // Reconcile
                let lastIndex = 0;
                const usedOldIndices = new Set();

                newChildren.forEach((newChild, newIndex) => {
                    const key = newChild.key ?? newIndex;
                    const oldEntry = oldKeyMap.get(key);

                    if (oldEntry) {
                        // Found matching old child
                        usedOldIndices.add(oldEntry.index);
                        const oldElement = oldElements[oldEntry.index];

                        // Patch in place
                        this._patch(oldEntry.vnode, newChild, parent, oldElement);

                        // Move if needed
                        if (oldEntry.index < lastIndex) {
                            // Need to move this element
                            const referenceNode = oldElements[newIndex] || null;
                            if (oldElement !== referenceNode) {
                                parent.insertBefore(oldElement, referenceNode);
                            }
                        }
                        lastIndex = Math.max(lastIndex, oldEntry.index);
                    } else {
                        // New child - create and insert
                        const newElement = this._createRealElement(newChild);
                        const referenceNode = oldElements[newIndex] || null;
                        parent.insertBefore(newElement, referenceNode);
                    }
                });

                // Remove unused old children
                oldChildren.forEach((child, index) => {
                    if (!usedOldIndices.has(index)) {
                        const oldElement = oldElements[index];
                        if (oldElement && oldElement.parentNode === parent) {
                            parent.removeChild(oldElement);
                        }
                    }
                });
            },

            /**
             * Cleanup VNode tree
             */
            dispose(vnodeId) {
                const vnode = this._vnodes.get(vnodeId);
                if (!vnode) return;

                // Recursively dispose children
                for (const childId of vnode.children) {
                    this.dispose(childId);
                }

                this._vnodes.delete(vnodeId);
                this._mountedRoots.delete(vnodeId);
            },
        },

        // Signal-based reactivity (Phase 5)
        signal: {
            _signals: new Map(),
            _nextSignalId: 1,
            _subscribers: new Map(),
            _nextSubId: 1,
            _batchDepth: 0,
            _pendingNotifications: new Set(),
            _computedCache: new Map(),
            _effects: new Map(),
            _currentEffect: null,
            _dependencies: new Map(),  // effect -> Set<signalId>

            /**
             * Create a new signal with initial value
             */
            create(initialValue) {
                const id = this._nextSignalId++;
                this._signals.set(id, {
                    value: initialValue,
                    subscribers: new Set(),
                });
                return id;
            },

            /**
             * Get signal value
             */
            get(signalId) {
                const signal = this._signals.get(signalId);
                if (!signal) return 0n;

                // Track dependency if inside an effect
                if (this._currentEffect !== null) {
                    let deps = this._dependencies.get(this._currentEffect);
                    if (!deps) {
                        deps = new Set();
                        this._dependencies.set(this._currentEffect, deps);
                    }
                    deps.add(signalId);
                    signal.subscribers.add(this._currentEffect);
                }

                return signal.value;
            },

            /**
             * Set signal value and notify subscribers
             */
            set(signalId, newValue) {
                const signal = this._signals.get(signalId);
                if (!signal) return;

                const oldValue = signal.value;
                if (oldValue === newValue) return;

                signal.value = newValue;

                if (this._batchDepth > 0) {
                    // Defer notifications
                    for (const subId of signal.subscribers) {
                        this._pendingNotifications.add(subId);
                    }
                } else {
                    // Notify immediately
                    this._notifySubscribers(signal.subscribers);
                }
            },

            /**
             * Subscribe to signal changes
             */
            subscribe(signalId, callbackTableIdx) {
                const signal = this._signals.get(signalId);
                if (!signal) return 0;

                const subId = this._nextSubId++;
                this._subscribers.set(subId, {
                    signalId,
                    callback: callbackTableIdx,
                });
                signal.subscribers.add(subId);
                return subId;
            },

            /**
             * Unsubscribe from signal
             */
            unsubscribe(subId) {
                const sub = this._subscribers.get(subId);
                if (!sub) return;

                const signal = this._signals.get(sub.signalId);
                if (signal) {
                    signal.subscribers.delete(subId);
                }
                this._subscribers.delete(subId);
            },

            /**
             * Start a batch update
             */
            batch_start() {
                this._batchDepth++;
            },

            /**
             * End batch update and flush notifications
             */
            batch_end() {
                this._batchDepth--;
                if (this._batchDepth === 0 && this._pendingNotifications.size > 0) {
                    const toNotify = new Set(this._pendingNotifications);
                    this._pendingNotifications.clear();
                    this._notifySubscribers(toNotify);
                }
            },

            /**
             * Create a computed signal (derived from other signals)
             */
            computed(computeFnTableIdx) {
                const id = this._nextSignalId++;
                const computedSignal = {
                    value: 0n,
                    subscribers: new Set(),
                    computeFn: computeFnTableIdx,
                    dirty: true,
                };
                this._signals.set(id, computedSignal);
                this._computedCache.set(id, computedSignal);

                // Initial computation
                this._recompute(id);

                return id;
            },

            /**
             * Recompute a computed signal
             */
            _recompute(signalId) {
                const signal = this._computedCache.get(signalId);
                if (!signal || !signal.dirty) return;

                const prevEffect = this._currentEffect;
                this._currentEffect = signalId;

                // Clear old dependencies
                const oldDeps = this._dependencies.get(signalId);
                if (oldDeps) {
                    for (const depId of oldDeps) {
                        const depSignal = this._signals.get(depId);
                        if (depSignal) {
                            depSignal.subscribers.delete(signalId);
                        }
                    }
                }
                this._dependencies.set(signalId, new Set());

                // Run compute function
                const computeFn = instance.exports.__indirect_function_table?.get(signal.computeFn);
                if (computeFn) {
                    signal.value = computeFn();
                }

                signal.dirty = false;
                this._currentEffect = prevEffect;
            },

            /**
             * Create an effect that runs when signals change
             */
            effect(effectFnTableIdx) {
                const id = this._nextSubId++;
                const effectInfo = {
                    fn: effectFnTableIdx,
                    dependencies: new Set(),
                };
                this._effects.set(id, effectInfo);

                // Run effect immediately to collect dependencies
                this._runEffect(id);

                return id;
            },

            /**
             * Run an effect
             */
            _runEffect(effectId) {
                const effect = this._effects.get(effectId);
                if (!effect) return;

                const prevEffect = this._currentEffect;
                this._currentEffect = effectId;

                // Clear old dependencies
                const oldDeps = this._dependencies.get(effectId);
                if (oldDeps) {
                    for (const depId of oldDeps) {
                        const depSignal = this._signals.get(depId);
                        if (depSignal) {
                            depSignal.subscribers.delete(effectId);
                        }
                    }
                }
                this._dependencies.set(effectId, new Set());

                // Run effect
                const effectFn = instance.exports.__indirect_function_table?.get(effect.fn);
                if (effectFn) {
                    effectFn();
                }

                this._currentEffect = prevEffect;
            },

            /**
             * Notify all subscribers
             */
            _notifySubscribers(subscribers) {
                for (const subId of subscribers) {
                    // Check if it's a regular subscriber
                    const sub = this._subscribers.get(subId);
                    if (sub) {
                        const callback = instance.exports.__indirect_function_table?.get(sub.callback);
                        if (callback) callback();
                        continue;
                    }

                    // Check if it's a computed signal
                    const computed = this._computedCache.get(subId);
                    if (computed) {
                        computed.dirty = true;
                        this._recompute(subId);
                        // Notify computed's subscribers
                        if (computed.subscribers.size > 0) {
                            this._notifySubscribers(computed.subscribers);
                        }
                        continue;
                    }

                    // Check if it's an effect
                    if (this._effects.has(subId)) {
                        this._runEffect(subId);
                    }
                }
            },
        },

        // Async operations (Phase 6)
        async: {
            _promises: new Map(),
            _nextPromiseId: 1,
            _tasks: new Map(),
            _nextTaskId: 1,
            _taskQueue: [],
            _isProcessing: false,

            /**
             * Create a new promise
             */
            promise_new() {
                const id = this._nextPromiseId++;
                let resolveFn, rejectFn;
                const promise = new Promise((resolve, reject) => {
                    resolveFn = resolve;
                    rejectFn = reject;
                });
                this._promises.set(id, {
                    promise,
                    resolve: resolveFn,
                    reject: rejectFn,
                    state: 'pending',
                    value: null,
                });
                return id;
            },

            /**
             * Resolve a promise with a value
             */
            promise_resolve(promiseId, value) {
                const p = this._promises.get(promiseId);
                if (!p || p.state !== 'pending') return;

                p.state = 'fulfilled';
                p.value = value;
                p.resolve(value);
            },

            /**
             * Reject a promise with an error message
             */
            promise_reject(promiseId, errorPtr, errorLen) {
                const p = this._promises.get(promiseId);
                if (!p || p.state !== 'pending') return;

                const errorMsg = readString(errorPtr, errorLen);
                p.state = 'rejected';
                p.value = errorMsg;
                p.reject(new Error(errorMsg));
            },

            /**
             * Chain .then() on a promise
             */
            promise_then(promiseId, onFulfilledTableIdx, onRejectedTableIdx) {
                const p = this._promises.get(promiseId);
                if (!p) return 0;

                const newId = this._nextPromiseId++;
                let newResolve, newReject;
                const newPromise = new Promise((resolve, reject) => {
                    newResolve = resolve;
                    newReject = reject;
                });

                p.promise.then(
                    (value) => {
                        if (onFulfilledTableIdx > 0) {
                            const fn = instance.exports.__indirect_function_table?.get(onFulfilledTableIdx);
                            if (fn) {
                                try {
                                    const result = fn(value);
                                    newResolve(result);
                                } catch (e) {
                                    newReject(e);
                                }
                            } else {
                                newResolve(value);
                            }
                        } else {
                            newResolve(value);
                        }
                    },
                    (error) => {
                        if (onRejectedTableIdx > 0) {
                            const fn = instance.exports.__indirect_function_table?.get(onRejectedTableIdx);
                            if (fn) {
                                try {
                                    const result = fn(0n); // Pass 0 for error
                                    newResolve(result);
                                } catch (e) {
                                    newReject(e);
                                }
                            } else {
                                newReject(error);
                            }
                        } else {
                            newReject(error);
                        }
                    }
                );

                this._promises.set(newId, {
                    promise: newPromise,
                    resolve: newResolve,
                    reject: newReject,
                    state: 'pending',
                    value: null,
                });

                return newId;
            },

            /**
             * Chain .catch() on a promise
             */
            promise_catch(promiseId, onRejectedTableIdx) {
                return this.promise_then(promiseId, 0, onRejectedTableIdx);
            },

            /**
             * Promise.all - wait for all promises in an array
             */
            promise_all(arrayId) {
                const arr = allocations.get(arrayId);
                if (!arr) return 0;

                const promises = arr.map(id => {
                    const p = this._promises.get(Number(id));
                    return p ? p.promise : Promise.resolve(id);
                });

                const newId = this._nextPromiseId++;
                let newResolve, newReject;
                const newPromise = new Promise((resolve, reject) => {
                    newResolve = resolve;
                    newReject = reject;
                });

                Promise.all(promises)
                    .then(values => {
                        // Store result as array
                        const resultId = nextAllocId++;
                        allocations.set(resultId, values.map(v => BigInt(v)));
                        newResolve(BigInt(resultId));
                    })
                    .catch(newReject);

                this._promises.set(newId, {
                    promise: newPromise,
                    resolve: newResolve,
                    reject: newReject,
                    state: 'pending',
                    value: null,
                });

                return newId;
            },

            /**
             * Promise.race - resolve with first completed
             */
            promise_race(arrayId) {
                const arr = allocations.get(arrayId);
                if (!arr) return 0;

                const promises = arr.map(id => {
                    const p = this._promises.get(Number(id));
                    return p ? p.promise : Promise.resolve(id);
                });

                const newId = this._nextPromiseId++;
                let newResolve, newReject;
                const newPromise = new Promise((resolve, reject) => {
                    newResolve = resolve;
                    newReject = reject;
                });

                Promise.race(promises)
                    .then(value => newResolve(BigInt(value)))
                    .catch(newReject);

                this._promises.set(newId, {
                    promise: newPromise,
                    resolve: newResolve,
                    reject: newReject,
                    state: 'pending',
                    value: null,
                });

                return newId;
            },

            /**
             * Spawn a new async task (cooperative multitasking)
             */
            spawn(taskFnTableIdx) {
                const taskId = this._nextTaskId++;
                this._tasks.set(taskId, {
                    fn: taskFnTableIdx,
                    state: 'pending',
                });
                this._taskQueue.push(taskId);

                // Process queue if not already processing
                if (!this._isProcessing) {
                    this._processQueue();
                }

                return taskId;
            },

            /**
             * Yield execution to other tasks
             */
            yield_now() {
                // In browser, this is a no-op since we're single-threaded
                // Real impl would use queueMicrotask or similar
            },

            /**
             * Process task queue
             */
            async _processQueue() {
                if (this._isProcessing) return;
                this._isProcessing = true;

                while (this._taskQueue.length > 0) {
                    const taskId = this._taskQueue.shift();
                    const task = this._tasks.get(taskId);
                    if (!task) continue;

                    task.state = 'running';
                    const fn = instance.exports.__indirect_function_table?.get(task.fn);
                    if (fn) {
                        try {
                            fn();
                            task.state = 'completed';
                        } catch (e) {
                            task.state = 'failed';
                            console.error('Task failed:', e);
                        }
                    }

                    // Yield to browser
                    await new Promise(resolve => setTimeout(resolve, 0));
                }

                this._isProcessing = false;
            },
        },
    };

    // Load and instantiate WASM module
    const response = await fetch(wasmPath);
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes, imports);

    // Get memory export
    memory = instance.exports.memory;

    // Create Sigil instance API
    const sigil = {
        // Raw WASM instance
        _instance: instance,
        _memory: memory,

        // Exported functions
        exports: instance.exports,

        // Call main function
        main() {
            if (instance.exports.main) {
                return instance.exports.main();
            }
            console.warn('No main function exported');
        },

        // Call any exported function
        call(name, ...args) {
            const fn = instance.exports[name];
            if (fn) {
                return fn(...args);
            }
            throw new Error(`Function not found: ${name}`);
        },

        // Utilities
        utils: {
            readString,
            writeString,
            getEvidence,
            getValue,
            tagValue,
            getElement,
        },

        // Debug info
        debug: {
            elements,
            allocations,
            fetchRequests,
            eventCallbacks,
        },
    };

    return sigil;
}

/**
 * Create a reactive Sigil component
 * For use with sigil-web-interface
 */
export function createComponent(sigil, componentName, props = {}) {
    const component = {
        name: componentName,
        props,
        element: null,

        mount(container) {
            // Call WASM component's render function
            const renderFn = sigil.exports[`${componentName}_render`];
            if (renderFn) {
                const elemHandle = renderFn();
                this.element = sigil.utils.getElement(elemHandle);
                if (this.element) {
                    container.appendChild(this.element);
                }
            }
            return this;
        },

        update(newProps) {
            Object.assign(this.props, newProps);
            const updateFn = sigil.exports[`${componentName}_update`];
            if (updateFn) {
                updateFn();
            }
        },

        unmount() {
            if (this.element && this.element.parentNode) {
                this.element.parentNode.removeChild(this.element);
            }
        },
    };

    return component;
}

// Default export
export default { initSigil, createComponent, EVIDENCE };
