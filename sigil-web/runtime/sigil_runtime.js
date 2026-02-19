/**
 * Sigil Web Runtime
 *
 * JavaScript runtime for Sigil WASM modules.
 * Implements all imports required by the Sigil WASM compiler.
 */

export class SigilRuntime {
  constructor() {
    // WASM instance reference (set when module loads)
    this.instance = null;
    this.memory = null;

    // String interning table: id -> string
    this.strings = new Map();
    this.stringCounter = 0x10000; // Start after data section

    // DOM element tracking: id -> Element
    this.elements = new Map();
    this.elementCounter = 1;

    // VNode tracking: id -> VNode object
    this.vnodes = new Map();
    this.vnodeCounter = 1;

    // Signal tracking: id -> { value, subscribers }
    this.signals = new Map();
    this.signalCounter = 1;
    this.signalBatchDepth = 0;
    this.pendingEffects = [];

    // Event listener tracking: id -> { element, type, handler }
    this.listeners = new Map();
    this.listenerCounter = 1;

    // Timer tracking
    this.timers = new Map();
    this.timerCounter = 1;

    // Fetch tracking: id -> { promise, status, body }
    this.fetches = new Map();
    this.fetchCounter = 1;

    // Array tracking for morphemes
    this.arrays = new Map();
    this.arrayCounter = 1;

    // Function table for callbacks
    this.callbacks = new Map();

    // Async state machine continuation tracking
    // id -> { framePtr, state, promise, promiseId }
    this.continuations = new Map();
    this.continuationCounter = 1;

    // Memory allocation tracking (simple bump allocator)
    this.heapBase = 0x20000; // Start heap after data section
    this.heapPtr = this.heapBase;
  }

  /**
   * Initialize runtime with WASM instance
   */
  init(instance) {
    this.instance = instance;
    this.memory = instance.exports.memory;
    this._loadDataSectionStrings();
  }

  /**
   * Load string constants from WASM data section
   */
  _loadDataSectionStrings() {
    if (!this.memory) return;
    const view = new Uint8Array(this.memory.buffer);
    // Scan for length-prefixed strings in data section (typically 0x10000+)
    for (let i = 0x10000; i < Math.min(view.length - 4, 0x20000); i++) {
      const len = view[i] | (view[i+1] << 8) | (view[i+2] << 16) | (view[i+3] << 24);
      if (len > 0 && len < 1000 && i + 4 + len <= view.length) {
        try {
          const bytes = view.slice(i + 4, i + 4 + len);
          const str = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
          if (/^[\x20-\x7E\n\r\t]*$/.test(str)) {
            this.strings.set(i, str);
          }
        } catch {}
      }
    }
  }

  /**
   * Read a string from WASM memory at given pointer
   */
  readString(ptr) {
    if (this.strings.has(ptr)) {
      return this.strings.get(ptr);
    }
    if (!this.memory) return '';
    const view = new DataView(this.memory.buffer);
    const len = view.getUint32(ptr, true);
    const bytes = new Uint8Array(this.memory.buffer, ptr + 4, len);
    const str = new TextDecoder().decode(bytes);
    this.strings.set(ptr, str);
    return str;
  }

  /**
   * Allocate and write a string to WASM memory, return pointer
   */
  allocString(str) {
    const id = ++this.stringCounter;
    this.strings.set(id, str);
    return id;
  }

  /**
   * Get all imports for WASM instantiation
   */
  getImports() {
    return {
      console: this._consoleImports(),
      string: this._stringImports(),
      dom: this._domImports(),
      events: this._eventImports(),
      timing: this._timingImports(),
      fetch: this._fetchImports(),
      storage: this._storageImports(),
      router: this._routerImports(),
      memory: this._memoryImports(),
      morpheme: this._morphemeImports(),
      math: this._mathImports(),
      vdom: this._vdomImports(),
      signal: this._signalImports(),
      async: this._asyncImports(),
      browser: this._browserImports(),
      env: this._envImports(),
      crate: this._crateImports(),
    };
  }

  // ========== Environment Imports ==========
  _envImports() {
    return {
      document: () => 1, // Placeholder document handle
      local_storage: () => 1, // Storage handle
      session_storage: () => 2, // Storage handle
      location: () => this.allocString(window.location?.href || ''),
      history: () => 1, // History handle
      navigator: () => 1, // Navigator handle
      inner_width: () => BigInt(window.innerWidth || 800),
      inner_height: () => BigInt(window.innerHeight || 600),
      outer_width: () => BigInt(window.outerWidth || 800),
      outer_height: () => BigInt(window.outerHeight || 600),
      scroll_x: () => BigInt(window.scrollX || 0),
      scroll_y: () => BigInt(window.scrollY || 0),
      device_pixel_ratio: () => window.devicePixelRatio || 1,
      alert: (msgPtr) => {
        const msg = this.strings.get(Number(msgPtr)) || this.readString(Number(msgPtr));
        window.alert?.(msg);
      },
      confirm: (msgPtr) => {
        const msg = this.strings.get(Number(msgPtr)) || this.readString(Number(msgPtr));
        return window.confirm?.(msg) ? 1 : 0;
      },
      prompt: (msgPtr, defaultPtr) => {
        const msg = this.strings.get(Number(msgPtr)) || this.readString(Number(msgPtr));
        const def = this.strings.get(Number(defaultPtr)) || this.readString(Number(defaultPtr));
        const result = window.prompt?.(msg, def) || '';
        return this.allocString(result);
      },
      open: (urlPtr, targetPtr, featuresPtr) => {
        const url = this.strings.get(Number(urlPtr)) || this.readString(Number(urlPtr));
        const target = this.strings.get(Number(targetPtr)) || this.readString(Number(targetPtr));
        const features = this.strings.get(Number(featuresPtr)) || this.readString(Number(featuresPtr));
        window.open?.(url, target, features);
        return BigInt(0);
      },
      close: () => window.close?.(),
      match_media: (queryPtr) => {
        const query = this.strings.get(Number(queryPtr)) || this.readString(Number(queryPtr));
        return window.matchMedia?.(query).matches ? 1 : 0;
      },
      user_agent: () => this.allocString(navigator.userAgent || ''),
      language: () => this.allocString(navigator.language || 'en'),
      languages: () => {
        const arr = ++this.arrayCounter;
        this.arrays.set(arr, (navigator.languages || ['en']).map(l => this.allocString(l)));
        return arr;
      },
      online: () => navigator.onLine ? 1 : 0,
      platform: () => this.allocString(navigator.platform || ''),
      hardware_concurrency: () => BigInt(navigator.hardwareConcurrency || 1),
      cookie_enabled: () => navigator.cookieEnabled ? 1 : 0,
      do_not_track: () => navigator.doNotTrack === '1' ? 1 : 0,
      orientation: () => BigInt(screen.orientation?.angle || 0),
      screen_width: () => BigInt(screen.width || 800),
      screen_height: () => BigInt(screen.height || 600),
      color_depth: () => BigInt(screen.colorDepth || 24),
      focus: () => window.focus?.(),
      blur: () => window.blur?.(),
      print: () => window.print?.(),
      scroll_to: (x, y) => window.scrollTo?.(Number(x), Number(y)),
      scroll_by: (x, y) => window.scrollBy?.(Number(x), Number(y)),
      hostname: () => this.allocString(window.location?.hostname || ''),
      port: () => this.allocString(window.location?.port || ''),
      protocol: () => this.allocString(window.location?.protocol || ''),
      pathname: () => this.allocString(window.location?.pathname || ''),
      search: () => this.allocString(window.location?.search || ''),
      hash: () => this.allocString(window.location?.hash || ''),
      origin: () => this.allocString(window.location?.origin || ''),
      href: () => this.allocString(window.location?.href || ''),
      set_href: (urlPtr) => {
        const url = this.strings.get(Number(urlPtr)) || this.readString(Number(urlPtr));
        if (window.location) window.location.href = url;
      },
      reload: () => window.location?.reload?.(),
      back: () => window.history?.back?.(),
      forward: () => window.history?.forward?.(),
      go: (delta) => window.history?.go?.(Number(delta)),
      push_state: (statePtr, titlePtr, urlPtr) => {
        const url = this.strings.get(Number(urlPtr)) || this.readString(Number(urlPtr));
        const title = this.strings.get(Number(titlePtr)) || this.readString(Number(titlePtr));
        window.history?.pushState?.({}, title, url);
      },
      replace_state: (statePtr, titlePtr, urlPtr) => {
        const url = this.strings.get(Number(urlPtr)) || this.readString(Number(urlPtr));
        const title = this.strings.get(Number(titlePtr)) || this.readString(Number(titlePtr));
        window.history?.replaceState?.({}, title, url);
      },
      history_length: () => BigInt(window.history?.length || 0),
      copy_to_clipboard: (textPtr) => {
        const text = this.strings.get(Number(textPtr)) || this.readString(Number(textPtr));
        navigator.clipboard?.writeText?.(text);
      },
      read_clipboard: () => {
        // Async, returns empty string synchronously
        return this.allocString('');
      },
      vibrate: (pattern) => navigator.vibrate?.(Number(pattern)),
      share: (titlePtr, textPtr, urlPtr) => {
        const title = this.strings.get(Number(titlePtr)) || this.readString(Number(titlePtr));
        const text = this.strings.get(Number(textPtr)) || this.readString(Number(textPtr));
        const url = this.strings.get(Number(urlPtr)) || this.readString(Number(urlPtr));
        navigator.share?.({ title, text, url });
      },
      entry_type: () => BigInt(0), // Navigation type
      domain: () => this.allocString(window.location?.hostname || ''),
      time_remaining: () => BigInt(1000), // Idle callback time
      set_start: (typePtr, namePtr, timePtr) => {}, // Performance mark
      // Clone operations
      clone: (ptr) => ptr, // Identity clone for reference types
      clone_contents: (ptr) => ptr, // Clone contents (shallow)
      get_item: (collectionPtr, index) => {
        // Get item from collection - placeholder
        return BigInt(0);
      },
    };
  }

  // ========== Crate Imports ==========
  _crateImports() {
    return {
      Window: () => BigInt(1), // Window handle placeholder
      Closure: () => BigInt(1), // Closure handle placeholder
    };
  }

  // ========== Console Imports ==========
  _consoleImports() {
    return {
      log_i64: (val) => console.log(Number(val)),
      log_f64: (val) => console.log(val),
      log_str: (ptr, len) => console.log(this.readString(ptr)),
      print: (val) => {
        const str = this.strings.get(Number(val));
        if (str !== undefined) {
          console.log(str);
        } else {
          console.log(Number(val));
        }
      },
    };
  }

  // ========== String Imports ==========
  _stringImports() {
    return {
      concat: (a, b) => {
        const sa = this.strings.get(a) || '';
        const sb = this.strings.get(b) || '';
        return this.allocString(sa + sb);
      },
      length: (s) => (this.strings.get(s) || '').length,
      slice: (s, start, end) => this.allocString((this.strings.get(s) || '').slice(start, end)),
      eq: (a, b) => (this.strings.get(a) || '') === (this.strings.get(b) || '') ? 1 : 0,
      from_int: (n) => this.allocString(String(n)),
      from_float: (n) => this.allocString(String(n)),
      parse_int: (s) => BigInt(parseInt(this.strings.get(s) || '0')),
      parse_float: (s) => parseFloat(this.strings.get(s) || '0'),
      lines: (s) => {
        const arr = (this.strings.get(s) || '').split('\n');
        const id = ++this.arrayCounter;
        this.arrays.set(id, arr.map(line => this.allocString(line)));
        return id;
      },
      split_whitespace: (s) => {
        const arr = (this.strings.get(s) || '').trim().split(/\s+/);
        const id = ++this.arrayCounter;
        this.arrays.set(id, arr.map(part => this.allocString(part)));
        return id;
      },
      split: (s, sep) => {
        const arr = (this.strings.get(s) || '').split(this.strings.get(sep) || '');
        const id = ++this.arrayCounter;
        this.arrays.set(id, arr.map(part => this.allocString(part)));
        return id;
      },
      trim: (s) => this.allocString((this.strings.get(s) || '').trim()),
      trim_start: (s) => this.allocString((this.strings.get(s) || '').trimStart()),
      trim_end: (s) => this.allocString((this.strings.get(s) || '').trimEnd()),
      to_uppercase: (s) => this.allocString((this.strings.get(s) || '').toUpperCase()),
      to_lowercase: (s) => this.allocString((this.strings.get(s) || '').toLowerCase()),
      contains: (s, substr) => (this.strings.get(s) || '').includes(this.strings.get(substr) || '') ? 1 : 0,
      starts_with: (s, prefix) => (this.strings.get(s) || '').startsWith(this.strings.get(prefix) || '') ? 1 : 0,
      ends_with: (s, suffix) => (this.strings.get(s) || '').endsWith(this.strings.get(suffix) || '') ? 1 : 0,
      replace: (s, from, to) => this.allocString(
        (this.strings.get(s) || '').replaceAll(this.strings.get(from) || '', this.strings.get(to) || '')
      ),
      chars: (s) => {
        const arr = [...(this.strings.get(s) || '')];
        const id = ++this.arrayCounter;
        this.arrays.set(id, arr.map(c => this.allocString(c)));
        return id;
      },
    };
  }

  // ========== DOM Imports ==========
  _domImports() {
    return {
      create_element: (tagPtr, tagLen) => {
        const tag = this.readString(tagPtr);
        const el = document.createElement(tag);
        const id = ++this.elementCounter;
        this.elements.set(id, el);
        return id;
      },
      create_text: (textPtr, textLen) => {
        const text = this.readString(textPtr);
        const node = document.createTextNode(text);
        const id = ++this.elementCounter;
        this.elements.set(id, node);
        return id;
      },
      set_attribute: (elId, namePtr, nameLen, valPtr, valLen) => {
        const el = this.elements.get(elId);
        if (el) {
          const name = this.readString(namePtr);
          const val = this.readString(valPtr);
          el.setAttribute(name, val);
        }
      },
      remove_attribute: (elId, namePtr, nameLen) => {
        const el = this.elements.get(elId);
        if (el) el.removeAttribute(this.readString(namePtr));
      },
      set_property: (elId, namePtr, nameLen, val) => {
        const el = this.elements.get(elId);
        if (el) el[this.readString(namePtr)] = Number(val);
      },
      append_child: (parentId, childId) => {
        const parent = this.elements.get(parentId);
        const child = this.elements.get(childId);
        if (parent && child) parent.appendChild(child);
      },
      insert_before: (parentId, childId, refId) => {
        const parent = this.elements.get(parentId);
        const child = this.elements.get(childId);
        const ref = this.elements.get(refId);
        if (parent && child) parent.insertBefore(child, ref);
      },
      remove_child: (parentId, childId) => {
        const parent = this.elements.get(parentId);
        const child = this.elements.get(childId);
        if (parent && child) parent.removeChild(child);
      },
      replace_child: (parentId, newId, oldId) => {
        const parent = this.elements.get(parentId);
        const newChild = this.elements.get(newId);
        const oldChild = this.elements.get(oldId);
        if (parent && newChild && oldChild) parent.replaceChild(newChild, oldChild);
      },
      set_text_content: (elId, textPtr, textLen) => {
        const el = this.elements.get(elId);
        if (el) el.textContent = this.readString(textPtr);
      },
      get_element_by_id: (idPtr, idLen) => {
        const el = document.getElementById(this.readString(idPtr));
        if (!el) return 0;
        const id = ++this.elementCounter;
        this.elements.set(id, el);
        return id;
      },
      query_selector: (selectorPtr, selectorLen) => {
        const el = document.querySelector(this.readString(selectorPtr));
        if (!el) return 0;
        const id = ++this.elementCounter;
        this.elements.set(id, el);
        return id;
      },
      clone_node: (elId, deep) => {
        const el = this.elements.get(elId);
        if (!el) return 0;
        const clone = el.cloneNode(!!deep);
        const id = ++this.elementCounter;
        this.elements.set(id, clone);
        return id;
      },
    };
  }

  // ========== Event Imports ==========
  _eventImports() {
    return {
      add_listener: (elId, typePtr, typeLen, callbackIdx) => {
        const el = this.elements.get(elId);
        if (!el) return 0;
        const type = this.readString(typePtr);
        const handler = (event) => {
          // Store event for retrieval
          const eventId = ++this.elementCounter;
          this.elements.set(eventId, event);
          // Call WASM callback via function table
          if (this.instance.exports.__indirect_function_table) {
            this.instance.exports.__indirect_function_table.get(callbackIdx)(eventId);
          }
        };
        el.addEventListener(type, handler);
        const listenerId = ++this.listenerCounter;
        this.listeners.set(listenerId, { element: el, type, handler });
        return listenerId;
      },
      remove_listener: (listenerId) => {
        const listener = this.listeners.get(listenerId);
        if (listener) {
          listener.element.removeEventListener(listener.type, listener.handler);
          this.listeners.delete(listenerId);
        }
      },
      prevent_default: (eventId) => {
        const event = this.elements.get(eventId);
        if (event && event.preventDefault) event.preventDefault();
      },
      stop_propagation: (eventId) => {
        const event = this.elements.get(eventId);
        if (event && event.stopPropagation) event.stopPropagation();
      },
      get_target: (eventId) => {
        const event = this.elements.get(eventId);
        if (!event || !event.target) return 0;
        const id = ++this.elementCounter;
        this.elements.set(id, event.target);
        return id;
      },
      get_value: (eventId, bufPtr) => {
        const event = this.elements.get(eventId);
        if (!event || !event.target) return 0;
        const value = event.target.value || '';
        return this.allocString(value);
      },
    };
  }

  // ========== Timing Imports ==========
  _timingImports() {
    return {
      now: () => performance.now(),
      set_timeout: (callbackIdx, ms) => {
        const timerId = setTimeout(() => {
          if (this.instance.exports.__indirect_function_table) {
            this.instance.exports.__indirect_function_table.get(callbackIdx)();
          }
        }, ms);
        const id = ++this.timerCounter;
        this.timers.set(id, timerId);
        return id;
      },
      clear_timeout: (id) => {
        const timerId = this.timers.get(id);
        if (timerId !== undefined) {
          clearTimeout(timerId);
          this.timers.delete(id);
        }
      },
      set_interval: (callbackIdx, ms) => {
        const timerId = setInterval(() => {
          if (this.instance.exports.__indirect_function_table) {
            this.instance.exports.__indirect_function_table.get(callbackIdx)();
          }
        }, ms);
        const id = ++this.timerCounter;
        this.timers.set(id, timerId);
        return id;
      },
      clear_interval: (id) => {
        const timerId = this.timers.get(id);
        if (timerId !== undefined) {
          clearInterval(timerId);
          this.timers.delete(id);
        }
      },
      request_animation_frame: (callbackIdx) => {
        const frameId = requestAnimationFrame((timestamp) => {
          if (this.instance.exports.__indirect_function_table) {
            this.instance.exports.__indirect_function_table.get(callbackIdx)(timestamp);
          }
        });
        return frameId;
      },
    };
  }

  // ========== Fetch Imports ==========
  _fetchImports() {
    return {
      start: (urlPtr, urlLen, method) => {
        const url = this.readString(urlPtr);
        const id = ++this.fetchCounter;
        const fetchInfo = { status: 0, body: null, done: false };
        this.fetches.set(id, fetchInfo);

        fetch(url, { method: method === 1 ? 'POST' : 'GET' })
          .then(async (response) => {
            fetchInfo.status = response.status;
            fetchInfo.body = await response.text();
            fetchInfo.done = true;
          })
          .catch((err) => {
            fetchInfo.status = 0;
            fetchInfo.body = err.message;
            fetchInfo.done = true;
          });

        return id;
      },
      poll: (id) => {
        const fetchInfo = this.fetches.get(id);
        return fetchInfo && fetchInfo.done ? 1 : 0;
      },
      get_status: (id) => {
        const fetchInfo = this.fetches.get(id);
        return fetchInfo ? fetchInfo.status : 0;
      },
      get_body: (id, bufPtr) => {
        const fetchInfo = this.fetches.get(id);
        if (!fetchInfo || !fetchInfo.body) return 0;
        return this.allocString(fetchInfo.body);
      },
      abort: (id) => {
        this.fetches.delete(id);
      },
    };
  }

  // ========== Storage Imports ==========
  _storageImports() {
    return {
      local_get: (keyPtr, keyLen, bufPtr) => {
        const key = this.readString(keyPtr);
        const value = localStorage.getItem(key);
        if (value === null) return 0;
        return this.allocString(value);
      },
      local_set: (keyPtr, keyLen, valPtr, valLen) => {
        const key = this.readString(keyPtr);
        const val = this.readString(valPtr);
        localStorage.setItem(key, val);
      },
      local_remove: (keyPtr, keyLen) => {
        const key = this.readString(keyPtr);
        localStorage.removeItem(key);
      },
    };
  }

  // ========== Router Imports ==========
  _routerImports() {
    return {
      push_state: (pathPtr, pathLen) => {
        const path = this.readString(pathPtr);
        history.pushState(null, '', path);
      },
      replace_state: (pathPtr, pathLen) => {
        const path = this.readString(pathPtr);
        history.replaceState(null, '', path);
      },
      get_pathname: (bufPtr) => {
        return this.allocString(location.pathname);
      },
    };
  }

  // ========== Memory Imports ==========
  _memoryImports() {
    return {
      alloc: (size) => {
        // Simple bump allocator for async state machine frames
        // Align to 8 bytes for i64 locals
        const alignedSize = (size + 7) & ~7;
        const ptr = this.heapPtr;
        this.heapPtr += alignedSize;

        // Grow memory if needed
        if (this.memory && this.heapPtr > this.memory.buffer.byteLength) {
          const pages = Math.ceil((this.heapPtr - this.memory.buffer.byteLength) / 65536);
          this.memory.grow(pages);
        }

        return ptr;
      },
      realloc: (ptr, newSize) => {
        // Simple realloc - just allocate new block (no copy for now)
        return this._memoryImports().alloc(newSize);
      },
      free: (ptr) => {
        // No-op for bump allocator - frames are short-lived
      },
      heap_alloc: (size) => {
        return this._memoryImports().alloc(size);
      },
    };
  }

  // ========== Morpheme (Array) Imports ==========
  _morphemeImports() {
    return {
      array_new: () => {
        const id = ++this.arrayCounter;
        this.arrays.set(id, []);
        return id;
      },
      array_push: (arrId, val) => {
        const arr = this.arrays.get(arrId);
        if (arr) arr.push(val);
      },
      array_get: (arrId, idx) => {
        const arr = this.arrays.get(arrId);
        return arr ? BigInt(arr[idx] || 0) : BigInt(0);
      },
      array_set: (arrId, idx, val) => {
        const arr = this.arrays.get(arrId);
        if (arr) arr[idx] = val;
      },
      array_len: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr ? arr.length : 0;
      },
      array_map: (arrId, fnIdx) => {
        const arr = this.arrays.get(arrId);
        if (!arr) return 0;
        const table = this.instance.exports.__indirect_function_table;
        const newArr = arr.map(v => table ? table.get(fnIdx)(v) : v);
        const newId = ++this.arrayCounter;
        this.arrays.set(newId, newArr);
        return newId;
      },
      array_filter: (arrId, fnIdx) => {
        const arr = this.arrays.get(arrId);
        if (!arr) return 0;
        const table = this.instance.exports.__indirect_function_table;
        const newArr = arr.filter(v => table ? table.get(fnIdx)(v) : false);
        const newId = ++this.arrayCounter;
        this.arrays.set(newId, newArr);
        return newId;
      },
      array_parallel_map: (arrId, fnIdx) => this._morphemeImports().array_map(arrId, fnIdx),
      array_parallel_filter: (arrId, fnIdx) => this._morphemeImports().array_filter(arrId, fnIdx),
      array_parallel_reduce: (arrId, fnIdx, init) => this._morphemeImports().array_reduce(arrId, fnIdx, init),
      array_reduce: (arrId, fnIdx, init) => {
        const arr = this.arrays.get(arrId);
        if (!arr) return init;
        const table = this.instance.exports.__indirect_function_table;
        return arr.reduce((acc, v) => table ? table.get(fnIdx)(acc, v) : acc, init);
      },
      array_sort: (arrId) => {
        const arr = this.arrays.get(arrId);
        if (arr) arr.sort((a, b) => Number(a) - Number(b));
        return arrId;
      },
      array_first: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr && arr.length > 0 ? BigInt(arr[0]) : BigInt(0);
      },
      array_last: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr && arr.length > 0 ? BigInt(arr[arr.length - 1]) : BigInt(0);
      },
      array_nth: (arrId, n) => {
        const arr = this.arrays.get(arrId);
        return arr && n < arr.length ? BigInt(arr[n]) : BigInt(0);
      },
      array_sum: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr ? BigInt(arr.reduce((a, b) => Number(a) + Number(b), 0)) : BigInt(0);
      },
      array_product: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr ? BigInt(arr.reduce((a, b) => Number(a) * Number(b), 1)) : BigInt(1);
      },
      array_min: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr && arr.length > 0 ? BigInt(Math.min(...arr.map(Number))) : BigInt(0);
      },
      array_max: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr && arr.length > 0 ? BigInt(Math.max(...arr.map(Number))) : BigInt(0);
      },
      array_all: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr && arr.every(v => !!v) ? 1 : 0;
      },
      array_any: (arrId) => {
        const arr = this.arrays.get(arrId);
        return arr && arr.some(v => !!v) ? 1 : 0;
      },
      array_random_element: (arrId) => {
        const arr = this.arrays.get(arrId);
        if (!arr || arr.length === 0) return BigInt(0);
        return BigInt(arr[Math.floor(Math.random() * arr.length)]);
      },
      vec_join: (arrId, sepPtr) => {
        // Join array of string pointers with separator
        const arr = this.arrays.get(arrId);
        if (!arr || arr.length === 0) return this.allocString('');
        const sep = this.strings.get(Number(sepPtr)) || this.readString(Number(sepPtr));
        const strings = arr.map(ptr => this.strings.get(Number(ptr)) || this.readString(Number(ptr)));
        return this.allocString(strings.join(sep));
      },
    };
  }

  // ========== Math Imports ==========
  _mathImports() {
    return {
      sqrt: Math.sqrt,
      sin: Math.sin,
      cos: Math.cos,
      tan: Math.tan,
      pow: Math.pow,
      exp: Math.exp,
      log: Math.log,
      floor: Math.floor,
      ceil: Math.ceil,
      round: Math.round,
      abs: Math.abs,
      abs_int: (x) => x < 0n ? -x : x,
      random: Math.random,
      clamp: (x, min, max) => Math.max(min, Math.min(max, x)),
      clamp_int: (x, min, max) => x < min ? min : x > max ? max : x,
      min: Math.min,
      max: Math.max,
      min_int: (a, b) => a < b ? a : b,
      max_int: (a, b) => a > b ? a : b,
      signum: Math.sign,
      signum_int: (x) => x < 0n ? -1n : x > 0n ? 1n : 0n,
    };
  }

  // ========== VDOM Imports ==========
  _vdomImports() {
    return {
      create_vnode: (tagStrRef) => {
        const tag = this.strings.get(Number(tagStrRef)) || this.readString(Number(tagStrRef));
        const id = ++this.vnodeCounter;
        this.vnodes.set(id, {
          type: 'element',
          tag,
          props: {},
          children: [],
          dom: null,
        });
        return id;
      },
      create_text_vnode: (textStrRef) => {
        const text = this.strings.get(Number(textStrRef)) || this.readString(Number(textStrRef));
        const id = ++this.vnodeCounter;
        this.vnodes.set(id, {
          type: 'text',
          text,
          dom: null,
        });
        return id;
      },
      create_fragment: () => {
        const id = ++this.vnodeCounter;
        this.vnodes.set(id, {
          type: 'fragment',
          children: [],
          dom: null,
        });
        return id;
      },
      set_vnode_prop: (vnodeId, nameStrRef, value) => {
        const vnode = this.vnodes.get(vnodeId);
        if (vnode && vnode.props) {
          const name = this.strings.get(Number(nameStrRef)) || this.readString(Number(nameStrRef));
          vnode.props[name] = Number(value);
        }
      },
      set_vnode_str_prop: (vnodeId, nameStrRef, valueStrRef) => {
        const vnode = this.vnodes.get(vnodeId);
        if (vnode && vnode.props) {
          const name = this.strings.get(Number(nameStrRef)) || this.readString(Number(nameStrRef));
          const value = this.strings.get(Number(valueStrRef)) || this.readString(Number(valueStrRef));
          vnode.props[name] = value;
        }
      },
      append_vnode_child: (parentId, childId) => {
        const parent = this.vnodes.get(parentId);
        const child = this.vnodes.get(childId);
        if (parent && parent.children && child) {
          parent.children.push(child);
        }
      },
      diff_and_patch: (oldId, newId, parentDomId) => {
        const oldVnode = this.vnodes.get(oldId);
        const newVnode = this.vnodes.get(newId);
        const parentDom = this.elements.get(parentDomId);
        if (newVnode && parentDom) {
          this._patchVnode(oldVnode, newVnode, parentDom);
        }
      },
      mount_vnode: (vnodeId, selectorStrRef) => {
        const vnode = this.vnodes.get(vnodeId);
        const selector = this.strings.get(Number(selectorStrRef)) || this.readString(Number(selectorStrRef));
        const container = document.querySelector(selector);
        if (!vnode || !container) return 0;

        const dom = this._createDom(vnode);
        container.appendChild(dom);
        vnode.dom = dom;

        const domId = ++this.elementCounter;
        this.elements.set(domId, dom);
        return domId;
      },
      dispose: (vnodeId) => {
        const vnode = this.vnodes.get(vnodeId);
        if (vnode && vnode.dom && vnode.dom.parentNode) {
          vnode.dom.parentNode.removeChild(vnode.dom);
        }
        this.vnodes.delete(vnodeId);
      },
    };
  }

  /**
   * Create DOM from VNode
   */
  _createDom(vnode) {
    if (vnode.type === 'text') {
      return document.createTextNode(vnode.text);
    }

    if (vnode.type === 'fragment') {
      const frag = document.createDocumentFragment();
      for (const child of vnode.children || []) {
        frag.appendChild(this._createDom(child));
      }
      return frag;
    }

    const el = document.createElement(vnode.tag);
    for (const [key, value] of Object.entries(vnode.props || {})) {
      if (key.startsWith('on')) {
        // Event handler
        el.addEventListener(key.slice(2).toLowerCase(), value);
      } else if (key === 'className') {
        el.className = value;
      } else if (key === 'style' && typeof value === 'object') {
        Object.assign(el.style, value);
      } else {
        el.setAttribute(key, value);
      }
    }

    for (const child of vnode.children || []) {
      el.appendChild(this._createDom(child));
    }

    vnode.dom = el;
    return el;
  }

  /**
   * Patch VNode differences
   */
  _patchVnode(oldVnode, newVnode, parent) {
    if (!oldVnode) {
      // Mount new
      const dom = this._createDom(newVnode);
      parent.appendChild(dom);
      return;
    }

    if (!newVnode) {
      // Remove old
      if (oldVnode.dom) parent.removeChild(oldVnode.dom);
      return;
    }

    if (oldVnode.type !== newVnode.type || oldVnode.tag !== newVnode.tag) {
      // Replace
      const dom = this._createDom(newVnode);
      if (oldVnode.dom) {
        parent.replaceChild(dom, oldVnode.dom);
      } else {
        parent.appendChild(dom);
      }
      return;
    }

    // Update existing
    newVnode.dom = oldVnode.dom;

    if (newVnode.type === 'text') {
      if (oldVnode.text !== newVnode.text) {
        newVnode.dom.textContent = newVnode.text;
      }
      return;
    }

    // Update props
    const oldProps = oldVnode.props || {};
    const newProps = newVnode.props || {};

    for (const key of Object.keys(newProps)) {
      if (oldProps[key] !== newProps[key]) {
        if (key === 'className') {
          newVnode.dom.className = newProps[key];
        } else {
          newVnode.dom.setAttribute(key, newProps[key]);
        }
      }
    }

    for (const key of Object.keys(oldProps)) {
      if (!(key in newProps)) {
        newVnode.dom.removeAttribute(key);
      }
    }

    // Update children
    const oldChildren = oldVnode.children || [];
    const newChildren = newVnode.children || [];
    const maxLen = Math.max(oldChildren.length, newChildren.length);

    for (let i = 0; i < maxLen; i++) {
      this._patchVnode(oldChildren[i], newChildren[i], newVnode.dom);
    }
  }

  // ========== Signal Imports ==========
  _signalImports() {
    return {
      create: (initialValue) => {
        const id = ++this.signalCounter;
        this.signals.set(id, {
          value: initialValue,
          subscribers: new Set(),
        });
        return id;
      },
      get: (signalId) => {
        const signal = this.signals.get(signalId);
        return signal ? signal.value : BigInt(0);
      },
      set: (signalId, value) => {
        const signal = this.signals.get(signalId);
        if (signal) {
          signal.value = value;
          if (this.signalBatchDepth === 0) {
            this._notifySubscribers(signalId);
          } else {
            this.pendingEffects.push(() => this._notifySubscribers(signalId));
          }
        }
      },
      subscribe: (signalId, callbackIdx) => {
        const signal = this.signals.get(signalId);
        if (!signal) return 0;
        const subId = ++this.signalCounter;
        signal.subscribers.add({ id: subId, callback: callbackIdx });
        return subId;
      },
      unsubscribe: (subId) => {
        for (const signal of this.signals.values()) {
          for (const sub of signal.subscribers) {
            if (sub.id === subId) {
              signal.subscribers.delete(sub);
              return;
            }
          }
        }
      },
      batch_start: () => {
        this.signalBatchDepth++;
      },
      batch_end: () => {
        this.signalBatchDepth--;
        if (this.signalBatchDepth === 0) {
          const effects = this.pendingEffects;
          this.pendingEffects = [];
          for (const effect of effects) {
            effect();
          }
        }
      },
      computed: (fnIdx) => {
        // Create a signal that recomputes when dependencies change
        const id = ++this.signalCounter;
        const compute = () => {
          const table = this.instance.exports.__indirect_function_table;
          return table ? table.get(fnIdx)() : BigInt(0);
        };
        this.signals.set(id, {
          value: compute(),
          subscribers: new Set(),
          compute,
        });
        return id;
      },
      effect: (fnIdx) => {
        // Run effect immediately and whenever dependencies change
        const table = this.instance.exports.__indirect_function_table;
        if (table) table.get(fnIdx)();
        return ++this.signalCounter;
      },
    };
  }

  _notifySubscribers(signalId) {
    const signal = this.signals.get(signalId);
    if (!signal) return;
    const table = this.instance.exports.__indirect_function_table;
    for (const sub of signal.subscribers) {
      if (table) table.get(sub.callback)(signal.value);
    }
  }

  // ========== Async Imports ==========
  _asyncImports() {
    return {
      promise_new: () => {
        const id = ++this.fetchCounter;
        let resolve, reject;
        const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
        this.fetches.set(id, { promise, resolve, reject });
        return id;
      },
      promise_resolve: (promiseId, value) => {
        const p = this.fetches.get(promiseId);
        if (p && p.resolve) p.resolve(value);
      },
      promise_reject: (promiseId, errPtr, errLen) => {
        const p = this.fetches.get(promiseId);
        if (p && p.reject) p.reject(new Error(this.readString(errPtr)));
      },
      promise_then: (promiseId, onFulfillIdx, onRejectIdx) => {
        const p = this.fetches.get(promiseId);
        if (!p || !p.promise) return 0;
        const newId = ++this.fetchCounter;
        const table = this.instance.exports.__indirect_function_table;
        const newPromise = p.promise.then(
          (val) => table ? table.get(onFulfillIdx)(val) : val,
          (err) => table && onRejectIdx ? table.get(onRejectIdx)(err) : Promise.reject(err)
        );
        this.fetches.set(newId, { promise: newPromise });
        return newId;
      },
      promise_catch: (promiseId, onRejectIdx) => {
        const p = this.fetches.get(promiseId);
        if (!p || !p.promise) return 0;
        const newId = ++this.fetchCounter;
        const table = this.instance.exports.__indirect_function_table;
        const newPromise = p.promise.catch((err) => table ? table.get(onRejectIdx)(err) : Promise.reject(err));
        this.fetches.set(newId, { promise: newPromise });
        return newId;
      },
      promise_all: (arrId) => {
        const arr = this.arrays.get(arrId);
        if (!arr) return 0;
        const promises = arr.map(id => this.fetches.get(id)?.promise).filter(Boolean);
        const newId = ++this.fetchCounter;
        this.fetches.set(newId, { promise: Promise.all(promises) });
        return newId;
      },
      promise_race: (arrId) => {
        const arr = this.arrays.get(arrId);
        if (!arr) return 0;
        const promises = arr.map(id => this.fetches.get(id)?.promise).filter(Boolean);
        const newId = ++this.fetchCounter;
        this.fetches.set(newId, { promise: Promise.race(promises) });
        return newId;
      },
      spawn: (fnIdx) => {
        const table = this.instance.exports.__indirect_function_table;
        if (table) queueMicrotask(() => table.get(fnIdx)());
        return 0;
      },
      yield_now: () => {
        // No-op in single-threaded JS
      },

      // ========== State Machine Async Support ==========
      // These functions support explicit state machine transformation
      // as defined in ASYNC-STATE-MACHINE-SPEC.md §4.4

      /**
       * Create a continuation for an async state machine suspension.
       *
       * @param {number} framePtr - Pointer to the suspension frame in WASM memory
       * @param {number} state - Next state to resume at
       * @param {BigInt} promise - The promise/future value being awaited (as i64)
       * @returns {number} Continuation ID to encode in return value
       */
      async_create_continuation: (framePtr, state, promise) => {
        const contId = ++this.continuationCounter;

        // Convert promise i64 to a JS Promise
        // The promise value is an ID referencing a tracked promise
        const promiseId = Number(promise);
        const promiseEntry = this.fetches.get(promiseId);

        this.continuations.set(contId, {
          framePtr,
          state,
          promiseId,
          promise: promiseEntry?.promise || Promise.resolve(promise),
        });

        return contId;
      },

      /**
       * Run an async state machine function to completion.
       *
       * @param {string} funcName - Name of the exported WASM function
       * @param {...any} args - Initial arguments to the function
       * @returns {Promise<BigInt>} The final result value
       */
      async_run: async (funcName, ...args) => {
        const func = this.instance.exports[funcName];
        if (!func) throw new Error(`Function ${funcName} not found`);

        // Initial call: frame_ptr = 0, resume_value = 0
        let result = func(0, BigInt(0), ...args);

        // Check for suspension (bit 32 set = SUSPENDED_FLAG)
        const SUSPENDED_FLAG = BigInt(1) << BigInt(32);
        const CONT_MASK = BigInt(0xFFFFFFFF);

        while ((result & SUSPENDED_FLAG) !== BigInt(0)) {
          const contId = Number(result & CONT_MASK);
          const cont = this.continuations.get(contId);

          if (!cont) {
            throw new Error(`Continuation ${contId} not found`);
          }

          // Wait for the promise to resolve
          let resolvedValue;
          try {
            resolvedValue = await cont.promise;
            // Convert resolved value to i64
            if (typeof resolvedValue === 'bigint') {
              // Already BigInt
            } else if (typeof resolvedValue === 'number') {
              resolvedValue = BigInt(resolvedValue);
            } else {
              resolvedValue = BigInt(0);
            }
          } catch (err) {
            // On error, we could propagate via a different mechanism
            // For now, just return 0
            console.error('Async error:', err);
            resolvedValue = BigInt(0);
          }

          // Clean up continuation
          this.continuations.delete(contId);

          // Resume: call with frame_ptr and resolved value
          result = func(cont.framePtr, resolvedValue);
        }

        // Complete - result is the final value (bits 31-0)
        return result & CONT_MASK;
      },

      await_promise: (promiseId) => {
        // Synchronous await not possible in JS - would need Asyncify
        const p = this.fetches.get(promiseId);
        return p && p.result !== undefined ? p.result : BigInt(0);
      },
    };
  }

  // ========== Browser Imports ==========
  _browserImports() {
    return {
      window: () => {
        const id = ++this.elementCounter;
        this.elements.set(id, window);
        return id;
      },
      document: () => {
        const id = ++this.elementCounter;
        this.elements.set(id, document);
        return id;
      },
      inner_width: (winId) => window.innerWidth,
      inner_height: (winId) => window.innerHeight,
      add_event_listener: (winId, typePtr, typeLen, callbackIdx) => {
        const type = this.readString(typePtr);
        const handler = (event) => {
          const eventId = ++this.elementCounter;
          this.elements.set(eventId, event);
          const table = this.instance.exports.__indirect_function_table;
          if (table) table.get(callbackIdx)(eventId);
        };
        window.addEventListener(type, handler);
        const listenerId = ++this.listenerCounter;
        this.listeners.set(listenerId, { element: window, type, handler });
        return listenerId;
      },
      remove_event_listener: (winId, listenerId) => {
        const listener = this.listeners.get(listenerId);
        if (listener) {
          window.removeEventListener(listener.type, listener.handler);
          this.listeners.delete(listenerId);
        }
      },
      match_media: (queryPtr, queryLen) => {
        const query = this.readString(queryPtr);
        const mql = window.matchMedia(query);
        const id = ++this.elementCounter;
        this.elements.set(id, mql);
        return id;
      },
      mql_matches: (mqlId) => {
        const mql = this.elements.get(mqlId);
        return mql && mql.matches ? 1 : 0;
      },
      mql_add_listener: (mqlId, callbackIdx) => {
        const mql = this.elements.get(mqlId);
        if (!mql) return 0;
        const handler = (event) => {
          const table = this.instance.exports.__indirect_function_table;
          if (table) table.get(callbackIdx)(event.matches ? 1 : 0);
        };
        mql.addEventListener('change', handler);
        const listenerId = ++this.listenerCounter;
        this.listeners.set(listenerId, { element: mql, type: 'change', handler });
        return listenerId;
      },
      mql_remove_listener: (mqlId, listenerId) => {
        const mql = this.elements.get(mqlId);
        const listener = this.listeners.get(listenerId);
        if (mql && listener) {
          mql.removeEventListener('change', listener.handler);
          this.listeners.delete(listenerId);
        }
      },
    };
  }
}

/**
 * Load and run a Sigil WASM module
 */
export async function loadSigilModule(wasmPath, containerSelector = '#app') {
  const runtime = new SigilRuntime();
  const response = await fetch(wasmPath);
  const wasmBuffer = await response.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(wasmBuffer, runtime.getImports());

  runtime.init(instance);

  // Run main if exported
  if (instance.exports.main) {
    instance.exports.main();
  }

  return { runtime, instance };
}

// Default export for ES modules
export default SigilRuntime;
