# Sigil Concurrency Specification

## 1. Philosophy: Concurrent Metaphors

Concurrency concepts exist across cultures, often expressed through powerful metaphors:

| Metaphor | Culture | Concurrency Concept |
|----------|---------|---------------------|
| **Weaving** | Global (esp. Andean, West African) | Interleaving threads, synchronization |
| **Water Flow** | Chinese (Dao), Japanese | Streams, backpressure, confluence |
| **Ancestor Communication** | African, Indigenous | Message passing, channels |
| **Collective Labor** | Andean (Ayni/Minka) | Work stealing, task distribution |
| **Dance/Ceremony** | Many traditions | Choreographed coordination, barriers |
| **Market/Bazaar** | Middle Eastern, African | Producer-consumer, negotiation |

Sigil embraces these metaphors alongside traditional CS terminology.

---

## 2. Async/Await (Futures)

### 2.1 Basic Async

```sigil
// Async function returns a Future
async fn fetch_user(id: u64) -> User~ {
    let response~ = http·get!("/users/{id}")|await
    response~|json·parse
}

// Await suspends until ready
async fn main() {
    let user~ = fetch_user(42)|await
    print!("Got user: {user~.name}")
}
```

### 2.2 Polysynthetic Async Syntax

```sigil
// Traditional chained awaits
let a = fetch_a()|await
let b = fetch_b(a)|await
let c = fetch_c(b)|await

// Polysynthetic: await morpheme (⌛ or ·await)
let c = fetch_a()⌛|fetch_b⌛|fetch_c⌛

// Parallel await (all at once)
let (a, b, c) = (fetch_a(), fetch_b(), fetch_c())|await·all

// Race (first to complete)
let first = (fetch_a(), fetch_b(), fetch_c())|await·race

// With timeout
let result? = fetch_slow()|await·timeout(5·sec)
```

### 2.3 Future Combinators

```sigil
// Map over future result
let future_name = fetch_user(42)|τ·async{.name}

// Filter future
let future_active = fetch_user(42)|φ·async{.active}

// Chain futures (flatMap/andThen)
let future_posts = fetch_user(42)|then{u => fetch_posts(u.id)}

// Handle errors
let result = fetch_user(42)
    |catch{NetworkError => default_user()}
    |await
```

---

## 3. Weaving (Thread Model)

Inspired by textile traditions where threads interlace to create fabric.

### 3.1 Threads as Warp and Weft

```sigil
use concurrency·weave·{Thread, Loom}

// Spawn a thread (a single "thread" in the weave)
let handle = Thread·spin {
    // Work happens here
    heavy_computation()
}

// Join threads (bring threads together)
let result = handle|weave·join

// Multiple threads on a loom
let loom = Loom·new()
let threads = (0..4)|τ{i =>
    loom|spin{ compute_chunk(i) }
}

// Gather all results (complete the fabric)
let results = threads|weave·gather
```

### 3.2 Thread Pools (Collective Labor)

```sigil
use concurrency·minka·{WorkGroup, Task}

// Minka: Andean collective labor tradition
// Work is distributed, results gathered

let minka = WorkGroup·new(num_cpus())

// Submit tasks
let futures = data·chunks(100)|τ{chunk =>
    minka|submit{ process(chunk) }
}

// Ayni: reciprocal exchange (work stealing)
let ayni_pool = WorkGroup·with_stealing(num_cpus())
```

---

## 4. Water Flow (Streams)

Streams as continuous data flow, inspired by Daoist water philosophy.

### 4.1 Basic Streams

```sigil
use concurrency·flow·{Stream, Source, Sink}

// Create a stream
let numbers: Stream<i32> = Stream·from([1, 2, 3, 4, 5])

// Infinite stream
let naturals: Stream<u64> = Stream·iterate(0, |n| n + 1)

// From async source
let events: Stream<Event~> = websocket·connect(url)|stream
```

### 4.2 Stream Operators (Flow Morphemes)

```sigil
// Transform stream
events
|τ·flow{parse_event}      // Map each element
|φ·flow{.is_valid}        // Filter
|σ·flow·window(10)        // Sort within windows
|ρ·flow{acc, x => acc + x} // Running reduction

// Backpressure (water backing up)
events
|buffer(100)              // Buffer up to 100
|throttle(10·per_sec)     // Limit rate
|debounce(100·ms)         // Collapse rapid events

// Confluence (merging streams)
let merged = stream_a|confluence(stream_b)
let merged = [stream_a, stream_b, stream_c]|confluence·all

// Branching (splitting stream)
let (evens, odds) = numbers|fork{n => n % 2 == 0}
```

### 4.3 Stream Consumption

```sigil
// Collect all (blocking)
let all_events = events|collect·await

// Process each as it arrives
events|for_each·await{event =>
    handle(event)
}

// Sink (drain to destination)
events|drain·to(file_sink)
events|drain·to(kafka_sink)
```

---

## 5. Ancestor Channels (Message Passing)

Communication between concurrent entities, inspired by ancestor communication traditions.

### 5.1 Basic Channels

```sigil
use concurrency·voice·{Channel, Sender, Receiver}

// Create channel (bounded)
let (send, recv): (Sender<Message>, Receiver<Message>) = Channel·new(100)

// Create unbounded channel
let (send, recv) = Channel·unbounded()

// Send message (speak to ancestors)
send|voice(Message { data: 42 })

// Receive message (hear from ancestors)
let msg? = recv|listen  // Non-blocking
let msg! = recv|listen·await  // Blocking

// Close channel (complete the ritual)
send|silence
```

### 5.2 Channel Patterns

```sigil
// Multiple producers, single consumer (MPSC)
let (send, recv) = Channel·mpsc(100)
let senders = (0..4)|τ{_ => send·clone}

// Multiple producers, multiple consumers (MPMC)
let (send, recv) = Channel·mpmc(100)

// Oneshot (single message, like a prophecy)
let (promise, fulfillment) = Channel·oneshot()
promise|voice(result)
let result = fulfillment|listen·await

// Broadcast (one speaker, many listeners)
let broadcast = Channel·broadcast(100)
let listener1 = broadcast|subscribe
let listener2 = broadcast|subscribe
broadcast|voice(announcement)
```

### 5.3 Select (Listen to Many Voices)

```sigil
// Wait for first of multiple channels
let result = select! {
    msg = channel_a|listen => handle_a(msg),
    msg = channel_b|listen => handle_b(msg),
    () = timeout(1·sec) => handle_timeout(),
    default => handle_none(),
}

// Polysynthetic select
let result = [channel_a, channel_b]|listen·first
```

---

## 6. Dance (Synchronization)

Coordination primitives, inspired by ceremonial dance where participants synchronize.

### 6.1 Barriers (Ceremony Points)

```sigil
use concurrency·dance·{Barrier, Latch}

// All dancers must arrive before proceeding
let barrier = Barrier·new(num_threads)

// In each thread:
do_preparation()
barrier|gather        // Wait for all
do_main_ritual()      // All proceed together

// Countdown latch (ceremony begins after N arrivals)
let latch = Latch·new(3)
latch|arrive          // Signal arrival
latch|await_ceremony  // Wait for all arrivals
```

### 6.2 Locks (Taking Turns)

```sigil
use concurrency·dance·{Mutex, RwLock, Semaphore}

// Mutex (one dancer at a time)
let shared = Mutex·new(data)
{
    let guard = shared|hold  // Acquire
    guard|mutate()
}   // Released when guard drops

// RwLock (many watchers, one performer)
let shared = RwLock·new(data)
{
    let read = shared|observe      // Many can observe
    let write = shared|perform     // Only one can perform
}

// Semaphore (limited participants)
let sem = Semaphore·new(3)  // Max 3 concurrent
sem|enter               // Wait for permission
// ... do work ...
sem|leave               // Release slot
```

### 6.3 Condition Variables (Waiting for Cue)

```sigil
use concurrency·dance·{Condition, Signal}

let cond = Condition·new()
let mutex = Mutex·new(state)

// Wait for cue
{
    let guard = mutex|hold
    while !ready(guard) {
        guard = cond|await_cue(guard)
    }
}

// Give cue
cond|cue_one    // Wake one waiter
cond|cue_all    // Wake all waiters
```

---

## 7. Market (Actor Model)

Actors as independent merchants in a bazaar, communicating through messages.

### 7.1 Actor Definition

```sigil
use concurrency·market·{Actor, Address, Message}

// Define an actor
actor Counter {
    state: i64 = 0

    // Handle messages
    on Increment(n: i64) {
        self.state += n
    }

    on Decrement(n: i64) {
        self.state -= n
    }

    on GetValue -> i64 {
        self.state
    }
}

// Spawn actor (open shop)
let counter: Address<Counter> = Counter·open()

// Send message (visit shop)
counter|tell(Increment(5))
counter|tell(Decrement(2))

// Ask with response (negotiate)
let value = counter|ask(GetValue)|await  // 3
```

### 7.2 Actor Supervision (Guild System)

```sigil
use concurrency·market·{Supervisor, Strategy}

// Supervisor manages actor lifecycle
let supervisor = Supervisor·new()
    |strategy(Strategy·OneForOne)  // Restart failed individual
    |max_restarts(3, per: 1·minute)

// Register children
supervisor|apprentice(Counter·open)
supervisor|apprentice(Logger·open)

// Supervision strategies
Strategy·OneForOne    // Restart only failed actor
Strategy·OneForAll    // Restart all if one fails
Strategy·RestForOne   // Restart actor and those started after it
```

### 7.3 Actor Patterns

```sigil
// Router (distribute to workers)
let router = Router·round_robin([worker1, worker2, worker3])
router|tell(Work(data))

// Aggregator (collect from many)
let aggregator = Aggregator·new(expected: 10)
workers|τ{w => w|tell(Report·to(aggregator))}
let results = aggregator|collect·await

// Saga (distributed transaction)
saga {
    step debit_account(amount) compensate credit_account(amount)
    step reserve_inventory(item) compensate release_inventory(item)
    step charge_payment(amount) compensate refund_payment(amount)
}
```

---

## 8. Parallel Iteration

### 8.1 Parallel Morphemes

```sigil
// Sequential (default)
data|τ{process}        // One at a time

// Parallel
data|par·τ{process}    // All at once
data|τ·par{process}    // Same thing

// Parallel with concurrency limit
data|par(4)·τ{process} // At most 4 concurrent

// Parallel filter, sort, reduce
data|par·φ{predicate}
data|par·σ.field
data|par·ρ(+)
```

### 8.2 Parallel Pipelines

```sigil
// Each stage runs in parallel
data
|par·τ{stage1}
|par·φ{stage2}
|par·τ{stage3}
|collect

// Chunked parallelism
data
|chunks(1000)
|par·τ{chunk => chunk|τ{process}|collect}
|flatten
```

### 8.3 Work Distribution

```sigil
// Work stealing (ayni)
data|par·ayni·τ{process}

// Static partitioning
data|par·static(num_cpus())·τ{process}

// Dynamic scheduling
data|par·dynamic·τ{process}
```

---

## 9. Structured Concurrency

All concurrent operations are scoped — no orphan tasks.

### 9.1 Task Scopes

```sigil
use concurrency·scope·{Scope, Task}

// All tasks must complete before scope exits
scope |s| {
    s|spawn{ task_a() }
    s|spawn{ task_b() }
    s|spawn{ task_c() }
}  // Waits for all tasks

// With cancellation
let result = scope |s| {
    let a = s|spawn{ fetch_a() }
    let b = s|spawn{ fetch_b() }

    // If any fails, cancel others
    s|cancel_on_error

    (a|await, b|await)
}
```

### 9.2 Cancellation

```sigil
// Cancellation token
let token = CancelToken·new()

async fn cancellable_work(cancel: CancelToken) {
    loop {
        if cancel|is_cancelled { break }
        do_work()
        yield  // Check cancellation
    }
}

// Cancel from outside
token|cancel

// Timeout as cancellation
scope |s| {
    s|timeout(5·sec)
    // All spawned tasks cancelled after 5 seconds
}
```

---

## 10. Memory Ordering

### 10.1 Atomics

```sigil
use concurrency·atom·{Atomic, Ordering}

let counter = Atomic·new(0i64)

// Operations
counter|load(Ordering·Relaxed)
counter|store(42, Ordering·Release)
counter|fetch_add(1, Ordering·SeqCst)
counter|compare_exchange(expected, new, Ordering·AcqRel)

// Atomic morphemes
counter|atom·inc              // fetch_add(1)
counter|atom·dec              // fetch_sub(1)
counter|atom·swap(new)        // exchange
counter|atom·cas(old, new)    // compare_exchange
```

### 10.2 Memory Fences

```sigil
use concurrency·fence·{fence, compiler_fence}

fence(Ordering·Acquire)      // Prevent reordering
fence(Ordering·Release)
fence(Ordering·SeqCst)

compiler_fence(Ordering·Release)  // Compiler only, no CPU fence
```

---

## 11. Evidentiality in Concurrency

### 11.1 Thread-Safe Evidence

```sigil
// Values from other threads are "reported"
async fn fetch_from_thread() -> Data~ {
    let handle = Thread·spin{ compute() }
    handle|join~  // Result is external to this thread
}

// Channel receives are reported
let msg~ = channel|listen·await
let trusted! = msg~|validate!  // Must validate to promote

// Shared state access is uncertain
let value? = atomic|load  // Might have changed
```

### 11.2 Data Race Evidence

```sigil
// The type system tracks potential races
struct SharedState {
    counter: Atomic<i64>,  // Safe to share
    data: Mutex<Data>,     // Protected access
    cache: RwLock<Cache>,  // Read-many, write-exclusive
}

// Unprotected sharing is a compile error
// let bad: &mut Data = shared across threads  // ERROR
```

---

## 12. Complete Example

```sigil
//! Concurrent web scraper using multiple paradigms

use concurrency·{flow, voice, market, scope}
use http·Client

actor Fetcher {
    client: Client

    on Fetch(url: str) -> Response~ {
        self.client|get(url)|await~
    }
}

async fn scrape_site(urls: [str]) -> [Page~] {
    // Create channel for results
    let (send, recv) = voice·Channel·mpsc(100)

    // Spawn fetcher actors
    let fetchers = (0..4)|τ{_ =>
        Fetcher { client: Client·new() }|market·open
    }

    // Distribute work
    scope |s| {
        // Producer: send URLs to fetchers
        s|spawn{
            urls|enumerate|for_each{(i, url) =>
                let fetcher = fetchers[i % 4]
                let response = fetcher|ask(Fetch(url))|await
                send|voice(response|parse_page)
            }
            send|silence  // Close when done
        }

        // Consumer: collect results
        s|spawn{
            recv
            |flow·stream
            |φ·flow{.is_valid}
            |collect·await
        }
    }
}

fn main() {
    let urls = load_urls("sitemap.xml")

    let pages = scrape_site(urls)
        |await·timeout(5·minutes)
        |unwrap_or([])

    print!("Scraped {pages.len} pages")
}
```

---

## 13. Concurrency Morpheme Summary

| Morpheme | Meaning | Example |
|----------|---------|---------|
| `⌛` / `·await` | Await future | `fetch()⌛` |
| `·await·all` | Await all | `futures\|await·all` |
| `·await·race` | First complete | `futures\|await·race` |
| `par·` | Parallel | `data\|par·τ{f}` |
| `·flow` | Stream operation | `stream\|τ·flow{f}` |
| `·spin` | Spawn thread | `Thread·spin{...}` |
| `·voice` | Send message | `send\|voice(msg)` |
| `·listen` | Receive message | `recv\|listen` |
| `·hold` | Acquire lock | `mutex\|hold` |
| `·gather` | Join all | `threads\|gather` |
| `·tell` | Actor fire-and-forget | `actor\|tell(msg)` |
| `·ask` | Actor request-response | `actor\|ask(msg)` |
