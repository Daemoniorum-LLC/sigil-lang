# Holographic Data Structures

> *"In Indra's Net, every jewel reflects every other jewel, infinitely."*
> — Huayan Buddhism

## 1. Overview

Holographic data structures embody the principle that **every part contains information about the whole**. In Sigil, this manifests through:

1. **Erasure-coded structures** - Reconstruct complete data from partial shards
2. **Probabilistic sketches** - Approximate the whole from streaming samples
3. **Content-addressed graphs** - Every node proves its subtree
4. **Superposition types** - Values exist in multiple states until observed

### 1.1 Core Operators

| Symbol | Name | Meaning | Example |
|--------|------|---------|---------|
| `∀` | Universal | Whole from part (deterministic) | `shards\|∀` reconstructs |
| `◊` | Possibility | Approximate/probabilistic access | `sketch\|◊count` estimates |
| `□` | Necessity | Verified/proven whole | `merkle\|□verify` proves |
| `⊛` | Convolution | Combine partial views | `a ⊛ b` merges shards |

### 1.2 Evidentiality Integration

Holographic operations interact with the evidentiality system:

```
Reconstruction Evidence:
┌─────────────────────────────────────────────────────────────┐
│  Source          Operation       Result Evidence            │
├─────────────────────────────────────────────────────────────┤
│  Shards!         ∀ (enough)      T!  (deterministic)        │
│  Shards!         ∀ (degraded)    T?  (uncertain)            │
│  Shards~         ∀ (any)         T~  (reported origin)      │
│  Sketch!         ◊               T?  (inherently approx)    │
│  Merkle~         □verify         T!  (cryptographic proof)  │
│  Superposition   |observe        T!  (collapsed)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Erasure-Coded Structures

### 2.1 Hologram Type

A hologram distributes data across `N` shards where any `K` suffice for reconstruction.

```sigil
/// A holographic encoding of type T
/// K = minimum shards needed, N = total shards
type Hologram<T, const K: usize, const N: usize> where K <= N {
    shards: [Shard<T>; N],
    encoding: ErasureScheme,
}

struct Shard<T> {
    index: u8,
    data: [u8],
    checksum: Hash128,
    origin: Evidence,      // Track where this shard came from
}

enum ErasureScheme {
    ReedSolomon,           // Classic polynomial interpolation
    Fountain(Degree),      // Rateless (LT codes, Raptor)
    LDPC(ParityMatrix),    // Low-density parity check
    Polar(FrozenBits),     // Polar codes
}
```

### 2.2 Creating Holograms

```sigil
use holograph·{Hologram, encode, scatter}

fn distribute_data() {
    let secret! = "The spice must flow"

    // Encode into 7 shards, any 4 reconstruct
    let hologram: Hologram<str, 4, 7> = secret!|encode(ReedSolomon)

    // Scatter to storage nodes
    let locations~ = hologram.shards
        |enumerate
        |τ{(i, shard) => storage_nodes[i]|store(shard)}
        |◊gather   // Probabilistic gather (some may fail)
}
```

### 2.3 Universal Reconstruction (∀)

The `∀` operator reconstructs the whole from available parts:

```sigil
/// Reconstruct with the universal operator
fn recover_data(available: [Shard<T>]) -> T? {
    // ∀ attempts reconstruction
    // Returns T! if enough shards, T? if degraded, Error if impossible
    available|∀
}

// Explicit threshold checking
fn recover_with_threshold<T, const K: usize>(
    shards: [Shard<T>]
) -> Result<T!, ReconstructionError> {
    if shards.len >= K {
        Ok(shards|∀!)  // Guaranteed reconstruction
    } else {
        // Attempt degraded reconstruction with uncertainty
        match shards|∀? {
            Some(data?) => Ok(data?|trust_degraded),
            None => Err(InsufficientShards(shards.len, K)),
        }
    }
}
```

### 2.4 Shard Convolution (⊛)

Combine compatible shards from different sources:

```sigil
/// Merge shards with convolution
fn merge_sources(
    from_alice~: [Shard<T>],
    from_bob~: [Shard<T>],
) -> [Shard<T>]~ {
    // ⊛ combines shards, deduplicating by index
    // Evidence: both sources are reported, result is reported
    from_alice~ ⊛ from_bob~
}

// Convolution rules:
// shard! ⊛ shard! = shard!  (both known, indices must match)
// shard! ⊛ shard~ = shard?  (mixed evidence)
// shard~ ⊛ shard~ = shard~  (both reported)
// shard? ⊛ shard? = shard?  (uncertainty propagates)
```

---

## 3. Probabilistic Sketches

Sketches trade exactness for space efficiency—the quintessential "approximate whole."

### 3.1 Sketch Types

```sigil
/// Cardinality estimation (count distinct)
type HyperLogLog<const P: usize> = {
    registers: [u8; 2**P],
    count_hint: u64◊,  // ◊ marks inherent uncertainty
}

/// Frequency estimation
type CountMinSketch<const W: usize, const D: usize> = {
    tables: [[u32; W]; D],
    total: u64!,
}

/// Membership testing
type BloomFilter<const M: usize, const K: usize> = {
    bits: BitArray<M>,
    hash_seeds: [u64; K],
}

/// Quantile estimation
type TDigest = {
    centroids: Vec<(f64, u64)>,
    compression: f64,
}

/// Top-K frequent items
type SpaceSaving<T, const K: usize> = {
    counters: [(T, u64); K],
    min_count: u64,
}
```

### 3.2 Possibility Queries (◊)

The `◊` operator extracts approximate answers:

```sigil
use sketch·{HyperLogLog, CountMinSketch, BloomFilter}

fn analyze_stream(events: Stream<Event~>) {
    // Build sketches from stream
    let hll = HyperLogLog::<14>·new()
    let cms = CountMinSketch::<1024, 4>·new()
    let bloom = BloomFilter::<10000, 7>·new()

    events|for_each{event =>
        hll|insert(event.user_id)
        cms|insert(event.item_id)
        bloom|insert(event.session_id)
    }

    // ◊ queries return uncertain results
    let unique_users◊ = hll|◊count        // ~1-2% error
    let item_freq◊ = cms|◊frequency(42)   // Upper bound estimate
    let maybe_seen◊ = bloom|◊contains(session)  // False positives possible

    // Uncertainty is explicit in the type
    print!("Approximately {unique_users◊} unique users")

    // Can bound the uncertainty
    let (low, high) = hll|◊count_bounds(confidence: 0.95)
}
```

### 3.3 Sketch Composition

Sketches are inherently holographic—substreams sketch the whole:

```sigil
/// Merge sketches from distributed nodes
fn global_cardinality(
    node_sketches: [HyperLogLog<14>~]
) -> u64◊ {
    // HLL union is a simple max over registers
    let merged = node_sketches
        |ρ{HyperLogLog·union}  // Reduce with union

    merged|◊count
}

/// Intersection cardinality (inclusion-exclusion)
fn overlap_estimate(
    a: HyperLogLog<14>,
    b: HyperLogLog<14>,
) -> u64◊ {
    let union◊ = (a ⊛ b)|◊count
    let a_count◊ = a|◊count
    let b_count◊ = b|◊count

    // |A ∩ B| ≈ |A| + |B| - |A ∪ B|
    // Uncertainty compounds
    (a_count◊ + b_count◊ - union◊)◊
}
```

---

## 4. Content-Addressed Structures (Merkle)

### 4.1 Merkle Types

Every node cryptographically commits to its entire subtree:

```sigil
/// Content identifier (hash-based address)
type Cid = {
    codec: Multicodec,
    hash: Multihash,
}

/// A Merkle node - contains proof of subtree
type MerkleNode<T> = {
    content: T,
    cid: Cid!,              // Computed locally = known
    links: [Cid~],          // External references = reported
}

/// Merkle DAG (directed acyclic graph)
type MerkleDag<T> = {
    root: Cid,
    store: ContentStore<T>,
}

/// IPLD-style flexible nodes
enum IpldNode {
    Null,
    Bool(bool),
    Integer(i128),
    Float(f64),
    String(str),
    Bytes([u8]),
    List([IpldNode]),
    Map(HashMap<str, IpldNode>),
    Link(Cid),
}
```

### 4.2 Necessity Verification (□)

The `□` operator verifies and promotes evidence:

```sigil
use merkle·{MerkleNode, verify, resolve}

/// Fetch and verify a node
async fn fetch_verified<T>(cid: Cid~) -> MerkleNode<T>! {
    // Fetch from untrusted source
    let node~ = ipfs·get(cid~)|await

    // □ verifies hash and promotes evidence
    // If hash matches, we KNOW the content is what was claimed
    let verified! = node~|□verify

    verified!
}

/// Recursive verification of entire subtree
async fn verify_tree<T>(root: Cid~) -> MerkleDag<T>! {
    let dag~ = fetch_dag(root~)|await

    // □* verifies entire tree recursively
    dag~|□*verify
}

// Verification rules:
// node~|□verify  →  node!  (if hash matches)
// node~|□verify  →  Error  (if hash mismatch)
// node?|□verify  →  node?  (uncertainty persists even if hash ok)
```

### 4.3 Merkle Proofs

Prove membership without revealing the whole:

```sigil
/// Inclusion proof for a leaf
struct MerkleProof<T> {
    leaf: T,
    path: [(Hash256, Direction)],
    root: Hash256,
}

enum Direction { Left, Right }

/// Create proof that element exists in tree
fn prove_inclusion<T>(tree: MerkleTree<T>, index: usize) -> MerkleProof<T> {
    let leaf = tree.leaves[index]
    let path = tree|path_to_root(index)
    MerkleProof { leaf, path, root: tree.root }
}

/// Verify proof without full tree
fn verify_proof<T>(proof: MerkleProof<T>) -> bool! {
    let computed! = proof.path
        |fold(proof.leaf|hash, {acc, (sibling, dir) =>
            match dir {
                Left => hash(sibling ++ acc),
                Right => hash(acc ++ sibling),
            }
        })

    computed! == proof.root
}
```

---

## 5. Superposition Types

### 5.1 Quantum-Inspired Semantics

Values exist in multiple states until observation collapses them:

```sigil
/// A value in superposition
type Superposition<T> = {
    states: [(T, Amplitude)],
    collapsed: Option<T!>,
}

/// Complex amplitude (probability = |amplitude|²)
type Amplitude = Complex<f64>

/// Create superposition
fn superpose<T>(states: [(T, Amplitude)]) -> Superposition<T> {
    // Normalize amplitudes
    let norm = states|τ{(_, a) => a.norm_sq()}|Σ|sqrt
    let normalized = states|τ{(t, a) => (t, a / norm)}

    Superposition {
        states: normalized,
        collapsed: None,
    }
}

/// Equal superposition (like |+⟩ state)
fn uniform<T>(options: [T]) -> Superposition<T> {
    let amp = Complex(1.0 / (options.len as f64).sqrt(), 0.0)
    options|τ{t => (t, amp)}|superpose
}
```

### 5.2 Observation and Collapse

```sigil
/// Observe collapses superposition to definite state
fn observe<T>(sup: &mut Superposition<T>) -> T! {
    match sup.collapsed {
        Some(value!) => value!,
        None => {
            // Probabilistic collapse based on |amplitude|²
            let probabilities = sup.states
                |τ{(_, a) => a.norm_sq()}

            let chosen! = sup.states
                |weighted_sample(probabilities)
                |clone

            sup.collapsed = Some(chosen!.clone())
            chosen!
        }
    }
}

/// Peek without collapsing (clone one possibility)
fn peek<T: Clone>(sup: &Superposition<T>) -> T◊ {
    // ◊ indicates this is one possible outcome
    sup.states|random_sample|clone◊
}
```

### 5.3 Quantum Operations

```sigil
/// Apply operation to all branches (like quantum gate)
fn map_superposition<T, U, F>(
    sup: Superposition<T>,
    f: F,
) -> Superposition<U>
where F: Fn(T) -> U {
    Superposition {
        states: sup.states|τ{(t, a) => (f(t), a)},
        collapsed: None,
    }
}

/// Interference - combine superpositions
fn interfere<T: Eq + Hash>(
    a: Superposition<T>,
    b: Superposition<T>,
) -> Superposition<T> {
    let mut combined: HashMap<T, Amplitude> = HashMap·new()

    for (state, amp) in a.states ++ b.states {
        combined
            |entry(state)
            |or_insert(Complex·zero())
            |add_assign(amp)
    }

    // Amplitudes can cancel (destructive interference)!
    combined
        |into_iter
        |φ{(_, a) => a.norm_sq() > 1e-10}  // Filter near-zero
        |collect
        |superpose
}

/// Entangle two values
type Entangled<A, B> = Superposition<(A, B)>

fn entangle<A, B>(
    sup_a: Superposition<A>,
    sup_b: Superposition<B>,
) -> Entangled<A, B> {
    // Tensor product of state spaces
    let states = sup_a.states
        |flat_map{(a, amp_a) =>
            sup_b.states|τ{(b, amp_b) =>
                ((a.clone(), b), amp_a * amp_b)
            }
        }
        |collect

    superpose(states)
}
```

---

## 6. Cultural Foundations

### 6.1 Indra's Net (Buddhist/Hindu)

```sigil
/// Every node reflects all others
trait IndraJewel {
    fn reflect(&self, other: &Self) -> Reflection;
    fn facets(&self) -> impl Iterator<Reflection>;
}

/// Distributed hash table as Indra's Net
type IndraNet<K, V> = {
    local: HashMap<K, V>,
    peers: [PeerId],
    reflections: u8,  // Replication factor
}

impl<K: Hash, V: Clone> IndraNet<K, V> {
    /// Store value across the net
    fn enshrine(&mut self, key: K, value: V) {
        let responsible = key|hash|closest_peers(self.reflections)
        responsible|for_each{peer =>
            peer|replicate(key, value.clone())
        }
    }

    /// Retrieve from any reflection
    fn divine(&self, key: K) -> V~ {
        let peer = key|hash|any_peer
        peer|fetch(key)~
    }
}
```

### 6.2 Akashic Records (Theosophical)

```sigil
/// Append-only universal memory
type AkashicLog<T> = {
    entries: MerkleDag<LogEntry<T>>,
    root: Cid,
}

struct LogEntry<T> {
    timestamp: Instant,
    content: T,
    prev: Option<Cid>,
}

impl<T> AkashicLog<T> {
    /// Nothing is ever lost, only appended
    fn inscribe(&mut self, content: T) -> Cid! {
        let entry = LogEntry {
            timestamp: Instant·now(),
            content,
            prev: Some(self.root),
        }
        let cid! = entry|hash_to_cid
        self.entries|insert(cid!, entry)
        self.root = cid!
        cid!
    }

    /// Query across all time
    fn recall(&self, predicate: fn(&T) -> bool) -> impl Iterator<&T> {
        self.entries
            |traverse_from(self.root)
            |τ{entry => &entry.content}
            |φ{predicate}
    }
}
```

### 6.3 Dreamtime (Aboriginal Australian)

```sigil
/// All moments exist simultaneously
type Dreamtime<T> = {
    now: T,
    was: [T],      // Past states (ancestors)
    might: [T◊],   // Possible futures
}

impl<T: Clone> Dreamtime<T> {
    /// Walk into a past moment
    fn walk_back(&self, steps: usize) -> T? {
        self.was|get(steps)
    }

    /// See possible futures (probabilistic)
    fn dream_forward(&self) -> impl Iterator<T◊> {
        self.might|iter|cloned
    }

    /// The eternal now contains all
    fn eternal(&self) -> Superposition<T> {
        let all_moments = [self.now.clone()]
            ++ self.was.clone()
            ++ self.might|τ{t◊ => t◊|assume}

        uniform(all_moments)
    }
}
```

### 6.4 I Ching (Chinese)

```sigil
/// 64 hexagrams contain all transformations
type Hexagram = Cycle<64>

/// Six lines, each yin or yang, stable or changing
struct Line {
    polarity: Polarity,
    changing: bool,
}

enum Polarity { Yin, Yang }

/// The Book of Changes as state machine
type IChing = Superposition<Hexagram>

impl IChing {
    /// Cast the oracle
    fn divine() -> (Hexagram!, Option<Hexagram!>) {
        let lines = (0..6)|τ{_ => cast_line()}|collect
        let present = lines_to_hexagram(&lines)

        let changing: Vec<_> = lines
            |enumerate
            |φ{(_, line) => line.changing}
            |τ{(i, _) => i}
            |collect

        let future = if changing.is_empty() {
            None
        } else {
            Some(lines|flip_changing|lines_to_hexagram)
        };

        (present!, future)
    }

    /// All possible transformations from current state
    fn mutations(&self, current: Hexagram) -> [Hexagram] {
        (0..64)|τ{i => Hexagram(i)}
            |φ{h => can_transform(current, h)}
            |collect
    }
}

fn cast_line() -> Line {
    // Traditional yarrow stalk or coin method
    let value = random(0..16)
    match value {
        0 => Line { polarity: Yin, changing: true },   // Old Yin
        1..5 => Line { polarity: Yang, changing: false }, // Young Yang
        5..11 => Line { polarity: Yin, changing: false }, // Young Yin
        11..16 => Line { polarity: Yang, changing: true }, // Old Yang
    }
}
```

---

## 7. Integration with Sigil Systems

### 7.1 Holographic Actors

Each actor holds a partial view of shared state:

```sigil
actor HolographicReplica<T, const K: usize, const N: usize> {
    my_shards: Vec<Shard<T>>,
    peer_hints: HashMap<PeerId, [u8]>,  // Which shards peers have

    /// Receive a shard from peer
    on ReceiveShard(shard: Shard<T>~) {
        let verified! = shard~|□verify_checksum
        self.my_shards|push(verified!)

        // Check if we can reconstruct
        if self.my_shards.len >= K {
            let whole! = self.my_shards|∀
            self|broadcast(Reconstructed(whole!))
        }
    }

    /// Query: can we reconstruct?
    on CanReconstruct -> bool {
        self.my_shards.len >= K
    }

    /// Get current approximation
    on Approximate -> T◊ {
        if self.my_shards.len >= K {
            self.my_shards|∀|as_certain◊
        } else {
            self.my_shards|◊interpolate
        }
    }
}
```

### 7.2 Holographic Streams

```sigil
/// Stream that maintains running sketches
fn holographic_stream<T>(
    source: Stream<T~>,
) -> HoloStream<T> {
    let hll = HyperLogLog::<14>·new()
    let cms = CountMinSketch::<2048, 5>·new()
    let reservoir = ReservoirSample::<1000>·new()

    HoloStream {
        source,
        sketches: (hll, cms, reservoir),
    }
}

struct HoloStream<T> {
    source: Stream<T~>,
    sketches: (HyperLogLog<14>, CountMinSketch<2048, 5>, ReservoirSample<1000, T>),
}

impl<T> HoloStream<T> {
    /// Process element, update all sketches
    async fn next(&mut self) -> Option<T~> {
        match self.source|next|await {
            Some(item~) => {
                self.sketches.0|insert(&item~)
                self.sketches.1|insert(&item~)
                self.sketches.2|insert(item~.clone())
                Some(item~)
            }
            None => None,
        }
    }

    /// Query the holographic state
    fn cardinality(&self) -> u64◊ { self.sketches.0|◊count }
    fn frequency(&self, item: &T) -> u64◊ { self.sketches.1|◊frequency(item) }
    fn sample(&self) -> [&T]◊ { self.sketches.2|◊sample }
}
```

### 7.3 Evidentiality Transitions

```sigil
/// Complete evidence transition rules for holographic ops
trait HolographicEvidence {
    /// Universal reconstruction
    fn reconstruct_evidence(shards: [Evidence]) -> Evidence {
        let all_known = shards|all{e => e == Known}
        let any_reported = shards|any{e => e == Reported}

        match (all_known, any_reported) {
            (true, _) => Known,      // All shards known → result known
            (_, true) => Reported,   // Any reported → result reported
            _ => Uncertain,          // Mixed/uncertain → uncertain
        }
    }

    /// Possibility query
    fn sketch_evidence() -> Evidence {
        Uncertain  // Sketches are inherently approximate
    }

    /// Necessity verification
    fn verify_evidence(input: Evidence, hash_matches: bool) -> Evidence {
        match (input, hash_matches) {
            (_, false) => Paradox,   // Hash mismatch = contradiction
            (Reported, true) => Known,  // Verified reported → known
            (e, true) => e,          // Others unchanged
        }
    }

    /// Superposition observation
    fn observe_evidence() -> Evidence {
        Known  // Observation produces definite result
    }
}
```

---

## 8. Standard Library: `holograph`

```sigil
//! Holographic data structure primitives

pub mod erasure {
    pub type Hologram<T, K, N>;
    pub type Shard<T>;
    pub trait ErasureCode;
    pub struct ReedSolomon;
    pub struct Fountain;

    pub fn encode<T, K, N>(data: T) -> Hologram<T, K, N>;
    pub fn decode<T, K, N>(shards: [Shard<T>]) -> Result<T, ReconstructError>;
}

pub mod sketch {
    pub type HyperLogLog<P>;
    pub type CountMinSketch<W, D>;
    pub type BloomFilter<M, K>;
    pub type TDigest;
    pub type SpaceSaving<T, K>;
    pub type ReservoirSample<N, T>;

    pub trait Sketch {
        fn insert(&mut self, item: &impl Hash);
        fn merge(&mut self, other: &Self);
    }
}

pub mod merkle {
    pub type Cid;
    pub type MerkleNode<T>;
    pub type MerkleDag<T>;
    pub type MerkleTree<T>;
    pub type MerkleProof<T>;

    pub fn hash_to_cid<T: Serialize>(data: &T) -> Cid;
    pub fn verify<T>(node: MerkleNode<T>~) -> Result<MerkleNode<T>!, VerifyError>;
}

pub mod superposition {
    pub type Superposition<T>;
    pub type Amplitude;
    pub type Entangled<A, B>;

    pub fn superpose<T>(states: [(T, Amplitude)]) -> Superposition<T>;
    pub fn uniform<T>(options: [T]) -> Superposition<T>;
    pub fn observe<T>(sup: &mut Superposition<T>) -> T!;
    pub fn interfere<T>(a: Superposition<T>, b: Superposition<T>) -> Superposition<T>;
}

pub mod cultural {
    pub type IndraNet<K, V>;
    pub type AkashicLog<T>;
    pub type Dreamtime<T>;
    pub type Hexagram;
    pub type IChing;
}
```

---

## 9. Syntax Summary

```sigil
// === Operators ===

// Universal (whole from part)
let whole! = shards|∀              // Deterministic reconstruction
let maybe? = degraded_shards|∀?   // Uncertain reconstruction

// Possibility (approximate)
let count◊ = hll|◊count           // Approximate cardinality
let freq◊ = cms|◊frequency(x)     // Approximate frequency
let member◊ = bloom|◊contains(x)  // Possible membership

// Necessity (verified)
let proven! = node~|□verify       // Hash verification
let tree! = dag~|□*verify         // Recursive verification

// Convolution (merge)
let merged = shards_a ⊛ shards_b  // Combine shard sets
let union = sketch_a ⊛ sketch_b   // Merge sketches

// === Type Annotations ===

fn returns_approximate() -> u64◊   // Inherently uncertain
fn returns_maybe() -> T?           // Might fail
fn returns_external() -> T~        // From untrusted source
fn returns_known() -> T!           // Definitely known

// === Superposition ===

let sup = [1, 2, 3]|uniform       // Equal superposition
let observed! = sup|observe        // Collapse to definite
let peeked◊ = sup|peek            // Sample without collapse
let mapped = sup|τ{_ * 2}         // Transform all branches
let combined = sup_a|interfere(sup_b)  // Quantum interference
```

---

## 10. Design Rationale

### Why These Symbols?

| Symbol | Logic Origin | Holographic Meaning |
|--------|--------------|---------------------|
| `∀` | Universal quantifier | "For all parts, the whole exists" |
| `◊` | Modal possibility | "This might be the answer" |
| `□` | Modal necessity | "This must be true (proven)" |
| `⊛` | Convolution operator | "Combine partial information" |

### Evidentiality Philosophy

The holographic system respects Sigil's core insight: **know what you know**.

- Reconstruction from local shards → Known (`!`)
- Reconstruction with external data → Reported (`~`)
- Approximate queries → Uncertain (`◊` implies `?`)
- Cryptographic verification → Promotes `~` to `!`
- Superposition observation → Produces definite `!`

This ensures that even exotic data structures maintain honest provenance tracking.
