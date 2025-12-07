# Quantum Computing Primitives

> *"God does not play dice with the universe." — Einstein*
> *"Stop telling God what to do." — Bohr*

## 1. Overview

Sigil provides first-class quantum computing primitives that:

1. **Type-encode quantum constraints** — No-cloning, linearity, measurement collapse
2. **Compose like classical pipelines** — Gates chain with `|`
3. **Integrate with evidentiality** — Measurement produces `!`, superposition is `◊`
4. **Support multiple backends** — Simulation, real hardware, tensor networks

### 1.1 Core Types

```sigil
/// A quantum bit - the fundamental unit
type Qubit = linear  // Cannot be cloned or dropped silently

/// Quantum state vector (2^n amplitudes for n qubits)
type StateVector<const N: usize> = [Complex; 2**N]

/// Density matrix (mixed states, decoherence)
type DensityMatrix<const N: usize> = [[Complex; 2**N]; 2**N]

/// A quantum register of N qubits
type QRegister<const N: usize> = [Qubit; N]

/// Classical bit (measurement result)
type Cbit = bool!  // Measurement is definite
```

### 1.2 Quantum Operators

| Symbol | Gate | Matrix | Effect |
|--------|------|--------|--------|
| `H` | Hadamard | `1/√2 [[1,1],[1,-1]]` | Creates superposition |
| `X` | Pauli-X | `[[0,1],[1,0]]` | Bit flip (NOT) |
| `Y` | Pauli-Y | `[[0,-i],[i,0]]` | Bit+phase flip |
| `Z` | Pauli-Z | `[[1,0],[0,-1]]` | Phase flip |
| `S` | Phase | `[[1,0],[0,i]]` | π/2 rotation |
| `T` | π/8 | `[[1,0],[0,e^(iπ/4)]]` | π/4 rotation |
| `⊗` | Tensor | Kronecker product | Combine systems |
| `†` | Adjoint | Conjugate transpose | Reverse gate |

---

## 2. Linear Types for Qubits

### 2.1 The No-Cloning Theorem

Quantum mechanics forbids copying arbitrary quantum states. Sigil enforces this at the type level:

```sigil
/// Qubit is a LINEAR type - must be used exactly once
type Qubit = linear {
    // Internal representation (not directly accessible)
    _state: *QuantumState,
}

// ❌ This won't compile - cloning is forbidden
fn try_clone(q: Qubit) -> (Qubit, Qubit) {
    (q, q)  // ERROR: Qubit used twice
}

// ❌ This won't compile - dropping is forbidden
fn try_drop(q: Qubit) {
    // ERROR: Qubit not used (implicit drop)
}

// ✅ Must consume the qubit
fn use_qubit(q: Qubit) -> Cbit {
    q|measure  // Consumes q, returns classical bit
}

// ✅ Or pass it on
fn transform_qubit(q: Qubit) -> Qubit {
    q|H|T  // Each gate consumes input, returns new qubit
}
```

### 2.2 Linearity Syntax

```sigil
/// Linear type declaration
type Linear<T> = linear T

/// Affine type (use at most once, can drop)
type Affine<T> = affine T

/// Relevant type (use at least once, can clone)
type Relevant<T> = relevant T

/// Normal type (use any number of times)
type Normal<T> = T  // Default

// Function that consumes linear argument
fn consume(q: linear Qubit) -> Cbit {
    q|measure
}

// Function that borrows without consuming
fn peek(q: &Qubit) -> Probability {
    q|probability_of(|1⟩)  // Read-only, doesn't collapse
}
```

---

## 3. Quantum Gates

### 3.1 Single-Qubit Gates

```sigil
use quantum·gate·{H, X, Y, Z, S, T, Rx, Ry, Rz, Phase}

fn single_qubit_ops(q: Qubit) -> Qubit {
    q
    |H              // Hadamard: |0⟩ → |+⟩, |1⟩ → |-⟩
    |X              // Pauli-X: |0⟩ ↔ |1⟩
    |Y              // Pauli-Y: |0⟩ → i|1⟩, |1⟩ → -i|0⟩
    |Z              // Pauli-Z: |1⟩ → -|1⟩
    |S              // Phase: |1⟩ → i|1⟩
    |T              // T-gate: |1⟩ → e^(iπ/4)|1⟩
    |Rx(θ)          // Rotation around X-axis
    |Ry(θ)          // Rotation around Y-axis
    |Rz(θ)          // Rotation around Z-axis
    |Phase(φ)       // Global phase
}

/// Gate composition
fn compose_gates() {
    // S = T|T (two T gates = one S gate)
    let S_equiv = T · T

    // Z = S|S
    let Z_equiv = S · S

    // X = H|Z|H
    let X_equiv = H · Z · H

    // Gates are unitary: U|U† = I
    let identity = H · H†  // H is self-adjoint, so H† = H
}
```

### 3.2 Multi-Qubit Gates

```sigil
use quantum·gate·{CNOT, CZ, SWAP, Toffoli, Fredkin}

/// Controlled-NOT (entangling gate)
fn cnot_example(control: Qubit, target: Qubit) -> (Qubit, Qubit) {
    // CNOT flips target if control is |1⟩
    (control, target)|CNOT
}

/// Create Bell state (maximally entangled)
fn bell_state(q0: Qubit, q1: Qubit) -> Entangled<Qubit, Qubit> {
    let q0 = q0|H           // |0⟩ → |+⟩
    let (q0, q1) = (q0, q1)|CNOT  // Creates |00⟩ + |11⟩

    // Return as entangled pair (linear, cannot separate)
    Entangled(q0, q1)
}

/// Controlled-Z
fn cz_example(a: Qubit, b: Qubit) -> (Qubit, Qubit) {
    // CZ: adds -1 phase to |11⟩
    (a, b)|CZ
}

/// SWAP gate
fn swap_example(a: Qubit, b: Qubit) -> (Qubit, Qubit) {
    // Exchanges quantum states
    (a, b)|SWAP
}

/// Toffoli (CCNOT) - universal for classical computation
fn toffoli_example(c1: Qubit, c2: Qubit, target: Qubit) -> (Qubit, Qubit, Qubit) {
    // Flips target only if both controls are |1⟩
    (c1, c2, target)|Toffoli
}

/// Fredkin (CSWAP) - controlled swap
fn fredkin_example(control: Qubit, a: Qubit, b: Qubit) -> (Qubit, Qubit, Qubit) {
    (control, a, b)|Fredkin
}
```

### 3.3 Parameterized Gates

```sigil
use quantum·gate·{Rx, Ry, Rz, U3, CRx, CRy, CRz}
use math·π

/// Rotation gates
fn rotations(q: Qubit, θ: f64) -> Qubit {
    q
    |Rx(θ)           // e^(-iθX/2)
    |Ry(θ)           // e^(-iθY/2)
    |Rz(θ)           // e^(-iθZ/2)
}

/// Universal single-qubit gate
fn universal_gate(q: Qubit, θ: f64, φ: f64, λ: f64) -> Qubit {
    // U3(θ, φ, λ) can represent any single-qubit unitary
    q|U3(θ, φ, λ)
}

/// Controlled rotations
fn controlled_rotation(control: Qubit, target: Qubit, θ: f64) -> (Qubit, Qubit) {
    (control, target)|CRz(θ)
}

/// Build gates from parameters
fn variational_layer(qubits: [Qubit; N], params: [f64; N * 3]) -> [Qubit; N] {
    qubits
        |enumerate
        |τ{(i, q) =>
            q|Rx(params[i * 3])
             |Ry(params[i * 3 + 1])
             |Rz(params[i * 3 + 2])
        }
}
```

---

## 4. Entanglement

### 4.1 Entangled Types

```sigil
/// Entangled pair - cannot be separated without measurement
type Entangled<A, B> = linear {
    _pair: (A, B),
    _correlation: EntanglementType,
}

enum EntanglementType {
    Bell(BellState),      // Maximally entangled
    GHZ,                  // Greenberger–Horne–Zeilinger
    W,                    // W state
    Cluster,              // Cluster state
    Partial(f64),         // Partially entangled
}

enum BellState {
    PhiPlus,   // |Φ+⟩ = (|00⟩ + |11⟩) / √2
    PhiMinus,  // |Φ-⟩ = (|00⟩ - |11⟩) / √2
    PsiPlus,   // |Ψ+⟩ = (|01⟩ + |10⟩) / √2
    PsiMinus,  // |Ψ-⟩ = (|01⟩ - |10⟩) / √2
}
```

### 4.2 Creating Entanglement

```sigil
/// Create all four Bell states
fn bell_states(q0: Qubit, q1: Qubit, which: BellState) -> Entangled<Qubit, Qubit> {
    let (q0, q1) = match which {
        PhiPlus => {
            let q0 = q0|H
            (q0, q1)|CNOT
        }
        PhiMinus => {
            let q0 = q0|X|H
            (q0, q1)|CNOT
        }
        PsiPlus => {
            let q0 = q0|H
            let (q0, q1) = (q0, q1)|CNOT
            (q0, q1|X)
        }
        PsiMinus => {
            let q0 = q0|X|H
            let (q0, q1) = (q0, q1)|CNOT
            (q0, q1|X)
        }
    }

    Entangled::new(q0, q1, Bell(which))
}

/// GHZ state: (|000...0⟩ + |111...1⟩) / √2
fn ghz_state<const N: usize>(qubits: [Qubit; N]) -> Entangled<[Qubit; N]> {
    let first = qubits[0]|H
    let rest = qubits[1..]

    // CNOT cascade
    let all = [first] ++ rest
        |windows(2)
        |fold(all[0], {acc, pair => (acc, pair[1])|CNOT})

    Entangled::ghz(all)
}

/// W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
fn w_state<const N: usize>(qubits: [Qubit; N]) -> Entangled<[Qubit; N]> {
    // More complex construction...
    quantum·prepare·w_state(qubits)
}
```

### 4.3 Entanglement Constraints

```sigil
/// Entangled qubits MUST be measured together or disentangled
fn use_entangled(pair: Entangled<Qubit, Qubit>) -> (Cbit, Cbit) {
    // ✅ Measure both
    pair|measure_both
}

// ❌ Cannot extract one qubit from entangled pair
fn try_extract(pair: Entangled<Qubit, Qubit>) -> Qubit {
    pair.0  // ERROR: Cannot destructure Entangled
}

/// Must explicitly disentangle (if possible)
fn disentangle(pair: Entangled<Qubit, Qubit>) -> Option<(Qubit, Qubit)> {
    // Only possible for product states (not truly entangled)
    pair|try_separate
}
```

---

## 5. Measurement

### 5.1 Measurement Operations

```sigil
use quantum·measure·{measure, measure_basis, measure_partial}

/// Standard Z-basis measurement
fn measure_qubit(q: Qubit) -> Cbit! {
    // Consumes qubit, returns definite classical bit
    // Evidence: measurement produces KNOWN result
    q|measure
}

/// Measurement in arbitrary basis
fn measure_in_basis(q: Qubit, basis: Basis) -> Cbit! {
    q|measure_basis(basis)
}

enum Basis {
    Z,              // Computational: |0⟩, |1⟩
    X,              // Hadamard: |+⟩, |-⟩
    Y,              // Circular: |i⟩, |-i⟩
    Custom(Unitary), // Any orthonormal basis
}

/// Partial measurement of register
fn measure_some<const N: usize, const M: usize>(
    reg: QRegister<N>,
    which: [usize; M],
) -> ([Cbit; M]!, QRegister<{N - M}>) {
    // Measures M qubits, collapses rest to consistent state
    reg|measure_partial(which)
}
```

### 5.2 Measurement and Evidentiality

```sigil
/// Measurement transitions in evidentiality lattice
///
/// Before measurement:
///   Superposition<T> has evidence ◊ (possibility)
///
/// After measurement:
///   Result has evidence ! (known)
///   This is the ONLY way to go from ◊ to !

fn evidence_example() {
    let q: Qubit = init_qubit(|0⟩)

    // After Hadamard, qubit is in superposition
    let q_super: Qubit◊ = q|H

    // Probability queries return uncertain results
    let prob◊: f64◊ = q_super|probability_of(|1⟩)

    // Measurement collapses to definite result
    let result!: Cbit! = q_super|measure

    // Result is KNOWN (not uncertain, not reported)
    assert!(result!.evidence == Known)
}

/// Cannot "peek" at superposition without collapsing
fn no_peeking(q: Qubit) -> Qubit {
    // ❌ This is NOT allowed
    let value = q|secretly_observe  // ERROR: No such operation

    // ✅ Can only get probabilities (which are ◊)
    let prob◊ = q|probability_of(|1⟩)

    q
}
```

### 5.3 Deferred Measurement

```sigil
/// Principle: measurements can be deferred to end of circuit
/// (but entanglement with measured qubit affects results)

type DeferredMeasurement = {
    qubit_id: QubitId,
    basis: Basis,
    result: Cell<Option<Cbit>>,
}

fn deferred_example() -> Circuit {
    circuit! {
        let q0 = qubit()
        let q1 = qubit()

        q0|H
        (q0, q1)|CNOT

        // Mark for measurement but don't collapse yet
        let m0 = q0|defer_measure
        let m1 = q1|defer_measure

        // Classical control based on deferred measurement
        if m0 {
            q1|X  // Will be applied conditionally at runtime
        }

        // Actual measurement happens when circuit executes
        (m0, m1)
    }
}
```

---

## 6. Quantum Circuits

### 6.1 Circuit DSL

```sigil
use quantum·circuit·{Circuit, circuit, qubit, cbit}

/// Build circuit with macro
fn build_circuit() -> Circuit {
    circuit! {
        // Allocate quantum resources
        let q0 = qubit()
        let q1 = qubit()
        let q2 = qubit()

        // Apply gates
        q0|H
        (q0, q1)|CNOT
        q1|T
        (q1, q2)|CNOT
        q2|measure -> c0  // Measure into classical bit

        // Classical control
        if c0 {
            q0|Z
        }

        // Return remaining qubits
        (q0, q1)
    }
}
```

### 6.2 Circuit Composition

```sigil
/// Circuits compose horizontally (sequence)
fn sequence(a: Circuit, b: Circuit) -> Circuit {
    a · b  // a then b
}

/// Circuits compose vertically (tensor product)
fn parallel(a: Circuit, b: Circuit) -> Circuit {
    a ⊗ b  // a and b on separate qubits
}

/// Controlled version of any circuit
fn controlled(circuit: Circuit) -> Circuit {
    circuit|control(1)  // Add one control qubit
}

/// Inverse of reversible circuit
fn inverse(circuit: Circuit) -> Circuit {
    circuit†  // Adjoint (reverse all gates)
}

/// Example: Quantum Fourier Transform
fn qft<const N: usize>() -> Circuit {
    circuit! {
        let qubits = qubits(N)

        for i in 0..N {
            qubits[i]|H
            for j in (i+1)..N {
                let k = j - i + 1
                (qubits[j], qubits[i])|CRz(π / 2.pow(k))
            }
        }

        // Reverse qubit order
        for i in 0..(N/2) {
            (qubits[i], qubits[N-1-i])|SWAP
        }

        qubits
    }
}
```

### 6.3 Circuit Optimization

```sigil
use quantum·optimize·{simplify, decompose, transpile}

fn optimize_circuit(circuit: Circuit) -> Circuit {
    circuit
        |simplify           // Cancel adjacent inverse gates
        |decompose(native_gates)  // Break into hardware gates
        |transpile(topology)      // Route for hardware connectivity
}

/// Gate cancellation rules
const CANCEL_RULES: [Rule] = [
    // X|X = I
    (X, X) => I,
    // H|H = I
    (H, H) => I,
    // Z|Z = I
    (Z, Z) => I,
    // T|T|T|T|T|T|T|T = I
    (T, T, T, T, T, T, T, T) => I,
    // CNOT|CNOT = I
    (CNOT, CNOT) => I,
]

/// Decomposition into native gate set
fn decompose_to_native(gate: Gate, native: [Gate]) -> [Gate] {
    match (gate, native) {
        // Decompose Toffoli into Clifford+T
        (Toffoli, [H, CNOT, T, Tdg]) => [
            H(2), CNOT(1,2), Tdg(2), CNOT(0,2), T(2),
            CNOT(1,2), Tdg(2), CNOT(0,2), T(1), T(2), H(2),
            CNOT(0,1), T(0), Tdg(1), CNOT(0,1)
        ],
        // ... more decompositions
    }
}
```

---

## 7. Quantum Algorithms

### 7.1 Grover's Search

```sigil
use quantum·algorithm·grover

/// Search for marked item in unstructured database
fn grover_search<const N: usize>(
    oracle: fn(QRegister<N>) -> QRegister<N>,
) -> usize! {
    let iterations = (π / 4.0 * (2.pow(N) as f64).sqrt()) as usize

    circuit! {
        let qubits = qubits(N)|τ{H}  // Uniform superposition

        for _ in 0..iterations {
            qubits|oracle              // Mark solution
            qubits|grover·diffusion    // Amplify amplitude
        }

        qubits|measure_all
    }|execute|interpret_as_usize
}

/// Grover diffusion operator
fn diffusion<const N: usize>(qubits: QRegister<N>) -> QRegister<N> {
    qubits
        |τ{H}           // Apply H to all
        |τ{X}           // Apply X to all
        |multi_controlled_z  // Z if all |1⟩
        |τ{X}           // Apply X to all
        |τ{H}           // Apply H to all
}
```

### 7.2 Shor's Algorithm (Period Finding)

```sigil
use quantum·algorithm·shor

/// Factor integer N using quantum period finding
fn factor(N: u64) -> (u64, u64)? {
    // Classical preprocessing
    if N|is_even { return Some((2, N / 2)) }
    if let Some(k) = N|is_prime_power { return Some(k) }

    // Quantum period finding
    loop {
        let a = random(2..N)
        if gcd(a, N) > 1 { return Some((gcd(a, N), N / gcd(a, N))) }

        let r = quantum_period_find(a, N)?

        if r|is_odd { continue }

        let x = a.pow(r / 2)
        if x % N == N - 1 { continue }

        let p = gcd(x - 1, N)
        let q = gcd(x + 1, N)

        if p * q == N { return Some((p, q)) }
    }
}

/// Quantum part: find period of f(x) = a^x mod N
fn quantum_period_find(a: u64, N: u64) -> usize? {
    let n = N|bit_length * 2  // Precision qubits

    circuit! {
        let input = qubits(n)|τ{H}   // Superposition of all x
        let output = qubits(n)

        // Modular exponentiation in superposition
        (input, output)|modular_exp(a, N)

        // QFT on input register
        input|qft_inverse

        // Measure and classically post-process
        let result = input|measure_all
        result|continued_fractions|find_period(N)
    }
}
```

### 7.3 Variational Quantum Eigensolver (VQE)

```sigil
use quantum·algorithm·vqe
use quantum·hamiltonian·Hamiltonian

/// Find ground state energy of Hamiltonian
fn vqe(
    hamiltonian: Hamiltonian,
    ansatz: fn([f64]) -> Circuit,
    optimizer: Optimizer,
) -> f64! {
    let n_params = ansatz·param_count

    // Classical-quantum hybrid loop
    optimizer·minimize(n_params, {params =>
        // Quantum: prepare state and measure expectation
        let circuit = ansatz(params)
        let energy◊ = circuit|execute|expectation(hamiltonian)

        // Return energy for classical optimizer
        energy◊|assume_for_optimization
    })
}

/// Hardware-efficient ansatz
fn hardware_efficient_ansatz<const N: usize>(params: [f64]) -> Circuit {
    let layers = params.len / (N * 3)

    circuit! {
        let qubits = qubits(N)

        for layer in 0..layers {
            // Single-qubit rotations
            for i in 0..N {
                let idx = layer * N * 3 + i * 3
                qubits[i]|Rx(params[idx])
                        |Ry(params[idx + 1])
                        |Rz(params[idx + 2])
            }

            // Entangling layer
            for i in 0..(N-1) {
                (qubits[i], qubits[i+1])|CNOT
            }
        }

        qubits
    }
}
```

---

## 8. Quantum Error Correction

### 8.1 Logical Qubits

```sigil
/// Logical qubit encoded in physical qubits
type LogicalQubit<Code> = {
    physical: [Qubit],
    code: Code,
}

/// Error correction codes
trait QEC {
    const PHYSICAL_QUBITS: usize;
    const LOGICAL_QUBITS: usize;
    const DISTANCE: usize;

    fn encode(logical: Qubit) -> [Qubit; Self::PHYSICAL_QUBITS];
    fn decode(physical: [Qubit; Self::PHYSICAL_QUBITS]) -> Qubit;
    fn detect_errors(physical: &[Qubit]) -> Syndrome;
    fn correct(physical: &mut [Qubit], syndrome: Syndrome);
}

/// 3-qubit bit-flip code
struct BitFlipCode;

impl QEC for BitFlipCode {
    const PHYSICAL_QUBITS: usize = 3;
    const LOGICAL_QUBITS: usize = 1;
    const DISTANCE: usize = 3;

    fn encode(q: Qubit) -> [Qubit; 3] {
        // |0⟩ → |000⟩, |1⟩ → |111⟩
        let (q, q1) = (q, fresh_qubit())|CNOT
        let (q, q2) = (q, fresh_qubit())|CNOT
        [q, q1, q2]
    }

    fn detect_errors(physical: &[Qubit; 3]) -> Syndrome {
        // Measure ZZI and IZZ stabilizers
        let s1 = measure_stabilizer(physical, ZZI)
        let s2 = measure_stabilizer(physical, IZZ)
        (s1, s2)
    }

    fn correct(physical: &mut [Qubit; 3], syndrome: Syndrome) {
        match syndrome {
            (0, 0) => {},           // No error
            (1, 0) => physical[0]|X, // Error on qubit 0
            (1, 1) => physical[1]|X, // Error on qubit 1
            (0, 1) => physical[2]|X, // Error on qubit 2
        }
    }
}
```

### 8.2 Surface Codes

```sigil
/// Topological surface code
struct SurfaceCode<const D: usize>;  // Distance D

impl<const D: usize> QEC for SurfaceCode<D> {
    const PHYSICAL_QUBITS: usize = D * D + (D-1) * (D-1);
    const LOGICAL_QUBITS: usize = 1;
    const DISTANCE: usize = D;

    // Surface code implementation...
}

/// Lattice surgery for logical operations
fn logical_cnot(
    control: LogicalQubit<SurfaceCode<D>>,
    target: LogicalQubit<SurfaceCode<D>>,
) -> (LogicalQubit<SurfaceCode<D>>, LogicalQubit<SurfaceCode<D>>) {
    // Merge and split boundaries
    lattice_surgery·cnot(control, target)
}
```

---

## 9. Execution Backends

### 9.1 Backend Trait

```sigil
trait QuantumBackend {
    /// Execute circuit and return measurement results
    fn execute(&self, circuit: Circuit, shots: usize) -> [Measurement]~;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    /// Native gate set
    fn native_gates(&self) -> [Gate];

    /// Qubit connectivity
    fn topology(&self) -> Graph<QubitId>;
}

struct BackendCapabilities {
    max_qubits: usize,
    max_depth: usize,
    gate_fidelities: HashMap<Gate, f64>,
    t1_times: [Duration],  // Relaxation
    t2_times: [Duration],  // Dephasing
}
```

### 9.2 Simulators

```sigil
/// State vector simulator (exact, exponential memory)
struct StateVectorSim {
    max_qubits: usize,  // ~30-40 on classical hardware
}

impl QuantumBackend for StateVectorSim {
    fn execute(&self, circuit: Circuit, shots: usize) -> [Measurement]~ {
        let state = circuit|simulate_statevector
        (0..shots)|τ{_ => state|sample}|collect
    }
}

/// Tensor network simulator (approximate, for structured circuits)
struct TensorNetworkSim {
    bond_dimension: usize,
    truncation_threshold: f64,
}

/// Clifford simulator (efficient for stabilizer circuits)
struct CliffordSim;  // Polynomial time for Clifford gates only
```

### 9.3 Real Hardware

```sigil
/// IBM Quantum backend
struct IBMQuantum {
    api_key: str~,
    device: str,
}

impl QuantumBackend for IBMQuantum {
    fn execute(&self, circuit: Circuit, shots: usize) -> [Measurement]~ {
        // Transpile for hardware
        let transpiled = circuit
            |transpile(self.topology, self.native_gates)

        // Submit job
        let job_id~ = self.api|submit(transpiled, shots)|await

        // Wait for results
        self.api|wait_for_result(job_id~)|await
    }
}

/// IonQ trapped ion backend
struct IonQ {
    api_key: str~,
}

/// Rigetti Aspen backend
struct Rigetti {
    api_key: str~,
    device: str,
}
```

---

## 10. Integration with Sigil

### 10.1 Quantum Types in Sigil

```sigil
/// Quantum computations return possibility evidence
fn quantum_evidence() {
    let q = qubit(|0⟩)
    let q = q|H  // Now in superposition

    // Type: Qubit with implicit ◊ evidence
    // Cannot extract classical value without measurement

    let result!: Cbit! = q|measure  // Collapse: ◊ → !
}

/// Entangled values are linear
fn linear_example() {
    let pair: Entangled<Qubit, Qubit> = bell_state()

    // Must use both together
    let (a!, b!): (Cbit, Cbit) = pair|measure_both

    // Classical correlation is now known
    assert!(a! == b! || a! != b!)  // Depends on Bell state type
}
```

### 10.2 Quantum Actors

```sigil
/// Actor that manages quantum resources
actor QuantumProcessor {
    backend: Box<dyn QuantumBackend>,
    job_queue: Vec<PendingJob>,

    on Execute(circuit: Circuit, shots: usize) -> JobId {
        let job_id = JobId·new()
        self.job_queue|push(PendingJob { id: job_id, circuit, shots })
        job_id
    }

    on GetResult(job_id: JobId) -> Option<[Measurement]~> {
        self.completed|get(job_id)
    }

    // Background task: process job queue
    async fn process_queue(&mut self) {
        loop {
            if let Some(job) = self.job_queue|pop {
                let result~ = self.backend|execute(job.circuit, job.shots)|await
                self.completed|insert(job.id, result~)
            }
            yield  // Allow other tasks
        }
    }
}
```

### 10.3 Quantum Streams

```sigil
/// Stream of quantum measurement results
fn measurement_stream(
    circuit: Circuit,
    shots: usize,
) -> Stream<Measurement!> {
    stream·generate(async {
        let backend = get_backend()
        for _ in 0..shots {
            let result~ = backend|execute(circuit, 1)|await
            yield result~[0]|verify!
        }
    })
}

/// Aggregate quantum results with sketches
fn quantum_statistics(results: Stream<Measurement!>) -> Statistics◊ {
    let hll = HyperLogLog·new()
    let cms = CountMinSketch·new()

    results|for_each{m! =>
        hll|insert(m!|as_bitstring)
        cms|insert(m!|as_usize)
    }

    Statistics {
        unique_outcomes◊: hll|◊count,
        frequencies◊: cms,
    }
}
```

---

## 11. Standard Library: `quantum`

```sigil
//! Quantum computing primitives

pub mod types {
    pub type Qubit;
    pub type Cbit;
    pub type QRegister<N>;
    pub type Entangled<A, B>;
    pub type StateVector<N>;
    pub type DensityMatrix<N>;
}

pub mod gate {
    // Single-qubit
    pub fn H, X, Y, Z, S, T, Sdg, Tdg;
    pub fn Rx, Ry, Rz, Phase, U3;

    // Multi-qubit
    pub fn CNOT, CZ, CY, SWAP, iSWAP;
    pub fn Toffoli, Fredkin;
    pub fn CRx, CRy, CRz, CU3;

    // Composition
    pub fn control, tensor, adjoint;
}

pub mod measure {
    pub fn measure, measure_basis, measure_partial;
    pub fn probability_of, expectation;
}

pub mod circuit {
    pub type Circuit;
    pub macro circuit!;
    pub fn compose, parallel, inverse;
    pub fn optimize, transpile;
}

pub mod algorithm {
    pub mod grover;
    pub mod shor;
    pub mod vqe;
    pub mod qaoa;
    pub mod qft;
}

pub mod error_correction {
    pub trait QEC;
    pub struct BitFlipCode;
    pub struct PhaseFlipCode;
    pub struct ShorCode;
    pub struct SteaneCode;
    pub struct SurfaceCode<D>;
}

pub mod backend {
    pub trait QuantumBackend;
    pub struct StateVectorSim;
    pub struct TensorNetworkSim;
    pub struct CliffordSim;
}
```
