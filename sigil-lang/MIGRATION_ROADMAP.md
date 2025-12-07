# Sigil Migration Roadmap

## Overview

This roadmap outlines the phased migration of persona-framework components to Sigil, prioritized by impact, complexity, and feature demonstration value.

**Total Identified Candidates:** 90+ components  
**Estimated Total Effort:** 6-9 months  
**Current Sigil Status:** Native Rust performance (1:1 parity achieved)

---

## Phase 1: Foundation & Quick Wins (Weeks 1-4)

### Goals
- Validate Sigil in production-like scenarios
- Build migration patterns and best practices
- Demonstrate all core Sigil features

### 1.1 AI-Worker Migration (Week 1-2)
**Source:** `/ai-worker/` (Python, 500 LOC)  
**Effort:** 3-5 days  
**Priority:** ★★★★★

```sigil
// Target architecture
actor SDXLWorker {
    queue: RequestQueue
    
    on Generate(req: GenerationRequest~) -> Image! {
        req |φ{validate} |τ{prepare} |τ{infer}~ |τ{upload}!
    }
}
```

**Demonstrates:**
- Actor-based concurrency
- Pipe chain processing
- Evidentiality tracking (request~ → output!)
- Async/await patterns

**Deliverables:**
- [ ] Port request validation logic
- [ ] Implement inference pipeline as pipe chain
- [ ] Add S3 upload with confirmation tracking
- [ ] Benchmark against Python baseline

### 1.2 Shell Script Migration (Week 2-3)
**Source:** `/scripts/*.sh` (12 scripts)  
**Effort:** 2-4 days  
**Priority:** ★★★★☆

**Target Scripts:**
| Script | Purpose | Sigil Benefit |
|--------|---------|---------------|
| `smoke.sh` | Integration testing | Error propagation via evidentiality |
| `submit-task-to-hydra.sh` | Task submission | Pipeline composition |
| `refresh-aws-credentials.sh` | Credential mgmt | Trust boundary tracking |

**Deliverables:**
- [ ] Create `sigil script` runner mode
- [ ] Port 3 representative scripts
- [ ] Document shell→Sigil patterns

### 1.3 Orpheus Frequency Analyzer (Week 3-4)
**Source:** `/orpheus/packages/audio-analysis/` (TypeScript)  
**Effort:** 5-8 days  
**Priority:** ★★★★☆

```sigil
fn analyze_spectrum(samples: [f64]) -> Spectrum {
    samples
        |τ{apply_window(_, HannWindow)}
        |τ{fft}
        |τ{magnitude_db}
        |φ{freq > 20.0 && freq < 20000.0}
        |σ.magnitude
}
```

**Demonstrates:**
- DSP pipeline patterns
- Performance-critical code (JIT benefits)
- Mathematical operations

**Deliverables:**
- [ ] Port FFT wrapper
- [ ] Implement windowing functions
- [ ] Benchmark against TypeScript (target: 10x speedup)

---

## Phase 2: Crypto & Security (Weeks 5-8)

### Goals
- Prove Sigil for security-critical code
- Establish crypto primitive patterns
- Validate evidentiality for trust tracking

### 2.1 Arcanum Cipher Pipelines (Week 5-6)
**Source:** `/arcanum/` (Rust, 20K LOC)  
**Effort:** 10-15 days  
**Priority:** ★★★★☆

**Target Modules:**
- `arcanum-symmetric` → AES/ChaCha20 pipelines
- `arcanum-hash` → SHA/Blake3 streaming
- `arcanum-kdf` → Key derivation chains

```sigil
fn encrypt_authenticated(
    plaintext: [u8]!,      // Known input
    key: Key!,             // Known key
    nonce: Nonce!
) -> Ciphertext! {
    plaintext
        |τ{pad_pkcs7}
        |τ{aes_encrypt(_, key, nonce)}
        |τ{compute_mac(_, key)}
        |τ{append_mac}!     // Confirmed output
}
```

**Deliverables:**
- [ ] Port symmetric encryption suite
- [ ] Implement streaming hash interface
- [ ] Add constant-time operation markers
- [ ] Security audit of generated code

### 2.2 Sitra Trust Boundaries (Week 7-8)
**Source:** `/sitra/` (Rust, 12K LOC)  
**Effort:** 12-18 days  
**Priority:** ★★★★★

**Evidentiality Mapping:**
| Data Source | Marker | Meaning |
|-------------|--------|---------|
| User keys | `!` | Known, trusted |
| Tor circuit | `~` | Reported by network |
| External nodes | `?` | Uncertain provenance |
| Validated data | `!` | Confirmed after verification |

```sigil
fn validate_tor_response(
    response: TorResponse~,  // Reported from Tor
    circuit: Circuit!        // Our known circuit
) -> ValidatedData! {
    response
        |φ{verify_signature(_, circuit)}?  // Uncertain until verified
        |τ{decrypt(_, circuit.key)}
        |φ{check_integrity}!               // Now confirmed
}
```

**Deliverables:**
- [ ] Port Tor client wrapper
- [ ] Implement trust propagation rules
- [ ] Add circuit validation logic
- [ ] Document evidentiality patterns for crypto

---

## Phase 3: Data Pipelines & Streaming (Weeks 9-14)

### Goals
- Migrate high-throughput data processing
- Prove Sigil for real-time systems
- Establish Kafka/streaming patterns

### 3.1 Infernum Token Streaming (Week 9-11)
**Source:** `/infernum/crates/abaddon/` (Rust)  
**Effort:** 15-20 days  
**Priority:** ★★★★★

```sigil
actor InferenceEngine {
    model: LoadedModel
    
    on Generate(prompt: String!, params: SamplingParams) -> TokenStream~ {
        prompt
            |τ{tokenize}
            |τ{embed(_, self.model)}
            |stream{generate_tokens(_, params)}~  // Streaming output
            |τ{decode_token}
    }
}
```

**Deliverables:**
- [ ] Port tokenization pipeline
- [ ] Implement streaming token generation
- [ ] Add batching support
- [ ] Benchmark latency (target: <5ms first token)

### 3.2 Moloch Event Producer (Week 12-14)
**Source:** `/moloch/crates/moloch-kafka/` (Rust)  
**Effort:** 12-15 days  
**Priority:** ★★★★☆

```sigil
actor BlockchainEventProducer {
    kafka: KafkaProducer
    
    on BlockValidated(block: Block!) {
        block
            |τ{serialize_event}
            |τ{add_metadata(_, now())}
            |τ{publish(_, self.kafka, "blocks")}!
    }
    
    on TransactionReceived(tx: Transaction~) {
        tx
            |φ{validate_signature}?     // Uncertain until validated
            |τ{validate_nonce}
            |τ{publish(_, self.kafka, "pending-txs")}~
    }
}
```

**Deliverables:**
- [ ] Port Kafka producer wrapper
- [ ] Implement event serialization
- [ ] Add transaction validation pipeline
- [ ] Benchmark throughput (target: 100K events/sec)

---

## Phase 4: Application Layer (Weeks 15-20)

### Goals
- Migrate user-facing business logic
- Prove Sigil for complex domain models
- Establish frontend-backend integration patterns

### 4.1 Farming-App Recommendation Engine (Week 15-17)
**Source:** `/farming-app/src/` (React/TypeScript)  
**Effort:** 8-12 days  
**Priority:** ★★★☆☆

```sigil
fn recommend_companions(
    plant: Plant!,
    soil: SoilData~,        // Sensor data (reported)
    history: GrowthHistory~  // Historical (reported)
) -> Vec<Recommendation>~ {
    get_companion_candidates(plant)
        |φ{compatible_with_soil(_, soil)}
        |φ{no_historical_issues(_, history)}
        |τ{score_compatibility(_, plant)}
        |σ.score
        |take(5)
        |τ{build_recommendation}~  // Model output
}
```

**Deliverables:**
- [ ] Extract recommendation logic to Sigil service
- [ ] Implement companion plant database
- [ ] Add sensor data integration
- [ ] Create REST API bridge

### 4.2 Vulcan-App Quote Calculator (Week 18-20)
**Source:** `/vulcan-app/src/` (React/TypeScript)  
**Effort:** 10-15 days  
**Priority:** ★★★☆☆

```sigil
fn calculate_quote(items: Vec<QuoteItem>!, config: PricingConfig!) -> Quote! {
    items
        |τ{apply_base_pricing(_, config)}
        |τ{apply_quantity_discounts}
        |τ{apply_material_markups(_, config)}
        |τ{calculate_labor}
        |ρ{sum_totals}
        |τ{apply_tax(_, config.tax_rate)}
        |τ{build_quote}!
}
```

**Deliverables:**
- [ ] Port pricing calculation engine
- [ ] Implement discount rules
- [ ] Add batch operation support
- [ ] Benchmark calculation speed

---

## Phase 5: Core Infrastructure (Weeks 21-30)

### Goals
- Migrate critical backend systems
- Prove Sigil at scale
- Establish actor supervision patterns

### 5.1 Leviathan GraphQL Resolvers (Week 21-25)
**Source:** `/leviathan/` (Kotlin/Spring, 30K LOC)  
**Effort:** 15-20 days  
**Priority:** ★★★★☆

**Migration Strategy:** Incremental resolver replacement

```sigil
// GraphQL resolver as Sigil function
fn resolve_workspace(
    ctx: Context!,
    id: WorkspaceId!
) -> Workspace? {
    ctx.auth
        |φ{has_permission(_, "workspace:read")}
        |τ{fetch_workspace(_, id)}?
        |φ{user_has_access(_, ctx.user)}
        |τ{hydrate_relations}
}
```

**Deliverables:**
- [ ] Create Sigil-Spring bridge
- [ ] Port 10 representative resolvers
- [ ] Implement permission checking
- [ ] Benchmark query latency

### 5.2 Nyx Agent Framework (Week 26-30)
**Source:** `/nyx/agents/` (Rust, 50K LOC)  
**Effort:** 20-30 days  
**Priority:** ★★★★☆

```sigil
actor GuardianAgent {
    policies: Vec<Policy>
    resources: ResourcePool
    
    on PolicyCheck(request: AccessRequest~) -> Decision! {
        request
            |τ{extract_claims}~
            |φ{validate_token}?
            |τ{match_policies(_, self.policies)}
            |ρ{combine_decisions}!
    }
    
    on ResourceRequest(req: ResourceReq!) -> Resource? {
        req
            |φ{check_quota(_, self.resources)}
            |τ{allocate}?
            |τ{track_usage}
    }
}
```

**Deliverables:**
- [ ] Port agent base framework
- [ ] Implement policy engine
- [ ] Add resource management
- [ ] Create supervision hierarchy

---

## Phase 6: Advanced Systems (Weeks 31-40)

### 6.1 Aether-Engine Asset Pipeline (Week 31-35)
**Source:** `/aether-engine/engine/aether-asset/` (Rust)  
**Effort:** 20-25 days

```sigil
fn load_asset_pipeline(path: Path!) -> LoadedAsset? {
    path
        |τ{read_file}?
        |τ{detect_format}
        |τ{parse_asset}?
        |τ{validate_asset}
        |τ{preprocess}
        |τ{optimize_for_gpu}
        |τ{upload_to_vram}?
}
```

### 6.2 Moloch Consensus Engine (Week 36-40)
**Source:** `/moloch/crates/moloch-consensus/` (Rust)  
**Effort:** 25-30 days

```sigil
actor ConsensusEngine<C: Consensus> {
    state: ConsensusState
    validators: Vec<Validator>
    
    on ProposeBlock(txs: Vec<Transaction>~) -> Block? {
        txs
            |φ{validate_all}?
            |τ{order_transactions}
            |τ{execute_state_transition(_, self.state)}
            |τ{build_block}
            |τ{sign_block}!
    }
    
    on ReceiveVote(vote: Vote~) -> ConsensusResult? {
        vote
            |φ{verify_validator(_, self.validators)}~
            |τ{tally_vote(_, self.state)}
            |φ{check_threshold}
            |τ{finalize_if_ready}
    }
}
```

---

## Success Metrics

### Performance Targets
| Benchmark | Baseline | Target | Status |
|-----------|----------|--------|--------|
| fib(35) | Rust 27ms | 26ms | ✅ Achieved |
| primes(100K) | Rust 5ms | 5ms | ✅ Achieved |
| Token streaming | - | <5ms first token | Pending |
| Kafka throughput | - | 100K events/sec | Pending |

### Code Quality Targets
| Metric | Target |
|--------|--------|
| Lines of code reduction | 30-50% |
| Cyclomatic complexity | -40% |
| Test coverage | >80% |
| Evidentiality coverage | 100% on trust boundaries |

### Migration Progress
| Phase | Components | Status |
|-------|------------|--------|
| Phase 1 | 15 | Not Started |
| Phase 2 | 8 | Not Started |
| Phase 3 | 10 | Not Started |
| Phase 4 | 12 | Not Started |
| Phase 5 | 25 | Not Started |
| Phase 6 | 20 | Not Started |

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|------------|
| GPU code incompatibility | Keep GPU kernels in Rust, orchestrate from Sigil |
| Async runtime differences | Build tokio bridge layer |
| FFI overhead | Use AOT compilation, minimize crossings |
| Crypto timing attacks | Audit generated assembly, add constant-time primitives |

### Organizational Risks
| Risk | Mitigation |
|------|------------|
| Learning curve | Phase 1 builds expertise before critical systems |
| Parallel development | Feature flags for gradual rollout |
| Rollback capability | Keep Rust implementations during transition |

---

## Next Steps

1. **Immediate:** Begin ai-worker migration (Phase 1.1)
2. **Week 2:** Start shell script migration toolkit
3. **Week 3:** Begin Orpheus frequency analyzer
4. **Week 5:** Phase 2 kickoff with Arcanum

---

*Last Updated: 2025-12-01*  
*Sigil Version: 0.1.0 (LLVM backend, native performance)*
