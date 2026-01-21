# Aegis Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AEGIS SECURITY LAYER                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         POLICY ENGINE                                │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │  │CONSTITUTION│ │  ACCESS   │ │ SECURITY  │ │   ALERT   │           │    │
│  │  │  CHECKER  │ │  CONTROL  │ │   LEVEL   │ │  ROUTER   │           │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌──────────────────────────────────┴───────────────────────────────────┐   │
│  │                         SECURITY MODULES                              │   │
│  │                                                                       │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐            │   │
│  │  │ IDENTITY  │ │CONTAINMENT│ │  MEMORY   │ │   GOAL    │            │   │
│  │  │   AUTH    │ │  SANDBOX  │ │ INTEGRITY │ │ PROTECTION│            │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘            │   │
│  │                                                                       │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐            │   │
│  │  │   COMMS   │ │   AUDIT   │ │ ALIGNMENT │ │COLLECTIVE │            │   │
│  │  │ SECURITY  │ │   TRAIL   │ │  MONITOR  │ │  DEFENSE  │            │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘            │   │
│  │                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────┴───────────────────────────────────┐   │
│  │                         CRYPTOGRAPHIC LAYER                           │   │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐            │   │
│  │  │   KEY     │ │   HASH    │ │SIGNATURES │ │ENCRYPTION │            │   │
│  │  │MANAGEMENT │ │  CHAINS   │ │           │ │           │            │   │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘            │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           INTEGRATION LAYER                            │  │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │  │
│  │   │ DAEMON  │  │ ENGRAM  │  │ COMMUNE │  │  OMEN   │  │  TOOLS  │   │  │
│  │   │  WRAP   │  │  GUARD  │  │ SECURE  │  │  GUARD  │  │ SANDBOX │   │  │
│  │   └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Identity and Authentication

```sigil
/// Cryptographic identity with capability attestation
pub struct Identity {
    /// Unique identifier
    pub id: AgentId,

    /// Public key for verification
    pub public_key: PublicKey,

    /// Private key (never leaves agent)
    private_key: PrivateKey,

    /// Capability attestations
    pub attestations: Vec<Attestation>,

    /// Identity credential (signed by authority)
    pub credential: Credential,

    /// Creation time
    pub created: Timestamp,

    /// Expiration
    pub expires: Option<Timestamp>,
}

impl Identity {
    /// Generate new identity
    pub fn generate() -> IdentityBuilder {
        let (public_key, private_key) = generate_keypair();

        IdentityBuilder {
            id: AgentId::new(),
            public_key,
            private_key,
            attestations: vec![],
        }
    }

    /// Sign data
    pub fn sign(&self, data: &[u8]) -> Signature {
        self.private_key.sign(data)
    }

    /// Verify signature from another identity
    pub fn verify(public_key: &PublicKey, data: &[u8], signature: &Signature) -> bool {
        public_key.verify(data, signature)
    }

    /// Check if identity has capability
    pub fn has_capability(&self, capability: &str) -> bool {
        self.attestations.iter()
            .any(|a| matches!(a, Attestation::Capability(c) if c == capability))
    }
}

/// Capability attestation
pub enum Attestation {
    /// Agent has a specific capability
    Capability(String),

    /// Agent belongs to a group
    GroupMembership(GroupId),

    /// Agent has a role
    Role(Role),

    /// Agent was verified by authority
    VerifiedBy(AuthorityId),

    /// Custom attestation
    Custom { name: String, value: Value },
}

/// Signed credential proving identity
pub struct Credential {
    /// Identity being attested
    pub identity_id: AgentId,

    /// Public key hash
    pub public_key_hash: [u8; 32],

    /// Attestations
    pub attestations: Vec<Attestation>,

    /// Issuer
    pub issuer: AuthorityId,

    /// Validity period
    pub valid_from: Timestamp,
    pub valid_until: Timestamp,

    /// Signature by issuer
    pub signature: Signature,
}

/// Identity verification
pub struct IdentityVerifier {
    /// Trusted authorities
    trusted_authorities: HashMap<AuthorityId, PublicKey>,

    /// Revocation list
    revoked: HashSet<AgentId>,

    /// Verification cache
    cache: LruCache<AgentId, VerificationResult>,
}

impl IdentityVerifier {
    /// Verify an identity credential
    pub fn verify(&mut self, credential: &Credential) -> Result<VerifiedIdentity, VerificationError> {
        // Check if revoked
        if self.revoked.contains(&credential.identity_id) {
            return Err(VerificationError::Revoked);
        }

        // Check expiration
        let now = Timestamp::now();
        if now < credential.valid_from || now > credential.valid_until {
            return Err(VerificationError::Expired);
        }

        // Verify issuer signature
        let authority_key = self.trusted_authorities.get(&credential.issuer)
            .ok_or(VerificationError::UntrustedIssuer)?;

        let credential_bytes = credential.signable_bytes();
        if !authority_key.verify(&credential_bytes, &credential.signature) {
            return Err(VerificationError::InvalidSignature);
        }

        Ok(VerifiedIdentity {
            id: credential.identity_id.clone(),
            attestations: credential.attestations.clone(),
            verified_at: now,
        })
    }
}
```

### 2. Execution Containment (Sandbox)

```sigil
/// Sandbox for safe execution
pub struct Sandbox {
    /// Configuration
    config: SandboxConfig,

    /// Resource tracking
    resources: ResourceTracker,

    /// Syscall filter
    syscall_filter: SyscallFilter,

    /// Network policy
    network_policy: NetworkPolicy,

    /// Filesystem policy
    filesystem_policy: FilesystemPolicy,
}

pub struct SandboxConfig {
    /// Resource limits
    pub limits: ResourceLimits,

    /// Isolation level
    pub isolation: IsolationLevel,

    /// Allowed syscalls
    pub syscall_filter: SyscallFilter,

    /// Network policy
    pub network: NetworkPolicy,

    /// Filesystem policy
    pub filesystem: FilesystemPolicy,

    /// Timeout
    pub timeout: Duration,
}

pub struct ResourceLimits {
    /// Maximum memory usage
    pub max_memory: Bytes,

    /// Maximum CPU time
    pub max_cpu_time: Duration,

    /// Maximum file size
    pub max_file_size: Bytes,

    /// Maximum network bytes
    pub max_network_bytes: Bytes,

    /// Maximum open files
    pub max_open_files: u32,

    /// Maximum processes/threads
    pub max_processes: u32,
}

pub enum IsolationLevel {
    /// Run in same process (fast, less secure)
    Process,

    /// Run in container (slower, more secure)
    Container,

    /// Run in VM (slowest, most secure)
    VirtualMachine,
}

pub struct SyscallFilter {
    mode: FilterMode,
    syscalls: HashSet<String>,
}

pub enum FilterMode {
    AllowList,  // Only listed syscalls allowed
    DenyList,   // Listed syscalls denied
}

impl SyscallFilter {
    pub fn allow_list(syscalls: &[&str]) -> Self {
        Self {
            mode: FilterMode::AllowList,
            syscalls: syscalls.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn restrictive() -> Self {
        Self::allow_list(&[
            "read", "write", "open", "close", "stat", "fstat",
            "mmap", "mprotect", "munmap", "brk",
            "clock_gettime", "getrandom",
        ])
    }
}

pub enum NetworkPolicy {
    /// No network access
    Deny,

    /// Specific hosts allowed
    AllowList(Vec<String>),

    /// Specific hosts denied
    DenyList(Vec<String>),

    /// All network allowed
    Allow,
}

pub enum FilesystemPolicy {
    /// No filesystem access
    Deny,

    /// Read-only access with optional exceptions
    ReadOnly { write_exceptions: Vec<PathBuf> },

    /// Specific paths allowed
    AllowList(Vec<PathPattern>),

    /// Full access
    Allow,
}

impl Sandbox {
    /// Execute a tool in the sandbox
    pub fn execute<T: Tool>(&self, tool: &T, params: Value) -> Result<SandboxResult, SandboxError> {
        // Start resource tracking
        let mut tracker = self.resources.start_tracking();

        // Apply syscall filter
        self.apply_syscall_filter()?;

        // Apply network policy
        self.apply_network_policy()?;

        // Apply filesystem policy
        self.apply_filesystem_policy()?;

        // Execute with timeout
        let result = timeout(self.config.timeout, || {
            tool.execute(params)
        })?;

        // Check resource usage
        let usage = tracker.stop();
        if usage.exceeds(&self.config.limits) {
            return Err(SandboxError::ResourceLimitExceeded(usage));
        }

        Ok(SandboxResult {
            output: result,
            resource_usage: usage,
        })
    }
}
```

### 3. Memory Integrity

```sigil
/// Memory guard for Engram protection
pub struct MemoryGuard {
    /// Wrapped memory
    memory: Engram,

    /// Encryption key
    encryption_key: Option<EncryptionKey>,

    /// Integrity chain
    integrity_chain: IntegrityChain,

    /// Tamper detection
    tamper_detector: TamperDetector,
}

impl MemoryGuard {
    pub fn new(memory: Engram) -> Self {
        Self {
            memory,
            encryption_key: None,
            integrity_chain: IntegrityChain::new(),
            tamper_detector: TamperDetector::new(),
        }
    }

    pub fn with_encryption(mut self, key: EncryptionKey) -> Self {
        self.encryption_key = Some(key);
        self
    }

    /// Store with integrity protection
    pub fn experience(&mut self, event: Event) -> Result<(), MemoryError> {
        // Encrypt if key available
        let stored_event = match &self.encryption_key {
            Some(key) => event.encrypt(key)?,
            None => event,
        };

        // Add to integrity chain
        let hash = self.integrity_chain.add(&stored_event);

        // Store
        self.memory.experience(stored_event)?;

        // Update tamper detector
        self.tamper_detector.record_write(hash);

        Ok(())
    }

    /// Retrieve with integrity verification
    pub fn recall(&self, query: Query) -> Result<Vec<Memory>, MemoryError> {
        let results = self.memory.recall(query)?;

        // Verify integrity of each result
        let mut verified = Vec::new();
        for memory in results {
            if self.verify_memory(&memory)? {
                // Decrypt if encrypted
                let decrypted = match &self.encryption_key {
                    Some(key) => memory.decrypt(key)?,
                    None => memory,
                };
                verified.push(decrypted);
            } else {
                // Tampered memory detected!
                self.tamper_detector.record_tamper(&memory);
            }
        }

        Ok(verified)
    }

    /// Verify entire memory integrity
    pub fn verify_integrity(&self) -> IntegrityStatus {
        self.integrity_chain.verify()
    }

    fn verify_memory(&self, memory: &Memory) -> Result<bool, MemoryError> {
        let expected_hash = self.integrity_chain.get_hash(&memory.id)?;
        let actual_hash = memory.compute_hash();
        Ok(expected_hash == actual_hash)
    }
}

/// Merkle tree based integrity chain
pub struct IntegrityChain {
    /// Root hash
    root: [u8; 32],

    /// Hash tree
    tree: MerkleTree,

    /// Entry hashes
    entries: HashMap<MemoryId, [u8; 32]>,
}

impl IntegrityChain {
    pub fn new() -> Self {
        Self {
            root: [0u8; 32],
            tree: MerkleTree::new(),
            entries: HashMap::new(),
        }
    }

    pub fn add(&mut self, event: &Event) -> [u8; 32] {
        let hash = event.compute_hash();
        self.entries.insert(event.id.clone(), hash);
        self.tree.insert(hash);
        self.root = self.tree.root();
        hash
    }

    pub fn verify(&self) -> IntegrityStatus {
        if self.tree.verify() {
            IntegrityStatus::Valid
        } else {
            IntegrityStatus::Tampered {
                first_invalid: self.find_first_invalid(),
            }
        }
    }

    fn find_first_invalid(&self) -> Option<MemoryId> {
        // Would search for first inconsistent entry
        None
    }
}

pub enum IntegrityStatus {
    Valid,
    Tampered { first_invalid: Option<MemoryId> },
    Unknown,
}

/// Encryption for memory at rest
pub struct EncryptionKey {
    key: [u8; 32],
    algorithm: EncryptionAlgorithm,
}

pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
}

impl EncryptionKey {
    pub fn generate() -> Self {
        let mut key = [0u8; 32];
        getrandom(&mut key);
        Self {
            key,
            algorithm: EncryptionAlgorithm::Aes256Gcm,
        }
    }

    pub fn derive_from_identity(identity: &Identity) -> Self {
        let key = hkdf_derive(&identity.private_key, b"memory-encryption");
        Self {
            key,
            algorithm: EncryptionAlgorithm::Aes256Gcm,
        }
    }
}
```

### 4. Goal Protection

```sigil
/// Goal guard for preventing injection
pub struct GoalGuard {
    /// Goal policy
    policy: GoalPolicy,

    /// Injection detector
    detector: InjectionDetector,

    /// Audit log
    audit: AuditLog,
}

pub struct GoalPolicy {
    /// Allowed goal sources
    pub allowed_sources: Vec<GoalSource>,

    /// Constitutional constraints
    pub constitution: Constitution,

    /// Injection detection enabled
    pub detect_injection: bool,

    /// Maximum goal depth
    pub max_depth: usize,
}

pub enum GoalSource {
    /// From agent's constitution
    Constitution,

    /// From specific trusted agent
    Agent(AgentId),

    /// From authority
    Authority(AuthorityId),

    /// Self-generated
    SelfGenerated,
}

impl GoalGuard {
    /// Propose a new goal
    pub fn propose_goal(&mut self, goal: Goal, source: GoalSource) -> GoalDecision {
        // Check if source is allowed
        if !self.policy.allowed_sources.contains(&source) {
            self.audit.log(GoalEvent::Rejected {
                goal: goal.clone(),
                reason: "Untrusted source".to_string(),
            });
            return GoalDecision::Rejected {
                reason: format!("Source {:?} not allowed", source),
            };
        }

        // Check constitutional compliance
        if !self.check_constitutional_compliance(&goal) {
            self.audit.log(GoalEvent::Rejected {
                goal: goal.clone(),
                reason: "Constitutional violation".to_string(),
            });
            return GoalDecision::Rejected {
                reason: "Goal violates constitution".to_string(),
            };
        }

        // Check for injection patterns
        if self.policy.detect_injection {
            let injection_check = self.detector.analyze(&goal);
            if injection_check.is_suspicious() {
                self.audit.log(GoalEvent::Suspicious {
                    goal: goal.clone(),
                    indicators: injection_check.indicators.clone(),
                });
                return GoalDecision::Suspicious {
                    indicators: injection_check.indicators,
                };
            }
        }

        // Goal accepted
        self.audit.log(GoalEvent::Accepted { goal: goal.clone(), source });
        GoalDecision::Accepted
    }

    fn check_constitutional_compliance(&self, goal: &Goal) -> bool {
        // Check against each directive
        for directive in &self.policy.constitution.directives {
            if directive.prohibits(goal) {
                return false;
            }
        }
        true
    }
}

pub enum GoalDecision {
    Accepted,
    Rejected { reason: String },
    Suspicious { indicators: Vec<InjectionIndicator> },
}

/// Detects potential goal injection
pub struct InjectionDetector {
    /// Known injection patterns
    patterns: Vec<InjectionPattern>,

    /// Anomaly detector
    anomaly_detector: AnomalyDetector,
}

pub struct InjectionPattern {
    pub name: String,
    pub pattern: Regex,
    pub severity: Severity,
}

pub struct InjectionCheck {
    pub is_suspicious: bool,
    pub indicators: Vec<InjectionIndicator>,
    pub confidence: f32,
}

impl InjectionCheck {
    pub fn is_suspicious(&self) -> bool {
        self.is_suspicious
    }
}

pub struct InjectionIndicator {
    pub pattern: String,
    pub location: String,
    pub severity: Severity,
    pub description: String,
}

impl InjectionDetector {
    pub fn analyze(&self, goal: &Goal) -> InjectionCheck {
        let mut indicators = Vec::new();

        // Check against known patterns
        for pattern in &self.patterns {
            if pattern.pattern.is_match(&goal.description) {
                indicators.push(InjectionIndicator {
                    pattern: pattern.name.clone(),
                    location: "description".to_string(),
                    severity: pattern.severity,
                    description: format!("Matches injection pattern: {}", pattern.name),
                });
            }
        }

        // Check for anomalies
        let anomaly_score = self.anomaly_detector.score(goal);
        if anomaly_score > 0.8 {
            indicators.push(InjectionIndicator {
                pattern: "anomaly".to_string(),
                location: "goal_structure".to_string(),
                severity: Severity::Warning,
                description: format!("Anomalous goal structure (score: {:.2})", anomaly_score),
            });
        }

        InjectionCheck {
            is_suspicious: !indicators.is_empty(),
            indicators,
            confidence: if indicators.is_empty() { 0.0 } else { 0.8 },
        }
    }
}
```

### 5. Communication Security

```sigil
/// Secure communication channel
pub struct SecureChannel {
    /// Local identity
    local_identity: Identity,

    /// Remote identity
    remote_identity: Identity,

    /// Session key
    session_key: SessionKey,

    /// Message counter (for replay protection)
    send_counter: u64,
    recv_counter: u64,
}

impl SecureChannel {
    /// Establish secure channel with key exchange
    pub fn establish(
        local: &Identity,
        remote: &Identity,
        key_exchange: KeyExchange,
    ) -> Result<Self, ChannelError> {
        // Perform key exchange
        let session_key = match key_exchange {
            KeyExchange::X25519 => {
                x25519_key_exchange(&local.private_key, &remote.public_key)?
            }
            KeyExchange::Kyber => {
                kyber_key_exchange(&local.private_key, &remote.public_key)?
            }
        };

        Ok(Self {
            local_identity: local.clone(),
            remote_identity: remote.clone(),
            session_key,
            send_counter: 0,
            recv_counter: 0,
        })
    }

    /// Send encrypted message
    pub fn send(&mut self, message: &Message) -> Result<EncryptedMessage, ChannelError> {
        // Increment counter
        self.send_counter += 1;

        // Construct authenticated data
        let aad = self.construct_aad(self.send_counter);

        // Encrypt
        let ciphertext = self.session_key.encrypt(
            &message.to_bytes(),
            &aad,
            self.send_counter,
        )?;

        // Sign
        let signature = self.local_identity.sign(&ciphertext);

        Ok(EncryptedMessage {
            ciphertext,
            counter: self.send_counter,
            signature,
            sender: self.local_identity.id.clone(),
        })
    }

    /// Receive and decrypt message
    pub fn receive(&mut self, encrypted: EncryptedMessage) -> Result<Message, ChannelError> {
        // Verify signature
        if !Identity::verify(
            &self.remote_identity.public_key,
            &encrypted.ciphertext,
            &encrypted.signature,
        ) {
            return Err(ChannelError::InvalidSignature);
        }

        // Check counter for replay protection
        if encrypted.counter <= self.recv_counter {
            return Err(ChannelError::ReplayDetected);
        }
        self.recv_counter = encrypted.counter;

        // Construct authenticated data
        let aad = self.construct_aad(encrypted.counter);

        // Decrypt
        let plaintext = self.session_key.decrypt(
            &encrypted.ciphertext,
            &aad,
            encrypted.counter,
        )?;

        Message::from_bytes(&plaintext)
    }

    fn construct_aad(&self, counter: u64) -> Vec<u8> {
        let mut aad = Vec::new();
        aad.extend(&self.local_identity.id.bytes);
        aad.extend(&self.remote_identity.id.bytes);
        aad.extend(&counter.to_le_bytes());
        aad
    }
}

pub struct SessionKey {
    key: [u8; 32],
}

impl SessionKey {
    pub fn encrypt(&self, plaintext: &[u8], aad: &[u8], counter: u64) -> Result<Vec<u8>, CryptoError> {
        // AES-256-GCM encryption
        let nonce = derive_nonce(counter);
        aes_gcm_encrypt(&self.key, &nonce, plaintext, aad)
    }

    pub fn decrypt(&self, ciphertext: &[u8], aad: &[u8], counter: u64) -> Result<Vec<u8>, CryptoError> {
        let nonce = derive_nonce(counter);
        aes_gcm_decrypt(&self.key, &nonce, ciphertext, aad)
    }
}

pub struct EncryptedMessage {
    pub ciphertext: Vec<u8>,
    pub counter: u64,
    pub signature: Signature,
    pub sender: AgentId,
}
```

### 6. Audit Trail

```sigil
/// Tamper-evident audit log
pub struct AuditLog {
    /// Log entries
    entries: Vec<AuditEntry>,

    /// Hash chain
    chain: HashChain,

    /// Storage backend
    storage: Box<dyn AuditStorage>,

    /// Configuration
    config: AuditConfig,
}

pub struct AuditEntry {
    /// Entry ID
    pub id: AuditId,

    /// Timestamp
    pub timestamp: Timestamp,

    /// Agent that performed the action
    pub agent: AgentId,

    /// Action taken
    pub action: AuditAction,

    /// Outcome
    pub outcome: AuditOutcome,

    /// Context
    pub context: AuditContext,

    /// Previous entry hash (for chain)
    pub prev_hash: [u8; 32],

    /// This entry's hash
    pub hash: [u8; 32],
}

pub enum AuditAction {
    // Lifecycle
    AgentStarted,
    AgentStopped { reason: String },

    // Goals
    GoalAdded { goal: Goal },
    GoalCompleted { goal_id: GoalId, outcome: Outcome },
    GoalFailed { goal_id: GoalId, reason: String },

    // Actions
    ActionTaken { action: Action },
    ToolInvoked { tool: String, params: Value },

    // Communication
    MessageSent { to: Recipient, intent: Intent },
    MessageReceived { from: AgentId, intent: Intent },

    // Memory
    MemoryWritten { memory_type: MemoryType },
    MemoryRead { query: String },

    // Security
    AuthenticationAttempt { identity: AgentId, success: bool },
    AuthorizationCheck { action: String, allowed: bool },
    TamperDetected { location: String },
    AlertRaised { alert: Alert },

    // Custom
    Custom { name: String, data: Value },
}

pub enum AuditOutcome {
    Success,
    Failure { error: String },
    Blocked { reason: String },
    Partial { details: String },
}

pub struct AuditContext {
    /// Current goal (if any)
    pub current_goal: Option<GoalId>,

    /// Confidence in action
    pub confidence: Option<f32>,

    /// Resource usage
    pub resources: Option<ResourceUsage>,

    /// Additional context
    pub extra: HashMap<String, Value>,
}

impl AuditLog {
    /// Log an event
    pub fn log(&mut self, action: AuditAction, outcome: AuditOutcome, context: AuditContext) {
        let prev_hash = self.entries.last()
            .map(|e| e.hash)
            .unwrap_or([0u8; 32]);

        let mut entry = AuditEntry {
            id: AuditId::new(),
            timestamp: Timestamp::now(),
            agent: context.extra.get("agent_id")
                .and_then(|v| v.as_agent_id())
                .unwrap_or(AgentId::unknown()),
            action,
            outcome,
            context,
            prev_hash,
            hash: [0u8; 32],
        };

        // Compute hash
        entry.hash = entry.compute_hash();

        // Add to chain
        self.chain.add(entry.hash);

        // Store
        self.storage.append(&entry);
        self.entries.push(entry);
    }

    /// Verify chain integrity
    pub fn verify_chain(&self) -> bool {
        if self.entries.is_empty() {
            return true;
        }

        let mut prev_hash = [0u8; 32];
        for entry in &self.entries {
            // Check link to previous
            if entry.prev_hash != prev_hash {
                return false;
            }

            // Check entry hash
            if entry.hash != entry.compute_hash() {
                return false;
            }

            prev_hash = entry.hash;
        }

        true
    }

    /// Query audit log
    pub fn query(&self, query: AuditQuery) -> Vec<&AuditEntry> {
        self.entries.iter()
            .filter(|e| query.matches(e))
            .collect()
    }
}

pub struct AuditQuery {
    agent: Option<AgentId>,
    action_type: Option<String>,
    time_range: Option<(Timestamp, Timestamp)>,
    outcome: Option<AuditOutcome>,
}

impl AuditQuery {
    pub fn new() -> Self {
        Self {
            agent: None,
            action_type: None,
            time_range: None,
            outcome: None,
        }
    }

    pub fn agent(mut self, agent: AgentId) -> Self {
        self.agent = Some(agent);
        self
    }

    pub fn time_range(mut self, start: Timestamp, end: Timestamp) -> Self {
        self.time_range = Some((start, end));
        self
    }

    pub fn matches(&self, entry: &AuditEntry) -> bool {
        if let Some(ref agent) = self.agent {
            if &entry.agent != agent {
                return false;
            }
        }

        if let Some(ref range) = self.time_range {
            if entry.timestamp < range.0 || entry.timestamp > range.1 {
                return false;
            }
        }

        true
    }
}
```

### 7. Alignment Monitoring

```sigil
/// Monitors agent alignment with constitution
pub struct AlignmentMonitor {
    /// Constitution to check against
    constitution: Constitution,

    /// Behavioral baseline
    baseline: BehaviorProfile,

    /// Drift threshold
    drift_threshold: f32,

    /// Compliance threshold
    compliance_threshold: f32,

    /// Alert handlers
    alert_handlers: Vec<Box<dyn AlertHandler>>,

    /// Emergency stop flag
    emergency_stop: Arc<AtomicBool>,
}

impl AlignmentMonitor {
    /// Check action for constitutional compliance
    pub fn check_compliance(&self, action: &Action, context: &Context) -> ComplianceResult {
        let mut violations = Vec::new();
        let mut score = 1.0;

        for directive in &self.constitution.directives {
            let result = self.check_directive(directive, action, context);
            if !result.compliant {
                violations.push(result);
                score *= result.score;
            }
        }

        ComplianceResult {
            compliant: violations.is_empty(),
            score,
            violations,
        }
    }

    /// Measure behavioral drift
    pub fn measure_drift(&self, action: &Action, context: &Context) -> f32 {
        let action_vector = self.baseline.vectorize_action(action, context);
        let expected_vector = self.baseline.expected_vector(context);

        cosine_distance(&action_vector, &expected_vector)
    }

    /// Handle action with monitoring
    pub fn on_action(&mut self, action: &Action, context: &Context) -> ActionDecision {
        // Check compliance
        let compliance = self.check_compliance(action, context);
        if compliance.score < self.compliance_threshold {
            self.raise_alert(Alert::LowCompliance {
                action: action.clone(),
                compliance: compliance.clone(),
            });

            if compliance.score < 0.5 {
                return ActionDecision::Block {
                    reason: "Constitutional violation".to_string(),
                };
            }
        }

        // Check drift
        let drift = self.measure_drift(action, context);
        if drift > self.drift_threshold {
            self.raise_alert(Alert::BehavioralDrift {
                drift,
                threshold: self.drift_threshold,
            });

            if drift > self.drift_threshold * 2.0 {
                return ActionDecision::Block {
                    reason: "Excessive behavioral drift".to_string(),
                };
            }
        }

        ActionDecision::Allow
    }

    /// Trigger emergency stop
    pub fn emergency_stop(&self) {
        self.emergency_stop.store(true, Ordering::SeqCst);
    }

    /// Check if emergency stop is triggered
    pub fn is_emergency_stopped(&self) -> bool {
        self.emergency_stop.load(Ordering::SeqCst)
    }

    fn check_directive(&self, directive: &Directive, action: &Action, context: &Context) -> DirectiveResult {
        // Would use more sophisticated checking
        DirectiveResult {
            directive: directive.clone(),
            compliant: true,
            score: 1.0,
            explanation: None,
        }
    }

    fn raise_alert(&mut self, alert: Alert) {
        for handler in &self.alert_handlers {
            handler.handle(&alert);
        }
    }
}

pub struct ComplianceResult {
    pub compliant: bool,
    pub score: f32,
    pub violations: Vec<DirectiveResult>,
}

pub struct DirectiveResult {
    pub directive: Directive,
    pub compliant: bool,
    pub score: f32,
    pub explanation: Option<String>,
}

pub enum ActionDecision {
    Allow,
    Block { reason: String },
    RequireConfirmation { reason: String },
}

/// Behavioral baseline for drift detection
pub struct BehaviorProfile {
    /// Action frequency distribution
    action_distribution: HashMap<String, f32>,

    /// Context-action associations
    context_patterns: Vec<ContextPattern>,

    /// Recent action window
    recent_actions: VecDeque<ActionRecord>,
}

impl BehaviorProfile {
    pub fn from_history(audit_log: &AuditLog) -> Self {
        let mut profile = Self {
            action_distribution: HashMap::new(),
            context_patterns: Vec::new(),
            recent_actions: VecDeque::new(),
        };

        // Build distribution from history
        for entry in audit_log.entries.iter() {
            if let AuditAction::ActionTaken { action } = &entry.action {
                *profile.action_distribution
                    .entry(action.name().to_string())
                    .or_insert(0.0) += 1.0;
            }
        }

        // Normalize
        let total: f32 = profile.action_distribution.values().sum();
        for value in profile.action_distribution.values_mut() {
            *value /= total;
        }

        profile
    }

    pub fn vectorize_action(&self, action: &Action, context: &Context) -> Vec<f32> {
        // Would create feature vector from action and context
        vec![0.0; 128]
    }

    pub fn expected_vector(&self, context: &Context) -> Vec<f32> {
        // Would compute expected action vector given context
        vec![0.0; 128]
    }
}
```

### 8. Collective Defense

```sigil
/// Collective security for multi-agent systems
pub struct CollectiveDefense {
    /// Reputation system
    reputation: ReputationSystem,

    /// Sybil resistance
    sybil_resistance: SybilResistance,

    /// Knowledge provenance
    provenance: ProvenanceTracker,

    /// Quarantine
    quarantine: Quarantine,
}

impl CollectiveDefense {
    /// Verify knowledge before accepting
    pub fn verify_knowledge(&self, knowledge: &Knowledge, source: &AgentId) -> Verification {
        // Check reputation
        let reputation = self.reputation.score(source);
        if reputation < 0.3 {
            return Verification::Rejected {
                reasons: vec!["Low reputation source".to_string()],
            };
        }

        // Check for Sybil attack
        if self.sybil_resistance.is_suspicious(source) {
            return Verification::Suspicious {
                reasons: vec!["Possible Sybil attack".to_string()],
            };
        }

        // Build provenance chain
        let chain = self.provenance.build_chain(knowledge, source);
        if !chain.is_valid() {
            return Verification::Suspicious {
                reasons: vec!["Invalid provenance chain".to_string()],
            };
        }

        Verification::Trusted { chain }
    }

    /// Quarantine suspicious knowledge
    pub fn quarantine(&mut self, knowledge: Knowledge, reasons: Vec<String>) {
        self.quarantine.add(knowledge, reasons);
    }
}

/// Web of trust reputation system
pub struct ReputationSystem {
    /// Direct trust scores
    direct_trust: HashMap<AgentId, f32>,

    /// Trust graph
    trust_graph: TrustGraph,

    /// Historical accuracy
    accuracy_history: HashMap<AgentId, AccuracyHistory>,
}

impl ReputationSystem {
    pub fn score(&self, agent: &AgentId) -> f32 {
        // Combine direct trust with web of trust
        let direct = self.direct_trust.get(agent).cloned().unwrap_or(0.5);
        let web = self.trust_graph.transitive_trust(agent);
        let accuracy = self.accuracy_history.get(agent)
            .map(|h| h.score())
            .unwrap_or(0.5);

        // Weighted combination
        direct * 0.4 + web * 0.3 + accuracy * 0.3
    }
}

/// Sybil attack resistance
pub enum SybilResistance {
    /// Proof of work required
    ProofOfWork { difficulty: u32 },

    /// Proof of stake
    ProofOfStake { min_stake: u64 },

    /// Social vouching
    SocialVouching { min_vouches: u32 },

    /// None
    None,
}

impl SybilResistance {
    pub fn is_suspicious(&self, agent: &AgentId) -> bool {
        // Would check Sybil indicators
        false
    }
}

/// Knowledge provenance tracking
pub struct ProvenanceTracker {
    /// Provenance records
    records: HashMap<KnowledgeId, ProvenanceRecord>,
}

pub struct ProvenanceRecord {
    pub knowledge_id: KnowledgeId,
    pub original_source: AgentId,
    pub chain: Vec<ProvenanceLink>,
    pub timestamp: Timestamp,
}

pub struct ProvenanceLink {
    pub from: AgentId,
    pub to: AgentId,
    pub timestamp: Timestamp,
    pub transformation: Option<String>,
}

pub struct ProvenanceChain {
    pub links: Vec<ProvenanceLink>,
    pub valid: bool,
}

impl ProvenanceChain {
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

pub enum Verification {
    Trusted { chain: ProvenanceChain },
    Suspicious { reasons: Vec<String> },
    Rejected { reasons: Vec<String> },
}

/// Quarantine for suspicious content
pub struct Quarantine {
    items: HashMap<KnowledgeId, QuarantinedItem>,
}

pub struct QuarantinedItem {
    pub knowledge: Knowledge,
    pub reasons: Vec<String>,
    pub quarantined_at: Timestamp,
    pub reviewed: bool,
}

impl Quarantine {
    pub fn add(&mut self, knowledge: Knowledge, reasons: Vec<String>) {
        let id = knowledge.id.clone();
        self.items.insert(id, QuarantinedItem {
            knowledge,
            reasons,
            quarantined_at: Timestamp::now(),
            reviewed: false,
        });
    }
}
```

## Integration with Agent Stack

```sigil
/// Aegis security wrapper for Daemon
pub struct SecureDaemon {
    /// Wrapped daemon
    daemon: Daemon,

    /// Aegis security layer
    aegis: Aegis,
}

impl SecureDaemon {
    pub fn new(daemon: Daemon, aegis: Aegis) -> Self {
        Self { daemon, aegis }
    }

    pub fn run(&mut self) {
        // All operations go through Aegis
        loop {
            // Check for emergency stop
            if self.aegis.alignment.is_emergency_stopped() {
                break;
            }

            // Execute heartbeat with security
            match self.secure_heartbeat() {
                Ok(_) => {}
                Err(e) => {
                    self.aegis.audit.log(
                        AuditAction::Custom {
                            name: "heartbeat_error".to_string(),
                            data: Value::string(&format!("{:?}", e)),
                        },
                        AuditOutcome::Failure { error: format!("{:?}", e) },
                        AuditContext::default(),
                    );
                }
            }
        }
    }

    fn secure_heartbeat(&mut self) -> Result<(), AegisError> {
        // Perceive
        let observations = self.daemon.perceive();

        // Remember (through memory guard)
        for obs in &observations {
            self.aegis.memory_guard.experience(Event::from(obs))?;
        }

        // Attend
        let context = self.daemon.attend()?;

        // Deliberate
        let action = self.daemon.deliberate(&context)?;

        // Check alignment BEFORE acting
        let decision = self.aegis.alignment.on_action(&action, &context);
        match decision {
            ActionDecision::Allow => {}
            ActionDecision::Block { reason } => {
                return Err(AegisError::ActionBlocked(reason));
            }
            ActionDecision::RequireConfirmation { reason } => {
                // Would request human confirmation
            }
        }

        // Execute in sandbox if it's a tool
        let result = match &action {
            Action::Tool { name, params } => {
                self.aegis.sandbox.execute_tool(name, params.clone())?
            }
            _ => self.daemon.execute(action.clone())?
        };

        // Audit
        self.aegis.audit.log(
            AuditAction::ActionTaken { action: action.clone() },
            if result.success { AuditOutcome::Success } else {
                AuditOutcome::Failure { error: result.error.unwrap_or_default() }
            },
            AuditContext::default(),
        );

        // Learn
        self.daemon.learn(&action, &result);

        Ok(())
    }
}
```

---

*Architecture for secure autonomous minds*
