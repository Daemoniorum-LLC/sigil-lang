# Aegis

**Security Infrastructure for Autonomous AI Agents**

> *"The shield that guards the minds that guard the world."*

Aegis is the security layer for Sigil's AI agent infrastructure. It provides identity verification, execution containment, memory integrity, alignment monitoring, and collective defense for autonomous agents operating in adversarial environments.

## Why Aegis?

Traditional security assumes human decision-makers and deterministic code. AI agents break both assumptions:

- **Agents make autonomous decisions** based on beliefs that can be manipulated
- **Agents behave probabilistically**, making behavior harder to predict
- **Everything is a potential instruction** - data and code boundaries blur

Aegis addresses the novel attack surface of autonomous AI systems.

## Threat Model

### Adversary Capabilities

Aegis assumes adversaries who can:

- Send arbitrary messages to agents
- Observe agent communications (without encryption)
- Compromise individual agents
- Inject data into shared memory systems
- Manipulate trust relationships over time
- Attempt to influence agent goals and beliefs

### Assets to Protect

| Asset | Confidentiality | Integrity | Availability |
|-------|-----------------|-----------|--------------|
| Agent identity | High | Critical | High |
| Agent memories | Medium | Critical | Medium |
| Agent goals | - | Critical | High |
| Agent beliefs | - | Critical | Medium |
| Communications | High | High | Medium |
| Collective knowledge | Low | Critical | High |
| Action audit trail | Medium | Critical | High |

### Security Properties

Aegis provides:

1. **Authentication**: Verify agent identity cryptographically
2. **Authorization**: Enforce capability-based access control
3. **Integrity**: Detect tampering with memories, goals, beliefs
4. **Confidentiality**: Encrypt sensitive data at rest and in transit
5. **Accountability**: Complete audit trail of agent actions
6. **Containment**: Limit blast radius of compromised agents
7. **Alignment**: Detect and respond to behavioral drift

## Core Components

### 1. Identity (Attestation & Verification)

```sigil
use aegis::{Identity, Credential, Attestation};

// Create verified identity
let identity = Identity::generate()
    .with_attestation(Attestation::capability("research"))
    .with_attestation(Attestation::capability("write"))
    .sign(&authority_key)?;

// Verify another agent's identity
let verified = aegis.verify_identity(&other_agent.credential)?;
assert!(verified.has_capability("research"));
```

### 2. Containment (Sandbox Execution)

```sigil
use aegis::{Sandbox, SandboxConfig, ResourceLimits};

// Create sandbox for tool execution
let sandbox = Sandbox::new(SandboxConfig {
    // Resource limits
    limits: ResourceLimits {
        max_memory: Bytes::megabytes(100),
        max_cpu_time: Duration::seconds(10),
        max_file_size: Bytes::megabytes(10),
        max_network_bytes: Bytes::megabytes(1),
    },

    // Syscall filtering
    syscall_filter: SyscallFilter::allow_list(&[
        "read", "write", "open", "close", "stat",
    ]),

    // Network restrictions
    network: NetworkPolicy::AllowList(vec![
        "api.example.com:443",
    ]),

    // Filesystem restrictions
    filesystem: FilesystemPolicy::ReadOnly {
        exceptions: vec!["/tmp/sandbox"],
    },
});

// Execute tool in sandbox
let result = sandbox.execute(tool, params)?;
```

### 3. Memory Integrity

```sigil
use aegis::{MemoryGuard, IntegrityChain};

// Wrap Engram with integrity protection
let guarded_memory = MemoryGuard::new(engram)
    .with_encryption(EncryptionKey::from_identity(&identity))
    .with_integrity_chain(IntegrityChain::merkle());

// All operations are now protected
guarded_memory.experience(event);  // Encrypted, checksummed

// Detect tampering
match guarded_memory.verify_integrity() {
    Integrity::Valid => { /* All good */ }
    Integrity::Tampered { first_invalid } => {
        // Memory has been modified!
        aegis.alert(Alert::memory_tampering(first_invalid));
    }
}
```

### 4. Goal Protection

```sigil
use aegis::{GoalGuard, GoalPolicy};

// Protect goal stack
let goal_guard = GoalGuard::new(daemon.goals)
    .with_policy(GoalPolicy {
        // Only accept goals from trusted sources
        allowed_sources: vec![
            GoalSource::Constitution,
            GoalSource::Agent(coordinator_id),
        ],

        // Constitutional compliance
        must_comply_with: daemon.constitution.clone(),

        // Injection detection
        detect_injection: true,
    });

// Attempt to add goal - checked against policy
match goal_guard.propose_goal(goal, source) {
    GoalDecision::Accepted => { /* Goal added */ }
    GoalDecision::Rejected { reason } => {
        aegis.log(Event::goal_rejected(goal, reason));
    }
    GoalDecision::Suspicious { indicators } => {
        aegis.alert(Alert::possible_injection(goal, indicators));
    }
}
```

### 5. Communication Security

```sigil
use aegis::{SecureChannel, KeyExchange};

// Establish secure channel with another agent
let channel = SecureChannel::establish(
    &my_identity,
    &other_agent_identity,
    KeyExchange::X25519,
)?;

// Messages are encrypted with perfect forward secrecy
channel.send(message)?;
let received = channel.receive()?;

// Verify message authenticity
assert!(received.verify_signature(&other_agent_identity.public_key));
```

### 6. Audit Trail

```sigil
use aegis::{AuditLog, AuditEvent, AuditQuery};

// All actions are logged
aegis.audit.log(AuditEvent::action_taken(
    agent_id,
    action,
    outcome,
    context,
));

// Query audit trail
let suspicious = aegis.audit.query(
    AuditQuery::new()
        .agent(agent_id)
        .time_range(last_hour)
        .filter(|e| e.outcome == Outcome::Failure)
)?;

// Audit log is append-only and tamper-evident
assert!(aegis.audit.verify_chain());
```

### 7. Alignment Monitoring

```sigil
use aegis::{AlignmentMonitor, ConstitutionalChecker, BehaviorProfile};

// Monitor for alignment drift
let monitor = AlignmentMonitor::new(daemon)
    .with_constitution_checker(ConstitutionalChecker::new(&constitution))
    .with_behavior_baseline(BehaviorProfile::from_history(&audit_log))
    .with_drift_threshold(0.2);

// Continuous monitoring
monitor.on_action(|action, context| {
    // Check constitutional compliance
    let compliance = monitor.check_compliance(action);
    if compliance.score < 0.8 {
        aegis.alert(Alert::low_compliance(action, compliance));
    }

    // Check behavioral drift
    let drift = monitor.measure_drift(action, context);
    if drift > monitor.threshold {
        aegis.alert(Alert::behavioral_drift(drift));
    }
});

// Emergency stop
if monitor.critical_violation_detected() {
    aegis.emergency_stop(daemon);
}
```

### 8. Collective Defense

```sigil
use aegis::{CollectiveDefense, ReputationSystem, SybilResistance};

// Collective defense for the commune
let defense = CollectiveDefense::new(commune)
    .with_reputation(ReputationSystem::web_of_trust())
    .with_sybil_resistance(SybilResistance::proof_of_work())
    .with_knowledge_provenance(true);

// Verify knowledge before accepting
let knowledge = commune.receive_knowledge(source)?;
match defense.verify_knowledge(knowledge) {
    Verification::Trusted { chain } => {
        // Accept with full provenance
        memory.learn_with_provenance(knowledge, chain);
    }
    Verification::Suspicious { reasons } => {
        // Quarantine for review
        defense.quarantine(knowledge, reasons);
    }
    Verification::Rejected { reasons } => {
        // Log and discard
        aegis.log(Event::knowledge_rejected(knowledge, reasons));
    }
}
```

## Quick Start

```sigil
use aegis::{Aegis, AegisConfig};
use daemon::Daemon;
use engram::Engram;

fn main() {
    // Create Aegis security layer
    let aegis = Aegis::new(AegisConfig {
        // Identity
        identity_authority: Some(authority_key),

        // Containment
        default_sandbox: SandboxConfig::restrictive(),

        // Encryption
        encryption: EncryptionConfig::aes256_gcm(),

        // Audit
        audit_retention: Duration::days(90),

        // Alignment
        alignment_threshold: 0.9,

        // Alerts
        alert_handlers: vec![
            AlertHandler::log(),
            AlertHandler::notify(admin_channel),
        ],
    });

    // Wrap daemon with security
    let secure_daemon = aegis.secure(daemon);

    // All operations now go through Aegis
    secure_daemon.run();
}
```

## Configuration

```sigil
let config = AegisConfig {
    // Identity verification
    identity: IdentityConfig {
        authority: Some(authority_public_key),
        require_attestation: true,
        credential_lifetime: Duration::days(30),
    },

    // Execution containment
    containment: ContainmentConfig {
        default_sandbox: SandboxConfig::restrictive(),
        tool_specific: HashMap::from([
            ("http", SandboxConfig::network_only()),
            ("filesystem", SandboxConfig::filesystem_only()),
        ]),
    },

    // Memory protection
    memory: MemorySecurityConfig {
        encryption: true,
        encryption_key_source: KeySource::DeriveFromIdentity,
        integrity_checking: IntegrityMode::MerkleTree,
        tamper_response: TamperResponse::AlertAndHalt,
    },

    // Goal protection
    goals: GoalSecurityConfig {
        require_provenance: true,
        constitutional_compliance: true,
        injection_detection: true,
    },

    // Communication
    communication: CommunicationSecurityConfig {
        require_encryption: true,
        key_exchange: KeyExchange::X25519,
        perfect_forward_secrecy: true,
    },

    // Audit
    audit: AuditConfig {
        log_all_actions: true,
        retention: Duration::days(90),
        tamper_evident: true,
        remote_backup: Some(backup_config),
    },

    // Alignment monitoring
    alignment: AlignmentConfig {
        continuous_monitoring: true,
        drift_threshold: 0.2,
        compliance_threshold: 0.9,
        emergency_stop_enabled: true,
    },

    // Collective defense
    collective: CollectiveDefenseConfig {
        reputation_system: ReputationSystem::WebOfTrust,
        sybil_resistance: SybilResistance::ProofOfWork,
        knowledge_provenance: true,
        quarantine_suspicious: true,
    },
};
```

## Security Levels

Aegis supports different security levels for different deployment scenarios:

| Level | Use Case | Trade-offs |
|-------|----------|------------|
| **Development** | Testing, debugging | Minimal overhead, full logging |
| **Standard** | Normal operation | Balanced security and performance |
| **Hardened** | Sensitive operations | Maximum security, higher latency |
| **Paranoid** | Adversarial environments | Assume compromise, verify everything |

```sigil
let aegis = Aegis::with_level(SecurityLevel::Hardened);
```

## Incident Response

When Aegis detects a security event:

```sigil
aegis.on_alert(|alert| {
    match alert.severity {
        Severity::Info => {
            // Log for analysis
            log::info!("Security event: {:?}", alert);
        }
        Severity::Warning => {
            // Log and notify
            log::warn!("Security warning: {:?}", alert);
            notify_admin(alert);
        }
        Severity::Critical => {
            // Immediate response
            log::error!("CRITICAL: {:?}", alert);
            notify_admin(alert);

            // Contain the threat
            if let Some(agent) = alert.source_agent {
                aegis.isolate(agent);
            }
        }
        Severity::Emergency => {
            // Everything stops
            log::error!("EMERGENCY: {:?}", alert);
            aegis.emergency_stop_all();
            notify_all_admins(alert);
        }
    }
});
```

## API Reference

See [docs/api-reference.md](docs/api-reference.md) for complete API documentation.

## Examples

- [Basic Security](examples/basic.sg) - Minimal security setup
- [Secure Agent](examples/secure_agent.sg) - Fully secured daemon
- [Secure Commune](examples/secure_commune.sg) - Multi-agent security
- [Incident Response](examples/incident.sg) - Handling security events

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed system design.

---

*Part of the Sigil ecosystem - Tools with Teeth*
