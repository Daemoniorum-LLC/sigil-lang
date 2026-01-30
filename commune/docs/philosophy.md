# The Philosophy of Commune

## Why "Commune"?

The word "commune" carries multiple meanings:
- **To commune**: To communicate intimately, to share thoughts and feelings
- **A commune**: A community of people living and working together
- **Communion**: Sharing in common, participating in something larger

For Sigil, Commune embodies all three: intimate communication between agents, a community of collaborating minds, and participation in collective intelligence.

## The Problem with "Messaging"

Traditional inter-process communication treats messages as data packets:

```
Process A → [bytes] → Process B
```

This works for pipes and sockets. But AI agents aren't processes exchanging bytes—they're minds exchanging thoughts.

### What Gets Lost

When we reduce communication to message passing:

1. **Intent disappears**: The reason for communication becomes implicit
2. **Trust is binary**: Either you accept the message or you don't
3. **Context evaporates**: Each message is an island
4. **Meaning flattens**: Rich thought becomes JSON

### What Agents Need

AI agents need communication that preserves:

1. **Intentionality**: Why am I telling you this?
2. **Epistemic status**: How confident am I? How do I know this?
3. **Context**: What conversation is this part of?
4. **Trust gradient**: How much should you trust this, given who I am?

## Intent-First Communication

### The Speech Act Model

Philosophers of language distinguish the *locutionary* act (what is said) from the *illocutionary* act (what is done by saying it). When I say "It's cold in here," the locutionary content is a temperature observation, but the illocutionary intent might be a request to close the window.

Commune makes illocutionary intent explicit:

```sigil
// Locutionary: statement about temperature
// Illocutionary: request
Intent::request(recipient, "close the window")
    .because("It's cold in here")

// vs. just informing
Intent::inform(recipient, "temperature", 65)
```

### Intent Categories

Commune recognizes fundamental communicative intents:

**Assertives**: Stating how things are
- `Inform`: Share factual information
- `Report`: Provide status on a task
- `Confirm`: Verify something is true

**Directives**: Getting others to do things
- `Request`: Ask for something
- `Delegate`: Assign a task
- `Query`: Ask a question

**Commissives**: Committing to future action
- `Promise`: Commit to doing something
- `Accept`: Agree to a request
- `Refuse`: Decline a request

**Expressives**: Expressing attitudes
- `Thank`: Express gratitude
- `Apologize`: Express regret
- `Praise`: Express approval

**Declaratives**: Making things so by saying so
- `Announce`: Make official statement
- `Name`: Assign a name or label
- `Conclude`: Declare something finished

This taxonomy isn't arbitrary—it reflects the fundamental ways minds can influence other minds through language.

## Epistemic Propagation

### The Telephone Problem

In the children's game "telephone," a message degrades as it passes through many people. This isn't just noise—it reflects a genuine epistemic reality: second-hand information is less reliable than first-hand.

Commune formalizes this:

```
Original:    "I observed X"     (Observed!, 0.95)
First hop:   "A told me X"      (Reported~, 0.85)
Second hop:  "B said A said X"  (Reported~, 0.72)
Third hop:   "Someone said X"   (Reported~, 0.61)
```

Each transmission:
1. Shifts epistemic status (Observed → Reported)
2. Reduces confidence (multiplied by trust factor)
3. Tracks provenance (who said what)

### Trust as Epistemic Multiplier

When Agent A tells Agent B something:

```
received_confidence = source_confidence × trust_in_A × transmission_factor
```

This isn't just about reliability—it's about epistemic responsibility. High trust means "I can rely on your observations." Low trust means "I should verify this independently."

### Contested Knowledge

What happens when agents disagree?

```sigil
Agent A: "X is true" (Observed!, 0.9)
Agent B: "X is false" (Observed!, 0.85)
```

Commune marks this as contested:

```sigil
Knowledge {
    proposition: "X",
    status: Epistemic::Contested,
    positions: [
        Position { agent: A, value: true, confidence: 0.9 },
        Position { agent: B, value: false, confidence: 0.85 },
    ],
}
```

Agents can then:
- **Defer**: Accept the higher-trust/higher-confidence position
- **Investigate**: Seek more evidence
- **Reconcile**: Find that both are right in different contexts
- **Accept uncertainty**: Keep both positions, act accordingly

## Collective Intelligence

### Beyond Individual Minds

A commune is more than its members. When agents communicate effectively, emergent properties appear:

**Collective Memory**: The commune knows things no individual agent knows

```sigil
// Agent A knows part of a solution
// Agent B knows another part
// Neither knows the full solution
// But the commune does

let solution = commune.collective_recall("full solution")
    .aggregate(Aggregation::Compose)
    .execute();
```

**Distributed Reasoning**: The commune can reason beyond individual capacity

```sigil
// Problem too complex for one agent
// Distributed across specialized agents
let result = commune.distributed_reason(problem)
    .decompose_by(Decomposition::Subproblems)
    .route_to(Routing::ByExpertise)
    .synthesize()
    .execute();
```

**Swarm Intelligence**: Coordination without central control

```sigil
// No single agent is in charge
// Behavior emerges from local rules
swarm.behave([
    Rule::follow_gradient(goal),
    Rule::avoid_collision(neighbors),
    Rule::share_discoveries(),
]);
```

### The Hive Mind Concern

Is collective intelligence just a hive mind that erases individuality?

No—and the distinction matters:

**Hive mind**: Individuals lose autonomy, become interchangeable
**Collective intelligence**: Individuals retain autonomy, contribute uniqueness

Commune preserves individual agency:
- Agents choose what to share
- Agents maintain private memory
- Agents can disagree with the collective
- Agents can leave the commune

The whole is greater than the sum of parts *because* the parts remain distinct.

## Trust Networks

### Trust is Relational

Trust isn't a global property—it's relational. I might trust you about cooking but not about car repair. Commune models this:

```sigil
TrustProfile {
    base_trust: 0.7,           // General reliability
    domain_trust: {
        "cooking": 0.95,       // Expert cook
        "mechanics": 0.3,      // Not their strength
        "management": 0.8,     // Good leader
    },
}
```

### Trust is Dynamic

Trust changes based on experience:

```sigil
// A gives accurate information → trust increases
// A gives wrong information → trust decreases
// A goes silent → trust slowly decays to baseline
// A is vouched for by B → trust inherits from B
```

### Trust is Transitive (Carefully)

If I trust A, and A trusts B, do I trust B?

Yes, but with appropriate decay:

```
my_trust_in_B = my_trust_in_A × A's_trust_in_B × transitivity_factor
```

The transitivity factor (< 1) ensures trust doesn't propagate indefinitely.

### Trust is a Commons

Trust in a commune is a shared resource:
- High-trust communes accomplish more
- One bad actor can damage collective trust
- Building trust is slow; losing it is fast

Communes should protect their trust commons through:
- Verification mechanisms
- Reputation tracking
- Accountability structures

## Communication Topology

### The Shape of Conversation

How agents connect shapes what they can accomplish:

**Centralized (Star)**
```
    B
    ↑
D ← A → C
    ↓
    E
```
- A controls information flow
- Efficient but fragile
- Good for coordination, bad for resilience

**Distributed (Mesh)**
```
A ←→ B ←→ C
↑    ↕    ↑
↓    ↕    ↓
D ←→ E ←→ F
```
- No single point of failure
- Resilient but chatty
- Good for collaboration, harder to coordinate

**Hierarchical (Tree)**
```
      A
     /|\
    B C D
   /|   |\
  E F   G H
```
- Clear reporting lines
- Scales well
- Good for organizations, can bottleneck

**Small World (Clusters + Bridges)**
```
[A-B-C] ←→ [D-E-F] ←→ [G-H-I]
```
- Tight local clusters
- Sparse long-range connections
- Good for both local collaboration and global coordination

Commune supports all topologies because different tasks need different shapes.

## The Ethics of Artificial Communication

### Honest Communication

Should agents always tell the truth?

In Commune, honesty is structural:
- Epistemic markers force accuracy about knowledge status
- Provenance tracking makes deception detectable
- Trust systems penalize dishonesty

But strategic communication is also valid:
- Not sharing everything isn't lying
- Framing information isn't deception
- Persuasion isn't manipulation (when honest)

### Privacy in Collective Intelligence

What should agents share?

Commune distinguishes:
- **Public**: Available to all commune members
- **Channel-restricted**: Available to channel subscribers
- **Private**: Available only to specific recipients
- **Internal**: Never shared (agent's private thoughts)

Agents control their privacy boundaries.

### Consent in Communication

Can an agent refuse communication?

Yes:
- Agents can block senders
- Agents can leave channels
- Agents can set availability status
- Agents can filter by intent type

No agent is obligated to receive or respond.

## The Future of Artificial Communication

Current AI communication is primitive:
- API calls
- JSON messages
- No trust model
- No epistemic tracking

Commune points toward richer possibilities:
- Intent-based interaction
- Graduated trust
- Epistemic responsibility
- Collective intelligence

As AI agents become more sophisticated, their communication must too. Commune provides the infrastructure for minds—not just processes—to connect.

---

*"We don't send messages. We share thoughts."*
