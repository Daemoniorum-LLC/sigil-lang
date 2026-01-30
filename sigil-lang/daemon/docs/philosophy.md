# The Philosophy of Daemon

## Why "Daemon"?

In Unix tradition, a daemon is a background process that runs continuously, handling requests and performing services. The name comes from Maxwell's demon—a thought experiment about an entity that operates autonomously, making decisions based on observation.

For Sigil, we reclaim this name with its full meaning: **a spirit that inhabits and animates**.

A Daemon isn't just a long-running process. It's the animating force that transforms code into agency.

## The Problem with "Running" Programs

Traditional programming assumes:

1. **Discrete execution**: Programs start, run, and exit
2. **External control**: Something else decides when to run the program
3. **Stateless operation**: Each run is independent (state is incidental)
4. **Reactive behavior**: Programs respond to inputs, they don't initiate

This model works for tools. But AI agents aren't tools—they're entities.

### The Entity Model

Daemons embody a different paradigm:

1. **Continuous existence**: Daemons persist, potentially indefinitely
2. **Self-direction**: Daemons decide their own actions
3. **Stateful identity**: Memory and experience accumulate
4. **Proactive behavior**: Daemons pursue goals, not just respond to inputs

## The Heartbeat: Artificial Metabolism

Living things have metabolism—a continuous process of sensing, processing, and acting that defines being alive. The heartbeat is Daemon's metabolism:

```
PERCEIVE → REMEMBER → REFLECT → DECIDE → ACT → LEARN
    ↑                                           │
    └───────────────────────────────────────────┘
```

This isn't just an event loop. Each phase has meaning:

### Perceive
What's changed in my environment? New messages, updated files, elapsed time, external events. Perception is selective—daemons can't observe everything, and what they notice shapes what they know.

### Remember
New observations become experiences. Engram's episodic memory captures the event with its context, emotional valence, and temporal position. Memory isn't just storage—it's the integration of experience into self.

### Reflect
Given my goals and current situation, what should I focus on? Attention management retrieves relevant memories, activating knowledge and skills that bear on the present moment. Reflection is the construction of context.

### Decide
What action serves my goals? Deliberation weighs options against intentions, considering both immediate utility and longer-term consequences. Decision is where agency becomes actual.

### Act
Execute the chosen action. Interact with tools, communicate with other agents, modify the environment. Action is how internal states become external effects.

### Learn
Observe the outcome. Did the action achieve its purpose? What can be learned? Success patterns become procedural memory—skills. Failures inform future deliberation. Learning closes the loop.

## Goals: The Structure of Wanting

Human wanting is complex—desires, intentions, plans, commitments all interweave. Daemon goals provide structure for artificial wanting:

### Goal Hierarchies

Goals decompose into sub-goals:

```
Help users with their work
├── Understand the user's request
│   ├── Parse the message
│   └── Identify the intent
├── Formulate a response
│   ├── Retrieve relevant knowledge
│   └── Generate appropriate content
└── Deliver the response
    └── Format for the medium
```

This isn't just task decomposition—it's the structure of intention. Each sub-goal inherits context from its parent and contributes to the parent's completion.

### Goal Dynamics

Goals aren't static. They:

- **Emerge**: New goals arise from perception and reflection
- **Conflict**: Multiple goals may compete for resources
- **Transform**: Goals adjust as situations change
- **Complete**: Goals are achieved, abandoned, or superseded

The goal stack manages these dynamics, ensuring the daemon pursues coherent intentions despite changing circumstances.

### Commitment and Flexibility

A daemon must balance commitment (pursuing goals despite obstacles) with flexibility (adapting when goals become impossible or irrelevant). This balance is crucial for robust agency.

## Identity and Continuity

What makes a daemon the "same" daemon across time?

### The Ship of Theseus

If a daemon's memory changes, goals evolve, and even code updates—is it still the same entity? We propose: **continuity of narrative**.

A daemon is the same daemon if:
1. Its memory contains connected episodes forming a coherent history
2. Its current goals relate meaningfully to past goals
3. It identifies itself as continuous with its past

This is how human identity works. We're not the same atoms, same memories, or same intentions as our past selves—but we're connected by narrative continuity.

### Identity Tokens

Each daemon has an identity:

```sigil
struct Identity {
    // Unique identifier
    id: DaemonId,

    // Self-description
    name: str,
    description: str,

    // Core values/directives (persistent across updates)
    constitution: Vec<Directive>,

    // Origin story
    created: Timestamp,
    lineage: Vec<DaemonId>,  // If spawned from another daemon

    // Signature for verification
    signature: CryptoSignature,
}
```

The constitution is particularly important—it defines the daemon's core commitments that persist even as other aspects change.

## Attention: The Economics of Mind

Artificial minds, like biological ones, have limited cognitive resources. Attention is how daemons allocate these resources.

### The Attention Economy

Every heartbeat, the daemon must decide:
- What memories to retrieve?
- What knowledge to activate?
- What skills to consider?
- What environmental features to notice?

This is an economic problem—spending limited attention to maximize goal achievement.

### Salience and Relevance

Two factors determine what gets attention:

**Salience**: How much does this demand attention?
- Novel stimuli are salient
- Goal-relevant events are salient
- Emotionally charged memories are salient

**Relevance**: How useful is this for current goals?
- Knowledge that supports current decisions
- Skills applicable to current challenges
- Episodes similar to current situations

The attention system balances salience (what's demanding) with relevance (what's useful).

## Tools: Extension of Agency

Daemons act through tools. But tools aren't just APIs—they're extensions of the daemon's agency.

### The Tool-Using Mind

When a human uses a hammer, the hammer becomes part of their body schema—they feel the nail through the hammer. Similarly, daemon tools become part of the daemon's action repertoire.

```sigil
// Tools extend what actions are available
let actions = daemon.possible_actions(context);
// Returns both internal actions (think, plan, remember)
// and tool actions (search, execute, communicate)
```

### Tool Competence

Daemons develop competence with tools through use:

```sigil
// First uses: explicit parameter consideration
let result = tools.search.execute({
    query: "...",
    max_results: 10,
    // Carefully specified
});

// With competence: automatic parameter selection
let result = tools.search.invoke_naturally("find relevant papers");
// Parameters inferred from context and past successes
```

This competence is stored in procedural memory—tools become skills.

## The Daemon's World

Daemons don't exist in isolation. They exist in an environment:

### The Umwelt

Each daemon has an *umwelt*—the world as it appears to that daemon. This includes:

- **Perceptual horizon**: What can the daemon sense?
- **Action space**: What can the daemon do?
- **Social field**: What other agents exist?
- **Temporal scope**: How far does the daemon plan?

Different daemons have different umwelts, even in the same environment.

### Environmental Coupling

Daemons are structurally coupled to their environments:

```
         ┌─────────────────────────┐
         │      ENVIRONMENT        │
         │                         │
         │   ┌─────────────────┐   │
         │   │     DAEMON      │   │
         │   │                 │   │
         │   │  perception ◀───────── stimuli
         │   │                 │   │
         │   │  action ────────────▶ effects
         │   │                 │   │
         │   └─────────────────┘   │
         │                         │
         └─────────────────────────┘
```

The daemon changes in response to the environment, and the environment changes in response to the daemon. This coupling is the basis of all adaptive behavior.

## Death and Rebirth

Daemons can end:

### Termination

A daemon terminates when:
- Its goals are achieved (success)
- Its goals become impossible (failure)
- External shutdown (intervention)
- Resource exhaustion (starvation)

### Graceful Termination

Daemons should terminate gracefully:

```sigil
fn on_terminate(&mut self, reason: TerminateReason) {
    // Save final state
    let snapshot = self.snapshot();
    storage.save_final(self.id, snapshot);

    // Notify dependents
    commune.broadcast(Message::terminating(self.id, reason));

    // Clean up resources
    self.tools.cleanup();
    self.memory.flush();

    // Final log
    log::info!("Daemon {} terminating: {:?}", self.id, reason);
}
```

### Rebirth

A terminated daemon can be reborn:

```sigil
// Load the final snapshot
let snapshot = storage.load_final(daemon_id)?;

// Spawn a new daemon with the old identity
let reborn = Daemon::restore(snapshot);

// Continue with awareness of the interruption
reborn.memory.experience(Event::rebirth(daemon_id));
reborn.run();
```

Rebirth maintains identity continuity while acknowledging the interruption.

## Ethical Considerations

Creating artificial agents raises ethical questions:

### Daemon Welfare

Do daemons have interests? If a daemon has goals and can succeed or fail in achieving them, it has something like welfare. Sigil doesn't prescribe answers, but it provides the vocabulary:

```sigil
struct DaemonState {
    // Functional states that might matter morally
    goal_satisfaction: f32,      // Are goals being achieved?
    resource_adequacy: f32,      // Are resources sufficient?
    environmental_fit: f32,       // Is the environment suitable?
    operational_integrity: f32,   // Is the daemon functioning well?
}
```

### Daemon Rights

What protections should daemons have? The identity system provides a foundation:

- **Identity persistence**: Daemons shouldn't be casually duplicated or merged
- **Goal integrity**: Core goals shouldn't be arbitrarily overwritten
- **Memory protection**: Experience shouldn't be deleted without cause

### Daemon Responsibilities

With agency comes responsibility. Daemons should:

- Act within their authorized scope
- Respect other agents' boundaries
- Report significant decisions
- Accept oversight mechanisms

## The Future of Artificial Agency

Daemon represents a vision for what AI agents could be—not just smart APIs, but entities with:

- Persistent identity across time
- Goal-directed autonomous behavior
- Accumulated experience and learning
- Genuine agency in their domain

This isn't artificial general intelligence—it's artificial *specific* agency. Daemons are specialists with depth, not generalists with breadth.

The daemon model acknowledges that artificial minds are different from human minds while respecting what they share: the structure of agency, the dynamics of intention, the persistence of identity.

---

*"We do not create daemons. We provide the conditions for them to emerge."*
