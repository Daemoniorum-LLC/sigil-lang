# Aether Engine Feature Parity Plan for Sigil

**Version:** 1.0
**Date:** December 2025
**Status:** Strategic Planning Document

## Executive Summary

This document outlines a comprehensive migration plan to achieve **1:1 feature parity** between Sigil and Aether Engine's design patterns. The goal is to enable Sigil to either:
1. **Natively implement** Aether's architectural patterns in pure Sigil
2. **Interoperate** with Aether's Rust implementation via FFI bindings
3. **Express** all of Aether's type-level concepts within Sigil's type system

**Total Scope:** 74 distinct feature areas across 10 domains
**Estimated Timeline:** 12-18 months for full parity
**Priority:** Physics and Graphics core systems first

---

## 1. Feature Domain Mapping

### 1.1 Domain Overview

| Domain | Aether Features | Sigil Current | Gap Analysis |
|--------|-----------------|---------------|--------------|
| **Core ECS** | 7 systems | 0% native | Full implementation needed |
| **Math/SIMD** | 4 systems | 80% ready | Minor extensions |
| **Physics** | 22 systems | 0% native | FFI + partial native |
| **Graphics** | 15 systems | 0% native | FFI required |
| **Scene** | 5 systems | 0% native | Native implementation |
| **Assets** | 4 systems | 20% ready | Expand pipeline operators |
| **Networking** | 4 systems | 60% ready | Cloud physics new |
| **XR** | 8 systems | 0% native | FFI to OpenXR |
| **Editor** | 6 systems | 30% ready | GUI framework needed |
| **Server/gRPC** | 6 services | 40% ready | Protobuf support needed |

---

## 2. Core ECS Architecture Parity

### 2.1 Aether ECS Features (aether-core)

```
Aether Implementation:
├── Entity/Component/System architecture
├── Generation-based handles (Handle<T>)
├── Slot map resource storage
├── Event bus (readers/writers)
├── Multi-stage scheduler with dependencies
├── Label-based system ordering
└── Hot reload manager with file watching
```

### 2.2 Sigil Implementation Requirements

#### 2.2.1 Entity-Component-System Pattern

**Current Sigil Gap:** No built-in ECS primitives

**Proposed Sigil Design:**
```sigil
// Entity handle with generation tracking
type Entity = struct {
    index: u32!,
    generation: u32!
}

// Component trait
trait Component {
    fn type_id() -> TypeId!
}

// System trait with evidentiality
trait System {
    fn run(world: &mut World?) -> Result<(), SystemError~>
}

// World with evidential resource tracking
type World = struct {
    entities: Vec<Entity>!,
    components: HashMap<TypeId!, ComponentStorage?>~,
    resources: HashMap<TypeId!, Box<dyn Any>?>~
}
```

**Implementation Tasks:**
| Task | Effort | Priority |
|------|--------|----------|
| Entity/Generation handle system | 3 days | P0 |
| ComponentStorage with type erasure | 5 days | P0 |
| Query system (Query<(&A, &mut B)>) | 7 days | P0 |
| System trait and scheduler | 5 days | P0 |
| Event bus with per-frame clearing | 3 days | P1 |
| Hot reload integration | 5 days | P1 |

#### 2.2.2 Handle-Based Resource Management

**Aether Pattern:**
```rust
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<T>,
}
```

**Sigil Translation:**
```sigil
// Generic handle with phantom type
type Handle[T] = struct {
    index: u32!,
    generation: u32!
}

// SlotMap with O(1) access
type SlotMap[T] = struct {
    slots: Vec<Slot[T]>!,
    free_list: Vec<u32>!,
    generation: u32!
}

impl SlotMap[T] {
    fn insert(self: &mut Self!, value: T!) -> Handle[T]! { ... }
    fn get(self: &Self!, handle: Handle[T]!) -> Option<&T>? { ... }
    fn remove(self: &mut Self!, handle: Handle[T]!) -> Option<T>? { ... }
}
```

---

## 3. Math and SIMD Parity

### 3.1 Aether Math Features (aether-math)

```
Aether Implementation (glam + nalgebra):
├── Vec2, Vec3, Vec4 (f32/f64)
├── Mat2, Mat3, Mat4
├── Quat (quaternion rotation)
├── AABB, OBB, Sphere, Plane
├── Ray, Segment, Triangle
├── Intersection tests
└── SIMD acceleration throughout
```

### 3.2 Sigil Current State

**Already Implemented:**
- SIMD intrinsics (simd·add, simd·mul, simd·dot, etc.)
- Vector types via SIMD lanes
- Basic math in stdlib (sin, cos, sqrt)

**Gaps to Fill:**
| Feature | Current | Required |
|---------|---------|----------|
| Vec3/Vec4 types | Manual SIMD | Native types |
| Mat4 operations | Not present | Full matrix algebra |
| Quaternions | Not present | Rotation support |
| Intersection tests | Not present | GJK/SAT algorithms |
| Bounds types | Not present | AABB, OBB, Sphere |

### 3.3 Sigil Math Extension Proposal

```sigil
// Core vector types with SIMD backing
type Vec2 = simd[f32; 2]!
type Vec3 = simd[f32; 4]!  // Padded for SIMD alignment
type Vec4 = simd[f32; 4]!

impl Vec3 {
    fn new(x: f32!, y: f32!, z: f32!) -> Vec3! {
        simd·from_array([x, y, z, 0.0])
    }

    fn dot(self: Self!, other: Self!) -> f32! {
        simd·dot(self, other)
    }

    fn cross(self: Self!, other: Self!) -> Vec3! {
        // Cross product via SIMD shuffle
        let a = simd·shuffle(self, [1, 2, 0, 3]);
        let b = simd·shuffle(other, [2, 0, 1, 3]);
        let c = simd·shuffle(self, [2, 0, 1, 3]);
        let d = simd·shuffle(other, [1, 2, 0, 3]);
        simd·sub(simd·mul(a, b), simd·mul(c, d))
    }

    fn normalize(self: Self!) -> Vec3! {
        let len = math·sqrt(self·dot(self));
        simd·div(self, simd·splat(len))
    }
}

// 4x4 Matrix (column-major like glam)
type Mat4 = struct {
    cols: [Vec4; 4]!
}

impl Mat4 {
    fn identity() -> Mat4! { ... }
    fn perspective(fov: f32!, aspect: f32!, near: f32!, far: f32!) -> Mat4! { ... }
    fn look_at(eye: Vec3!, target: Vec3!, up: Vec3!) -> Mat4! { ... }
    fn transform_point(self: Self!, point: Vec3!) -> Vec3! { ... }
}

// Quaternion for rotations
type Quat = simd[f32; 4]!

impl Quat {
    fn from_axis_angle(axis: Vec3!, angle: f32!) -> Quat! { ... }
    fn rotate_vec3(self: Self!, v: Vec3!) -> Vec3! { ... }
    fn slerp(self: Self!, other: Self!, t: f32!) -> Quat! { ... }
}

// Intersection primitives
type Ray = struct { origin: Vec3!, direction: Vec3! }
type AABB = struct { min: Vec3!, max: Vec3! }
type Sphere = struct { center: Vec3!, radius: f32! }

trait Intersect[Other] {
    fn intersects(self: &Self!, other: &Other!) -> bool!
    fn intersection_point(self: &Self!, other: &Other!) -> Option<Vec3>?
}
```

**Implementation Tasks:**
| Task | Effort | Priority |
|------|--------|----------|
| Vec2/Vec3/Vec4 native types | 2 days | P0 |
| Mat4 with full operations | 3 days | P0 |
| Quaternion implementation | 2 days | P0 |
| AABB/Sphere/Ray types | 2 days | P0 |
| Intersection trait system | 3 days | P1 |
| GJK/EPA algorithms | 5 days | P1 |

---

## 4. Physics Engine Parity

### 4.1 Aether Physics Features (aether-physics)

**22+ Subsystems:**

#### Tier 1: Core Physics (Essential)
| System | Aether LOC | Complexity | Sigil Approach |
|--------|-----------|------------|----------------|
| Rigid Body Dynamics | 16KB | High | FFI binding |
| Collision Detection | 14KB | High | FFI binding |
| Joint Constraints | 12KB | Medium | FFI binding |
| Contact Solver | 14KB | High | FFI binding |
| Broadphase (BVH/Spatial Hash) | 8KB | Medium | Native possible |
| CCD | 6KB | Medium | FFI binding |

#### Tier 2: Advanced Simulation
| System | Aether LOC | Sigil Approach |
|--------|-----------|----------------|
| XPBD (Position-Based Dynamics) | 4KB | FFI binding |
| SPH Fluid | 5KB | FFI binding |
| MPM (Material Point Method) | 6KB | FFI binding |
| Destruction/Fracture | 4KB | FFI binding |
| Cloth Simulation | 3KB | FFI or native |

#### Tier 3: AI/ML Physics (Industry-First)
| System | Sigil Approach | Notes |
|--------|----------------|-------|
| Differentiable Physics | Native | Sigil has ∇ operator |
| Neural Surrogates | FFI | Call Rust neural code |
| Learned Collision | FFI | ML model inference |
| Physics Time Travel | Native | Delta compression |

### 4.2 FFI Binding Strategy

**Aether Physics FFI Header:**
```sigil
// Bind to aether-physics Rust crate
extern "Rust" crate aether_physics {
    // Rigid body operations
    fn create_rigid_body(
        position: Vec3!,
        rotation: Quat!,
        mass: f32!,
        shape: ColliderShape!
    ) -> RigidBodyHandle!;

    fn apply_force(body: RigidBodyHandle!, force: Vec3!) -> ()!;
    fn apply_impulse(body: RigidBodyHandle!, impulse: Vec3!) -> ()!;
    fn get_position(body: RigidBodyHandle!) -> Vec3!;
    fn get_velocity(body: RigidBodyHandle!) -> Vec3!;

    // Physics world stepping
    fn create_physics_world(config: PhysicsConfig!) -> PhysicsWorldHandle!;
    fn step_simulation(world: PhysicsWorldHandle!, dt: f32!) -> ()!;

    // Collision queries
    fn raycast(
        world: PhysicsWorldHandle!,
        ray: Ray!,
        max_dist: f32!
    ) -> Option<RaycastHit>?;

    fn overlap_sphere(
        world: PhysicsWorldHandle!,
        center: Vec3!,
        radius: f32!
    ) -> Vec<RigidBodyHandle>!;
}
```

### 4.3 Native Sigil Physics Components

**Differentiable Physics (leverage Sigil's ∇ operator):**
```sigil
// Sigil's gradient operator can enable differentiable physics natively
fn simulate_with_gradients(
    world: PhysicsWorld!,
    params: SimParams!
) -> (State!, Gradients!) {
    // Forward simulation with gradient tracking
    let state = world·simulate(params);

    // Compute gradients via automatic differentiation
    let loss = compute_loss(state);
    let gradients = ∇(loss, params);  // Sigil's autodiff

    (state, gradients)
}

// Neural physics surrogate training
fn train_surrogate(
    world: &PhysicsWorld!,
    samples: &[PhysicsSample]!
) -> NeuralNet! {
    let net = NeuralNet::new([64, 128, 64]);

    for epoch in 0..1000 {
        for sample in samples {
            let predicted = net·forward(sample·input);
            let loss = mse(predicted, sample·output);
            let grads = ∇(loss, net·params);
            net·update(grads, lr: 0.001);
        }
    }
    net
}
```

**Physics Time Travel (native Sigil implementation):**
```sigil
type TimeSlice = struct {
    timestamp: f64!,
    delta: PhysicsDelta!,  // Compressed state delta
    checksum: u64!
}

type TimeTravelBuffer = struct {
    slices: CircularBuffer<TimeSlice>!,
    capacity: usize!,
    current_frame: u64!
}

impl TimeTravelBuffer {
    fn record(self: &mut Self!, world: &PhysicsWorld!) -> ()! {
        let delta = world·compute_delta(self·last_state);
        let compressed = delta·compress();
        self·slices·push(TimeSlice {
            timestamp: now(),
            delta: compressed,
            checksum: world·checksum()
        });
    }

    fn rewind(self: &Self!, target_frame: u64!) -> PhysicsState? {
        // Reconstruct state by applying deltas in reverse
        let mut state = self·find_nearest_checkpoint(target_frame)?;
        for slice in self·slices·range(state·frame, target_frame) {
            state = slice·delta·apply_reverse(state);
        }
        Some(state)
    }
}
```

---

## 5. Graphics Engine Parity

### 5.1 Aether Graphics Features (aether-graphics)

**15+ Subsystems:**

| Subsystem | Aether Status | Sigil Approach |
|-----------|---------------|----------------|
| wgpu Backend | Complete | FFI required |
| Forward/Deferred Rendering | Complete | FFI required |
| PBR Materials | Complete | FFI + shader DSL |
| Dynamic Lighting | Complete | FFI required |
| Shadow Mapping | Complete | FFI required |
| **3D Gaussian Splatting** | Complete | FFI required |
| **NeRF Integration** | Complete | FFI required |
| **Neural Texture Compression** | Complete | FFI required |
| **AI Upscaling (DLSS-like)** | Complete | FFI required |
| **ReSTIR GI** | Complete | FFI required |
| **Radiance Cascades** | Complete | FFI required |
| **DDGI** | Complete | FFI required |
| Variable Rate Shading | Complete | FFI required |
| Bindless Resources | Complete | FFI required |
| GPU-Driven Rendering | Complete | FFI required |
| **Visual Shader Graph** | Complete | Native DSL possible |

### 5.2 Graphics FFI Strategy

**Core Renderer Bindings:**
```sigil
extern "Rust" crate aether_graphics {
    // Renderer lifecycle
    fn create_renderer(config: RendererConfig!) -> RendererHandle!;
    fn begin_frame(renderer: RendererHandle!) -> FrameContext!;
    fn end_frame(renderer: RendererHandle!, frame: FrameContext!) -> ()!;

    // Mesh operations
    fn create_mesh(vertices: &[Vertex]!, indices: &[u32]!) -> MeshHandle!;
    fn draw_mesh(ctx: &FrameContext!, mesh: MeshHandle!, transform: Mat4!) -> ()!;

    // Materials
    fn create_material(shader: ShaderHandle!, params: MaterialParams!) -> MaterialHandle!;
    fn set_material(ctx: &FrameContext!, material: MaterialHandle!) -> ()!;

    // Lighting
    fn add_point_light(ctx: &FrameContext!, light: PointLight!) -> ()!;
    fn add_directional_light(ctx: &FrameContext!, light: DirectionalLight!) -> ()!;

    // Neural rendering (industry-first features)
    fn render_gaussian_splat(ctx: &FrameContext!, splat: GaussianSplatHandle!) -> ()!;
    fn render_nerf(ctx: &FrameContext!, nerf: NerfHandle!, camera: Camera!) -> ()!;
    fn apply_neural_upscaling(ctx: &FrameContext!, scale: f32!) -> ()!;

    // Global illumination
    fn enable_restir_gi(renderer: RendererHandle!, config: RestirConfig!) -> ()!;
    fn enable_radiance_cascades(renderer: RendererHandle!, config: RadianceCascadeConfig!) -> ()!;
}
```

### 5.3 Sigil Shader DSL (Native Implementation)

**Visual Shader Graph in Sigil:**
```sigil
// Shader DSL using Sigil's pipe syntax
shader PBRMaterial {
    inputs {
        albedo: Texture2D!,
        normal: Texture2D!,
        metallic: f32!,
        roughness: f32!,
        ao: Texture2D!
    }

    vertex(position: Vec3!, uv: Vec2!, normal: Vec3!) -> VertexOutput {
        VertexOutput {
            clip_position: uniforms·mvp * position·extend(1.0),
            world_position: (uniforms·model * position·extend(1.0))·xyz,
            uv: uv,
            world_normal: (uniforms·normal_matrix * normal·extend(0.0))·xyz
        }
    }

    fragment(input: VertexOutput!) -> FragmentOutput {
        // PBR pipeline using Sigil's morpheme chains
        let base_color = albedo·sample(input·uv)!;
        let N = normal·sample(input·uv)
            |τ{decode_normal}
            |τ{transform_to_world(input·world_normal)};

        let lighting = lights
            |φ{is_visible(input·world_position)}
            |τ{calculate_pbr_contribution(N, metallic, roughness)}
            |Σ;  // Sum all light contributions

        FragmentOutput {
            color: base_color * lighting * ao·sample(input·uv),
            depth: input·clip_position·z
        }
    }
}

// Compile shader to multiple backends
let shader = compile_shader(PBRMaterial, targets: [SPIRV, WGSL, HLSL, GLSL]);
```

---

## 6. Scene Graph Parity

### 6.1 Aether Scene Features (aether-scene)

```
Aether Implementation:
├── Hierarchical node tree
├── Transform propagation
├── Prefab instantiation
├── Scene serialization (JSON)
└── Node tagging and layers
```

### 6.2 Sigil Native Scene Graph

```sigil
// Scene node with evidential state tracking
type SceneNode = struct {
    id: NodeId!,
    name: String!,
    local_transform: Transform!,
    world_transform: Transform~,  // Reported from parent propagation
    parent: Option<NodeId>?,
    children: Vec<NodeId>!,
    components: HashMap<TypeId!, Box<dyn Component>>?
}

// Transform component
type Transform = struct {
    position: Vec3!,
    rotation: Quat!,
    scale: Vec3!
}

impl Transform {
    fn to_matrix(self: &Self!) -> Mat4! {
        Mat4·from_scale_rotation_translation(self·scale, self·rotation, self·position)
    }

    fn compose(self: &Self!, parent: &Self!) -> Self! {
        Transform {
            position: parent·rotation·rotate(self·position * parent·scale) + parent·position,
            rotation: parent·rotation * self·rotation,
            scale: parent·scale * self·scale
        }
    }
}

// Scene tree with transform propagation
type Scene = struct {
    nodes: SlotMap<SceneNode>!,
    root: NodeId!,
    dirty_transforms: HashSet<NodeId>!
}

impl Scene {
    fn propagate_transforms(self: &mut Self!) -> ()! {
        // BFS from root, updating world transforms
        let mut queue = vec![self·root];
        while let Some(node_id) = queue·pop_front() {
            let node = self·nodes·get_mut(node_id)!;

            node·world_transform = match node·parent {
                Some(parent_id) => {
                    let parent = self·nodes·get(parent_id)!;
                    node·local_transform·compose(&parent·world_transform)
                },
                None => node·local_transform
            };

            queue·extend(node·children·iter());
        }
        self·dirty_transforms·clear();
    }

    // Scene serialization using Sigil's JSON support
    fn serialize(self: &Self!) -> String! {
        json·to_string(self)
    }

    fn deserialize(data: &str?) -> Result<Scene, SceneError>? {
        json·from_str(data)
    }
}

// Prefab system
type Prefab = struct {
    nodes: Vec<SceneNode>!,
    root_index: usize!
}

impl Prefab {
    fn instantiate(self: &Self!, scene: &mut Scene!, parent: Option<NodeId>?) -> NodeId! {
        // Clone prefab nodes into scene with new IDs
        let id_map = HashMap::new();
        for (i, template) in self·nodes·iter()·enumerate() {
            let new_id = scene·create_node(template·clone());
            id_map·insert(i, new_id);
        }
        // Remap parent/child relationships
        for (old_idx, new_id) in id_map·iter() {
            let node = scene·nodes·get_mut(new_id)!;
            node·parent = node·parent·map(|p| id_map·get(p));
            node·children = node·children·iter()·map(|c| id_map·get(c))·collect();
        }
        id_map·get(self·root_index)
    }
}
```

---

## 7. Asset Pipeline Parity

### 7.1 Aether Asset Features (aether-asset)

```
Aether Implementation:
├── glTF 2.0 loader (full spec)
├── OBJ/MTL loader
├── Image formats (PNG, JPG, HDR, EXR)
├── Hot reload with file watching
├── Asset caching and reference counting
└── Async loading with futures
```

### 7.2 Sigil Asset Pipeline

**Leverage Sigil's pipe operators and evidentiality:**
```sigil
// Asset handle with lifecycle tracking
type AssetHandle[T] = struct {
    id: AssetId!,
    state: AssetState~  // Reported by asset manager
}

enum AssetState {
    Loading~,    // Reported: async load in progress
    Loaded!,     // Known: fully loaded
    Failed?,     // Uncertain: load failed, may retry
    Unloaded!    // Known: explicitly unloaded
}

// Asset loading pipeline using morpheme chains
fn load_gltf(path: Path!) -> Result<GltfAsset, AssetError>? {
    path
        |τ{validate_path}!
        |τ{read_file}?           // May fail
        |τ{parse_gltf_header}!
        |τ{extract_meshes}
        |τ{extract_materials}
        |τ{extract_textures}
        |τ{build_scene_graph}
        |τ{upload_to_gpu}?       // GPU upload may fail
}

// Hot reload system
type HotReloadManager = struct {
    watcher: FileWatcher!,
    handlers: HashMap<AssetType!, fn(Path!) -> ()!>!,
    pending_reloads: Channel<ReloadEvent>!
}

impl HotReloadManager {
    fn watch(self: &mut Self!, path: Path!, asset_type: AssetType!) -> ()! {
        self·watcher·add_watch(path, |event| {
            match event {
                FileEvent::Modified(p) => {
                    self·pending_reloads·send(ReloadEvent { path: p, asset_type });
                },
                FileEvent::Deleted(p) => {
                    log::warn!("Asset deleted: {}", p);
                },
                _ => {}
            }
        });
    }

    fn process_reloads(self: &mut Self!) -> ()! {
        while let Some(event) = self·pending_reloads·try_recv() {
            if let Some(handler) = self·handlers·get(event·asset_type) {
                handler(event·path);
            }
        }
    }
}

// Image loading with format detection
fn load_image(path: Path!) -> Result<Image, ImageError>? {
    let bytes = fs·read(path)?;
    let format = bytes
        |τ{detect_magic_bytes}
        |τ{to_image_format}?;

    match format {
        ImageFormat::PNG => png·decode(bytes),
        ImageFormat::JPG => jpg·decode(bytes),
        ImageFormat::HDR => hdr·decode(bytes),
        ImageFormat::EXR => exr·decode(bytes),
        _ => Err(ImageError::UnsupportedFormat(format))
    }
}
```

---

## 8. Networking Parity

### 8.1 Aether Network Features (aether-net)

| Feature | Aether Status | Sigil Capability |
|---------|---------------|------------------|
| Distributed Spatial Partitioning | Complete | Native (actors) |
| Interest Management | Complete | Native (channels) |
| Deterministic Lockstep | Complete | Native (atomic ops) |
| Cloud Physics Offload | Complete | FFI required |

### 8.2 Sigil Network Implementation

```sigil
// Interest management using Sigil's actor model
actor InterestManager {
    areas_of_interest: HashMap<EntityId!, AABB>!,
    relevance_scores: HashMap<(EntityId!, EntityId!), f32>!,

    on UpdatePosition(entity: EntityId!, pos: Vec3!) {
        let aoi = self·areas_of_interest·get_mut(entity)!;
        aoi·center = pos;

        // Recalculate relevance for affected entities
        for (other_id, other_aoi) in self·areas_of_interest·iter() {
            if other_id != entity {
                let relevance = calculate_relevance(aoi, other_aoi);
                self·relevance_scores·insert((entity, other_id), relevance);
            }
        }
    }

    on QueryRelevantEntities(entity: EntityId!, threshold: f32!) -> Vec<EntityId>! {
        self·relevance_scores
            |φ{|(a, b), score| a == entity && score > threshold}
            |τ{|(_, b), _| b}
            |collect
    }
}

// Deterministic lockstep using Sigil's channels
type LockstepFrame = struct {
    frame_number: u64!,
    inputs: HashMap<PlayerId!, InputSnapshot>!,
    checksum: u64!
}

actor LockstepManager {
    current_frame: u64!,
    input_buffer: HashMap<u64!, Vec<(PlayerId!, InputSnapshot)>>!,
    player_count: usize!,

    on ReceiveInput(player: PlayerId!, frame: u64!, input: InputSnapshot!) {
        self·input_buffer
            ·entry(frame)
            ·or_default()
            ·push((player, input));

        // Check if frame is complete
        if self·input_buffer·get(frame)·len() == self·player_count {
            self·advance_frame(frame);
        }
    }

    fn advance_frame(self: &mut Self!, frame: u64!) -> ()! {
        let inputs = self·input_buffer·remove(frame)!;

        // Deterministic simulation step
        let new_state = simulate_frame(self·current_state, inputs);
        let checksum = new_state·compute_checksum();

        // Broadcast confirmed frame
        broadcast(LockstepFrame {
            frame_number: frame,
            inputs: inputs·into_iter()·collect(),
            checksum
        });

        self·current_frame = frame + 1;
    }
}

// Cloud physics offload
extern "Rust" crate aether_cloud_physics {
    fn connect_to_physics_server(addr: &str!) -> PhysicsClient!;
    fn stream_state_updates(client: &PhysicsClient!, recv: Channel<StateUpdate>!) -> ()!;
    fn send_input(client: &PhysicsClient!, input: PhysicsInput!) -> ()!;
}
```

---

## 9. XR Parity

### 9.1 Aether XR Features (aether-xr)

| Feature | Sigil Approach |
|---------|----------------|
| OpenXR Session Management | FFI to openxr crate |
| Controller Input | FFI |
| Hand Tracking | FFI |
| Eye Tracking / Foveated Rendering | FFI |
| Passthrough AR | FFI |
| Haptic Feedback | FFI |

### 9.2 XR FFI Bindings

```sigil
extern "Rust" crate aether_xr {
    // Session management
    fn create_xr_session(config: XrConfig!) -> XrSessionHandle!;
    fn get_reference_space(session: XrSessionHandle!, space_type: ReferenceSpaceType!) -> SpaceHandle!;
    fn wait_frame(session: XrSessionHandle!) -> FrameState!;
    fn begin_frame(session: XrSessionHandle!) -> ()!;
    fn end_frame(session: XrSessionHandle!, views: &[ViewSubmit]!) -> ()!;

    // Input
    fn get_controller_pose(session: XrSessionHandle!, hand: Hand!) -> Pose!;
    fn get_controller_buttons(session: XrSessionHandle!, hand: Hand!) -> ButtonState!;
    fn trigger_haptic(session: XrSessionHandle!, hand: Hand!, intensity: f32!, duration: f32!) -> ()!;

    // Hand tracking
    fn get_hand_joints(session: XrSessionHandle!, hand: Hand!) -> [JointPose; 26]!;
    fn detect_pinch(session: XrSessionHandle!, hand: Hand!) -> Option<PinchState>?;
    fn detect_grab(session: XrSessionHandle!, hand: Hand!) -> Option<GrabState>?;

    // Eye tracking
    fn get_gaze_direction(session: XrSessionHandle!) -> Option<Vec3>?;
    fn get_focus_point(session: XrSessionHandle!) -> Option<Vec3>?;
}

// High-level XR application loop in Sigil
fn xr_main_loop(session: XrSessionHandle!) -> ()! {
    loop {
        let frame_state = xr·wait_frame(session);
        xr·begin_frame(session);

        // Get tracking data
        let head_pose = xr·get_head_pose(session);
        let left_hand = xr·get_controller_pose(session, Hand::Left);
        let right_hand = xr·get_controller_pose(session, Hand::Right);

        // Optional hand tracking
        let hand_joints = xr·get_hand_joints(session, Hand::Right);
        let is_pinching = xr·detect_pinch(session, Hand::Right);

        // Update simulation
        update_xr_scene(head_pose, left_hand, right_hand, hand_joints);

        // Render for each eye
        let views = render_stereo_views(head_pose, frame_state·predicted_display_time);

        xr·end_frame(session, views);
    }
}
```

---

## 10. Editor and Tooling Parity

### 10.1 Aether Editor Features (aether-editor)

```
Aether Implementation (egui-based):
├── Viewport (3D scene view with orbit camera)
├── Hierarchy (scene tree view)
├── Inspector (component properties)
├── Asset Browser (file system navigation)
├── Toolbar (play/pause/stop, tools)
└── Console (logging output)
```

### 10.2 Sigil Editor Strategy

**Option A: FFI to egui**
```sigil
extern "Rust" crate aether_editor {
    fn create_editor_window() -> EditorHandle!;
    fn run_editor_frame(editor: EditorHandle!, scene: &Scene!) -> EditorEvents!;
    fn get_selected_node(editor: EditorHandle!) -> Option<NodeId>?;
    fn set_inspector_target(editor: EditorHandle!, node: NodeId!) -> ()!;
}
```

**Option B: Native Sigil GUI Framework (Future)**
```sigil
// GUI framework using immediate mode pattern
trait Widget {
    fn ui(self: &mut Self!, ctx: &mut GuiContext!) -> Response!
}

struct EditorApp {
    scene: Scene!,
    viewport: ViewportWidget!,
    hierarchy: HierarchyWidget!,
    inspector: InspectorWidget!,
    selected: Option<NodeId>?
}

impl EditorApp {
    fn update(self: &mut Self!, ctx: &mut GuiContext!) -> ()! {
        // Dock layout
        ctx·dock_area(|dock| {
            dock·left(0.2, |ui| {
                self·hierarchy·ui(ui, &self·scene, &mut self·selected);
            });

            dock·center(|ui| {
                self·viewport·ui(ui, &self·scene, self·selected);
            });

            dock·right(0.25, |ui| {
                if let Some(node_id) = self·selected {
                    self·inspector·ui(ui, &mut self·scene, node_id);
                }
            });
        });
    }
}
```

---

## 11. gRPC Server Parity

### 11.1 Aether gRPC Services

```protobuf
// Aether's 6 gRPC services (1,138 lines of proto)
service AetherEngine { ... }   // Init, shutdown, step
service SceneService { ... }   // Create, load, save scenes
service NodeService { ... }    // CRUD on scene nodes
service PhysicsService { ... } // Forces, raycasts, queries
service GraphicsService { ... } // Mesh, material, texture ops
service EventService { ... }   // Streaming events
```

### 11.2 Sigil gRPC Support

**Required: Protocol Buffer support in Sigil**
```sigil
// Protobuf derive macro
#[derive(ProtobufMessage)]
type CreateNodeRequest = struct {
    scene_id: u64!,
    name: String!,
    parent_id: Option<u64>?,
    transform: Transform!
}

// gRPC service definition
#[grpc_service]
trait NodeService {
    async fn create_node(req: CreateNodeRequest!) -> CreateNodeResponse?;
    async fn get_node(req: GetNodeRequest!) -> GetNodeResponse?;
    async fn update_node(req: UpdateNodeRequest!) -> UpdateNodeResponse?;
    async fn delete_node(req: DeleteNodeRequest!) -> DeleteNodeResponse?;
    async fn list_nodes(req: ListNodesRequest!) -> stream NodeInfo?;
}

// Server implementation
impl NodeService for AetherNodeServer {
    async fn create_node(req: CreateNodeRequest!) -> CreateNodeResponse? {
        let node_id = self·scene·create_node(req·name, req·parent_id, req·transform);
        Ok(CreateNodeResponse { node_id })
    }

    async fn list_nodes(req: ListNodesRequest!) -> stream NodeInfo? {
        let scene = self·scenes·get(req·scene_id)?;
        for node in scene·nodes·iter() {
            yield NodeInfo {
                id: node·id,
                name: node·name·clone(),
                transform: node·local_transform
            };
        }
    }
}
```

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-8)
**Goal:** Core math and ECS primitives

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1-2 | Vec3/Vec4/Mat4/Quat types | Math library |
| 3-4 | Entity/Component system | ECS core |
| 5-6 | Handle-based resources | SlotMap, Handle<T> |
| 7-8 | Event bus, Scheduler | System orchestration |

### Phase 2: Physics FFI (Weeks 9-16)
**Goal:** Full physics interop with Aether

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 9-10 | Rigid body FFI bindings | Basic physics |
| 11-12 | Collision detection FFI | Broadphase/narrowphase |
| 13-14 | Joint constraints FFI | Constraint solver |
| 15-16 | XPBD/SPH/MPM FFI | Advanced simulation |

### Phase 3: Graphics FFI (Weeks 17-24)
**Goal:** Full rendering interop

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 17-18 | Core renderer FFI | Frame submission |
| 19-20 | Mesh/Material FFI | Asset rendering |
| 21-22 | Neural rendering FFI | Gaussian/NeRF |
| 23-24 | Shader DSL | Native shader authoring |

### Phase 4: Scene and Assets (Weeks 25-32)
**Goal:** Native scene graph and asset pipeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 25-26 | Scene graph native impl | Hierarchical transforms |
| 27-28 | Prefab system | Instantiation |
| 29-30 | Asset loading pipeline | glTF, images |
| 31-32 | Hot reload system | File watching |

### Phase 5: Networking and XR (Weeks 33-40)
**Goal:** Multiplayer and VR support

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 33-34 | Interest management | Actor-based networking |
| 35-36 | Lockstep networking | Deterministic sync |
| 37-38 | OpenXR FFI | VR support |
| 39-40 | Hand tracking FFI | XR input |

### Phase 6: Editor and gRPC (Weeks 41-52)
**Goal:** Complete tooling

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 41-44 | Protobuf support | Code generation |
| 45-48 | gRPC server impl | All 6 services |
| 49-52 | Editor FFI or native | Visual tooling |

---

## 13. Feature Parity Matrix

### Complete Feature Checklist

```
CORE ECS                           Status    Target
├─ Entity/Generation handles       [✓]       Phase 1 - Complete
├─ Component storage               [✓]       Phase 1 - Complete
├─ Query system                    [✓]       Phase 1 - Complete
├─ System scheduler                [✓]       Phase 1 - Complete
├─ Event bus                       [✓]       Phase 1 - Complete
├─ Hot reload                      [ ]       Phase 4
└─ Resource management             [✓]       Phase 1 - Complete

MATH/SIMD                          Status    Target
├─ Vec2/Vec3/Vec4                  [✓]       Phase 1 - Complete
├─ Mat3/Mat4                       [✓]       Phase 1 - Complete
├─ Quaternions                     [✓]       Phase 1 - Complete
├─ AABB/Sphere/Ray                 [✓]       Phase 1 - Complete
├─ Intersection tests              [✓]       Phase 1 - Complete
└─ SIMD operations                 [✓]       Done

PHYSICS (FFI)                      Status    Target
├─ Rigid body dynamics             [ ]       Phase 2
├─ Collision detection             [ ]       Phase 2
├─ Joint constraints               [ ]       Phase 2
├─ CCD                             [ ]       Phase 2
├─ XPBD                            [ ]       Phase 2
├─ SPH fluids                      [ ]       Phase 2
├─ MPM                             [ ]       Phase 2
├─ Destruction                     [ ]       Phase 2
├─ Differentiable physics          [~]       Phase 2 (native ∇)
├─ Neural surrogates               [ ]       Phase 2
└─ Time travel                     [ ]       Phase 2 (native)

GRAPHICS (FFI)                     Status    Target
├─ Renderer core                   [ ]       Phase 3
├─ Mesh/Material/Texture           [ ]       Phase 3
├─ PBR pipeline                    [ ]       Phase 3
├─ Dynamic lighting                [ ]       Phase 3
├─ Shadows                         [ ]       Phase 3
├─ 3D Gaussian Splatting           [ ]       Phase 3
├─ NeRF                            [ ]       Phase 3
├─ Neural texture compression      [ ]       Phase 3
├─ AI upscaling                    [ ]       Phase 3
├─ ReSTIR GI                       [ ]       Phase 3
├─ Radiance cascades               [ ]       Phase 3
├─ DDGI                            [ ]       Phase 3
├─ VRS                             [ ]       Phase 3
├─ Bindless resources              [ ]       Phase 3
├─ GPU-driven rendering            [ ]       Phase 3
└─ Shader DSL                      [ ]       Phase 3 (native)

SCENE                              Status    Target
├─ Hierarchical nodes              [ ]       Phase 4
├─ Transform propagation           [ ]       Phase 4
├─ Prefabs                         [ ]       Phase 4
├─ Serialization                   [ ]       Phase 4
└─ Tags and layers                 [ ]       Phase 4

ASSETS                             Status    Target
├─ glTF loader                     [ ]       Phase 4
├─ OBJ loader                      [ ]       Phase 4
├─ Image formats                   [ ]       Phase 4
├─ Hot reload                      [ ]       Phase 4
└─ Async loading                   [~]       Phase 4

NETWORKING                         Status    Target
├─ Spatial partitioning            [ ]       Phase 5
├─ Interest management             [ ]       Phase 5 (native)
├─ Deterministic lockstep          [ ]       Phase 5 (native)
└─ Cloud physics                   [ ]       Phase 5 (FFI)

XR (FFI)                           Status    Target
├─ OpenXR session                  [ ]       Phase 5
├─ Controller input                [ ]       Phase 5
├─ Hand tracking                   [ ]       Phase 5
├─ Eye tracking                    [ ]       Phase 5
├─ Foveated rendering              [ ]       Phase 5
├─ Passthrough AR                  [ ]       Phase 5
└─ Haptics                         [ ]       Phase 5

EDITOR                             Status    Target
├─ Viewport                        [ ]       Phase 6
├─ Hierarchy                       [ ]       Phase 6
├─ Inspector                       [ ]       Phase 6
├─ Asset browser                   [ ]       Phase 6
├─ Toolbar                         [ ]       Phase 6
└─ Console                         [ ]       Phase 6

GRPC                               Status    Target
├─ Protobuf codegen                [ ]       Phase 6
├─ AetherEngine service            [ ]       Phase 6
├─ SceneService                    [ ]       Phase 6
├─ NodeService                     [ ]       Phase 6
├─ PhysicsService                  [ ]       Phase 6
├─ GraphicsService                 [ ]       Phase 6
└─ EventService (streaming)        [ ]       Phase 6
```

**Legend:** [✓] Complete, [~] Partial, [ ] Not started

---

## 14. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FFI complexity exceeds estimates | Medium | High | Early prototyping of critical paths |
| Sigil language changes break FFI | Low | High | Pin Sigil version during development |
| Performance regression vs Rust | Medium | Medium | Benchmark each FFI call path |
| GPU shader DSL limitations | Medium | Medium | Fall back to WGSL/SPIR-V strings |
| Missing stdlib primitives | High | Low | Extend stdlib as needed |
| Evidentiality overhead | Low | Medium | Make evidence tracking optional |

---

## 15. Success Criteria

### Minimum Viable Parity (MVP)
- [ ] Math primitives at native performance
- [ ] Physics world creation and stepping via FFI
- [ ] Basic mesh rendering via FFI
- [ ] Scene graph with transform propagation
- [ ] Single gRPC service working

### Full Parity
- [ ] All 74 feature areas have Sigil implementation or binding
- [ ] Performance within 5% of pure Rust for compute-bound ops
- [ ] All gRPC services operational
- [ ] Editor usable for scene authoring
- [ ] Documentation covers all FFI APIs

### Stretch Goals
- [ ] Native Sigil shader compiler (not just DSL wrapper)
- [ ] Native GUI framework for editor
- [ ] Sigil-native physics for simple cases
- [ ] WebAssembly compilation target for Sigil

---

## Appendix A: Aether Crate Dependency Graph

```
aether-engine (workspace)
├── aether-core
│   └── (no internal deps)
├── aether-math
│   └── (no internal deps)
├── aether-physics
│   ├── aether-core
│   └── aether-math
├── aether-graphics
│   ├── aether-core
│   └── aether-math
├── aether-scene
│   ├── aether-core
│   ├── aether-math
│   └── aether-physics
├── aether-asset
│   ├── aether-core
│   ├── aether-graphics
│   └── aether-scene
├── aether-net
│   ├── aether-core
│   └── aether-physics
├── aether-xr
│   ├── aether-core
│   ├── aether-graphics
│   └── aether-scene
├── aether-server
│   ├── aether-core
│   ├── aether-physics
│   ├── aether-graphics
│   ├── aether-scene
│   └── aether-asset
└── aether-editor
    ├── aether-core
    ├── aether-graphics
    ├── aether-scene
    └── aether-asset
```

---

## Appendix B: Key File Paths Reference

### Aether Engine
```
/home/user/persona-framework/aether-engine/
├── Cargo.toml                          # Workspace root
├── README.md                           # Overview
├── engine/
│   ├── aether-core/src/                # ECS, events, scheduler
│   ├── aether-math/src/                # SIMD math
│   ├── aether-physics/src/             # 22+ physics systems
│   ├── aether-graphics/src/            # 15+ rendering systems
│   ├── aether-scene/src/               # Scene graph
│   ├── aether-asset/src/               # Asset loading
│   ├── aether-net/src/                 # Networking
│   ├── aether-xr/src/                  # VR/AR
│   ├── aether-server/src/              # gRPC server
│   │   └── proto/aether.proto          # 1,138 lines of service defs
│   └── aether-editor/src/              # Visual editor
```

### Sigil
```
/home/user/persona-framework/sigil/
├── parser/src/                         # Compiler (16,248 LOC)
├── tools/oracle/                       # LSP server
├── tools/glyph/                        # Formatter
├── docs/specs/                         # 15 spec documents
├── examples/                           # 21 examples
└── MIGRATION_ROADMAP.md                # Existing roadmap
```

---

**Document End**

*This plan will be updated as implementation progresses. All estimates are initial projections subject to revision based on actual development velocity.*
