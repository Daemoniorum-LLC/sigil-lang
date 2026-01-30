// Rust Feature Benchmark - Direct comparison with Sigil
// Run with: cargo run --release

use std::time::Instant;
use std::hint::black_box;

fn bench(name: &str, iterations: usize, start: Instant) {
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let per_op = elapsed / iterations as f64 * 1000.0;
    println!("  {}: {:.3}ms total, {:.4}μs/op", name, elapsed, per_op);
}

// ============================================================================
// GRAPHICS MATH
// ============================================================================

#[derive(Clone, Copy)]
struct Vec3 { x: f64, y: f64, z: f64 }

#[derive(Clone, Copy)]
struct Vec4 { x: f64, y: f64, z: f64, w: f64 }

#[derive(Clone, Copy)]
struct Mat4 { m: [f64; 16] }

#[derive(Clone, Copy)]
struct Quat { x: f64, y: f64, z: f64, w: f64 }

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z } }
    fn dot(self, other: Self) -> f64 { self.x * other.x + self.y * other.y + self.z * other.z }
    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    fn normalize(self) -> Self {
        let len = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }
}

impl Mat4 {
    fn identity() -> Self {
        Self { m: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] }
    }
    fn translate(x: f64, y: f64, z: f64) -> Self {
        Self { m: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0] }
    }
    fn mul(self, other: Self) -> Self {
        let mut result = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i * 4 + j] += self.m[i * 4 + k] * other.m[k * 4 + j];
                }
            }
        }
        Self { m: result }
    }
}

impl Quat {
    fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        Self { x: axis.x * s, y: axis.y * s, z: axis.z * s, w: half.cos() }
    }
    fn rotate(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        let dot_uv = u.dot(v);
        let dot_uu = u.dot(u);
        let cross_uv = u.cross(v);
        Vec3 {
            x: 2.0 * dot_uv * u.x + (s * s - dot_uu) * v.x + 2.0 * s * cross_uv.x,
            y: 2.0 * dot_uv * u.y + (s * s - dot_uu) * v.y + 2.0 * s * cross_uv.y,
            z: 2.0 * dot_uv * u.z + (s * s - dot_uu) * v.z + 2.0 * s * cross_uv.z,
        }
    }
    fn slerp(self, other: Self, t: f64) -> Self {
        let mut dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
        let other = if dot < 0.0 { dot = -dot; Quat { x: -other.x, y: -other.y, z: -other.z, w: -other.w } } else { other };
        if dot > 0.9995 {
            return Quat {
                x: self.x + t * (other.x - self.x),
                y: self.y + t * (other.y - self.y),
                z: self.z + t * (other.z - self.z),
                w: self.w + t * (other.w - self.w),
            };
        }
        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();
        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;
        Quat {
            x: s0 * self.x + s1 * other.x,
            y: s0 * self.y + s1 * other.y,
            z: s0 * self.z + s1 * other.z,
            w: s0 * self.w + s1 * other.w,
        }
    }
}

fn benchmark_graphics_math() {
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│  GRAPHICS MATH (Vectors, Matrices, Quaternions)              │");
    println!("└───────────────────────────────────────────────────────────────┘");

    let n = 10000;
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);

    let start = Instant::now();
    for _ in 0..n { black_box(Vec3::new(1.0, 2.0, 3.0)); }
    bench("vec3 create", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(a.add(b)); }
    bench("vec3_add", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(a.dot(b)); }
    bench("vec3_dot", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(a.cross(b)); }
    bench("vec3_cross", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(a.normalize()); }
    bench("vec3_normalize", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(Mat4::identity()); }
    bench("mat4_identity", n, start);

    let m1 = Mat4::identity();
    let m2 = Mat4::translate(1.0, 2.0, 3.0);
    let start = Instant::now();
    for _ in 0..n { black_box(m1.mul(m2)); }
    bench("mat4_mul", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5)); }
    bench("quat_from_axis_angle", n, start);

    let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5);
    let start = Instant::now();
    for _ in 0..n { black_box(q.rotate(a)); }
    bench("quat_rotate", n, start);

    let q1 = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3);
    let q2 = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5);
    let start = Instant::now();
    for _ in 0..n { black_box(q1.slerp(q2, 0.5)); }
    bench("quat_slerp", n, start);

    println!();
}

// ============================================================================
// SIMD OPERATIONS
// ============================================================================

fn simd_add(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut r = [0.0; 8];
    for i in 0..8 { r[i] = a[i] + b[i]; }
    r
}

fn simd_mul(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut r = [0.0; 8];
    for i in 0..8 { r[i] = a[i] * b[i]; }
    r
}

fn simd_dot(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    let mut sum = 0.0;
    for i in 0..8 { sum += a[i] * b[i]; }
    sum
}

fn simd_hadd(a: &[f64; 8]) -> f64 {
    a.iter().sum()
}

fn benchmark_simd() {
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│  SIMD OPERATIONS (8-element arrays)                          │");
    println!("└───────────────────────────────────────────────────────────────┘");

    let n = 5000;
    let arr_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let arr_b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    let start = Instant::now();
    for _ in 0..n { black_box(simd_add(&arr_a, &arr_b)); }
    bench("simd_add", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(simd_mul(&arr_a, &arr_b)); }
    bench("simd_mul", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(simd_dot(&arr_a, &arr_b)); }
    bench("simd_dot", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(simd_hadd(&arr_a)); }
    bench("simd_hadd (sum)", n, start);

    println!();
}

// ============================================================================
// TENSOR OPERATIONS
// ============================================================================

fn outer_product(a: &[f64; 4], b: &[f64; 4]) -> [[f64; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = a[i] * b[j];
        }
    }
    result
}

fn hadamard_product(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

fn trace(mat: &[f64; 4], size: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..size { sum += mat[i * size + i]; }
    sum
}

fn benchmark_tensor() {
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│  TENSOR OPERATIONS                                           │");
    println!("└───────────────────────────────────────────────────────────────┘");

    let n = 5000;
    let vec_a = [1.0, 2.0, 3.0, 4.0];
    let vec_b = [5.0, 6.0, 7.0, 8.0];

    let start = Instant::now();
    for _ in 0..n { black_box(outer_product(&vec_a, &vec_b)); }
    bench("outer_product (4x4)", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(hadamard_product(&vec_a, &vec_b)); }
    bench("hadamard_product", n, start);

    let mat_flat = [1.0, 2.0, 3.0, 4.0];
    let start = Instant::now();
    for _ in 0..n { black_box(trace(&mat_flat, 2)); }
    bench("trace (2x2)", n, start);

    println!();
}

// ============================================================================
// PHYSICS
// ============================================================================

fn verlet_integrate(pos: Vec3, prev: Vec3, accel: Vec3, dt: f64) -> Vec3 {
    Vec3 {
        x: 2.0 * pos.x - prev.x + accel.x * dt * dt,
        y: 2.0 * pos.y - prev.y + accel.y * dt * dt,
        z: 2.0 * pos.z - prev.z + accel.z * dt * dt,
    }
}

fn spring_force(p1: Vec3, p2: Vec3, rest_length: f64, stiffness: f64) -> Vec3 {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let dz = p2.z - p1.z;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    let displacement = dist - rest_length;
    let force = stiffness * displacement;
    if dist > 0.0001 {
        Vec3 { x: force * dx / dist, y: force * dy / dist, z: force * dz / dist }
    } else {
        Vec3::new(0.0, 0.0, 0.0)
    }
}

fn ray_sphere_intersect(origin: Vec3, dir: Vec3, center: Vec3, radius: f64) -> f64 {
    let oc = Vec3 { x: origin.x - center.x, y: origin.y - center.y, z: origin.z - center.z };
    let a = dir.dot(dir);
    let b = 2.0 * oc.dot(dir);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 { -1.0 } else { (-b - discriminant.sqrt()) / (2.0 * a) }
}

fn benchmark_physics() {
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│  PHYSICS (Verlet, Springs, Raycasting)                       │");
    println!("└───────────────────────────────────────────────────────────────┘");

    let n = 10000;
    let pos = Vec3::new(0.0, 10.0, 0.0);
    let prev = Vec3::new(0.0, 10.0, 0.0);
    let accel = Vec3::new(0.0, -9.81, 0.0);

    let start = Instant::now();
    for _ in 0..n { black_box(verlet_integrate(pos, prev, accel, 0.016)); }
    bench("verlet_integrate", n, start);

    let p1 = Vec3::new(0.0, 0.0, 0.0);
    let p2 = Vec3::new(1.0, 1.0, 0.0);
    let start = Instant::now();
    for _ in 0..n { black_box(spring_force(p1, p2, 1.0, 100.0)); }
    bench("spring_force", n, start);

    let ray_origin = Vec3::new(0.0, 0.0, 0.0);
    let ray_dir = Vec3::new(0.0, 0.0, 1.0);
    let sphere_center = Vec3::new(0.0, 0.0, 5.0);
    let start = Instant::now();
    for _ in 0..n { black_box(ray_sphere_intersect(ray_origin, ray_dir, sphere_center, 1.0)); }
    bench("ray_sphere_intersect", n, start);

    println!();
}

// ============================================================================
// GEOMETRIC ALGEBRA - Cl(3,0,0)
// ============================================================================

#[derive(Clone, Copy)]
struct Multivector {
    s: f64,           // scalar
    e1: f64, e2: f64, e3: f64,  // vectors
    e12: f64, e13: f64, e23: f64,  // bivectors
    e123: f64,        // trivector
}

impl Multivector {
    fn new(s: f64, e1: f64, e2: f64, e3: f64, e12: f64, e13: f64, e23: f64, e123: f64) -> Self {
        Self { s, e1, e2, e3, e12, e13, e23, e123 }
    }

    fn geometric(self, b: Self) -> Self {
        Self {
            s: self.s*b.s + self.e1*b.e1 + self.e2*b.e2 + self.e3*b.e3 - self.e12*b.e12 - self.e13*b.e13 - self.e23*b.e23 - self.e123*b.e123,
            e1: self.s*b.e1 + self.e1*b.s - self.e2*b.e12 + self.e12*b.e2 - self.e3*b.e13 + self.e13*b.e3 - self.e23*b.e123 - self.e123*b.e23,
            e2: self.s*b.e2 + self.e1*b.e12 + self.e2*b.s - self.e12*b.e1 - self.e3*b.e23 + self.e13*b.e123 + self.e23*b.e3 + self.e123*b.e13,
            e3: self.s*b.e3 + self.e1*b.e13 + self.e2*b.e23 + self.e3*b.s - self.e12*b.e123 - self.e13*b.e1 - self.e23*b.e2 - self.e123*b.e12,
            e12: self.s*b.e12 + self.e1*b.e2 - self.e2*b.e1 + self.e12*b.s + self.e3*b.e123 + self.e13*b.e23 - self.e23*b.e13 + self.e123*b.e3,
            e13: self.s*b.e13 + self.e1*b.e3 - self.e3*b.e1 + self.e13*b.s - self.e2*b.e123 - self.e12*b.e23 + self.e23*b.e12 - self.e123*b.e2,
            e23: self.s*b.e23 + self.e2*b.e3 - self.e3*b.e2 + self.e23*b.s + self.e1*b.e123 + self.e12*b.e13 - self.e13*b.e12 + self.e123*b.e1,
            e123: self.s*b.e123 + self.e1*b.e23 + self.e2*b.e13 + self.e3*b.e12 + self.e12*b.e3 + self.e13*b.e2 + self.e23*b.e1 + self.e123*b.s,
        }
    }

    fn dual(self) -> Self {
        Self { s: self.e123, e1: self.e23, e2: -self.e13, e3: self.e12, e12: self.e3, e13: -self.e2, e23: self.e1, e123: self.s }
    }
}

fn rotor_from_axis_angle(axis: Vec3, angle: f64) -> Multivector {
    let half = angle / 2.0;
    let s = half.sin();
    Multivector::new(half.cos(), 0.0, 0.0, 0.0, -axis.z * s, axis.y * s, -axis.x * s, 0.0)
}

fn rotor_apply(rotor: Multivector, v: Vec3) -> Vec3 {
    let mv_v = Multivector::new(0.0, v.x, v.y, v.z, 0.0, 0.0, 0.0, 0.0);
    let rev = Multivector::new(rotor.s, -rotor.e1, -rotor.e2, -rotor.e3, -rotor.e12, -rotor.e13, -rotor.e23, rotor.e123);
    let result = rotor.geometric(mv_v).geometric(rev);
    Vec3::new(result.e1, result.e2, result.e3)
}

fn benchmark_geometric_algebra() {
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│  GEOMETRIC ALGEBRA (Cl(3,0,0) Multivectors & Rotors)         │");
    println!("└───────────────────────────────────────────────────────────────┘");

    let n = 10000;

    let start = Instant::now();
    for _ in 0..n { black_box(Multivector::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)); }
    bench("mv_new", n, start);

    let a = Multivector::new(1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.1);
    let b = Multivector::new(0.5, 1.5, 2.5, 3.5, 0.2, 0.3, 0.4, 0.05);

    let start = Instant::now();
    for _ in 0..n { black_box(a.geometric(b)); }
    bench("mv_geometric", n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(a.dual()); }
    bench("mv_dual", n, start);

    let axis = Vec3::new(0.0, 1.0, 0.0);
    let start = Instant::now();
    for _ in 0..n { black_box(rotor_from_axis_angle(axis, 0.5)); }
    bench("rotor_from_axis_angle", n, start);

    let rotor = rotor_from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.5);
    let vec = Vec3::new(1.0, 0.0, 0.0);
    let start = Instant::now();
    for _ in 0..n { black_box(rotor_apply(rotor, vec)); }
    bench("rotor_apply", n, start);

    println!();
}

// ============================================================================
// DIMENSIONAL ANALYSIS
// ============================================================================

#[derive(Clone, Copy)]
struct Quantity {
    value: f64,
    m: i8, kg: i8, s: i8, a: i8, k: i8, mol: i8, cd: i8,
}

impl Quantity {
    fn new(value: f64, unit: &str) -> Self {
        match unit {
            "m" => Self { value, m: 1, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 },
            "kg" => Self { value, m: 0, kg: 1, s: 0, a: 0, k: 0, mol: 0, cd: 0 },
            "s" => Self { value, m: 0, kg: 0, s: 1, a: 0, k: 0, mol: 0, cd: 0 },
            _ => Self { value, m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 },
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            m: self.m + other.m, kg: self.kg + other.kg, s: self.s + other.s,
            a: self.a + other.a, k: self.k + other.k, mol: self.mol + other.mol, cd: self.cd + other.cd,
        }
    }

    fn div(self, other: Self) -> Self {
        Self {
            value: self.value / other.value,
            m: self.m - other.m, kg: self.kg - other.kg, s: self.s - other.s,
            a: self.a - other.a, k: self.k - other.k, mol: self.mol - other.mol, cd: self.cd - other.cd,
        }
    }

    fn add(self, other: Self) -> Self {
        Self { value: self.value + other.value, ..self }
    }
}

fn benchmark_dimensional() {
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│  DIMENSIONAL ANALYSIS (SI Unit-Aware Math)                   │");
    println!("└───────────────────────────────────────────────────────────────┘");

    let n = 10000;

    let start = Instant::now();
    for _ in 0..n { black_box(Quantity::new(9.81, "m")); }
    bench("qty create", n, start);

    let mass = Quantity::new(10.0, "kg");
    let accel = Quantity::new(9.81, "m");
    let start = Instant::now();
    for _ in 0..n { black_box(mass.mul(accel)); }
    bench("qty_mul", n, start);

    let distance = Quantity::new(100.0, "m");
    let time_q = Quantity::new(10.0, "s");
    let start = Instant::now();
    for _ in 0..n { black_box(distance.div(time_q)); }
    bench("qty_div", n, start);

    let d1 = Quantity::new(50.0, "m");
    let d2 = Quantity::new(30.0, "m");
    let start = Instant::now();
    for _ in 0..n { black_box(d1.add(d2)); }
    bench("qty_add", n, start);

    println!();
}

// ============================================================================
// ECS
// ============================================================================

use std::collections::HashMap;
use std::any::Any;

struct World {
    next_entity: u64,
    components: HashMap<u64, HashMap<String, Box<dyn Any>>>,
}

impl World {
    fn new() -> Self { Self { next_entity: 0, components: HashMap::new() } }
    fn spawn(&mut self) -> u64 { let e = self.next_entity; self.next_entity += 1; self.components.insert(e, HashMap::new()); e }
    fn attach(&mut self, entity: u64, name: &str, component: Box<dyn Any>) {
        if let Some(comps) = self.components.get_mut(&entity) { comps.insert(name.to_string(), component); }
    }
    fn get(&self, entity: u64, name: &str) -> Option<&Box<dyn Any>> {
        self.components.get(&entity).and_then(|c| c.get(name))
    }
    fn has(&self, entity: u64, name: &str) -> bool {
        self.components.get(&entity).map_or(false, |c| c.contains_key(name))
    }
    fn query(&self, comp1: &str, comp2: &str) -> Vec<u64> {
        self.components.iter()
            .filter(|(_, c)| c.contains_key(comp1) && c.contains_key(comp2))
            .map(|(e, _)| *e)
            .collect()
    }
    fn count(&self) -> usize { self.components.len() }
}

fn benchmark_ecs() {
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│  ENTITY COMPONENT SYSTEM (Game Architecture)                 │");
    println!("└───────────────────────────────────────────────────────────────┘");

    let n = 1000;

    let start = Instant::now();
    for _ in 0..n { black_box(World::new()); }
    bench("ecs_world create", n, start);

    let mut world = World::new();
    let spawn_n = 10000;
    let start = Instant::now();
    for _ in 0..spawn_n { black_box(world.spawn()); }
    bench("ecs_spawn", spawn_n, start);

    let mut world = World::new();
    let mut entities = Vec::new();
    for _ in 0..1000 { entities.push(world.spawn()); }

    let start = Instant::now();
    for (i, &e) in entities.iter().enumerate() {
        world.attach(e, "Position", Box::new((i as f64, 0.0, 0.0)));
    }
    bench("ecs_attach", entities.len(), start);

    let start = Instant::now();
    for &e in &entities { black_box(world.get(e, "Position")); }
    bench("ecs_get", entities.len(), start);

    let start = Instant::now();
    for &e in &entities { black_box(world.has(e, "Position")); }
    bench("ecs_has", entities.len(), start);

    for &e in &entities { world.attach(e, "Velocity", Box::new((1.0, 0.0, 0.0))); }

    let query_n = 100;
    let start = Instant::now();
    for _ in 0..query_n { black_box(world.query("Position", "Velocity")); }
    bench("ecs_query", query_n, start);

    let start = Instant::now();
    for _ in 0..n { black_box(world.count()); }
    bench("ecs_count", n, start);

    println!();
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     RUST COMPREHENSIVE FEATURE BENCHMARKS                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    let total_start = Instant::now();

    benchmark_graphics_math();
    benchmark_simd();
    benchmark_tensor();
    benchmark_physics();
    benchmark_geometric_algebra();
    benchmark_dimensional();
    benchmark_ecs();

    let total_elapsed = total_start.elapsed().as_secs_f64() * 1000.0;
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  TOTAL TIME: {:.3} ms", total_elapsed);
    println!("╚═══════════════════════════════════════════════════════════════╝");
}
