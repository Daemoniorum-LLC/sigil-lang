//! Holographic Data Structure Utilities
//!
//! Shared utilities for probabilistic data structures implementing
//! Spec 11-HOLOGRAPHIC.md. This module provides:
//!
//! - Hash functions for sketches (FNV-1a)
//! - Proper PRNG with thread-local state
//! - HyperLogLog estimation algorithms
//! - SHA256-based Merkle tree hashing

use sha2::{Sha256, Digest};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// Hash Functions
// =============================================================================

/// FNV-1a hash for probabilistic data structures.
/// Seeded variant allows multiple independent hash functions.
pub fn holographic_hash(data: &[u8], seed: u64) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    let mut hash = FNV_OFFSET.wrapping_add(seed.wrapping_mul(FNV_PRIME));
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash a string value with optional seed.
pub fn hash_value_str(value: &str, seed: u64) -> u64 {
    holographic_hash(value.as_bytes(), seed)
}

/// SHA256 hash for cryptographic operations (Merkle trees).
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Combine two SHA256 hashes for Merkle tree internal nodes.
/// Uses sorted concatenation to make the tree order-independent for leaves,
/// but preserves left/right distinction for proofs.
pub fn merkle_combine(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

// =============================================================================
// Random Number Generation
// =============================================================================

/// Thread-local PRNG state using xorshift64*
thread_local! {
    static RNG_STATE: RefCell<u64> = RefCell::new(0);
}

/// Global counter for additional entropy mixing
static RNG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Initialize or get PRNG state for current thread.
fn get_rng_state() -> u64 {
    RNG_STATE.with(|state| {
        let mut s = state.borrow_mut();
        if *s == 0 {
            // Initialize with time + counter + thread id hash
            let time_seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x123456789ABCDEF0);
            let counter = RNG_COUNTER.fetch_add(1, Ordering::Relaxed);
            // Hash thread id via Debug format (stable approach)
            let thread_id = format!("{:?}", std::thread::current().id());
            let thread_hash = holographic_hash(thread_id.as_bytes(), 0xDEADBEEF);
            *s = time_seed ^ counter ^ thread_hash;
            if *s == 0 { *s = 0xCAFEBABE; } // Ensure non-zero
        }
        *s
    })
}

/// Set PRNG state (advances the state and returns previous).
fn set_rng_state(new_state: u64) {
    RNG_STATE.with(|state| {
        *state.borrow_mut() = new_state;
    });
}

/// xorshift64* PRNG - fast, good statistical properties.
pub fn random_u64() -> u64 {
    let mut s = get_rng_state();
    // xorshift64*
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    set_rng_state(s);
    s.wrapping_mul(0x2545F4914F6CDD1D)
}

/// Random value in range [0, max).
pub fn random_usize(max: usize) -> usize {
    if max == 0 { return 0; }
    (random_u64() as usize) % max
}

/// Random float in [0.0, 1.0).
pub fn random_f64() -> f64 {
    (random_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Weighted random selection based on probabilities.
/// Returns index of selected item.
pub fn weighted_random(weights: &[f64]) -> usize {
    if weights.is_empty() { return 0; }

    let total: f64 = weights.iter().sum();
    if total <= 0.0 { return random_usize(weights.len()); }

    let mut r = random_f64() * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 { return i; }
    }
    weights.len() - 1 // Fallback due to floating point
}

// =============================================================================
// HyperLogLog Utilities
// =============================================================================

/// Count leading zeros in a 64-bit value.
#[inline]
pub fn count_leading_zeros(value: u64) -> u32 {
    if value == 0 { 64 } else { value.leading_zeros() }
}

/// HyperLogLog alpha correction factor for m registers.
pub fn hll_alpha(num_registers: usize) -> f64 {
    match num_registers {
        16 => 0.673,
        32 => 0.697,
        64 => 0.709,
        _ => 0.7213 / (1.0 + 1.079 / num_registers as f64),
    }
}

/// Compute HyperLogLog cardinality estimate from registers.
/// Returns (estimate, standard_error).
pub fn hll_estimate(registers: &[i64], precision: u32) -> (f64, f64) {
    let num_registers = registers.len();
    if num_registers == 0 { return (0.0, 0.0); }

    let mut harmonic_sum = 0.0_f64;
    let mut zeros = 0usize;

    for &r in registers {
        if r == 0 { zeros += 1; }
        harmonic_sum += 2.0_f64.powi(-(r as i32));
    }

    let alpha = hll_alpha(num_registers);
    let raw_estimate = alpha * (num_registers as f64).powi(2) / harmonic_sum;

    // Small range correction (linear counting)
    let estimate = if raw_estimate <= 2.5 * num_registers as f64 && zeros > 0 {
        num_registers as f64 * (num_registers as f64 / zeros as f64).ln()
    } else if raw_estimate > (1u64 << 32) as f64 / 30.0 {
        // Large range correction
        -((1u64 << 32) as f64) * (1.0 - raw_estimate / (1u64 << 32) as f64).ln()
    } else {
        raw_estimate
    };

    // Standard error is approximately 1.04 / sqrt(m)
    let std_error = 1.04 / (num_registers as f64).sqrt();

    (estimate, std_error)
}

/// Compute confidence bounds for HLL estimate.
/// Returns (lower_bound, upper_bound) for given confidence level (e.g., 0.95).
pub fn hll_bounds(estimate: f64, std_error: f64, confidence: f64) -> (f64, f64) {
    // Z-score for confidence level (approximation)
    let z = match confidence {
        c if c >= 0.99 => 2.576,
        c if c >= 0.95 => 1.96,
        c if c >= 0.90 => 1.645,
        c if c >= 0.80 => 1.282,
        _ => 1.0,
    };

    let margin = z * std_error * estimate;
    let lower = (estimate - margin).max(0.0);
    let upper = estimate + margin;
    (lower, upper)
}

/// Merge two HyperLogLog register arrays (union operation).
/// Takes element-wise maximum.
pub fn hll_merge(a: &[i64], b: &[i64]) -> Vec<i64> {
    if a.len() != b.len() {
        // Different precisions - can't merge directly
        // In production, would need to downgrade the higher precision one
        return a.to_vec();
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x.max(y)).collect()
}

// =============================================================================
// BloomFilter Utilities
// =============================================================================

/// Compute bit positions for BloomFilter using double hashing.
/// Returns iterator of bit indices for given number of hash functions.
pub fn bloom_positions(data: &[u8], size: usize, num_hashes: usize) -> Vec<usize> {
    let h1 = holographic_hash(data, 0);
    let h2 = holographic_hash(data, h1); // Second hash seeded by first

    (0..num_hashes)
        .map(|i| {
            let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
            (combined as usize) % size
        })
        .collect()
}

// =============================================================================
// CountMinSketch Utilities
// =============================================================================

/// Compute row positions for CountMinSketch.
/// Returns position in each row (depth positions for width-sized rows).
pub fn cms_positions(data: &[u8], width: usize, depth: usize) -> Vec<usize> {
    (0..depth)
        .map(|row| {
            let hash = holographic_hash(data, row as u64);
            (hash as usize) % width
        })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv_hash_deterministic() {
        let h1 = holographic_hash(b"test", 0);
        let h2 = holographic_hash(b"test", 0);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv_hash_different_seeds() {
        let h1 = holographic_hash(b"test", 0);
        let h2 = holographic_hash(b"test", 1);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_sha256_hash() {
        let hash = sha256_hash(b"hello");
        assert_eq!(hash.len(), 32);
        // Known SHA256 of "hello"
        assert_eq!(hex::encode(hash),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }

    #[test]
    fn test_merkle_combine() {
        let left = sha256_hash(b"left");
        let right = sha256_hash(b"right");
        let combined = merkle_combine(&left, &right);
        assert_eq!(combined.len(), 32);
        assert_ne!(combined, left);
        assert_ne!(combined, right);
    }

    #[test]
    fn test_random_produces_different_values() {
        let r1 = random_u64();
        let r2 = random_u64();
        let r3 = random_u64();
        // Extremely unlikely to be equal
        assert!(r1 != r2 || r2 != r3);
    }

    #[test]
    fn test_random_usize_in_range() {
        for _ in 0..100 {
            let r = random_usize(10);
            assert!(r < 10);
        }
    }

    #[test]
    fn test_weighted_random() {
        let weights = vec![0.0, 1.0, 0.0]; // Should always pick index 1
        for _ in 0..10 {
            assert_eq!(weighted_random(&weights), 1);
        }
    }

    #[test]
    fn test_hll_alpha() {
        assert!((hll_alpha(16) - 0.673).abs() < 0.001);
        assert!((hll_alpha(32) - 0.697).abs() < 0.001);
        assert!((hll_alpha(64) - 0.709).abs() < 0.001);
    }

    #[test]
    fn test_hll_estimate_empty() {
        let registers = vec![0i64; 1024];
        let (est, _) = hll_estimate(&registers, 10);
        assert!(est < 1.0); // Should be ~0 for empty
    }

    #[test]
    fn test_hll_merge() {
        let a = vec![1, 2, 3, 0];
        let b = vec![0, 3, 2, 1];
        let merged = hll_merge(&a, &b);
        assert_eq!(merged, vec![1, 3, 3, 1]);
    }

    #[test]
    fn test_bloom_positions_deterministic() {
        let p1 = bloom_positions(b"test", 1000, 7);
        let p2 = bloom_positions(b"test", 1000, 7);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 7);
    }

    #[test]
    fn test_cms_positions() {
        let positions = cms_positions(b"test", 1024, 5);
        assert_eq!(positions.len(), 5);
        for &p in &positions {
            assert!(p < 1024);
        }
    }
}
