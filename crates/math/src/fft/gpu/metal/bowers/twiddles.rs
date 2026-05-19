//! Bitrev twiddles + coset powers for the Bowers-G NTT.
//!
//! Twiddles are stored in bit-reversed order, matching the layout the
//! `bowers_butterfly_indices` shader helper expects.

use crate::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field, traits::IsPrimeField,
};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

type Fp = FieldElement<Goldilocks64Field>;
type CosetCacheKey = (u32, u64);
type CosetCacheValue = Mutex<HashMap<CosetCacheKey, Vec<Fp>>>;

const GOLDILOCKS_PRIME: u64 = 18446744069414584321;

/// Modular exponentiation for computing primitive roots.
fn mod_pow(mut base: u128, mut exp: u128, modulus: u128) -> u128 {
    let mut result: u128 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        exp >>= 1;
        base = base * base % modulus;
    }
    result
}

/// Compute Bowers-G bit-reversed twiddles: `[w^0..w^(n/2-1)]` permuted by `bit_reverse`.
/// Uses generator 7 to match the CPU reference implementation.
/// Returns twiddles with canonicalized internal values in [0, p).
pub fn bowers_twiddles_goldilocks(log_n: u32) -> Vec<Fp> {
    let n = 1usize << log_n;
    let p = GOLDILOCKS_PRIME as u128;
    let generator: u128 = 7;

    // Compute primitive root: 7^((p-1)/n)
    let omega_raw = mod_pow(generator, (p - 1) / n as u128, p) as u64;
    let omega = Fp::new(omega_raw);

    let mut tw: Vec<Fp> = Vec::with_capacity(n / 2);
    let mut w = Fp::one();
    for _ in 0..(n / 2) {
        tw.push(w);
        w *= omega;
    }
    bit_reverse(&mut tw);

    // Canonicalize all twiddles so their internal values are in [0, p)
    tw.into_iter()
        .map(|t| Fp::new(Goldilocks64Field::canonical(t.value())))
        .collect()
}

/// Compute coset powers `[g^0..g^(n-1)]`.
/// Returns coset powers with canonicalized internal values in [0, p).
pub fn coset_powers_goldilocks(log_n: u32, coset_offset: Fp) -> Vec<Fp> {
    let n = 1usize << log_n;
    let mut out = Vec::with_capacity(n);
    let mut acc = Fp::one();
    for _ in 0..n {
        out.push(acc);
        acc *= coset_offset;
    }

    // Canonicalize all coset powers so their internal values are in [0, p)
    out.into_iter()
        .map(|c| Fp::new(Goldilocks64Field::canonical(c.value())))
        .collect()
}

fn bit_reverse<T>(data: &mut [T]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (u32::BITS - log_n);
        if i < j as usize {
            data.swap(i, j as usize);
        }
    }
}

static TWIDDLE_CACHE: Lazy<Mutex<HashMap<u32, Vec<Fp>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

static COSET_CACHE: Lazy<CosetCacheValue> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Thread-safe cached version of `bowers_twiddles_goldilocks`.
pub fn cached_bowers_twiddles_goldilocks(log_n: u32) -> Vec<Fp> {
    TWIDDLE_CACHE
        .lock()
        .expect("twiddle cache lock should never be poisoned")
        .entry(log_n)
        .or_insert_with(|| bowers_twiddles_goldilocks(log_n))
        .clone()
}

/// Thread-safe cached version of `coset_powers_goldilocks`.
pub fn cached_coset_powers_goldilocks(log_n: u32, coset_offset: Fp) -> Vec<Fp> {
    let key = (log_n, *coset_offset.value());
    COSET_CACHE
        .lock()
        .expect("coset cache lock should never be poisoned")
        .entry(key)
        .or_insert_with(|| coset_powers_goldilocks(log_n, coset_offset))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::cpu::ntt_bowers_goldilocks::compute_bowers_twiddles;

    #[test]
    fn bowers_twiddles_match_cpu_ref_log10() {
        let log_n = 10u32;
        let gpu = bowers_twiddles_goldilocks(log_n);
        let cpu = compute_bowers_twiddles(1usize << log_n, 7);

        assert_eq!(
            gpu.len(),
            cpu.len(),
            "GPU and CPU twiddle tables must have same length"
        );
        for (i, (a, b)) in gpu.iter().zip(cpu.iter()).enumerate() {
            // GPU twiddles are now guaranteed canonical, so direct comparison
            assert_eq!(*a.value(), *b, "Twiddle mismatch at index {}", i);
        }
    }

    #[test]
    fn coset_powers_basic() {
        let g = Fp::from(3u64);
        let cp = coset_powers_goldilocks(4, g);
        assert_eq!(cp.len(), 16);
        assert_eq!(cp[0], Fp::one());
        assert_eq!(cp[1], g);
        assert_eq!(cp[2], g * g);
    }

    #[test]
    fn twiddle_cache_returns_equal_vectors() {
        assert_eq!(
            cached_bowers_twiddles_goldilocks(8),
            cached_bowers_twiddles_goldilocks(8)
        );
    }

    #[test]
    fn bowers_twiddles_are_canonical() {
        const P: u64 = 18446744069414584321;
        let tw = bowers_twiddles_goldilocks(12);
        for (i, t) in tw.iter().enumerate() {
            assert!(*t.value() < P, "twiddle[{}] = {} >= P", i, *t.value());
        }
    }

    #[test]
    fn coset_powers_are_canonical() {
        const P: u64 = 18446744069414584321;
        let cp = coset_powers_goldilocks(12, FieldElement::from(7u64));
        for (i, c) in cp.iter().enumerate() {
            assert!(*c.value() < P, "coset[{}] = {} >= P", i, *c.value());
        }
    }
}
