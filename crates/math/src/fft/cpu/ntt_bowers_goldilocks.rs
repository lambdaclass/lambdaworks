//! AMO-Lean Generated Rust NTT — Bowers G ordering (raw u64 arithmetic)
//!
//! N = up to 2^20, p = 18446744069414584321 (Goldilocks)
//! Reduction: solinasFold (e-graph cost model)
//! Ordering: Bowers G (bit-reversed twiddles, DIF butterfly, sequential access)
//!
//! This is a standalone raw-u64 NTT for benchmarking against the lambdaworks
//! `FieldElement`-based Bowers FFT. It bypasses the `FieldElement` abstraction
//! to measure the overhead of the generic field layer.

const P: u64 = 18446744069414584321;

/// Goldilocks reduction — exploits p = 2^64 - 2^32 + 1.
#[inline(always)]
fn goldilocks_reduce(x: u128) -> u64 {
    let lo = x as u64;
    let hi = (x >> 64) as u64;
    let hh = hi >> 32;
    let hl = hi & 0xFFFFFFFF_u64;
    let (t0, borrow) = lo.overflowing_sub(hh);
    let t0 = if borrow {
        t0.wrapping_sub(0xFFFFFFFF_u64)
    } else {
        t0
    };
    let t1 = hl.wrapping_mul(0xFFFFFFFF_u64);
    let (result, carry) = t0.overflowing_add(t1);
    if carry || result >= P {
        result.wrapping_sub(P)
    } else {
        result
    }
}

/// DIF butterfly for Goldilocks.
#[inline(always)]
fn dif_butterfly(a: &mut u64, b: &mut u64, w: u64) {
    let va = *a;
    let vb = *b;
    let sum = va.wrapping_add(vb);
    *a = if sum >= P { sum - P } else { sum };
    *b = goldilocks_reduce(
        (if va >= vb {
            va - vb
        } else {
            P - vb + va
        }) as u128
            * w as u128,
    );
}

fn bit_reverse(data: &mut [u64]) {
    let n = data.len();
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

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

/// Compute bit-reversed twiddle table for Bowers G network.
pub fn compute_bowers_twiddles(n: usize, generator: u128) -> Vec<u64> {
    let p = P as u128;
    let omega = mod_pow(generator, (p - 1) / n as u128, p);
    let mut tw: Vec<u64> = (0..n / 2)
        .map(|i| mod_pow(omega, i as u128, p) as u64)
        .collect();
    bit_reverse(&mut tw);
    tw
}

/// Bowers G NTT: bit-reverse input, DIF stages small→large, sequential twiddle access.
pub fn ntt_bowers(data: &mut [u64], twiddles: &[u64]) {
    bit_reverse(data);
    ntt_bowers_butterflies(data, twiddles);
}

/// Butterfly-only kernel (no bit-reverse). For benchmarking the core FFT.
pub fn ntt_bowers_butterflies(data: &mut [u64], twiddles: &[u64]) {
    let n = data.len();
    let log_n = n.trailing_zeros() as usize;

    for log_half in 0..log_n {
        let half = 1_usize << log_half;
        let block_size = 2 * half;
        let num_blocks = n / block_size;

        for block in 0..num_blocks {
            let w: u64 = if block == 0 { 1 } else { twiddles[block] };
            let base = block * block_size;
            for j in 0..half {
                let i0 = base + j;
                let i1 = i0 + half;
                let (left, right) = data.split_at_mut(i1);
                dif_butterfly(&mut left[i0], &mut right[0], w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const G: u128 = 7; // Goldilocks multiplicative generator

    #[test]
    fn bowers_ntt_smoke_log4() {
        let n = 16usize;
        let tw = compute_bowers_twiddles(n, G);
        let mut data: Vec<u64> = (0..n as u64).collect();
        let original = data.clone();
        ntt_bowers(&mut data, &tw);
        assert_ne!(data, original);
        assert!(data.iter().all(|&x| x < P));
    }
}
