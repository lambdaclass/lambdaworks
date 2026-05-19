//! Fp3 (cubic extension of Goldilocks) Bowers G NTT — reference implementation.
//! Uses base-field bitrev twiddles, extension-field butterfly arithmetic.
//! Used as a golden vector source for Metal GPU correctness tests.

use crate::field::{
    element::FieldElement,
    fields::u64_goldilocks_field::{Degree3GoldilocksExtensionField, Goldilocks64Field},
};

type Fp = FieldElement<Goldilocks64Field>;
type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;

/// Bowers G NTT over Fp3 with base-field Fp twiddles.
/// `twiddles` must be `[ω^0, ω^1, …, ω^(n/2-1)]` in bit-reversed order.
/// Input is consumed natural-order; output is natural-order (matches `ntt_bowers` for Goldilocks).
pub fn ntt_bowers_fp3(data: &mut [Fp3], twiddles: &[Fp]) {
    let n = data.len();
    bit_reverse(data);
    let log_n = n.trailing_zeros() as usize;

    for log_half in 0..log_n {
        let half = 1usize << log_half;
        let block_size = 2 * half;
        let num_blocks = n / block_size;

        #[allow(clippy::needless_range_loop)]
        for block in 0..num_blocks {
            let w = if block == 0 {
                Fp::from(1u64)
            } else {
                twiddles[block]
            };
            let base = block * block_size;
            for j in 0..half {
                let i0 = base + j;
                let i1 = i0 + half;
                let a = data[i0];
                let bb = data[i1];
                let wbb = w * bb;
                data[i0] = a + wbb;
                data[i1] = a - wbb;
            }
        }
    }
}

fn bit_reverse<T: Clone>(data: &mut [T]) {
    let n = data.len();
    if n <= 1 { return; }
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (u32::BITS - log_n);
        if i < j as usize { data.swap(i, j as usize); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::cpu::ntt_bowers_goldilocks::compute_bowers_twiddles;

    #[test]
    fn fp3_bowers_smoke_log4() {
        let n = 16usize;
        let tw_u64 = compute_bowers_twiddles(n, 7);
        let twiddles: Vec<Fp> = tw_u64.iter().map(|&x| Fp::from(x)).collect();

        // Construct Fp3 elements from base field values: [c0, c1, c2]
        let mut data: Vec<Fp3> = (0..n as u64)
            .map(|i| {
                let base = Fp::from(i);
                FieldElement::new([base, Fp::from(0), Fp::from(0)])
            })
            .collect();

        let original = data.clone();
        ntt_bowers_fp3(&mut data, &twiddles);
        assert_ne!(data, original);
    }
}
