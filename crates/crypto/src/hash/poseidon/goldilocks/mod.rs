/// Poseidon hash for the Goldilocks field (p = 2^64 - 2^32 + 1).
///
/// This is a standalone implementation that matches Plonky2's Poseidon.
/// It does NOT use the existing `PermutationParameters` trait because that
/// trait hardcodes state_size=3 and x^3 S-box, whereas Goldilocks Poseidon
/// uses state_size=12 and x^7 S-box.
pub mod constants;

use alloc::vec::Vec;
use constants::*;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::u64_goldilocks_field::{Goldilocks64Field, GOLDILOCKS_PRIME},
    traits::IsPrimeField,
};

type FpE = FieldElement<Goldilocks64Field>;

#[derive(Clone, Default)]
pub struct PoseidonGoldilocks;

impl PoseidonGoldilocks {
    /// Reduce a 96-bit value (lo, hi) modulo the Goldilocks prime.
    ///
    /// For Goldilocks p = 2^64 - 2^32 + 1, we have 2^64 ≡ 2^32 - 1 (mod p).
    /// So (hi * 2^64 + lo) ≡ hi * (2^32 - 1) + lo (mod p).
    #[inline(always)]
    fn reduce_u96(lo: u64, hi: u32) -> FpE {
        // hi * (2^32 - 1) = hi * 0xFFFFFFFF
        let hi_times = (hi as u64) * 0xFFFF_FFFFu64;
        let (sum, carry) = lo.overflowing_add(hi_times);
        if carry {
            // sum + 2^64 ≡ sum + (2^32 - 1) (mod p)
            // = sum + 0xFFFFFFFF, which can't overflow again since sum < 2^64 and 0xFFFFFFFF < 2^32
            let val = sum.wrapping_add(0xFFFF_FFFF);
            // May still be >= p, canonicalize
            let canonical = if val >= GOLDILOCKS_PRIME {
                val - GOLDILOCKS_PRIME
            } else {
                val
            };
            FieldElement::from_raw(canonical)
        } else {
            let canonical = if sum >= GOLDILOCKS_PRIME {
                sum - GOLDILOCKS_PRIME
            } else {
                sum
            };
            FieldElement::from_raw(canonical)
        }
    }

    /// Compute one row of the MDS matrix–vector product.
    ///
    /// MDS = circ(MDS_MATRIX_CIRC) + diag(MDS_MATRIX_DIAG).
    /// Uses u128 accumulation to avoid intermediate modular reductions.
    #[inline(always)]
    fn mds_row_shf(r: usize, state: &[u64; SPONGE_WIDTH]) -> (u64, u32) {
        let mut acc = 0u128;
        for i in 0..SPONGE_WIDTH {
            acc += (state[(i + r) % SPONGE_WIDTH] as u128) * (MDS_MATRIX_CIRC[i] as u128);
        }
        acc += (state[r] as u128) * (MDS_MATRIX_DIAG[r] as u128);
        (acc as u64, (acc >> 64) as u32)
    }

    /// Apply the MDS layer to the state.
    #[inline(always)]
    fn mds_layer(state: &[FpE; SPONGE_WIDTH]) -> [FpE; SPONGE_WIDTH] {
        let mut raw = [0u64; SPONGE_WIDTH];
        for (raw_i, s) in raw.iter_mut().zip(state.iter()) {
            *raw_i = Goldilocks64Field::canonical(s.value());
        }

        let mut result = [FpE::zero(); SPONGE_WIDTH];
        for (r, result_r) in result.iter_mut().enumerate() {
            let (lo, hi) = Self::mds_row_shf(r, &raw);
            *result_r = Self::reduce_u96(lo, hi);
        }
        result
    }

    /// S-box: x^7 = x * x^2 * x^4
    #[inline(always)]
    fn sbox(x: FpE) -> FpE {
        let x2 = &x * &x;
        let x4 = &x2 * &x2;
        let x3 = &x * &x2;
        &x3 * &x4
    }

    /// Add round constants to state.
    #[inline(always)]
    fn constant_layer(state: &mut [FpE; SPONGE_WIDTH], round: usize) {
        let base = round * SPONGE_WIDTH;
        for (i, s) in state.iter_mut().enumerate() {
            *s = &*s + &FpE::from(ALL_ROUND_CONSTANTS[base + i]);
        }
    }

    /// Full round: add constants, apply S-box to all elements, MDS multiply.
    #[inline(always)]
    fn full_round(state: &mut [FpE; SPONGE_WIDTH], round: usize) {
        Self::constant_layer(state, round);
        for s in state.iter_mut() {
            *s = Self::sbox(*s);
        }
        *state = Self::mds_layer(state);
    }

    /// Run the Poseidon permutation on a width-12 state.
    ///
    /// Uses the fast partial round algorithm (Plonky2) which reduces the
    /// partial-round MDS from O(n^2) to O(n) per round using precomputed
    /// constants (FP_W_HATS, FP_VS, FP_INIT_MAT).
    pub fn permute(state: &mut [FpE; SPONGE_WIDTH]) {
        // Phase 1: First 4 full rounds
        for round in 0..HALF_N_FULL_ROUNDS {
            Self::full_round(state, round);
        }

        // Phase 2a: Add first partial round constants
        for i in 0..SPONGE_WIDTH {
            state[i] = &state[i] + &FpE::from(FP_FIRST_RC[i]);
        }

        // Phase 2b: Apply initial matrix to state[1..12]
        // Plonky2: result[c+1] = sum_r state[r+1] * INIT_MAT[r][c]
        let mut tmp = [FpE::zero(); SPONGE_WIDTH];
        tmp[0] = state[0];
        for c in 0..11 {
            let mut sum = FpE::zero();
            for r in 0..11 {
                sum = &sum + &(&state[r + 1] * &FpE::from(FP_INIT_MAT[r][c]));
            }
            tmp[c + 1] = sum;
        }
        *state = tmp;

        // Phase 2c: 22 partial rounds with fast sparse MDS
        for i in 0..N_PARTIAL_ROUNDS {
            state[0] = Self::sbox(state[0]);
            state[0] = &state[0] + &FpE::from(FP_RC[i]);

            let s0 = state[0];

            // d = mds0to0 * s0 + dot(w_hat, state[1..12])
            let mut d = &s0 * &FpE::from(25u64);
            for j in 0..11 {
                d = &d + &(&state[j + 1] * &FpE::from(FP_W_HATS[i][j]));
            }

            // state[j+1] += s0 * v[j]
            for j in 0..11 {
                state[j + 1] = &state[j + 1] + &(&s0 * &FpE::from(FP_VS[i][j]));
            }

            state[0] = d;
        }

        // Phase 3: Last 4 full rounds
        for round in (HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS)..N_ROUNDS {
            Self::full_round(state, round);
        }
    }

    /// Hash two field elements: H(a, b).
    pub fn hash(a: &FpE, b: &FpE) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        state[0] = *a;
        state[1] = *b;
        Self::permute(&mut state);
        state[0]
    }

    /// Hash a single field element.
    pub fn hash_single(x: &FpE) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        state[0] = *x;
        Self::permute(&mut state);
        state[0]
    }

    /// Hash a variable-length slice of field elements using a sponge construction.
    pub fn hash_many(inputs: &[FpE]) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];

        // Pad input with 1 followed by zeros to fill a rate block
        let mut values: Vec<FpE> = inputs.to_vec();
        values.push(FpE::one());
        while !values.len().is_multiple_of(SPONGE_RATE) {
            values.push(FpE::zero());
        }

        // Absorb each rate-sized block
        for block in values.chunks(SPONGE_RATE) {
            for i in 0..SPONGE_RATE {
                state[i] = &state[i] + &block[i];
            }
            Self::permute(&mut state);
        }

        state[0]
    }

    /// Hash a fixed-length slice of field elements without padding.
    /// Used for Merkle leaf hashing where the input length is known.
    pub fn hash_no_pad(inputs: &[FpE]) -> FpE {
        let mut state = [FpE::zero(); SPONGE_WIDTH];

        for block in inputs.chunks(SPONGE_RATE) {
            for (i, val) in block.iter().enumerate() {
                state[i] = &state[i] + val;
            }
            Self::permute(&mut state);
        }

        state[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permute_zero_state() {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        PoseidonGoldilocks::permute(&mut state);
        assert_ne!(state[0], FpE::zero());
    }

    #[test]
    fn test_hash_deterministic() {
        let a = FpE::from(1u64);
        let b = FpE::from(2u64);
        let h1 = PoseidonGoldilocks::hash(&a, &b);
        let h2 = PoseidonGoldilocks::hash(&a, &b);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = PoseidonGoldilocks::hash(&FpE::from(1u64), &FpE::from(2u64));
        let h2 = PoseidonGoldilocks::hash(&FpE::from(2u64), &FpE::from(1u64));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_many_vs_hash() {
        let a = FpE::from(1u64);
        let b = FpE::from(2u64);
        // hash_many pads with 1 then zeros; hash uses raw [a, b, 0...0].
        // They should NOT be equal.
        let h_pair = PoseidonGoldilocks::hash(&a, &b);
        let h_many = PoseidonGoldilocks::hash_many(&[a, b]);
        assert_ne!(h_pair, h_many);
    }

    #[test]
    fn test_sbox_is_x7() {
        let x = FpE::from(3u64);
        let expected = FpE::from(2187u64); // 3^7 = 2187
        assert_eq!(PoseidonGoldilocks::sbox(x), expected);
    }

    #[test]
    fn test_mds_layer_nonzero() {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            state[i] = FpE::from((i + 1) as u64);
        }
        let result = PoseidonGoldilocks::mds_layer(&state);
        for r in &result {
            assert_ne!(*r, FpE::zero());
        }
    }

    #[test]
    fn test_reduce_u96() {
        // Test: reduce_u96(0, 0) = 0
        assert_eq!(PoseidonGoldilocks::reduce_u96(0, 0), FpE::zero());

        // Test: reduce_u96(1, 0) = 1
        assert_eq!(PoseidonGoldilocks::reduce_u96(1, 0), FpE::one());

        // Test: reduce_u96(p, 0) = 0 (p reduces to 0)
        assert_eq!(
            PoseidonGoldilocks::reduce_u96(GOLDILOCKS_PRIME, 0),
            FpE::zero()
        );

        // Test: reduce_u96(0, 1) = 2^64 mod p = 2^32 - 1 = 0xFFFFFFFF
        let expected = FpE::from(0xFFFF_FFFFu64);
        assert_eq!(PoseidonGoldilocks::reduce_u96(0, 1), expected);
    }

    /// Verify permutation against Plonky2's official test vectors.
    fn check_plonky2_vector(input: [u64; SPONGE_WIDTH], expected: [u64; SPONGE_WIDTH]) {
        let mut state = [FpE::zero(); SPONGE_WIDTH];
        for i in 0..SPONGE_WIDTH {
            state[i] = FpE::from(input[i]);
        }
        PoseidonGoldilocks::permute(&mut state);
        for i in 0..SPONGE_WIDTH {
            let got = Goldilocks64Field::canonical(state[i].value());
            assert_eq!(
                got, expected[i],
                "Mismatch at index {i}: got {got:#018x}, expected {:#018x}",
                expected[i]
            );
        }
    }

    #[test]
    fn test_plonky2_vector_all_zeros() {
        check_plonky2_vector(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0x3c18a9786cb0b359,
                0xc4055e3364a246c3,
                0x7953db0ab48808f4,
                0xc71603f33a1144ca,
                0xd7709673896996dc,
                0x46a84e87642f44ed,
                0xd032648251ee0b3c,
                0x1c687363b207df62,
                0xdf8565563e8045fe,
                0x40f5b37ff4254dae,
                0xd070f637b431067c,
                0x1792b1c4342109d7,
            ],
        );
    }

    #[test]
    fn test_plonky2_vector_sequential() {
        check_plonky2_vector(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [
                0xd64e1e3efc5b8e9e,
                0x53666633020aaa47,
                0xd40285597c6a8825,
                0x613a4f81e81231d2,
                0x414754bfebd051f0,
                0xcb1f8980294a023f,
                0x6eb2a9e4d54a9d0f,
                0x1902bc3af467e056,
                0xf045d5eafdc6021f,
                0xe4150f77caaa3be5,
                0xc9bfd01d39b50cce,
                0x5c0a27fcb0e1459b,
            ],
        );
    }

    #[test]
    fn test_plonky2_vector_all_neg_one() {
        let neg_one = GOLDILOCKS_PRIME - 1; // p - 1 = -1 mod p
        check_plonky2_vector(
            [neg_one; SPONGE_WIDTH],
            [
                0xbe0085cfc57a8357,
                0xd95af71847d05c09,
                0xcf55a13d33c1c953,
                0x95803a74f4530e82,
                0xfcd99eb30a135df1,
                0xe095905e913a3029,
                0xde0392461b42919b,
                0x7d3260e24e81d031,
                0x10d3d0465d9deaa0,
                0xa87571083dfc2a47,
                0xe18263681e9958f8,
                0xe28e96f1ae5e60d3,
            ],
        );
    }

    /// Test that the fast partial round algorithm (now the production `permute()`)
    /// gives the same result as the standard O(n^2) algorithm.
    #[test]
    fn test_fast_partial_rounds_match_standard() {
        // Standard (slow) permutation for reference — uses full 12×12 MDS per partial round.
        fn permute_standard(state: &mut [FpE; SPONGE_WIDTH]) {
            let mut round = 0;
            for _ in 0..HALF_N_FULL_ROUNDS {
                PoseidonGoldilocks::constant_layer(state, round);
                for s in state.iter_mut() {
                    *s = PoseidonGoldilocks::sbox(*s);
                }
                *state = PoseidonGoldilocks::mds_layer(state);
                round += 1;
            }
            for _ in 0..N_PARTIAL_ROUNDS {
                PoseidonGoldilocks::constant_layer(state, round);
                state[0] = PoseidonGoldilocks::sbox(state[0]);
                *state = PoseidonGoldilocks::mds_layer(state);
                round += 1;
            }
            for _ in 0..HALF_N_FULL_ROUNDS {
                PoseidonGoldilocks::constant_layer(state, round);
                for s in state.iter_mut() {
                    *s = PoseidonGoldilocks::sbox(*s);
                }
                *state = PoseidonGoldilocks::mds_layer(state);
                round += 1;
            }
        }

        let test_inputs: [[u64; 12]; 3] = [
            [0; 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [
                0x8ccbbbea4fe5d2b7,
                0xc2af59ee9ec49970,
                0x90f7e1a9e658446a,
                0xdcc0630a3ab8b1b8,
                0x7ff8256bca20588c,
                0x5d99a7ca0c44ecfb,
                0x48452b17a70fbee3,
                0xeb09d654690b6c88,
                0x4a55d3a39c676a88,
                0xc0407a38d2285139,
                0xa234bac9356386d1,
                0xe1633f2bad98a52f,
            ],
        ];

        for (idx, input) in test_inputs.iter().enumerate() {
            let mut standard = [FpE::zero(); SPONGE_WIDTH];
            let mut fast = [FpE::zero(); SPONGE_WIDTH];
            for i in 0..SPONGE_WIDTH {
                standard[i] = FpE::from(input[i]);
                fast[i] = FpE::from(input[i]);
            }
            permute_standard(&mut standard);
            PoseidonGoldilocks::permute(&mut fast);
            for i in 0..SPONGE_WIDTH {
                assert_eq!(
                    standard[i], fast[i],
                    "Mismatch at test {idx} element {i}: standard={:?}, fast={:?}",
                    standard[i], fast[i]
                );
            }
        }
    }

    #[test]
    fn test_plonky2_vector_random() {
        check_plonky2_vector(
            [
                0x8ccbbbea4fe5d2b7,
                0xc2af59ee9ec49970,
                0x90f7e1a9e658446a,
                0xdcc0630a3ab8b1b8,
                0x7ff8256bca20588c,
                0x5d99a7ca0c44ecfb,
                0x48452b17a70fbee3,
                0xeb09d654690b6c88,
                0x4a55d3a39c676a88,
                0xc0407a38d2285139,
                0xa234bac9356386d1,
                0xe1633f2bad98a52f,
            ],
            [
                0xa89280105650c4ec,
                0xab542d53860d12ed,
                0x5704148e9ccab94f,
                0xd3a826d4b62da9f5,
                0x8a7a6ca87892574f,
                0xc7017e1cad1a674e,
                0x1f06668922318e34,
                0xa3b203bc8102676f,
                0xfcc781b0ce382bf2,
                0x934c69ff3ed14ba5,
                0x504688a5996e8f13,
                0x401f3f2ed524a2ba,
            ],
        );
    }
}
