use super::parameters::*;
use super::utils::*;
use super::Fp;
use super::MdsMethod;
use crate::alloc::vec::Vec;
use core::iter;
use lambdaworks_math::field::errors::FieldError;

/// Implementation of RPX (Rescue Prime eXtension) hash function.
///
/// Based on XHash-12 construction from paper: https://eprint.iacr.org/2023/1045
/// Reference implementation: https://github.com/0xMiden/crypto
///
/// RPX is ~2x faster than RPO by using cubic extension field arithmetic
/// in the extension rounds (E).
///
/// Permutation: (FB)(E)(FB)(E)(FB)(E)(M) where:
/// - FB: Full Block (MDS → add_const → S-box → MDS → add_const → inv_S-box)
/// - E: Extension round (uses cubic extension field x³ - x - 1)
/// - M: Middle round (MDS → add_const)
#[allow(dead_code)]
const NUM_ROUNDS: usize = 7;

pub struct Rpx256 {
    m: usize,
    capacity: usize,
    rate: usize,
    round_constants: &'static [Fp],
    mds_matrix: Vec<Vec<Fp>>,
    mds_vector: MdsVector,
    mds_method: MdsMethod,
}

impl Default for Rpx256 {
    fn default() -> Self {
        Self::new(MdsMethod::MatrixMultiplication).unwrap()
    }
}

impl Rpx256 {
    pub fn new(mds_method: MdsMethod) -> Result<Self, &'static str> {
        let security_level = SecurityLevel::Sec128;
        let m = get_state_size(&security_level);
        let capacity = get_capacity(&security_level);
        let rate = m - capacity;
        let mds_matrix = get_mds_matrix(&security_level);
        let round_constants = get_round_constants(&security_level);
        let mds_vector = get_mds_vector(security_level);

        Ok(Self {
            m,
            capacity,
            rate,
            round_constants,
            mds_matrix: match mds_matrix {
                MdsMatrix::Mds128(matrix) => matrix.iter().map(|&row| row.to_vec()).collect(),
                MdsMatrix::Mds160(matrix) => matrix.iter().map(|&row| row.to_vec()).collect(),
            },
            mds_vector,
            mds_method,
        })
    }

    pub fn apply_inverse_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA_INV);
        }
    }

    pub fn apply_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA);
        }
    }

    fn mds_matrix_vector_multiplication(&self, state: &[Fp]) -> Vec<Fp> {
        debug_assert_eq!(
            state.len(),
            self.m,
            "State size must match MDS matrix dimension"
        );
        let m = state.len();
        let mut new_state = vec![Fp::zero(); m];

        for (i, new_value) in new_state.iter_mut().enumerate() {
            for (j, state_value) in state.iter().enumerate() {
                *new_value += self.mds_matrix[i][j] * state_value;
            }
        }

        new_state
    }

    /// Performs MDS using Number Theoretic Transform.
    fn mds_ntt(&self, state: &[Fp]) -> Result<Vec<Fp>, FieldError> {
        let m = state.len();
        let omega = if m == 12 {
            Fp::from(281474976645120u64)
        } else {
            Fp::from(17293822564807737345u64)
        };
        let mds_vector = self.mds_vector.as_slice();

        let mds_ntt = ntt(mds_vector, omega);
        let state_rev: Vec<Fp> = iter::once(state[0])
            .chain(state[1..].iter().rev().cloned())
            .collect();
        let state_ntt = ntt(&state_rev, omega);

        let mut product_ntt = vec![Fp::zero(); m];
        for i in 0..m {
            product_ntt[i] = mds_ntt[i] * state_ntt[i];
        }

        let omega_inv = omega.inv()?;
        let result = intt(&product_ntt, omega_inv)?;

        Ok(iter::once(result[0])
            .chain(result[1..].iter().rev().cloned())
            .collect())
    }

    /// Performs MDS using the Karatsuba algorithm.
    fn mds_karatsuba(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let mds_vector = self.mds_vector.as_slice();
        let mds_rev: Vec<Fp> = iter::once(mds_vector[0])
            .chain(mds_vector[1..].iter().rev().cloned())
            .collect();

        let conv = karatsuba(&mds_rev, state);

        let mut result = vec![Fp::zero(); m];
        result[..m].copy_from_slice(&conv[..m]);
        for i in m..conv.len() {
            result[i - m] += conv[i];
        }

        result
    }

    fn apply_mds(&self, state: &mut [Fp]) -> Result<(), FieldError> {
        let new_state = match self.mds_method {
            MdsMethod::MatrixMultiplication => self.mds_matrix_vector_multiplication(state),
            MdsMethod::Ntt => self.mds_ntt(state)?,
            MdsMethod::Karatsuba => self.mds_karatsuba(state),
        };
        state.copy_from_slice(&new_state);
        Ok(())
    }

    fn add_round_constants(&self, state: &mut [Fp], round: usize, offset: usize) {
        let m = self.m;
        let rc = &self.round_constants[round * 2 * m + offset * m..round * 2 * m + offset * m + m];

        state
            .iter_mut()
            .zip(rc.iter())
            .take(m)
            .for_each(|(state_elem, &constant)| {
                *state_elem += constant;
            });
    }

    fn apply_full_block(&self, state: &mut [Fp], round: usize) {
        let _ = self.apply_mds(state);
        self.add_round_constants(state, round, 0);
        Self::apply_sbox(state);
        let _ = self.apply_mds(state);
        self.add_round_constants(state, round, 1);
        Self::apply_inverse_sbox(state);
    }

    fn apply_extension_round(&self, state: &mut [Fp], round: usize) {
        self.add_round_constants(state, round, 0);
        Self::apply_ext_sbox(state);
    }

    /// Applies the extension S-box using cubic extension field arithmetic.
    /// The state is interpreted as 4 elements in the cubic extension field
    /// (x³ - x - 1), and each element is raised to the 7th power.
    fn apply_ext_sbox(state: &mut [Fp]) {
        debug_assert!(state.len() >= 12, "State must have at least 12 elements");
        let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] = [
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
            state[8], state[9], state[10], state[11],
        ];

        let ext0 = cubic_ext_power7([s0, s1, s2]);
        let ext1 = cubic_ext_power7([s3, s4, s5]);
        let ext2 = cubic_ext_power7([s6, s7, s8]);
        let ext3 = cubic_ext_power7([s9, s10, s11]);

        state[0] = ext0[0];
        state[1] = ext0[1];
        state[2] = ext0[2];
        state[3] = ext1[0];
        state[4] = ext1[1];
        state[5] = ext1[2];
        state[6] = ext2[0];
        state[7] = ext2[1];
        state[8] = ext2[2];
        state[9] = ext3[0];
        state[10] = ext3[1];
        state[11] = ext3[2];
    }

    fn apply_middle_round(&self, state: &mut [Fp], round: usize) {
        let _ = self.apply_mds(state);
        self.add_round_constants(state, round, 0);
    }

    pub fn permutation(&self, state: &mut [Fp]) {
        debug_assert_eq!(state.len(), self.m, "State size must match state width");
        self.apply_full_block(state, 0);
        self.apply_extension_round(state, 1);
        self.apply_full_block(state, 2);
        self.apply_extension_round(state, 3);
        self.apply_full_block(state, 4);
        self.apply_extension_round(state, 5);
        self.apply_middle_round(state, 6);
    }

    pub fn hash(&self, input_sequence: &[Fp]) -> Vec<Fp> {
        let mut state = vec![Fp::zero(); self.m];

        let absorb_range = self.capacity..self.capacity + self.rate;

        for chunk in input_sequence.chunks_exact(self.rate) {
            state[absorb_range.clone()].copy_from_slice(chunk);
            self.permutation(&mut state);
        }

        let remainder = &input_sequence[input_sequence.len() / self.rate * self.rate..];
        if !remainder.is_empty() {
            debug_assert!(
                remainder.len() < self.rate,
                "Remainder must be smaller than rate"
            );
            let mut last_chunk = vec![Fp::zero(); self.rate];
            last_chunk[..remainder.len()].copy_from_slice(remainder);
            last_chunk[remainder.len()] = Fp::one();
            state[absorb_range.clone()].copy_from_slice(&last_chunk);
            self.permutation(&mut state);
        }

        state[self.capacity..self.capacity + self.rate / 2].to_vec()
    }

    pub fn hash_bytes(&self, input: &[u8]) -> Vec<Fp> {
        let field_elements = bytes_to_field_elements(input);
        self.hash(&field_elements)
    }
}

// CUBIC EXTENSION FIELD OPERATIONS
// ================================================================================================
// Implementation of cubic extension field arithmetic for RPX (XHash-12).
// Paper: https://eprint.iacr.org/2023/1045 (XHash: Efficient STARK-friendly Hash Function)
// Based on Miden Crypto implementation: https://github.com/0xMiden/crypto
//
// Cubic extension over the irreducible polynomial x³ - x - 1.
// Elements are represented as [a0, a1, a2] = a0 + a1*φ + a2*φ² where φ³ = φ + 1.

#[inline(always)]
fn cubic_ext_mul(a: [Fp; 3], b: [Fp; 3]) -> [Fp; 3] {
    let a0b0 = a[0] * b[0];
    let a1b1 = a[1] * b[1];
    let a2b2 = a[2] * b[2];

    let a0b0_a0b1_a1b0_a1b1 = (a[0] + a[1]) * (b[0] + b[1]);
    let a0b0_a0b2_a2b0_a2b2 = (a[0] + a[2]) * (b[0] + b[2]);
    let a1b1_a1b2_a2b1_a2b2 = (a[1] + a[2]) * (b[1] + b[2]);

    let a0b0_minus_a1b1 = a0b0 - a1b1;

    let a0b0_a1b2_a2b1 = a1b1_a1b2_a2b1_a2b2 + a0b0_minus_a1b1 - a2b2;
    let a0b1_a1b0_a1b2_a2b1_a2b2 = a0b0_a0b1_a1b0_a1b1 + a1b1_a1b2_a2b1_a2b2 - a1b1.double() - a0b0;
    let a0b2_a1b1_a2b0_a2b2 = a0b0_a0b2_a2b0_a2b2 - a0b0_minus_a1b1;

    [
        a0b0_a1b2_a2b1,
        a0b1_a1b0_a1b2_a2b1_a2b2,
        a0b2_a1b1_a2b0_a2b2,
    ]
}

#[inline(always)]
fn cubic_ext_square(a: [Fp; 3]) -> [Fp; 3] {
    let a0 = a[0];
    let a1 = a[1];
    let a2 = a[2];

    let a2_sq = a2.square();
    let a1_a2 = a1 * a2;

    let out0 = a0.square() + a1_a2.double();
    let out1 = (a0 * a1 + a1_a2).double() + a2_sq;
    let out2 = (a0 * a2).double() + a1.square() + a2_sq;

    [out0, out1, out2]
}

#[inline(always)]
fn cubic_ext_power7(a: [Fp; 3]) -> [Fp; 3] {
    let a2 = cubic_ext_square(a);
    let a3 = cubic_ext_mul(a2, a);
    let a6 = cubic_ext_square(a3);
    cubic_ext_mul(a6, a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rand_field_element(rng: &mut StdRng) -> Fp {
        Fp::from(rng.gen::<u64>())
    }

    #[test]
    fn test_apply_sbox() {
        let mut rng = StdRng::seed_from_u64(1);
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let mut state: Vec<Fp> = (0..rpx.m).map(|_| rand_field_element(&mut rng)).collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA));

        Rpx256::apply_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn test_apply_inverse_sbox() {
        let mut rng = StdRng::seed_from_u64(2);
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let mut state: Vec<Fp> = (0..rpx.m).map(|_| rand_field_element(&mut rng)).collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA_INV));

        Rpx256::apply_inverse_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn test_permutation() {
        let mut rng = StdRng::seed_from_u64(3);
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let mut state: Vec<Fp> = (0..rpx.m).map(|_| rand_field_element(&mut rng)).collect();

        let mut expected_state = state.clone();

        rpx.apply_full_block(&mut expected_state, 0);
        rpx.apply_extension_round(&mut expected_state, 1);
        rpx.apply_full_block(&mut expected_state, 2);
        rpx.apply_extension_round(&mut expected_state, 3);
        rpx.apply_full_block(&mut expected_state, 4);
        rpx.apply_extension_round(&mut expected_state, 5);
        rpx.apply_middle_round(&mut expected_state, 6);

        rpx.permutation(&mut state);

        assert_eq!(expected_state, state);
    }

    #[test]
    fn test_hash_single_chunk() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let input_sequence: Vec<Fp> = (0..8).map(Fp::from).collect();
        let hash_output = rpx.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_multiple_chunks() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let input_sequence: Vec<Fp> = (0..16).map(Fp::from).collect();
        let hash_output = rpx.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_with_padding() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let input_sequence: Vec<Fp> = (0..5).map(Fp::from).collect();
        let hash_output = rpx.hash(&input_sequence);
        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_bytes() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let input_bytes = b"Rescue Prime XHash";
        let hash_output = rpx.hash_bytes(input_bytes);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_mds_methods_consistency() {
        let rpx_matrix = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();
        let rpx_ntt = Rpx256::new(MdsMethod::Ntt).unwrap();
        let rpx_karatsuba = Rpx256::new(MdsMethod::Karatsuba).unwrap();

        let input = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
            Fp::from(5u64),
            Fp::from(6u64),
            Fp::from(7u64),
            Fp::from(8u64),
            Fp::from(9u64),
        ];

        let hash_matrix = rpx_matrix.hash(&input);
        let hash_ntt = rpx_ntt.hash(&input);
        let hash_karatsuba = rpx_karatsuba.hash(&input);

        assert_eq!(hash_matrix, hash_ntt);
        assert_eq!(hash_ntt, hash_karatsuba);
    }

    #[test]
    fn test_cubic_ext_power7_unit() {
        // Test with [1, 0, 0] - the multiplicative identity
        let x = [Fp::one(), Fp::zero(), Fp::zero()];
        let x7 = cubic_ext_power7(x);
        assert_eq!(x7, x, "1^7 should equal 1 in cubic extension");
    }

    #[test]
    fn test_cubic_ext_power7_phi() {
        // Test with [0, 1, 0] - just φ (root of x³ - x - 1)
        let phi = [Fp::zero(), Fp::one(), Fp::zero()];
        let phi7 = cubic_ext_power7(phi);
        // φ^7 should be some combination - verify it's not equal to φ
        assert_ne!(phi7, phi, "φ^7 should not equal φ");
    }

    #[test]
    fn test_cubic_ext_power7_consistency() {
        // Test that power7 is deterministic
        let x = [Fp::from(42u64), Fp::from(17u64), Fp::from(99u64)];
        let x7_a = cubic_ext_power7(x);
        let x7_b = cubic_ext_power7(x);
        assert_eq!(x7_a, x7_b, "power7 should be deterministic");
    }

    #[test]
    fn test_cubic_ext_square_mul_consistency() {
        // Test that square gives same result as mul with itself
        let a = [Fp::from(3u64), Fp::from(5u64), Fp::from(7u64)];
        let square_result = cubic_ext_square(a);
        let mul_result = cubic_ext_mul(a, a);
        assert_eq!(
            square_result, mul_result,
            "square(a) should equal mul(a, a)"
        );
    }

    #[test]
    fn test_cubic_ext_mul_commutative() {
        // Test commutativity of multiplication
        let a = [Fp::from(2u64), Fp::from(3u64), Fp::from(5u64)];
        let b = [Fp::from(7u64), Fp::from(11u64), Fp::from(13u64)];
        let ab = cubic_ext_mul(a, b);
        let ba = cubic_ext_mul(b, a);
        assert_eq!(ab, ba, "Multiplication should be commutative");
    }

    #[test]
    fn test_cubic_ext_mul_identity() {
        // Test that multiplying by identity gives same element
        let identity = [Fp::one(), Fp::zero(), Fp::zero()];
        let a = [Fp::from(42u64), Fp::from(17u64), Fp::from(99u64)];
        let result = cubic_ext_mul(a, identity);
        assert_eq!(result, a, "a * 1 should equal a");
    }
}
