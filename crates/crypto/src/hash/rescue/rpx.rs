use super::parameters::*;
use super::rescue_core::RescueCore;
use super::Fp;
use crate::alloc::vec::Vec;

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
pub struct Rpx256 {
    core: RescueCore,
}

impl Default for Rpx256 {
    fn default() -> Self {
        Self::new(MdsMethod::MatrixMultiplication)
    }
}

impl Rpx256 {
    pub fn new(mds_method: MdsMethod) -> Self {
        Self {
            core: RescueCore::new(&SecurityLevel::Sec128, mds_method),
        }
    }

    fn apply_full_block(&self, state: &mut [Fp], round: usize) {
        self.core.apply_mds(state);
        self.core.add_round_constants(state, round, 0);
        RescueCore::apply_sbox(state);
        self.core.apply_mds(state);
        self.core.add_round_constants(state, round, 1);
        RescueCore::apply_inverse_sbox(state);
    }

    /// Applies an extension round: ARK1 followed by the cubic extension S-box.
    ///
    /// Per XHash-12 spec (Section 3.2), E rounds only use ARK1 (offset=0).
    /// The ARK2 constants for E-round indices (rounds 1, 3, 5) are allocated
    /// in the round constants array but intentionally unused.
    fn apply_extension_round(&self, state: &mut [Fp], round: usize) {
        self.core.add_round_constants(state, round, 0);
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
        self.core.apply_mds(state);
        self.core.add_round_constants(state, round, 0);
    }

    pub fn permutation(&self, state: &mut [Fp]) {
        debug_assert_eq!(
            state.len(),
            self.core.m,
            "State size must match state width"
        );
        self.apply_full_block(state, 0);
        self.apply_extension_round(state, 1);
        self.apply_full_block(state, 2);
        self.apply_extension_round(state, 3);
        self.apply_full_block(state, 4);
        self.apply_extension_round(state, 5);
        self.apply_middle_round(state, 6);
    }

    pub fn hash(&self, input_sequence: &[Fp]) -> Vec<Fp> {
        self.core.hash(input_sequence, |s| self.permutation(s))
    }

    pub fn hash_bytes(&self, input: &[u8]) -> Vec<Fp> {
        self.core.hash_bytes(input, |s| self.permutation(s))
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
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rpx.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA));

        RescueCore::apply_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn test_apply_inverse_sbox() {
        let mut rng = StdRng::seed_from_u64(2);
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rpx.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA_INV));

        RescueCore::apply_inverse_sbox(&mut state);
        assert_eq!(expected, state);
    }

    #[test]
    fn test_permutation() {
        let mut rng = StdRng::seed_from_u64(3);
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rpx.core.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

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
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..8).map(Fp::from).collect();
        let hash_output = rpx.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_multiple_chunks() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..16).map(Fp::from).collect();
        let hash_output = rpx.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_with_padding() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..5).map(Fp::from).collect();
        let hash_output = rpx.hash(&input_sequence);
        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_hash_bytes() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        let input_bytes = b"Rescue Prime XHash";
        let hash_output = rpx.hash_bytes(input_bytes);

        assert_eq!(hash_output.len(), 4);
    }

    #[test]
    fn test_padding_collision_prevention() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);
        // 7 zeroes gets padding 1 appended → [0,0,0,0,0,0,0,1]
        // 8 elements [0,0,0,0,0,0,0,1] is rate-aligned, no padding
        // Without domain separation these would collide
        let hash_7_zeros = rpx.hash(&[Fp::zero(); 7]);
        let mut eight_elems = vec![Fp::zero(); 7];
        eight_elems.push(Fp::one());
        let hash_8_with_one = rpx.hash(&eight_elems);
        assert_ne!(
            hash_7_zeros, hash_8_with_one,
            "hash([0;7]) must differ from hash([0,0,0,0,0,0,0,1])"
        );
    }

    #[test]
    fn hash_padding() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);

        let input1 = vec![1u8, 2, 3];
        let input2 = vec![1u8, 2, 3, 0];
        let hash1 = rpx.hash_bytes(&input1);
        let hash2 = rpx.hash_bytes(&input2);
        assert_ne!(hash1, hash2);

        let input1 = vec![1_u8, 2, 3, 4, 5, 6];
        let input2 = vec![1_u8, 2, 3, 4, 5, 6, 0];
        let hash1 = rpx.hash_bytes(&input1);
        let hash2 = rpx.hash_bytes(&input2);
        assert_ne!(hash1, hash2);

        let input1 = vec![1_u8, 2, 3, 4, 5, 6, 7, 0, 0];
        let input2 = vec![1_u8, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0];
        let hash1 = rpx.hash_bytes(&input1);
        let hash2 = rpx.hash_bytes(&input2);
        assert_ne!(hash1, hash2);
    }

    #[cfg(feature = "std")]
    #[test]
    fn sponge_zeroes_collision() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);

        let mut zeroes = Vec::new();
        let mut hashes = std::collections::HashSet::new();

        for _ in 0..255 {
            let hash = rpx.hash(&zeroes);
            assert!(hashes.insert(hash));
            zeroes.push(Fp::zero());
        }
    }

    #[test]
    fn test_mds_methods_consistency() {
        let rpx_matrix = Rpx256::new(MdsMethod::MatrixMultiplication);
        let rpx_ntt = Rpx256::new(MdsMethod::Ntt);
        let rpx_karatsuba = Rpx256::new(MdsMethod::Karatsuba);

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

    /// Cross-validation test vectors generated from miden-crypto's Rpx256::apply_permutation.
    /// Only the raw permutation is compared — sponge layouts differ between implementations
    /// (our capacity at 0..3, rate at 4..11 vs miden's rate at 0..7, capacity at 8..11).
    #[test]
    fn test_permutation_cross_validation_miden() {
        let rpx = Rpx256::new(MdsMethod::MatrixMultiplication);

        // Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        let mut state: Vec<Fp> = (1..=12).map(|i| Fp::from(i as u64)).collect();
        rpx.permutation(&mut state);
        let expected: Vec<Fp> = [
            5437748534614640079u64,
            854874938920055048,
            18278654462140408466,
            17240697175332752171,
            7310175166461302633,
            18290390891494061033,
            10686820761628507650,
            15328173731076229406,
            4281259797668742483,
            8756723097944267591,
            7079891540869279681,
            12686994217342534069,
        ]
        .iter()
        .map(|&v| Fp::from(v))
        .collect();
        assert_eq!(state, expected, "permutation mismatch for input [1..12]");

        // Input: [0; 12]
        let mut state = vec![Fp::zero(); 12];
        rpx.permutation(&mut state);
        let expected: Vec<Fp> = [
            8760086638283468260u64,
            18228666152919569253,
            4041825754230271128,
            16906183286731764961,
            4664375192219530269,
            271590372761485506,
            5612474514543166805,
            8933101171974180471,
            1556877437237031065,
            7026397410864970258,
            15101742939622740655,
            4524429088483979565,
        ]
        .iter()
        .map(|&v| Fp::from(v))
        .collect();
        assert_eq!(state, expected, "permutation mismatch for input [0;12]");
    }
}
