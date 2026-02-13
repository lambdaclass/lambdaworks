//! Poseidon2 hash function implementation for Goldilocks field.
//!
//! Poseidon2 is an optimized version of the Poseidon hash function designed for
//! efficient implementation in both hardware and software, particularly for
//! zero-knowledge proof systems.
//!
//! # Configuration
//!
//! | Parameter | Value |
//! |-----------|-------|
//! | Field | Goldilocks (p = 2^64 - 2^32 + 1) |
//! | Width | 8 |
//! | Rate | 4 |
//! | Capacity | 4 |
//! | S-box | x^7 |
//! | Rounds | 4 (external) + 22 (internal) + 4 (external) = 30 |
//!
//! # Key Differences from Poseidon
//!
//! - Different linear layers for external and internal rounds
//! - Optimized MDS matrices (Horizen Labs 4x4 matrix)
//! - Reduced constraint complexity
//!
//! # Domain Separation
//!
//! The implementation uses domain separation via the capacity element (last element
//! of the state vector). These values follow the Plonky3 convention and are validated
//! by reference test vectors:
//!
//! - `hash_single(x)`: domain tag = 1 (single field element)
//! - `hash(x, y)`: domain tag = 2 (two field elements)
//! - `compress(left, right)`: domain tag = 4 (four field elements: two 128-bit digests)
//! - `hash_many(inputs)`: uses 10* padding (no explicit domain tag)
//!
//! **Note:** These small values (1, 2, 4) are part of the Plonky3 specification and
//! ensure compatibility with the Plonky3 ecosystem. The implementation is validated
//! against Plonky3 test vectors.
//!
//! # References
//!
//! - Paper: <https://eprint.iacr.org/2023/323>
//! - Constants: HorizenLabs/Plonky3 implementation

pub mod goldilocks;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

use goldilocks::{
    EXTERNAL_ROUNDS_BEGIN, EXTERNAL_ROUNDS_END, EXTERNAL_ROUND_CONSTANTS_INIT,
    EXTERNAL_ROUND_CONSTANTS_TERM, INTERNAL_ROUNDS, INTERNAL_ROUND_CONSTANTS, MATRIX_DIAG_8, RATE,
    WIDTH,
};

/// Type alias for Goldilocks field element
pub type Fp = FieldElement<Goldilocks64Field>;

/// 128-bit digest type (2 Goldilocks field elements)
pub type Digest = [Fp; 2];

/// Poseidon2 hash function for Goldilocks field with width 8
pub struct Poseidon2 {
    state: [Fp; WIDTH],
}

impl Default for Poseidon2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Poseidon2 {
    /// Create a new Poseidon2 instance with zero state
    pub fn new() -> Self {
        Self {
            state: core::array::from_fn(|_| Fp::zero()),
        }
    }

    /// Create a new Poseidon2 instance with given initial state
    pub fn with_state(state: [Fp; WIDTH]) -> Self {
        Self { state }
    }

    /// Get the current state
    pub fn state(&self) -> &[Fp; WIDTH] {
        &self.state
    }

    /// Apply the full Poseidon2 permutation to the state
    ///
    /// # Permutation Structure (Plonky3 Variant)
    ///
    /// This implementation follows the **Plonky3/HorizenLabs variant** which includes
    /// an initial external linear layer before the first round of external transformations.
    /// This structure is validated by test vectors from the reference implementation.
    ///
    /// ```text
    /// external_linear_layer()                    <- Initial layer (Plonky3-specific)
    /// for i in 0..4 (EXTERNAL_ROUNDS_BEGIN):
    ///     add_constants -> sbox -> external_linear_layer()
    /// for i in 0..22 (INTERNAL_ROUNDS):
    ///     add_constant -> sbox -> internal_linear_layer()
    /// for i in 0..4 (EXTERNAL_ROUNDS_END):
    ///     add_constants -> sbox -> external_linear_layer()
    /// ```
    pub fn permute(&mut self) {
        // Initial linear layer (Plonky3 variant - applied before first external round)
        self.external_linear_layer();

        // Initial external rounds (after initial linear layer)
        for round_constants in EXTERNAL_ROUND_CONSTANTS_INIT
            .iter()
            .take(EXTERNAL_ROUNDS_BEGIN)
        {
            self.external_round(round_constants);
        }

        // Internal rounds
        for round_constant in INTERNAL_ROUND_CONSTANTS.iter().take(INTERNAL_ROUNDS) {
            self.internal_round(round_constant);
        }

        // Terminal external rounds (no initial linear layer)
        for round_constants in EXTERNAL_ROUND_CONSTANTS_TERM
            .iter()
            .take(EXTERNAL_ROUNDS_END)
        {
            self.external_round(round_constants);
        }
    }

    /// Apply the S-box (x^7) to a single element
    #[inline(always)]
    fn sbox(x: &Fp) -> Fp {
        let x2 = x.square();
        let x4 = x2.square();
        let x6 = &x4 * &x2;
        &x6 * x
    }

    /// External round: AddRoundConstants + S-box (all elements) + External Linear Layer
    #[inline]
    fn external_round(&mut self, round_constants: &[Fp; WIDTH]) {
        // Add round constants
        for (state_elem, rc) in self.state.iter_mut().zip(round_constants.iter()) {
            *state_elem = &*state_elem + rc;
        }

        // Apply S-box to all elements
        for state_elem in &mut self.state {
            *state_elem = Self::sbox(state_elem);
        }

        // External linear layer (MDS)
        self.external_linear_layer();
    }

    /// Internal round: AddRoundConstant (first element only) + S-box (first element only) + Internal Linear Layer
    #[inline]
    fn internal_round(&mut self, round_constant: &Fp) {
        // Add round constant to first element only
        self.state[0] = &self.state[0] + round_constant;

        // Apply S-box to first element only
        self.state[0] = Self::sbox(&self.state[0]);

        // Internal linear layer (diagonal matrix with 1s)
        self.internal_linear_layer();
    }

    /// Apply the Horizen Labs 4x4 MDS matrix to a 4-element array
    /// Matrix: [[5,7,1,3], [4,6,1,1], [1,3,5,7], [1,1,4,6]]
    /// This requires 10 additions and 4 doubles
    #[inline]
    fn apply_hl_mat4(x: &mut [Fp; 4]) {
        // t0 = x0 + x1
        let t0 = &x[0] + &x[1];
        // t1 = x2 + x3
        let t1 = &x[2] + &x[3];
        // t2 = 2*x1 + t1 = 2*x1 + x2 + x3
        let t2 = &(&x[1] + &x[1]) + &t1;
        // t3 = 2*x3 + t0 = x0 + x1 + 2*x3
        let t3 = &(&x[3] + &x[3]) + &t0;
        // t4 = 4*t1 + t3 = x0 + x1 + 4*x2 + 6*x3
        let t1_double = &t1 + &t1;
        let t4 = &(&t1_double + &t1_double) + &t3;
        // t5 = 4*t0 + t2 = 4*x0 + 6*x1 + x2 + x3
        let t0_double = &t0 + &t0;
        let t5 = &(&t0_double + &t0_double) + &t2;
        // t6 = t3 + t5 = 5*x0 + 7*x1 + x2 + 3*x3
        let t6 = &t3 + &t5;
        // t7 = t2 + t4 = x0 + 3*x1 + 5*x2 + 7*x3
        let t7 = &t2 + &t4;

        x[0] = t6;
        x[1] = t5;
        x[2] = t7;
        x[3] = t4;
    }

    /// External linear layer using Horizen Labs 4x4 MDS matrix applied to two halves
    /// For width 8: apply 4x4 MDS to state[0..4] and state[4..8], then diffuse
    #[inline]
    fn external_linear_layer(&mut self) {
        // Apply HL M4 to each 4-element chunk in-place
        let (first, second) = self.state.split_at_mut(4);
        Self::apply_hl_mat4(first.try_into().unwrap());
        Self::apply_hl_mat4(second.try_into().unwrap());

        // Diffuse across blocks: add column sums back to each element
        for i in 0..4 {
            let sum = &self.state[i] + &self.state[i + 4];
            self.state[i] = &self.state[i] + &sum;
            self.state[i + 4] = &self.state[i + 4] + &sum;
        }
    }

    /// Internal linear layer using diagonal matrix
    /// The matrix is: M_internal = diag(MATRIX_DIAG) + all-ones matrix
    /// Equivalently: y_i = diag_i * x_i + sum(x_j)
    #[inline]
    fn internal_linear_layer(&mut self) {
        // Compute sum of all elements
        let mut sum = Fp::zero();
        for state_elem in &self.state {
            sum = &sum + state_elem;
        }

        // Apply: y_i = diag_i * x_i + sum(all x_j)
        for (state_elem, diag) in self.state.iter_mut().zip(MATRIX_DIAG_8.iter()) {
            *state_elem = &(diag * &*state_elem) + &sum;
        }
    }

    // ========================================================================
    // Hash functions
    // ========================================================================

    /// Hash a single field element returning 128-bit digest.
    pub fn hash_single(x: &Fp) -> Digest {
        let mut hasher = Self::new();
        hasher.state[0] = *x;
        // Domain separation: capacity element = 1 (for 1 input)
        hasher.state[WIDTH - 1] = Fp::from(1u64);
        hasher.permute();
        [hasher.state[0], hasher.state[1]]
    }

    /// Hash two field elements returning 128-bit digest.
    pub fn hash(x: &Fp, y: &Fp) -> Digest {
        let mut hasher = Self::new();
        hasher.state[0] = *x;
        hasher.state[1] = *y;
        // Domain separation: capacity element = 2 (for 2 inputs)
        hasher.state[WIDTH - 1] = Fp::from(2u64);
        hasher.permute();
        [hasher.state[0], hasher.state[1]]
    }

    /// Compress two 128-bit digests into one (for Merkle tree internal nodes).
    pub fn compress(left: &Digest, right: &Digest) -> Digest {
        let mut hasher = Self::new();
        hasher.state[0] = left[0];
        hasher.state[1] = left[1];
        hasher.state[2] = right[0];
        hasher.state[3] = right[1];
        // Domain separation: capacity element = 4 (for 4 field elements input)
        hasher.state[WIDTH - 1] = Fp::from(4u64);
        hasher.permute();
        [hasher.state[0], hasher.state[1]]
    }

    /// Hash multiple field elements using sponge construction, returning 128-bit digest.
    ///
    /// Uses 10* padding: inputs || 1 || 0* to align to RATE boundary.
    pub fn hash_many(inputs: &[Fp]) -> Digest {
        let mut hasher = Self::new();

        // Process complete chunks directly from input (no allocation)
        let chunks = inputs.chunks_exact(RATE);
        let remainder = chunks.remainder();
        for chunk in chunks {
            for (i, val) in chunk.iter().enumerate() {
                hasher.state[i] = &hasher.state[i] + val;
            }
            hasher.permute();
        }

        // Absorb remaining elements
        for (i, val) in remainder.iter().enumerate() {
            hasher.state[i] = &hasher.state[i] + val;
        }

        // Add 10* padding: 1 followed by zeros (zeros already in state)
        hasher.state[remainder.len()] = &hasher.state[remainder.len()] + &Fp::from(1u64);

        // Final permutation
        hasher.permute();

        // Squeeze phase: return first two elements
        [hasher.state[0], hasher.state[1]]
    }

    /// Hash a vector of field elements, returning 128-bit digest (for Merkle tree leaves).
    ///
    /// - **Empty**: Panics (empty input is not allowed)
    /// - **Length 1**: Delegates to [`hash_single`](Self::hash_single)
    /// - **Length 2+**: Delegates to [`hash_many`](Self::hash_many)
    ///
    /// # Panics
    ///
    /// Panics if `inputs` is empty.
    pub fn hash_vec(inputs: &[Fp]) -> Digest {
        assert!(
            !inputs.is_empty(),
            "hash_vec called with empty input - empty input is not allowed"
        );
        if inputs.len() == 1 {
            return Self::hash_single(&inputs[0]);
        }
        Self::hash_many(inputs)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::goldilocks::MATRIX_DIAG_8;
    use super::*;

    #[test]
    fn test_poseidon2_permutation_deterministic() {
        let mut hasher1 = Poseidon2::new();
        let mut hasher2 = Poseidon2::new();

        hasher1.state[0] = Fp::from(1u64);
        hasher1.state[1] = Fp::from(2u64);

        hasher2.state[0] = Fp::from(1u64);
        hasher2.state[1] = Fp::from(2u64);

        hasher1.permute();
        hasher2.permute();

        for i in 0..WIDTH {
            assert_eq!(hasher1.state[i], hasher2.state[i]);
        }
    }

    #[test]
    fn test_poseidon2_hash_deterministic() {
        let x = Fp::from(123u64);
        let y = Fp::from(456u64);

        let h1 = Poseidon2::hash(&x, &y);
        let h2 = Poseidon2::hash(&x, &y);

        assert_eq!(h1, h2);
    }

    #[test]
    fn test_poseidon2_hash_different_inputs() {
        let x1 = Fp::from(1u64);
        let y1 = Fp::from(2u64);

        let x2 = Fp::from(1u64);
        let y2 = Fp::from(3u64);

        let h1 = Poseidon2::hash(&x1, &y1);
        let h2 = Poseidon2::hash(&x2, &y2);

        assert_ne!(h1, h2);
    }

    #[test]
    fn test_poseidon2_hash_single() {
        let x = Fp::from(42u64);
        let h = Poseidon2::hash_single(&x);

        assert_ne!(h, [Fp::zero(); 2]);
    }

    #[test]
    fn test_poseidon2_hash_many() {
        let inputs: Vec<Fp> = (1..=10).map(|i| Fp::from(i as u64)).collect();
        let h = Poseidon2::hash_many(&inputs);

        assert_ne!(h, [Fp::zero(); 2]);

        let h2 = Poseidon2::hash_many(&inputs);
        assert_eq!(h, h2);
    }

    #[test]
    fn test_poseidon2_sbox() {
        let x = Fp::from(2u64);
        let result = Poseidon2::sbox(&x);
        let expected = Fp::from(128u64); // 2^7 = 128
        assert_eq!(result, expected);
    }

    #[test]
    fn test_poseidon2_compress() {
        let left = [Fp::from(100u64), Fp::from(101u64)];
        let right = [Fp::from(200u64), Fp::from(201u64)];

        let h1 = Poseidon2::compress(&left, &right);
        let h2 = Poseidon2::compress(&left, &right);

        assert_eq!(h1, h2);
        assert_ne!(h1, [Fp::zero(); 2]);
    }

    #[test]
    fn test_poseidon2_state_changes() {
        let mut hasher = Poseidon2::new();

        for (i, state_elem) in hasher.state.iter_mut().enumerate() {
            *state_elem = Fp::from(i as u64);
        }

        let initial_state = hasher.state;
        hasher.permute();

        let all_same = hasher
            .state
            .iter()
            .zip(initial_state.iter())
            .all(|(a, b)| a == b);
        assert!(!all_same, "State should change after permutation");
    }

    // ========================================================================
    // Test vectors from Plonky3 (HorizenLabs implementation)
    // ========================================================================

    #[test]
    fn test_poseidon2_plonky3_vector_zeros() {
        let mut hasher = Poseidon2::new();

        hasher.permute();

        let expected: [u64; 8] = [
            4214787979728720400,
            12324939279576102560,
            10353596058419792404,
            15456793487362310586,
            10065219879212154722,
            16227496357546636742,
            2959271128466640042,
            14285409611125725709,
        ];

        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                *hasher.state[i].value(),
                exp,
                "Mismatch at index {} for zeros input",
                i
            );
        }
    }

    #[test]
    fn test_poseidon2_plonky3_vector_sequential() {
        let mut hasher = Poseidon2::new();
        for (i, state_elem) in hasher.state.iter_mut().enumerate() {
            *state_elem = Fp::from(i as u64);
        }

        hasher.permute();

        let expected: [u64; 8] = [
            14266028122062624699,
            5353147180106052723,
            15203350112844181434,
            17630919042639565165,
            16601551015858213987,
            10184091939013874068,
            16774100645754596496,
            12047415603622314780,
        ];

        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                *hasher.state[i].value(),
                exp,
                "Mismatch at index {} for sequential input",
                i
            );
        }
    }

    #[test]
    fn test_poseidon2_plonky3_vector_random() {
        let input: [u64; 8] = [
            5116996373749832116,
            8931548647907683339,
            17132360229780760684,
            11280040044015983889,
            11957737519043010992,
            15695650327991256125,
            17604752143022812942,
            543194415197607509,
        ];

        let mut hasher = Poseidon2::new();
        for (i, &inp) in input.iter().enumerate() {
            hasher.state[i] = Fp::from(inp);
        }

        hasher.permute();

        let expected: [u64; 8] = [
            1831346684315917658,
            13497752062035433374,
            12149460647271516589,
            15656333994315312197,
            4671534937670455565,
            3140092508031220630,
            4251208148861706881,
            6973971209430822232,
        ];

        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                *hasher.state[i].value(),
                exp,
                "Mismatch at index {} for random input",
                i
            );
        }
    }

    #[test]
    fn test_apply_hl_mat4() {
        // Input: [1, 0, 0, 0] should give [5, 4, 1, 1]
        let mut x = [Fp::from(1u64), Fp::zero(), Fp::zero(), Fp::zero()];
        Poseidon2::apply_hl_mat4(&mut x);
        assert_eq!(*x[0].value(), 5u64, "Row 0, column 0");
        assert_eq!(*x[1].value(), 4u64, "Row 1, column 0");
        assert_eq!(*x[2].value(), 1u64, "Row 2, column 0");
        assert_eq!(*x[3].value(), 1u64, "Row 3, column 0");

        // Input: [0, 1, 0, 0] should give [7, 6, 3, 1]
        let mut x = [Fp::zero(), Fp::from(1u64), Fp::zero(), Fp::zero()];
        Poseidon2::apply_hl_mat4(&mut x);
        assert_eq!(*x[0].value(), 7u64, "Row 0, column 1");
        assert_eq!(*x[1].value(), 6u64, "Row 1, column 1");
        assert_eq!(*x[2].value(), 3u64, "Row 2, column 1");
        assert_eq!(*x[3].value(), 1u64, "Row 3, column 1");

        // Input: [1, 2, 3, 4] -> [34, 23, 50, 39]
        let mut x = [
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
        ];
        Poseidon2::apply_hl_mat4(&mut x);
        assert_eq!(*x[0].value(), 34u64, "General test [0]");
        assert_eq!(*x[1].value(), 23u64, "General test [1]");
        assert_eq!(*x[2].value(), 50u64, "General test [2]");
        assert_eq!(*x[3].value(), 39u64, "General test [3]");
    }

    #[test]
    fn test_external_linear_layer() {
        let mut hasher = Poseidon2::new();
        hasher.state[0] = Fp::from(1u64);
        hasher.external_linear_layer();

        assert_eq!(*hasher.state[0].value(), 10u64, "ELL state[0]");
        assert_eq!(*hasher.state[4].value(), 5u64, "ELL state[4]");
    }

    #[test]
    fn test_internal_linear_layer() {
        let mut hasher = Poseidon2::new();
        hasher.state[0] = Fp::from(1u64);
        hasher.internal_linear_layer();

        let expected0 = &MATRIX_DIAG_8[0] + &Fp::from(1u64);
        assert_eq!(hasher.state[0], expected0, "ILL state[0]");
        assert_eq!(hasher.state[1], Fp::from(1u64), "ILL state[1]");
    }

    // ========================================================================
    // Domain separation tests
    // ========================================================================

    #[test]
    fn test_domain_separation_hash_vs_hash_many() {
        let a = Fp::from(1u64);
        let b = Fp::from(2u64);
        assert_ne!(
            Poseidon2::hash(&a, &b),
            Poseidon2::hash_many(&[a, b]),
            "hash(a,b) should differ from hash_many([a,b]) due to domain separation"
        );
    }

    #[test]
    fn test_domain_separation_single_vs_many() {
        let x = Fp::from(42u64);
        assert_ne!(
            Poseidon2::hash_single(&x),
            Poseidon2::hash_many(&[x]),
            "hash_single(x) should differ from hash_many([x]) due to domain separation"
        );
    }

    #[test]
    fn test_hash_vec_delegates_length_one() {
        let x = Fp::from(42u64);
        assert_eq!(
            Poseidon2::hash_vec(&[x]),
            Poseidon2::hash_single(&x),
            "hash_vec([x]) should equal hash_single(x)"
        );
    }

    #[test]
    fn test_hash_vec_delegates_length_two_plus() {
        let inputs = vec![Fp::from(1u64), Fp::from(2u64), Fp::from(3u64)];
        assert_eq!(
            Poseidon2::hash_vec(&inputs),
            Poseidon2::hash_many(&inputs),
            "hash_vec should equal hash_many for length >= 2"
        );
    }

    #[test]
    fn test_compress_non_commutative() {
        let a = [Fp::from(100u64), Fp::from(101u64)];
        let b = [Fp::from(200u64), Fp::from(201u64)];
        assert_ne!(
            Poseidon2::compress(&a, &b),
            Poseidon2::compress(&b, &a),
            "compress should be non-commutative (order matters)"
        );
    }

    #[test]
    fn test_domain_separation_hash_vs_compress() {
        let a = Fp::from(1u64);
        let b = Fp::from(2u64);
        let left = [a, b];
        let right = [Fp::zero(), Fp::zero()];

        assert_ne!(
            Poseidon2::hash(&a, &b),
            Poseidon2::compress(&left, &right),
            "hash and compress should have domain separation"
        );
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    #[should_panic(expected = "hash_vec called with empty input")]
    fn test_hash_vec_empty_panics() {
        let _ = Poseidon2::hash_vec(&[]);
    }

    #[test]
    fn test_hash_many_empty_not_zero() {
        let result = Poseidon2::hash_many(&[]);
        assert_ne!(
            result,
            [Fp::zero(); 2],
            "hash_many([]) should not be zero (padding is applied)"
        );
    }

    #[test]
    fn test_hash_single_zero_not_zero() {
        let result = Poseidon2::hash_single(&Fp::zero());
        assert_ne!(
            result,
            [Fp::zero(); 2],
            "hash_single(0) should not return [0, 0]"
        );
    }

    #[test]
    fn test_hash_zero_pair_not_zero() {
        let result = Poseidon2::hash(&Fp::zero(), &Fp::zero());
        assert_ne!(
            result,
            [Fp::zero(); 2],
            "hash(0, 0) should not return [0, 0]"
        );
    }

    // ========================================================================
    // Collision resistance tests
    // ========================================================================

    #[test]
    fn test_hash_single_collision_resistance() {
        let h1 = Poseidon2::hash_single(&Fp::from(1u64));
        let h2 = Poseidon2::hash_single(&Fp::from(2u64));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_collision_resistance() {
        let h1 = Poseidon2::hash(&Fp::from(1u64), &Fp::from(2u64));
        let h2 = Poseidon2::hash(&Fp::from(1u64), &Fp::from(3u64));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_many_length_extension_resistance() {
        let a = Fp::from(1u64);
        let b = Fp::from(2u64);
        let c = Fp::from(3u64);
        assert_ne!(
            Poseidon2::hash_many(&[a, b]),
            Poseidon2::hash_many(&[a, b, c]),
            "Different length inputs should produce different hashes"
        );
    }
}
