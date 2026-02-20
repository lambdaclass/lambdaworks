use super::parameters::*;
use super::utils::*;
use super::Fp;
use crate::alloc::vec::Vec;
use core::iter;

/// Shared core for Rescue-based hash functions (RPO, RPX).
///
/// Contains the S-box, MDS, round constant addition, and sponge logic
/// that is identical between RPO and RPX. The permutation itself differs
/// (RPO uses uniform full rounds, RPX uses FB/E/M round types), so it
/// is passed as a closure to the sponge.
pub(crate) struct RescueCore {
    pub(crate) m: usize,
    pub(crate) capacity: usize,
    pub(crate) rate: usize,
    pub(crate) round_constants: &'static [Fp],
    mds_matrix: Vec<Vec<Fp>>,
    mds_vector: MdsVector,
    mds_method: MdsMethod,
}

impl RescueCore {
    pub(crate) fn new(security_level: &SecurityLevel, mds_method: MdsMethod) -> Self {
        let m = get_state_size(security_level);
        let capacity = get_capacity(security_level);
        let rate = m - capacity;
        let mds_matrix = get_mds_matrix(security_level);
        let round_constants = get_round_constants(security_level);
        let mds_vector = get_mds_vector(security_level.clone());

        Self {
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
        }
    }

    pub(crate) fn apply_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA);
        }
    }

    pub(crate) fn apply_inverse_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA_INV);
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

    fn mds_ntt(&self, state: &[Fp]) -> Vec<Fp> {
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

        let omega_inv = omega.inv().expect("hardcoded omega is nonzero");
        let result = intt(&product_ntt, omega_inv);

        iter::once(result[0])
            .chain(result[1..].iter().rev().cloned())
            .collect()
    }

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

    pub(crate) fn apply_mds(&self, state: &mut [Fp]) {
        let new_state = match self.mds_method {
            MdsMethod::MatrixMultiplication => self.mds_matrix_vector_multiplication(state),
            MdsMethod::Ntt => self.mds_ntt(state),
            MdsMethod::Karatsuba => self.mds_karatsuba(state),
        };
        state.copy_from_slice(&new_state);
    }

    /// Adds round constants to the state.
    ///
    /// `offset` selects which set: 0 for ARK1, 1 for ARK2.
    pub(crate) fn add_round_constants(&self, state: &mut [Fp], round: usize, offset: usize) {
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

    /// Sponge-based hash. The `permutation` closure defines the specific
    /// permutation (RPO vs RPX).
    pub(crate) fn hash(&self, input_sequence: &[Fp], permutation: impl Fn(&mut [Fp])) -> Vec<Fp> {
        let mut state = vec![Fp::zero(); self.m];
        if !input_sequence.len().is_multiple_of(self.rate) {
            state[0] = Fp::one();
        }

        let absorb_range = self.capacity..self.capacity + self.rate;

        for chunk in input_sequence.chunks_exact(self.rate) {
            state[absorb_range.clone()].copy_from_slice(chunk);
            permutation(&mut state);
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
            permutation(&mut state);
        }

        state[self.capacity..self.capacity + self.rate / 2].to_vec()
    }

    pub(crate) fn hash_bytes(&self, input: &[u8], permutation: impl Fn(&mut [Fp])) -> Vec<Fp> {
        let field_elements = bytes_to_field_elements(input);
        self.hash(&field_elements, permutation)
    }
}
