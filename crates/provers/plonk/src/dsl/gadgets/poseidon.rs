//! Poseidon hash gadget for circuit-based hashing.
//!
//! This implements the Poseidon hash function as a circuit gadget,
//! following the HADES design (full rounds + partial rounds + full rounds).
//!
//! # Configuration
//!
//! Default parameters match StarkNet's Poseidon:
//! - State width: 3
//! - Rate: 2, Capacity: 1
//! - Alpha (S-box exponent): 3
//! - Full rounds: 8 (4 before partials, 4 after)
//! - Partial rounds: 83
//!
//! # Usage
//!
//! ```ignore
//! let hash = PoseidonHash::synthesize(&mut builder, vec![a, b])?;
//! ```

use crate::dsl::builder::CircuitBuilder;
use crate::dsl::gadgets::{Gadget, GadgetError};
use crate::dsl::types::FieldVar;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Poseidon hash gadget parameters.
///
/// These define the structure of the Poseidon permutation.
pub trait PoseidonParams<F: IsField>: Clone {
    /// State width (typically 3)
    const STATE_WIDTH: usize;

    /// Rate (number of input elements absorbed per permutation)
    const RATE: usize;

    /// S-box exponent (typically 3, 5, or 7)
    const ALPHA: u32;

    /// Number of full rounds
    const FULL_ROUNDS: usize;

    /// Number of partial rounds
    const PARTIAL_ROUNDS: usize;

    /// Get round constant for given round and position
    fn round_constant(round: usize, position: usize) -> FieldElement<F>;

    /// Get MDS matrix element at (row, col)
    fn mds_element(row: usize, col: usize) -> FieldElement<F>;
}

/// Default Poseidon parameters (simplified for testing).
///
/// For production use with specific curves, provide concrete parameters.
#[derive(Clone)]
pub struct DefaultPoseidonParams;

impl<F: IsField> PoseidonParams<F> for DefaultPoseidonParams {
    const STATE_WIDTH: usize = 3;
    const RATE: usize = 2;
    const ALPHA: u32 = 3;
    const FULL_ROUNDS: usize = 8;
    const PARTIAL_ROUNDS: usize = 83;

    fn round_constant(_round: usize, _position: usize) -> FieldElement<F> {
        // Simplified: use deterministic but non-zero constants
        // In production, use cryptographically derived constants
        FieldElement::<F>::from(1u64)
    }

    fn mds_element(row: usize, col: usize) -> FieldElement<F> {
        // Simplified MDS matrix: Cauchy matrix variant
        // M[i][j] = 1 / (x_i + y_j) where x and y are distinct
        // Simplified: just return sum + 1 for now
        FieldElement::<F>::from((row + col + 1) as u64)
    }
}

/// Poseidon hash gadget.
///
/// Hashes a variable number of field elements using the Poseidon permutation.
pub struct PoseidonHash<P = DefaultPoseidonParams> {
    _params: core::marker::PhantomData<P>,
}

impl<F: IsField, P: PoseidonParams<F>> Gadget<F> for PoseidonHash<P> {
    type Input = Vec<FieldVar>;
    type Output = FieldVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        inputs: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        if inputs.is_empty() {
            return Err(GadgetError::InvalidInput(
                "Poseidon requires at least one input".to_string(),
            ));
        }

        // Initialize state with capacity element (0) followed by rate elements
        let mut state: Vec<FieldVar> = Vec::with_capacity(P::STATE_WIDTH);

        // First element is capacity (initialized to 0)
        state.push(builder.constant(FieldElement::<F>::zero()));

        // Add input elements (pad with zeros if needed)
        for i in 0..P::RATE {
            if i < inputs.len() {
                state.push(inputs[i]);
            } else {
                state.push(builder.constant(FieldElement::<F>::zero()));
            }
        }

        // If more inputs than rate, need sponge construction
        if inputs.len() > P::RATE {
            // Apply initial permutation
            poseidon_permutation::<F, P>(builder, &mut state)?;

            // Absorb remaining inputs in chunks
            let mut idx = P::RATE;
            while idx < inputs.len() {
                // XOR (add) new inputs to rate portion
                for i in 0..P::RATE {
                    if idx + i < inputs.len() {
                        state[1 + i] = builder.add(&state[1 + i], &inputs[idx + i]);
                    }
                }
                idx += P::RATE;

                // Apply permutation
                poseidon_permutation::<F, P>(builder, &mut state)?;
            }
        } else {
            // Single permutation for small inputs
            poseidon_permutation::<F, P>(builder, &mut state)?;
        }

        // Output is first rate element (state[1])
        Ok(state[1])
    }

    fn constraint_count() -> usize {
        // Rough estimate based on rounds
        // Each full round: STATE_WIDTH S-box constraints + MDS
        // Each partial round: 1 S-box constraint + MDS
        let full_round_constraints = P::FULL_ROUNDS * P::STATE_WIDTH * 2;
        let partial_round_constraints = P::PARTIAL_ROUNDS * 2;
        full_round_constraints + partial_round_constraints
    }

    fn name() -> &'static str {
        "PoseidonHash"
    }
}

/// Applies the full Poseidon permutation to the state.
fn poseidon_permutation<F: IsField, P: PoseidonParams<F>>(
    builder: &mut CircuitBuilder<F>,
    state: &mut [FieldVar],
) -> Result<(), GadgetError> {
    let mut round_idx = 0;

    // First half of full rounds
    for _ in 0..P::FULL_ROUNDS / 2 {
        full_round::<F, P>(builder, state, round_idx)?;
        round_idx += 1;
    }

    // Partial rounds
    for _ in 0..P::PARTIAL_ROUNDS {
        partial_round::<F, P>(builder, state, round_idx)?;
        round_idx += 1;
    }

    // Second half of full rounds
    for _ in 0..P::FULL_ROUNDS / 2 {
        full_round::<F, P>(builder, state, round_idx)?;
        round_idx += 1;
    }

    Ok(())
}

/// Applies a full round: add constants, S-box on all, MDS mix.
fn full_round<F: IsField, P: PoseidonParams<F>>(
    builder: &mut CircuitBuilder<F>,
    state: &mut [FieldVar],
    round: usize,
) -> Result<(), GadgetError> {
    // Add round constants
    for (i, s) in state.iter_mut().enumerate() {
        let constant = P::round_constant(round, i);
        *s = builder.add_constant(s, constant);
    }

    // Apply S-box (x^alpha) to all state elements
    for s in state.iter_mut() {
        *s = apply_sbox::<F, P>(builder, s)?;
    }

    // Apply MDS matrix
    mds_mix::<F, P>(builder, state)?;

    Ok(())
}

/// Applies a partial round: add constant to last, S-box on last only, MDS mix.
fn partial_round<F: IsField, P: PoseidonParams<F>>(
    builder: &mut CircuitBuilder<F>,
    state: &mut [FieldVar],
    round: usize,
) -> Result<(), GadgetError> {
    // Add round constant only to last element
    let last_idx = state.len() - 1;
    let constant = P::round_constant(round, last_idx);
    state[last_idx] = builder.add_constant(&state[last_idx], constant);

    // Apply S-box only to last element
    state[last_idx] = apply_sbox::<F, P>(builder, &state[last_idx])?;

    // Apply MDS matrix
    mds_mix::<F, P>(builder, state)?;

    Ok(())
}

/// Applies the S-box (x^alpha) to a variable.
fn apply_sbox<F: IsField, P: PoseidonParams<F>>(
    builder: &mut CircuitBuilder<F>,
    x: &FieldVar,
) -> Result<FieldVar, GadgetError> {
    match P::ALPHA {
        3 => {
            // x^3 = x * x * x
            let x2 = builder.mul(x, x);
            Ok(builder.mul(&x2, x))
        }
        5 => {
            // x^5 = x^2 * x^2 * x
            let x2 = builder.mul(x, x);
            let x4 = builder.mul(&x2, &x2);
            Ok(builder.mul(&x4, x))
        }
        7 => {
            // x^7 = x^4 * x^2 * x
            let x2 = builder.mul(x, x);
            let x4 = builder.mul(&x2, &x2);
            let x6 = builder.mul(&x4, &x2);
            Ok(builder.mul(&x6, x))
        }
        _ => Err(GadgetError::InvalidInput(format!(
            "Unsupported S-box exponent: {}",
            P::ALPHA
        ))),
    }
}

/// Applies the MDS matrix to the state.
fn mds_mix<F: IsField, P: PoseidonParams<F>>(
    builder: &mut CircuitBuilder<F>,
    state: &mut [FieldVar],
) -> Result<(), GadgetError> {
    let n = state.len();
    let mut new_state = Vec::with_capacity(n);

    for i in 0..n {
        // Compute row i of matrix-vector product
        let mut sum = builder.constant(FieldElement::<F>::zero());

        for (j, state_j) in state.iter().enumerate() {
            let mds_val = P::mds_element(i, j);
            let scaled = builder.mul_constant(state_j, mds_val);
            sum = builder.add(&sum, &scaled);
        }

        new_state.push(sum);
    }

    // Copy back to state
    for (i, val) in new_state.into_iter().enumerate() {
        state[i] = val;
    }

    Ok(())
}

/// Two-to-one Poseidon hash (for Merkle trees).
///
/// Hashes exactly two field elements efficiently.
pub struct PoseidonTwoToOne<P = DefaultPoseidonParams> {
    _params: core::marker::PhantomData<P>,
}

impl<F: IsField, P: PoseidonParams<F>> Gadget<F> for PoseidonTwoToOne<P> {
    type Input = (FieldVar, FieldVar);
    type Output = FieldVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        (left, right): Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        PoseidonHash::<P>::synthesize(builder, vec![left, right])
    }

    fn constraint_count() -> usize {
        PoseidonHash::<P>::constraint_count()
    }

    fn name() -> &'static str {
        "PoseidonTwoToOne"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<65537>;

    #[test]
    fn test_poseidon_hash_single_input() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.public_input("x");
        let _hash =
            PoseidonHash::<DefaultPoseidonParams>::synthesize(&mut builder, vec![x]).unwrap();
    }

    #[test]
    fn test_poseidon_hash_two_inputs() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let _hash =
            PoseidonHash::<DefaultPoseidonParams>::synthesize(&mut builder, vec![a, b]).unwrap();
    }

    #[test]
    fn test_poseidon_two_to_one() {
        let mut builder = CircuitBuilder::<F>::new();

        let left = builder.public_input("left");
        let right = builder.public_input("right");
        let _hash =
            PoseidonTwoToOne::<DefaultPoseidonParams>::synthesize(&mut builder, (left, right))
                .unwrap();
    }

    #[test]
    fn test_poseidon_empty_input_error() {
        let mut builder = CircuitBuilder::<F>::new();

        let result = PoseidonHash::<DefaultPoseidonParams>::synthesize(&mut builder, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_poseidon_constraint_count() {
        let count = <PoseidonHash<DefaultPoseidonParams> as Gadget<F>>::constraint_count();
        assert!(count > 0);
    }
}
