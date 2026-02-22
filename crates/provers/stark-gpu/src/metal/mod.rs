pub mod buffers;
pub mod constraint_eval;
pub mod deep_composition;
pub mod fft;
pub mod fp3;
pub mod merkle;
pub mod phases;
pub mod prover;
pub mod state;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Compute `base^(2^log_power)` by repeated squaring.
///
/// For exponents that are powers of 2, this performs exactly `log_power`
/// squarings with zero multiplications, avoiding the overhead of the
/// generic binary exponentiation loop in `FieldElement::pow`.
#[inline]
pub(crate) fn exp_power_of_2<F: IsField>(
    base: &FieldElement<F>,
    log_power: u32,
) -> FieldElement<F> {
    let mut result = base.clone();
    for _ in 0..log_power {
        result = result.square();
    }
    result
}
