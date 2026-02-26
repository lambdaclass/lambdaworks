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
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::field::traits::IsPrimeField;

type FpE = FieldElement<Goldilocks64Field>;

/// Extract the canonical u64 representation from a Goldilocks field element.
#[inline]
pub(crate) fn canonical(fe: &FpE) -> u64 {
    Goldilocks64Field::canonical(fe.value())
}

/// Convert a slice of Goldilocks field elements to canonical u64 values.
#[inline]
pub(crate) fn to_raw_u64(elems: &[FpE]) -> Vec<u64> {
    elems.iter().map(canonical).collect()
}

/// Extract the raw u64 components from an Fp3 (degree-3 extension) element.
#[inline]
pub(crate) fn fp3_to_u64s(
    e: &FieldElement<
        lambdaworks_math::field::fields::u64_goldilocks_field::Degree3GoldilocksExtensionField,
    >,
) -> [u64; 3] {
    let comps = e.value();
    [*comps[0].value(), *comps[1].value(), *comps[2].value()]
}

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
