pub mod air;
pub mod fri;

use air::composition_poly::get_composition_poly;
use lambdaworks_math::polynomial::Polynomial;
use winterfell::{
    crypto::hashers::Blake3_256,
    math::{fields::f128::BaseElement, StarkField},
    prover::constraints::CompositionPoly,
    Air, AuxTraceRandElements, Serializable, Trace, TraceTable,
};


use lambdaworks_math::{
    field::fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
    unsigned_integer::element::U384,
};
use lambdaworks_math::field::element::FieldElement;

use lambdaworks_math::field::traits::IsField;
#[derive(Clone, Debug)]
pub struct MontgomeryConfig;
impl IsMontgomeryConfiguration for MontgomeryConfig {
    const MODULUS: U384 =
        U384::from("800000000000011000000000000000000000000000000000000000000000001");
    const MP: u64 = 18446744073709551615;
    const R2: U384 =
        U384::from("38e5f79873c0a6df47d84f8363000187545706677ffcc06cc7177d1406df18e");
}
const PRIME_GENERATOR_MONTGOMERY: U384 = U384::from("3");
type U384PrimeField = MontgomeryBackendPrimeField<MontgomeryConfig>;
type U384FieldElement = FieldElement<U384PrimeField>;
const MODULUS_MINUS_1: U384 =
    U384::from("800000000000011000000000000000000000000000000000000000000000000");

pub fn generate_vec_roots(subgroup_size: u64, coset_factor: u64) -> Vec<U384FieldElement> {
    let MODULUS_MINUS_1_FIELD: U384FieldElement = U384FieldElement::new(MODULUS_MINUS_1);
    let subgroup_size_u384: U384FieldElement = subgroup_size.into();
    let generator_field: U384FieldElement = 3.into();
    let coset_factor_u384: U384FieldElement = coset_factor.into();

    let exp = (MODULUS_MINUS_1_FIELD) / subgroup_size_u384;
    let exp_384 = *exp.value();

    let generator_of_subgroup = generator_field.pow(exp_384);

    let mut numbers = Vec::new();

    for exp in 0..subgroup_size {
        let ret = generator_of_subgroup.pow(exp) * &coset_factor_u384;
        numbers.push(ret.clone());
    }

    numbers
}

#[derive(Debug)]
pub struct StarkProof {
    // TODO: fill this when we know what a proof entails
}

pub fn prove<A>(air: A, trace: TraceTable<A::BaseField>, pub_inputs: A::PublicInputs) -> StarkProof
where
    A: Air<BaseField = BaseElement>,
{
    // * Generate composition polynomials using Winterfell
    let result_poly = get_composition_poly(air, trace, pub_inputs);

    // * Do Reed-Solomon on the trace and composition polynomials using some blowup factor
        // * Generate Coset

    // * Commit to both polynomials using a Merkle Tree
    // * Do FRI on the composition polynomials
    // * Sample q_1, ..., q_m using Fiat-Shamir
    // * For every q_i, do FRI decommitment
    // * For every trace polynomial t_i, provide the evaluations on every q_i, q_i * g, q_i * g^2

    StarkProof {}
}

pub fn verify() {}

#[cfg(test)]
mod tests {
    use super::prove;
    use winterfell::{FieldExtension, ProofOptions};

    #[test]
    fn test_prove() {}
}
