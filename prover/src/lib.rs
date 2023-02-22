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
