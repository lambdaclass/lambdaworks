pub use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

use super::fri_commitment::FriLayer;
#[derive(Debug, Clone)]
pub struct FriDecommitment<F: IsField> {
    pub layers_auth_paths_sym: Vec<Proof<F>>,
    pub layers_evaluations_sym: Vec<FieldElement<F>>,
    pub first_layer_evaluation: FieldElement<F>,
    pub first_layer_auth_path: Proof<F>,
}

pub fn open_layer<F: IsField>(layer: &FriLayer<F>, mut index: usize) -> (FieldElement<F>, Proof<F>) {
    index = index % layer.domain.len();
    let evaluation = layer.evaluation[index].clone();
    let auth_path = layer.merkle_tree.get_proof_by_pos(index).unwrap();
    (evaluation, auth_path)
}

// Integration test:
// * get an arbitrary polynomial
// * have a domain containing roots of the unity (# is power of two)
// p = 65_537
// * apply FRI commitment
// * apply FRI decommitment
// assert:
// * evaluations of the polynomials coincide with calculations from the decommitment
// * show a fail example: with a monomial

#[cfg(test)]
mod tests {
    use crate::fri::U64PrimeField;
    use lambdaworks_math::field::element::FieldElement;
    use std::collections::HashSet;
    const PRIME_GENERATOR: (u64, u64) = (0xFFFF_FFFF_0000_0001_u64, 2717_u64);
    pub type F = U64PrimeField<{ PRIME_GENERATOR.0 }>;
    pub type FeGoldilocks = FieldElement<F>;

    #[test]
    fn test() {
        let subgroup_size = 1024_u64;
        let generator_field = FeGoldilocks::new(PRIME_GENERATOR.1);
        let exp = (PRIME_GENERATOR.0 - 1) / subgroup_size;
        let generator_of_subgroup = generator_field.pow(exp);
        let mut numbers = HashSet::new();

        let mut i = 0;
        for exp in 0..1024_u64 {
            i += 1;
            let ret = generator_of_subgroup.pow(exp);
            numbers.insert(*ret.value());
            println!("{ret:?}");
        }

        let count = numbers.len();
        println!("count: {count}");
        println!("iter: {i}");
    }
}
