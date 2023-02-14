/// Poseidon implementation for curve BLS12381
use self::parameters::Parameters;

use super::traits::IsCryptoHash;
use lambdaworks_math::{
    elliptic_curve::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::{element::FieldElement, traits::IsField},
};
use std::ops::{Add, Mul};

mod parameters;

pub struct Poseidon<F: IsField> {
    phantom: std::marker::PhantomData<F>,
}

/// Implements hashing for BLS 12381's field.
/// Alpha = 5 and parameters are predefined for secure implementations
impl IsCryptoHash<BLS12381PrimeField> for Poseidon<BLS12381PrimeField> {
    fn hash_one(
        input: FieldElement<BLS12381PrimeField>,
    ) -> Result<FieldElement<BLS12381PrimeField>, String> {
        let params = Parameters::for_one_element()
            .expect("Error loading parameters for hashing one element");
        Self::hash(vec![input], &params, 5)
    }

    fn hash_two(
        left: FieldElement<BLS12381PrimeField>,
        right: FieldElement<BLS12381PrimeField>,
    ) -> Result<FieldElement<BLS12381PrimeField>, String> {
        let params = Parameters::for_two_elements()
            .expect("Error loading parameters for hashing two elements");

        Self::hash(vec![left, right], &params, 5)
    }
}

impl<F> Poseidon<F>
where
    F: IsField,
{
    fn ark(state: &mut Vec<FieldElement<F>>, round_constants: &[FieldElement<F>], round: usize) {
        for i in 0..state.len() {
            state[i] += round_constants[round + i].clone();
        }
    }

    fn sbox(
        n_rounds_f: usize,
        n_rounds_p: usize,
        state: &mut [FieldElement<F>],
        i: usize,
        alpha: u32,
    ) {
        if i < n_rounds_f / 2 || i >= n_rounds_f / 2 + n_rounds_p {
            for current_state in state.iter_mut() {
                let aux = current_state.clone();
                *current_state = current_state.pow(2u32);
                *current_state = current_state.pow(2u32);
                *current_state = current_state.clone().mul(aux);
            }
        } else {
            state[0] = state[0].pow(alpha);
        }
    }

    fn mix(state: &Vec<FieldElement<F>>, mds: &[Vec<FieldElement<F>>]) -> Vec<FieldElement<F>> {
        let mut new_state: Vec<FieldElement<F>> = Vec::new();
        for i in 0..state.len() {
            new_state.push(FieldElement::zero());
            for (j, current_state) in state.iter().enumerate() {
                let mut mij = mds[i][j].clone();
                mij = mij.mul(current_state);
                new_state[i] = new_state[i].clone().add(&mij);
            }
        }
        new_state.clone()
    }

    fn hash(
        inp: Vec<FieldElement<F>>,
        params: &Parameters<F>,
        alpha: u32,
    ) -> Result<FieldElement<F>, String> {
        let t = inp.len() + 1;
        if inp.is_empty() || inp.len() >= params.n_partial_rounds - 1 {
            return Err("Wrong inputs length".to_string());
        }

        let mut state = vec![FieldElement::zero(); t];
        state[1..].clone_from_slice(&inp);

        for i in 0..(params.n_full_rounds + params.n_partial_rounds) {
            Self::ark(&mut state, &params.round_constants, i * t);
            Self::sbox(
                params.n_full_rounds,
                params.n_partial_rounds,
                &mut state,
                i,
                alpha,
            );
            state = Self::mix(&state, &params.mds_matrix);
        }

        Ok(state[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poseidon_bls() {
        let a = FieldElement::<BLS12381PrimeField>::new_base("1");
        let b = FieldElement::<BLS12381PrimeField>::new_base("2");

        //assert_eq!(
        //    "0000000000000000000000000000000000000000000000000000000000000001",
        //    a
        //);

        let v = Poseidon::hash_one(a.clone()).unwrap();
        println!("{:?}", v);
        let v = Poseidon::hash_one(b.clone()).unwrap();
        println!("{:?}", v);

        let v = Poseidon::hash_two(a, b).unwrap();
        println!("{:?}", v);
    }
}
