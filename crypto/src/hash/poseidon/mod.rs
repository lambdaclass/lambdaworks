mod cairo_poseidon_constants;
pub mod parameters;
use self::parameters::PermutationParameters;

use lambdaworks_math::field::{element::FieldElement, traits::IsPrimeField};
use std::ops::{Add, Mul};

#[derive(Clone)]
pub struct Poseidon<F: IsPrimeField> {
    params: PermutationParameters<F>,
    // Suggestion: Add the state here
}

impl<F: IsPrimeField> Poseidon<F> {
    pub fn new_with_params(params: PermutationParameters<F>) -> Self {
        Poseidon { params }
    }

    pub fn hades_permutation(&self, state: &mut [FieldElement<F>]) {
        let mut round_number = 0;
        for _ in 0..self.params.n_full_rounds / 2 {
            self.full_round(state, round_number);
            round_number += 1;
        }
        for _ in 0..self.params.n_partial_rounds {
            self.partial_round(state, round_number);
            round_number += 1;
        }
        for _ in 0..self.params.n_full_rounds / 2 {
            self.full_round(state, round_number);
            round_number += 1;
        }
    }

    pub fn full_round(&self, state: &mut [FieldElement<F>], round_number: usize) {
        for i in 0..self.params.state_size {
            state[i] = &state[i] + &self.params.round_constants[round_number][i];
            state[i] = state[i].pow(self.params.alpha);
        }
        self.mix(state);
    }
    pub fn partial_round(&self, state: &mut [FieldElement<F>], round_number: usize) {
        for i in 0..self.params.state_size {
            state[i] = &state[i] + &self.params.round_constants[round_number][i];
        }

        state[self.params.state_size - 1] =
            state[self.params.state_size - 1].pow(self.params.alpha);

        self.mix(state);
    }

    pub fn mix(&self, state: &mut [FieldElement<F>]) {
        let mut new_state: Vec<FieldElement<F>> = Vec::with_capacity(self.params.state_size);
        for i in 0..self.params.state_size {
            new_state.push(FieldElement::zero());
            for (j, current_state) in state.iter().enumerate() {
                let mut mij = self.params.mds_matrix[i][j].clone();
                mij = mij.mul(current_state);
                new_state[i] = new_state[i].clone().add(&mij);
            }
        }
        state.clone_from_slice(&new_state[0..self.params.state_size]);
    }
    pub fn hash(&self, x: &FieldElement<F>, y: &FieldElement<F>) -> FieldElement<F> {
        let mut state: Vec<FieldElement<F>> = vec![x.clone(), y.clone(), FieldElement::from(2)];
        self.hades_permutation(&mut state);
        let x = &state[0];
        return x.clone();
    }

    pub fn hash_single(&self, x: &FieldElement<F>) -> FieldElement<F> {
        let mut state: Vec<FieldElement<F>> =
            vec![x.clone(), FieldElement::zero(), FieldElement::from(1)];
        self.hades_permutation(&mut state);
        let x = &state[0];
        return x.clone();
    }
    pub fn hash_many(&self, inputs: &Vec<FieldElement<F>>) -> FieldElement<F> {
        let r = self.params.rate; // chunk size
        let m = self.params.state_size; // state size

        // Pad input with 1 followed by 0's (if necessary).
        let mut values = inputs.clone();
        values.push(FieldElement::from(1));
        values.resize(((values.len() + r - 1) / r) * r, FieldElement::zero());

        assert!(values.len() % r == 0);
        let mut state: Vec<FieldElement<F>> = vec![FieldElement::zero(); m];

        // Process each block
        for block in values.chunks(r) {
            let mut block_state: Vec<FieldElement<F>> =
                state[0..r].iter().zip(block).map(|(s, b)| s + b).collect();
            block_state.extend_from_slice(&state[r..]);

            self.hades_permutation(&mut block_state);
            state = block_state;
        }

        state[0].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::poseidon::parameters::{DefaultPoseidonParams, PermutationParameters};
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    #[test]
    fn test_hades_permutation() {
        let params = PermutationParameters::new_with(DefaultPoseidonParams::CairoStark252);

        let poseidon = Poseidon::new_with_params(params);

        // Initialize a state to test. The exact contents will depend on your specific use case.
        let mut state: Vec<FieldElement<Stark252PrimeField>> = vec![
            FieldElement::<Stark252PrimeField>::from_hex("0x9").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xb").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x2").unwrap(),
        ];

        poseidon.hades_permutation(&mut state);

        // Compare the result to the expected output. You will need to know the expected output for your specific test case.
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x510f3a3faf4084e3b1e95fd44c30746271b48723f7ea9c8be6a9b6b5408e7e6",
        )
        .unwrap();
        let expected_state1 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x4f511749bd4101266904288021211333fb0a514cb15381af087462fa46e6bd9",
        )
        .unwrap();
        let expected_state2 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x186f6dd1a6e79cb1b66d505574c349272cd35c07c223351a0990410798bb9d8",
        )
        .unwrap();

        assert_eq!(state[0], expected_state0);
        assert_eq!(state[1], expected_state1);
        assert_eq!(state[2], expected_state2);
    }
    #[test]
    fn test_hash() {
        let params = PermutationParameters::new_with(DefaultPoseidonParams::CairoStark252);

        let poseidon = Poseidon::new_with_params(params);

        let x = FieldElement::<Stark252PrimeField>::from_hex("0x123456").unwrap();
        let y = FieldElement::<Stark252PrimeField>::from_hex("0x789101").unwrap();

        let z = poseidon.hash(&x, &y);

        // Compare the result to the expected output. You will need to know the expected output for your specific test case.
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x2fb6e1e8838d4b850877944f0a13340dd5810f01f5d4361c54b22b4abda3248",
        )
        .unwrap();

        assert_eq!(z, expected_state0);
    }

    #[test]
    fn test_hash_single() {
        let params = PermutationParameters::new_with(DefaultPoseidonParams::CairoStark252);

        let poseidon = Poseidon::new_with_params(params);

        let x = FieldElement::<Stark252PrimeField>::from_hex("0x9").unwrap();

        let z = poseidon.hash_single(&x);

        // Compare the result to the expected output. You will need to know the expected output for your specific test case.
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x3bb3b91c714cb47003947f36dadc98326176963c434cd0a10320b8146c948b3",
        )
        .unwrap();

        assert_eq!(z, expected_state0);
    }

    #[test]
    fn test_hash_many() {
        let params = PermutationParameters::new_with(DefaultPoseidonParams::CairoStark252);

        let poseidon = Poseidon::new_with_params(params);

        let a = FieldElement::<Stark252PrimeField>::from_hex("0x1").unwrap();
        let b = FieldElement::<Stark252PrimeField>::from_hex("0x2").unwrap();
        let c = FieldElement::<Stark252PrimeField>::from_hex("0x3").unwrap();
        let d = FieldElement::<Stark252PrimeField>::from_hex("0x4").unwrap();
        let e = FieldElement::<Stark252PrimeField>::from_hex("0x5").unwrap();
        let f = FieldElement::<Stark252PrimeField>::from_hex("0x6").unwrap();

        let ins = vec![a, b, c, d, e, f];
        let z = poseidon.hash_many(&ins);

        // Compare the result to the expected output. You will need to know the expected output for your specific test case.
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0xf50993f0797e4cc05734a47daeb214fde2d444ef6619a7c1f7c8e0924feb0b",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a];
        let z = poseidon.hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x579e8877c7755365d5ec1ec7d3a94a457eff5d1f40482bbe9729c064cdead2",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a, b];
        let z = poseidon.hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x371cb6995ea5e7effcd2e174de264b5b407027a75a231a70c2c8d196107f0e7",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a, b, c];
        let z = poseidon.hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x2f0d8840bcf3bc629598d8a6cc80cb7c0d9e52d93dab244bbf9cd0dca0ad082",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a, b, c, d];
        let z = poseidon.hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x26e3ad8b876e02bc8a4fc43dad40a8f81a6384083cabffa190bcf40d512ae1d",
        )
        .unwrap();

        assert_eq!(z, expected_state0);
    }
}
