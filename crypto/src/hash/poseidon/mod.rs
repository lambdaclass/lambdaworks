/// Poseidon implementation for curve BLS12381
use self::parameters::Parameters;

use super::traits::IsCryptoHash;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use std::ops::{Add, Mul};

mod parameters;
// mod test_helper;

pub struct Poseidon<F: IsField> {
    params: Parameters<F>,
}

/// Implements hashing for BLS 12381's field.
/// Alpha = 5 and parameters are predefined for secure implementations
impl<F: IsField> IsCryptoHash<F> for Poseidon<F> {
    fn hash_one(&self, input: FieldElement<F>) -> FieldElement<F> {
        self.hash(&[input]).unwrap().first().unwrap().clone()
    }

    fn hash_two(&self, left: FieldElement<F>, right: FieldElement<F>) -> FieldElement<F> {
        self.hash(&[left, right]).unwrap().first().unwrap().clone()
    }
}

impl<F> Poseidon<F>
where
    F: IsField,
{
    pub fn new(params: Parameters<F>) -> Self {
        Poseidon { params }
    }

    pub fn ark(&self, state: &mut [FieldElement<F>], round_number: usize) {
        for (i, state) in state.iter_mut().enumerate() {
            *state += self.params.round_constants[round_number + i].clone();
        }
    }

    pub fn sbox(&self, state: &mut [FieldElement<F>], round_number: usize) {
        if round_number < self.params.n_full_rounds / 2
            || round_number >= self.params.n_full_rounds / 2 + self.params.n_partial_rounds
        {
            for current_state in state.iter_mut() {
                *current_state = current_state.pow(self.params.alpha);
            }
        } else {
            state[0] = state[0].pow(self.params.alpha);
        }
    }

    pub fn mix(&self, state: &mut [FieldElement<F>]) {
        let mut new_state: Vec<FieldElement<F>> = Vec::with_capacity(state.len());
        for i in 0..state.len() {
            new_state.push(FieldElement::zero());
            for (j, current_state) in state.iter().enumerate() {
                let mut mij = self.params.mds_matrix[i][j].clone();
                mij = mij.mul(current_state);
                new_state[i] = new_state[i].clone().add(&mij);
            }
        }
        state.clone_from_slice(&new_state[0..state.len()]);
    }

    fn permute(&self, state: &mut [FieldElement<F>]) {
        let t = self.params.capacity + self.params.rate;
        for i in 0..(self.params.n_full_rounds + self.params.n_partial_rounds) {
            self.ark(state, i * t);
            self.sbox(state, i);
            self.mix(state);
        }
    }

    fn ensure_permuted(&self, state: &mut [FieldElement<F>], offset: &mut usize) {
        // offset should be <= rate, so really testing for equality
        if *offset >= self.params.rate {
            self.permute(state);
            *offset = 0;
        }
    }

    fn hash(&self, inputs: &[FieldElement<F>]) -> Result<Vec<FieldElement<F>>, String>
    where
        F: IsField,
    {
        let t = self.params.rate + self.params.capacity;
        if inputs.is_empty() || inputs.len() >= self.params.n_partial_rounds - 1 {
            return Err("Wrong input length".to_string());
        }

        let mut state = vec![FieldElement::zero(); t];
        let mut offset: usize = 0;

        let n_remaining = inputs.len() % self.params.rate;
        if n_remaining != 0 {
            return Err(format!(
                "Input length {} must be a multiple of the hash rate {}",
                inputs.len(),
                self.params.rate
            ));
        }

        // absorb
        for input in inputs {
            self.ensure_permuted(&mut state, &mut offset);
            state[offset] += input.clone();
            offset += 1;
        }

        // squeeze
        let mut result = vec![FieldElement::zero(); self.params.rate];
        for result_element in result.iter_mut().take(self.params.rate) {
            self.ensure_permuted(&mut state, &mut offset);
            *result_element = state[offset].clone();
            offset += 1;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        field::fields::u384_prime_field::{IsMontgomeryConfiguration, MontgomeryBackendPrimeField},
        unsigned_integer::element::U384,
    };

    use super::*;

    #[derive(Clone, Debug)]
    pub struct TestFieldConfig;
    impl IsMontgomeryConfiguration for TestFieldConfig {
        const MODULUS: U384 = U384::from(
            "14474011154664525231415395255581126252639794253786371766033694892385558855681",
        );
        const MP: u64 = 123;
        const R2: U384 = U384::from("123");
    }

    pub type PoseidonTestField = MontgomeryBackendPrimeField<TestFieldConfig>;
    type TestFieldElement = FieldElement<PoseidonTestField>;

    pub fn load_test_parameters() -> Result<Parameters<PoseidonTestField>, String> {
        let round_constants_csv = include_str!("s128b/round_constants.csv");
        let mds_constants_csv = include_str!("s128b/mds_matrix.csv");

        let round_constants = round_constants_csv
            .split(',')
            .map(|c| TestFieldElement::new(U384::from(c.trim())))
            .collect();

        let mut mds_matrix = vec![];

        for line in mds_constants_csv.lines() {
            let matrix_line = line
                .split(',')
                .map(|c| TestFieldElement::new(U384::from(c.trim())))
                .collect();

            mds_matrix.push(matrix_line);
        }

        Ok(Parameters {
            rate: 1,
            capacity: 2,
            alpha: 3,
            n_full_rounds: 8,
            n_partial_rounds: 83,
            round_constants,
            mds_matrix,
        })
    }

    #[test]
    fn test_poseidon_bls() {
        let poseidon = Poseidon::new(load_test_parameters().unwrap());
        let mut state = [
            TestFieldElement::new(U384::from("7")),
            TestFieldElement::new(U384::from("98")),
            TestFieldElement::new(U384::from("0")),
        ];
        poseidon.ark(&mut state, 0);
        let expected = [
            TestFieldElement::new(U384::from(
                "10187801339791605336251748402605479409606566396373491958667041943798551150218",
            )),
            TestFieldElement::new(U384::from(
                "8824452141556477327634835943439996420519135454314677708228322513850226510123",
            )),
            TestFieldElement::new(U384::from(
                "5264468709835621148349527988912247104353814123939106227116596276180070073104",
            )),
        ];
        // this needs to be asserted once the field definition is in place: assert_eq!(state, expected);
    }
}
