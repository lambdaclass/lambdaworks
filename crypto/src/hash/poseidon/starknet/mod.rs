pub mod parameters;
pub mod round_constants;
pub use parameters::Poseidon;
pub use parameters::PoseidonCairoStark252;

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    #[test]
    fn test_hades_permutation() {
        // Initialize a state to test. The exact contents will depend on your specific use case.
        let mut state: Vec<FieldElement<Stark252PrimeField>> = vec![
            FieldElement::<Stark252PrimeField>::from_hex("0x9").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xb").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x2").unwrap(),
        ];

        Poseidon::<PoseidonCairoStark252>::hades_permutation(&mut state);

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
        let x = FieldElement::<Stark252PrimeField>::from_hex("0x123456").unwrap();
        let y = FieldElement::<Stark252PrimeField>::from_hex("0x789101").unwrap();

        let z = Poseidon::<PoseidonCairoStark252>::hash(&x, &y);

        // Compare the result to the expected output. You will need to know the expected output for your specific test case.
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x2fb6e1e8838d4b850877944f0a13340dd5810f01f5d4361c54b22b4abda3248",
        )
        .unwrap();

        assert_eq!(z, expected_state0);
    }

    #[test]
    fn test_hash_single() {
        let x = FieldElement::<Stark252PrimeField>::from_hex("0x9").unwrap();

        let z = Poseidon::<PoseidonCairoStark252>::hash_single(&x);

        // Compare the result to the expected output. You will need to know the expected output for your specific test case.
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x3bb3b91c714cb47003947f36dadc98326176963c434cd0a10320b8146c948b3",
        )
        .unwrap();

        assert_eq!(z, expected_state0);
    }

    #[test]
    fn test_hash_many() {
        let a = FieldElement::<Stark252PrimeField>::from_hex("0x1").unwrap();
        let b = FieldElement::<Stark252PrimeField>::from_hex("0x2").unwrap();
        let c = FieldElement::<Stark252PrimeField>::from_hex("0x3").unwrap();
        let d = FieldElement::<Stark252PrimeField>::from_hex("0x4").unwrap();
        let e = FieldElement::<Stark252PrimeField>::from_hex("0x5").unwrap();
        let f = FieldElement::<Stark252PrimeField>::from_hex("0x6").unwrap();

        let ins = vec![a, b, c, d, e, f];
        let z = Poseidon::<PoseidonCairoStark252>::hash_many(&ins);

        // Compare the result to the expected output. You will need to know the expected output for your specific test case.
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0xf50993f0797e4cc05734a47daeb214fde2d444ef6619a7c1f7c8e0924feb0b",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a];
        let z = Poseidon::<PoseidonCairoStark252>::hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x579e8877c7755365d5ec1ec7d3a94a457eff5d1f40482bbe9729c064cdead2",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a, b];
        let z = Poseidon::<PoseidonCairoStark252>::hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x371cb6995ea5e7effcd2e174de264b5b407027a75a231a70c2c8d196107f0e7",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a, b, c];
        let z = Poseidon::<PoseidonCairoStark252>::hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x2f0d8840bcf3bc629598d8a6cc80cb7c0d9e52d93dab244bbf9cd0dca0ad082",
        )
        .unwrap();
        assert_eq!(z, expected_state0);

        let ins = vec![a, b, c, d];
        let z = Poseidon::<PoseidonCairoStark252>::hash_many(&ins);
        let expected_state0 = FieldElement::<Stark252PrimeField>::from_hex(
            "0x26e3ad8b876e02bc8a4fc43dad40a8f81a6384083cabffa190bcf40d512ae1d",
        )
        .unwrap();

        assert_eq!(z, expected_state0);
    }
}
