use lambdaworks_math::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, U384PrimeField},
    unsigned_integer::element::U384,
};

use super::*;

#[derive(Clone, Debug)]
pub struct TestFieldModulus;
impl IsModulus<U384> for TestFieldModulus {
    const MODULUS: U384 = U384::from_hex_unchecked(
        "2000000000000080000000000000000000000000000000000000000000000001",
    );
}

pub type PoseidonTestField = U384PrimeField<TestFieldModulus>;
type TestFieldElement = FieldElement<PoseidonTestField>;

pub fn load_test_parameters() -> Result<Parameters<PoseidonTestField>, String> {
    let round_constants_csv = include_str!("s128b/round_constants.csv");
    let mds_constants_csv = include_str!("s128b/mds_matrix.csv");

    let round_constants = round_constants_csv
        .split(',')
        .map(|c| TestFieldElement::new(U384::from_hex_unchecked(c.trim())))
        .collect();

    let mut mds_matrix = vec![];

    for line in mds_constants_csv.lines() {
        let matrix_line = line
            .split(',')
            .map(|c| TestFieldElement::new(U384::from_hex_unchecked(c.trim())))
            .collect();

        mds_matrix.push(matrix_line);
    }

    Ok(Parameters {
        rate: 2,
        capacity: 1,
        alpha: 3,
        n_full_rounds: 8,
        n_partial_rounds: 83,
        round_constants,
        mds_matrix,
    })
}

fn test_poseidon_s128b_t() {
    let mut state = [
        TestFieldElement::new(U384::from_u64(7)),
        TestFieldElement::new(U384::from_u64(98)),
        TestFieldElement::new(U384::from_u64(0)),
    ];
    let poseidon = Poseidon::new_with_params(load_test_parameters().unwrap());

    poseidon.ark(&mut state, 0);
    let expected = [
        TestFieldElement::new(U384::from_hex_unchecked(
            "16861759ea5568dd39dd92f9562a30b9e58e2ad98109ae4780b7fd8eac77fe8a",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "13827681995D5ADFFFC8397A3D00425A3DA43F76ABF28A64E4AB1A22F275092B",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "BA3956D2FAD4469E7F760A2277DC7CB2CAC75DC279B2D687A0DBE17704A8310",
        )),
    ];
    assert_eq!(state, expected);
}

fn test_mix() {
    let mut state = [
        TestFieldElement::new(U384::from_hex_unchecked(
            "13f891b043b3b740cc3e1b3051127d335f08e488322f360a776b3810b7dc690a",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "1bd24b7cb99acf0dbea719ff4007bd60105bcefef21ec509d2f8d4f9bb6a3a1a",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "110853eb2ebee0d940454fe420229a2a0974e666d16c92bab9f36cbd1a0eded",
        )),
    ];

    let poseidon = Poseidon::new_with_params(load_test_parameters().unwrap());

    poseidon.mix(&mut state);

    let expected = [
        TestFieldElement::new(U384::from_hex_unchecked(
            "1d30b34b465f8cddc8dc468f137891659c7e32b510cf41cec3aac0b26741681d",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "c445fa4dd2af583994272bede589b06b98fe9cd6d868bf718f6748ba6165620",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "1ed95ae0ea03bb892691f5200fb5902957ac17b3466afa62be808682801f97f9",
        )),
    ];
    assert_eq!(state, expected);
}

fn test_hash() {
    let poseidon: Poseidon<BLS12381PrimeField> = Poseidon::new();

    let a = FieldElement::one();
    let b = FieldElement::zero();

    poseidon.hash_new_parent(&a, &b);
}

fn test_permutation() {
    let poseidon = Poseidon::new_with_params(load_test_parameters().unwrap());

    let mut state = [
        TestFieldElement::new(U384::from_u64(7)),
        TestFieldElement::new(U384::from_u64(98)),
        TestFieldElement::new(U384::from_u64(0)),
    ];

    poseidon.permute(&mut state);

    let expected = [
        TestFieldElement::new(U384::from_hex_unchecked(
            "18700783647721BB9AD092B176BBEB5348401C21132CCF83C30134DFAB5A2DEB",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "1CC8856652601B3C81139AD5EC13E4A3A8F4A5DB242555521A09E002E7A10B2B",
        )),
        TestFieldElement::new(U384::from_hex_unchecked(
            "3DCB1CEC811FC2D7401CA7B9B084D167F33B6983D4428C8E0534C9C3CECF46D",
        )),
    ];

    assert_eq!(state, expected);
}
pub fn run_tests() {
    test_poseidon_s128b_t();
    test_mix();
    test_hash();
    test_permutation();
}
