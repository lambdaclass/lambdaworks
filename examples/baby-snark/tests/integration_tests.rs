use baby_snark::common::FrElement;
use baby_snark::scs::SquareConstraintSystem;
use baby_snark::ssp::SquareSpanProgram;
use baby_snark::utils::{i64_matrix_to_field, i64_vec_to_field};
use baby_snark::{setup, verify, Prover};
#[test]
fn identity_matrix() {
    let u = vec![i64_vec_to_field(&[1, 0]), i64_vec_to_field(&[0, 1])];
    let witness = i64_vec_to_field(&[1, 1]);
    let public = i64_vec_to_field(&[]);

    test_integration(u, witness, public, true)
}

#[test]
fn size_not_pow2() {
    let u: &[&[i64]] = &[
        &[1, 3, 2, 4, 5],
        &[-1, -2, 3, 4, -2],
        &[1, 2, 3, 2, 2],
        &[-3, -2, 0, 0, 0],
        &[0, 9, 2, -1, 3],
    ];
    let input: &[i64] = &[1, 2, 3, 4, 5];

    let witness = i64_vec_to_field(&[3, 4, 5]);
    let public = i64_vec_to_field(&[1, 2]);
    let input_field = i64_vec_to_field(input);
    let u_field = normalize(i64_matrix_to_field(u), &input_field);

    test_integration(u_field, witness, public, true);
}

#[test]
fn and_gate() {
    let u = vec![
        i64_vec_to_field(&[-1, 2, 0, 0]),
        i64_vec_to_field(&[-1, 0, 2, 0]),
        i64_vec_to_field(&[-1, 0, 0, 2]),
        i64_vec_to_field(&[-1, 2, 2, -4]),
    ];
    let witness = i64_vec_to_field(&[1, 1, 1]);
    let public = i64_vec_to_field(&[1]);

    test_integration(u, witness, public, true)
}

#[test]
fn invalid_proof() {
    let u: &[&[i64]] = &[
        &[1, 3, 2, 4, 5],
        &[-1, 8, 3, 4, -2],
        &[1, 2, 3, 2, 2],
        &[-3, -2, 0, 0, 0],
        &[0, 9, 2, -1, 3],
        &[3, 9, 2, -1, 3],
    ];
    let input: &[i64] = &[1, 4, 6, 0, 5];
    let mut witness = i64_vec_to_field(&[0, 5]);
    let public = i64_vec_to_field(&[1, 4, 6]);
    let input_field = i64_vec_to_field(input);
    let u_field = normalize(i64_matrix_to_field(u), &input_field);
    test_integration(u_field.clone(), witness.clone(), public.clone(), true);

    witness = i64_vec_to_field(&[0, 3]);
    test_integration(u_field, witness, public, false);
}

fn normalize(matrix: Vec<Vec<FrElement>>, input: &Vec<FrElement>) -> Vec<Vec<FrElement>> {
    let mut new_matrix = Vec::new();

    for row in matrix {
        let coef = row
            .iter()
            .zip(input)
            .map(|(a, b)| a * b)
            .reduce(|a, b| a + b)
            .unwrap();
        let new_row: Vec<FrElement> = row.iter().map(|x| x * coef.inv().unwrap()).collect();
        new_matrix.push(new_row);
    }

    new_matrix
}

pub fn test_integration(
    u: Vec<Vec<FrElement>>,
    witness: Vec<FrElement>,
    public: Vec<FrElement>,
    pass: bool,
) {
    let mut input = public.clone();
    input.extend(witness.clone());

    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrix(u, public.len()));
    let (proving_key, verifying_key) = setup(&ssp);

    let proof = Prover::prove(&input, &ssp, &proving_key);
    let verified = verify(&verifying_key, &proof, &public);
    if pass {
        assert!(verified);
    } else {
        assert!(!verified);
    }
}
