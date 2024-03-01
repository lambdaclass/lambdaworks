use baby_snark::utils;

#[test]
fn identity_matrix() {
    let u = vec![
        utils::i64_vec_to_field(&[1, 0]),
        utils::i64_vec_to_field(&[0, 1]),
    ];
    let witness = utils::i64_vec_to_field(&[1, 1]);
    let public = utils::i64_vec_to_field(&[]);

    utils::test_integration(u, witness, public, true)
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
    let witness = utils::i64_vec_to_field(&[3, 4, 5]);
    let public = utils::i64_vec_to_field(&[1, 2]);
    let mut u_field = utils::i64_matrix_to_field(u);
    let input_field = utils::i64_vec_to_field(input);
    utils::normalize(&mut u_field, &input_field);

    utils::test_integration(u_field, witness, public, true);
}

#[test]
fn and_gate() {
    let u = vec![
        utils::i64_vec_to_field(&[-1, 2, 0, 0]),
        utils::i64_vec_to_field(&[-1, 0, 2, 0]),
        utils::i64_vec_to_field(&[-1, 0, 0, 2]),
        utils::i64_vec_to_field(&[-1, 2, 2, -4]),
    ];
    let witness = utils::i64_vec_to_field(&[1, 1, 1]);
    let public = utils::i64_vec_to_field(&[1]);

    utils::test_integration(u, witness, public, true)
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
    let mut witness = utils::i64_vec_to_field(&[0, 5]);
    let public = utils::i64_vec_to_field(&[1, 4, 6]);
    let mut u_field = utils::i64_matrix_to_field(u);
    let input_field = utils::i64_vec_to_field(input);
    utils::normalize(&mut u_field, &input_field);
    utils::test_integration(u_field.clone(), witness.clone(), public.clone(), true);

    witness = utils::i64_vec_to_field(&[0, 3]);
    utils::test_integration(u_field, witness, public, false);
}
