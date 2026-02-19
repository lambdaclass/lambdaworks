use super::utils::{
    generate_domain, generate_permutation_coefficients, ORDER_R_MINUS_1_ROOT_UNITY,
};
use crate::setup::{CommonPreprocessedInput, Witness};
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{FrElement, FrField},
    field::{element::FieldElement, traits::IsFFTField},
    polynomial::Polynomial,
};

/// Create a circuit with n gates by using circuit_1 for the first 4 gates
/// and padding the rest with identity gates (no constraints)
pub fn test_common_preprocessed_input_size(n: usize) -> CommonPreprocessedInput<FrField> {
    assert!(n.is_power_of_two() && n >= 4, "n must be a power of 2 >= 4");

    let log_n = n.trailing_zeros() as u64;
    let omega = FrField::get_primitive_root_of_unity(log_n).unwrap();
    let domain = generate_domain(&omega, n);

    // Build permutation: use circuit_1's permutation for first 4 gates, identity for the rest
    let mut permutation: Vec<usize> = vec![0; n * 3];

    if n == 4 {
        permutation.copy_from_slice(&[11, 3, 0, 1, 2, 4, 6, 10, 5, 8, 7, 9]);
    } else {
        // Use circuit_1's permutation pattern for first 4 gates
        permutation[0] = n * 2 + 3;
        permutation[1] = 3;
        permutation[2] = 0;
        permutation[3] = 1;

        permutation[n] = 2;
        permutation[n + 1] = n;
        permutation[n + 2] = n + 2;
        permutation[n + 3] = n * 2 + 2;

        permutation[n * 2] = n + 1;
        permutation[n * 2 + 1] = n * 2;
        permutation[n * 2 + 2] = n + 3;
        permutation[n * 2 + 3] = n * 2 + 1;

        // Remaining gates: identity permutation (no copy constraints)
        for i in 4..n {
            permutation[i] = i;
            permutation[n + i] = n + i;
            permutation[n * 2 + i] = n * 2 + i;
        }
    }

    let permuted =
        generate_permutation_coefficients(&omega, n, &permutation, &ORDER_R_MINUS_1_ROOT_UNITY);

    let s1_lagrange: Vec<FrElement> = permuted[..n].to_vec();
    let s2_lagrange: Vec<FrElement> = permuted[n..n * 2].to_vec();
    let s3_lagrange: Vec<FrElement> = permuted[n * 2..].to_vec();

    // Selector polynomials: copy circuit_1 for first 4 gates, zeros for the rest
    let mut ql_vals = vec![FieldElement::zero(); n];
    let mut qr_vals = vec![FieldElement::zero(); n];
    let mut qo_vals = vec![FieldElement::zero(); n];
    let mut qm_vals = vec![FieldElement::zero(); n];
    let qc_vals = vec![FieldElement::zero(); n];

    // First 4 gates: circuit_1 selectors
    ql_vals[0] = -FieldElement::one();
    ql_vals[1] = -FieldElement::one();
    ql_vals[3] = FieldElement::one();

    qr_vals[3] = -FieldElement::one();

    qo_vals[2] = -FieldElement::one();

    qm_vals[2] = FieldElement::one();

    CommonPreprocessedInput {
        n,
        omega,
        domain,
        k1: ORDER_R_MINUS_1_ROOT_UNITY,
        ql: Polynomial::interpolate_fft::<FrField>(&ql_vals).unwrap(),
        qr: Polynomial::interpolate_fft::<FrField>(&qr_vals).unwrap(),
        qo: Polynomial::interpolate_fft::<FrField>(&qo_vals).unwrap(),
        qm: Polynomial::interpolate_fft::<FrField>(&qm_vals).unwrap(),
        qc: Polynomial::interpolate_fft::<FrField>(&qc_vals).unwrap(),
        s1: Polynomial::interpolate_fft::<FrField>(&s1_lagrange).unwrap(),
        s2: Polynomial::interpolate_fft::<FrField>(&s2_lagrange).unwrap(),
        s3: Polynomial::interpolate_fft::<FrField>(&s3_lagrange).unwrap(),
        s1_lagrange,
        s2_lagrange,
        s3_lagrange,
    }
}

/// Create witness for large circuit - same as circuit_1 for first 4 gates, zeros for padding
pub fn test_witness_size(x: FrElement, e: FrElement, n: usize) -> Witness<FrField> {
    assert!(n.is_power_of_two() && n >= 4);

    let y = &x * &e;
    let empty = x.clone();
    let zero = FieldElement::zero();

    let mut a = vec![zero.clone(); n];
    let mut b = vec![zero.clone(); n];
    let mut c = vec![zero; n];

    a[0] = x.clone();
    a[1] = y.clone();
    a[2] = x.clone();
    a[3] = y.clone();

    b[0] = empty.clone();
    b[1] = empty.clone();
    b[2] = e.clone();
    b[3] = &x * &e;

    c[0] = empty.clone();
    c[1] = empty.clone();
    c[2] = &x * &e;
    c[3] = empty;

    Witness { a, b, c }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::Prover;
    use crate::setup::setup;
    use crate::test_utils::utils::{test_srs, TestRandomFieldGenerator, KZG};
    use crate::verifier::Verifier;

    #[test]
    fn test_large_circuit_n16() {
        let n = 16;
        let common = test_common_preprocessed_input_size(n);
        let x = FieldElement::from(4_u64);
        let e = FieldElement::from(3_u64);
        let y = &x * &e;
        let witness = test_witness_size(x.clone(), e, n);
        let public_input = vec![x, y];
        let srs = test_srs(n);
        let kzg = KZG::new(srs);
        let vk = setup(&common, &kzg);
        let rng = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), rng);
        let proof = prover
            .prove(&witness, &public_input, &common, &vk)
            .expect("Proof generation should succeed");
        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(&proof, &public_input, &common, &vk));
    }

    #[test]
    fn test_large_circuit_n64() {
        let n = 64;
        let common = test_common_preprocessed_input_size(n);
        let x = FieldElement::from(4_u64);
        let e = FieldElement::from(3_u64);
        let y = &x * &e;
        let witness = test_witness_size(x.clone(), e, n);
        let public_input = vec![x, y];
        let srs = test_srs(n);
        let kzg = KZG::new(srs);
        let vk = setup(&common, &kzg);
        let rng = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), rng);
        let proof = prover
            .prove(&witness, &public_input, &common, &vk)
            .expect("Proof generation should succeed");
        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(&proof, &public_input, &common, &vk));
    }

    #[test]
    fn test_large_circuit_n256() {
        let n = 256;
        let common = test_common_preprocessed_input_size(n);
        let x = FieldElement::from(4_u64);
        let e = FieldElement::from(3_u64);
        let y = &x * &e;
        let witness = test_witness_size(x.clone(), e, n);
        let public_input = vec![x, y];
        let srs = test_srs(n);
        let kzg = KZG::new(srs);
        let vk = setup(&common, &kzg);
        let rng = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), rng);
        let proof = prover
            .prove(&witness, &public_input, &common, &vk)
            .expect("Proof generation should succeed");
        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(&proof, &public_input, &common, &vk));
    }

    #[test]
    fn test_large_circuit_n512() {
        let n = 512;
        let common = test_common_preprocessed_input_size(n);
        let x = FieldElement::from(4_u64);
        let e = FieldElement::from(3_u64);
        let y = &x * &e;
        let witness = test_witness_size(x.clone(), e, n);
        let public_input = vec![x, y];
        let srs = test_srs(n);
        let kzg = KZG::new(srs);
        let vk = setup(&common, &kzg);
        let rng = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), rng);
        let proof = prover
            .prove(&witness, &public_input, &common, &vk)
            .expect("Proof generation should succeed");
        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(&proof, &public_input, &common, &vk));
    }

    #[test]
    fn test_large_circuit_n4096() {
        let n = 4096;
        let common = test_common_preprocessed_input_size(n);
        let x = FieldElement::from(4_u64);
        let e = FieldElement::from(3_u64);
        let y = &x * &e;
        let witness = test_witness_size(x.clone(), e, n);
        let public_input = vec![x, y];
        let srs = test_srs(n);
        let kzg = KZG::new(srs);
        let vk = setup(&common, &kzg);
        let rng = TestRandomFieldGenerator {};
        let prover = Prover::new(kzg.clone(), rng);
        let proof = prover
            .prove(&witness, &public_input, &common, &vk)
            .expect("Proof generation should succeed");
        let verifier = Verifier::new(kzg);
        assert!(verifier.verify(&proof, &public_input, &common, &vk));
    }
}
