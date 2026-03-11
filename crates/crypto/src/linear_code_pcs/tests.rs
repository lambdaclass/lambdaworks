use alloc::vec::Vec;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use sha3::Keccak256;

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::merkle_tree::backends::field_element_vector::FieldElementVectorBackend;

use super::commit::{commit, tensor_vec};
use super::encoding::expander::ExpanderEncoding;
use super::encoding::reed_solomon::ReedSolomonEncoding;
use super::prove::prove;
use super::verify::verify;

// ---------- Ligero (Reed-Solomon) tests with Goldilocks ----------

mod ligero {
    use super::*;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

    type F = Goldilocks64Field;
    type FE = FieldElement<F>;
    type B = FieldElementVectorBackend<F, Keccak256, 32>;

    fn make_poly(n_vars: usize) -> (Vec<FE>, DenseMultilinearPolynomial<F>) {
        let n = 1usize << n_vars;
        let evals: Vec<FE> = (1..=n).map(|x| FE::from(x as u64)).collect();
        let poly = DenseMultilinearPolynomial::new(evals.clone());
        (evals, poly)
    }

    #[test]
    fn commit_open_verify_4_vars() {
        let n_vars = 4;
        let (evals, poly) = make_poly(n_vars);

        // n_cols = 2^(4-2) = 4, rho_inv = 2 => cw_len = 8
        let encoding = ReedSolomonEncoding::<F>::new(4, 2);
        let out = commit::<F, B, _>(&evals, &encoding);

        // Evaluation point
        let point: Vec<FE> = (1..=n_vars).map(|x| FE::from(x as u64)).collect();
        let claimed_value = poly.evaluate(point.clone()).expect("valid eval point");

        // Prove
        let mut transcript_p = DefaultTranscript::<F>::new(&[0x42]);
        let proof = prove(
            &out.commitment,
            &out.state,
            &encoding,
            &point,
            &mut transcript_p,
            128,
        );

        // Verify
        let mut transcript_v = DefaultTranscript::<F>::new(&[0x42]);
        let ok = verify(
            &out.commitment,
            &proof,
            &encoding,
            &point,
            &claimed_value,
            &mut transcript_v,
            128,
        );
        assert!(ok, "verification should pass for honest prover");
    }

    #[test]
    fn commit_open_verify_6_vars() {
        let n_vars = 6;
        let (evals, poly) = make_poly(n_vars);

        // n_cols = 2^3 = 8, rho_inv = 2 => cw_len = 16
        let encoding = ReedSolomonEncoding::<F>::new(8, 2);
        let out = commit::<F, B, _>(&evals, &encoding);

        let point: Vec<FE> = (10..10 + n_vars).map(|x| FE::from(x as u64)).collect();
        let claimed_value = poly.evaluate(point.clone()).expect("valid eval point");

        let mut transcript_p = DefaultTranscript::<F>::new(&[0x01]);
        let proof = prove(
            &out.commitment,
            &out.state,
            &encoding,
            &point,
            &mut transcript_p,
            128,
        );

        let mut transcript_v = DefaultTranscript::<F>::new(&[0x01]);
        let ok = verify(
            &out.commitment,
            &proof,
            &encoding,
            &point,
            &claimed_value,
            &mut transcript_v,
            128,
        );
        assert!(ok);
    }

    #[test]
    fn wrong_evaluation_fails() {
        let n_vars = 4;
        let (evals, _poly) = make_poly(n_vars);

        let encoding = ReedSolomonEncoding::<F>::new(4, 2);
        let out = commit::<F, B, _>(&evals, &encoding);

        let point: Vec<FE> = (1..=n_vars).map(|x| FE::from(x as u64)).collect();

        let mut transcript_p = DefaultTranscript::<F>::new(&[0x42]);
        let proof = prove(
            &out.commitment,
            &out.state,
            &encoding,
            &point,
            &mut transcript_p,
            128,
        );

        // Tamper with claimed value
        let wrong_value = FE::from(999u64);
        let mut transcript_v = DefaultTranscript::<F>::new(&[0x42]);
        let ok = verify(
            &out.commitment,
            &proof,
            &encoding,
            &point,
            &wrong_value,
            &mut transcript_v,
            128,
        );
        assert!(!ok, "verification should fail for wrong evaluation");
    }

    #[test]
    fn rate_4_code() {
        // Test with rate 1/4 code (more redundancy, fewer column openings needed)
        let n_vars = 4;
        let (evals, poly) = make_poly(n_vars);

        // n_cols = 4, rho_inv = 4 => cw_len = 16
        let encoding = ReedSolomonEncoding::<F>::new(4, 4);
        let out = commit::<F, B, _>(&evals, &encoding);

        let point: Vec<FE> = (1..=n_vars).map(|x| FE::from(x as u64)).collect();
        let claimed_value = poly.evaluate(point.clone()).expect("valid eval point");

        let mut transcript_p = DefaultTranscript::<F>::new(&[0xAB]);
        let proof = prove(
            &out.commitment,
            &out.state,
            &encoding,
            &point,
            &mut transcript_p,
            128,
        );

        let mut transcript_v = DefaultTranscript::<F>::new(&[0xAB]);
        let ok = verify(
            &out.commitment,
            &proof,
            &encoding,
            &point,
            &claimed_value,
            &mut transcript_v,
            128,
        );
        assert!(ok);
    }
}

// ---------- Brakedown (Expander) tests with Mersenne31 ----------

mod brakedown {
    use super::*;
    use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

    type F = Mersenne31Field;
    type FE = FieldElement<F>;
    type B = FieldElementVectorBackend<F, Keccak256, 32>;

    fn make_poly(n_vars: usize) -> (Vec<FE>, DenseMultilinearPolynomial<F>) {
        let n = 1usize << n_vars;
        let evals: Vec<FE> = (1..=n).map(|x| FE::from(x as u64)).collect();
        let poly = DenseMultilinearPolynomial::new(evals.clone());
        (evals, poly)
    }

    #[test]
    fn commit_open_verify_4_vars() {
        let n_vars = 4;
        let (evals, poly) = make_poly(n_vars);

        // n_cols = 4 for 4 vars (2^(4-2) = 4)
        let encoding = ExpanderEncoding::<F>::new(4, 0.25, 3, 1, 42);
        let out = commit::<F, B, _>(&evals, &encoding);

        let point: Vec<FE> = (1..=n_vars).map(|x| FE::from(x as u64)).collect();
        let claimed_value = poly.evaluate(point.clone()).expect("valid eval point");

        let mut transcript_p = DefaultTranscript::<F>::new(&[0x42]);
        let proof = prove(
            &out.commitment,
            &out.state,
            &encoding,
            &point,
            &mut transcript_p,
            128,
        );

        let mut transcript_v = DefaultTranscript::<F>::new(&[0x42]);
        let ok = verify(
            &out.commitment,
            &proof,
            &encoding,
            &point,
            &claimed_value,
            &mut transcript_v,
            128,
        );
        assert!(ok, "Brakedown verification should pass for honest prover");
    }

    #[test]
    fn commit_open_verify_6_vars() {
        let n_vars = 6;
        let (evals, poly) = make_poly(n_vars);

        // n_cols = 2^3 = 8
        let encoding = ExpanderEncoding::<F>::new(8, 0.25, 3, 1, 42);
        let out = commit::<F, B, _>(&evals, &encoding);

        let point: Vec<FE> = (5..5 + n_vars).map(|x| FE::from(x as u64)).collect();
        let claimed_value = poly.evaluate(point.clone()).expect("valid eval point");

        let mut transcript_p = DefaultTranscript::<F>::new(&[0x01]);
        let proof = prove(
            &out.commitment,
            &out.state,
            &encoding,
            &point,
            &mut transcript_p,
            128,
        );

        let mut transcript_v = DefaultTranscript::<F>::new(&[0x01]);
        let ok = verify(
            &out.commitment,
            &proof,
            &encoding,
            &point,
            &claimed_value,
            &mut transcript_v,
            128,
        );
        assert!(ok);
    }

    #[test]
    fn wrong_evaluation_fails() {
        let n_vars = 4;
        let (evals, _poly) = make_poly(n_vars);

        let encoding = ExpanderEncoding::<F>::new(4, 0.25, 3, 1, 42);
        let out = commit::<F, B, _>(&evals, &encoding);

        let point: Vec<FE> = (1..=n_vars).map(|x| FE::from(x as u64)).collect();

        let mut transcript_p = DefaultTranscript::<F>::new(&[0x42]);
        let proof = prove(
            &out.commitment,
            &out.state,
            &encoding,
            &point,
            &mut transcript_p,
            128,
        );

        let wrong_value = FE::from(12345u64);
        let mut transcript_v = DefaultTranscript::<F>::new(&[0x42]);
        let ok = verify(
            &out.commitment,
            &proof,
            &encoding,
            &point,
            &wrong_value,
            &mut transcript_v,
            128,
        );
        assert!(
            !ok,
            "Brakedown verification should fail for wrong evaluation"
        );
    }
}

// ---------- Tensor product tests ----------

mod tensor {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<101>;
    type FE = FieldElement<F>;

    #[test]
    fn tensor_vec_single() {
        // tensor(r) = (1-r, r)
        let r = FE::from(3u64);
        let t = tensor_vec(&[r]);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], FE::from(101 - 2)); // 1 - 3 mod 101 = 99
        assert_eq!(t[1], FE::from(3u64));
    }

    #[test]
    fn tensor_vec_two() {
        // tensor(r0, r1) = ((1-r0)(1-r1), (1-r0)*r1, r0*(1-r1), r0*r1)
        let r0 = FE::from(2u64);
        let r1 = FE::from(3u64);
        let t = tensor_vec(&[r0, r1]);
        assert_eq!(t.len(), 4);

        let one = FE::one();
        let one_minus_r0 = one.clone() - FE::from(2u64);
        let one_minus_r1 = one.clone() - FE::from(3u64);

        assert_eq!(t[0], one_minus_r0.clone() * one_minus_r1.clone());
        assert_eq!(t[1], one_minus_r0.clone() * FE::from(3u64));
        assert_eq!(t[2], FE::from(2u64) * one_minus_r1.clone());
        assert_eq!(t[3], FE::from(2u64) * FE::from(3u64));
    }

    #[test]
    fn tensor_product_evaluation() {
        // For a multilinear polynomial f with evaluations on the boolean hypercube,
        // f(r) = <tensor(r_L), M * tensor(r_R)> where M is the eval matrix.
        let evals = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let poly = DenseMultilinearPolynomial::new(evals.clone());

        let r = vec![FE::from(5u64), FE::from(7u64)];
        let expected = poly.evaluate(r.clone()).expect("valid");

        // Manual computation via tensor product:
        // a = tensor(r[0]) = (1-5, 5) = (97, 5) mod 101
        // b = tensor(r[1]) = (1-7, 7) = (95, 7) mod 101
        // M = [[1, 2], [3, 4]]
        // v = a^T * M = [97*1 + 5*3, 97*2 + 5*4] = [97+15, 194+20] = [112, 214] = [11, 12] mod 101
        // result = <v, b> = 11*95 + 12*7 = 1045 + 84 = 1129 = 1129 mod 101 = 1129 - 11*101 = 1129 - 1111 = 18
        let a = tensor_vec(&r[..1]);
        let b = tensor_vec(&r[1..]);

        let v0 = a[0].clone() * FE::from(1u64) + a[1].clone() * FE::from(3u64);
        let v1 = a[0].clone() * FE::from(2u64) + a[1].clone() * FE::from(4u64);
        let result = v0 * b[0].clone() + v1 * b[1].clone();
        assert_eq!(result, expected);
    }
}

// ---------- Matrix arrangement test ----------

mod matrix_tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    use crate::linear_code_pcs::matrix::Matrix;

    type F = U64PrimeField<101>;
    type FE = FieldElement<F>;

    #[test]
    fn eval_via_tensor_matrix_product() {
        // f(x0, x1, x2, x3) with 16 evaluations
        let evals: Vec<FE> = (1..=16).map(|x| FE::from(x as u64)).collect();
        let poly = DenseMultilinearPolynomial::new(evals.clone());

        // Matrix: 4 rows x 4 cols
        let n_rows = 4;
        let n_cols = 4;
        let m = Matrix::new(n_rows, n_cols, evals);

        let point: Vec<FE> = vec![
            FE::from(2u64),
            FE::from(3u64),
            FE::from(5u64),
            FE::from(7u64),
        ];
        let expected = poly.evaluate(point.clone()).expect("valid");

        let a = tensor_vec(&point[..2]);
        let b = tensor_vec(&point[2..]);

        // v = a^T * M
        let v = m.row_mul(&a);

        // result = <v, b>
        let result: FE = v
            .iter()
            .zip(b.iter())
            .fold(FE::zero(), |acc, (vi, bi)| acc + vi.clone() * bi.clone());

        assert_eq!(result, expected);
    }
}
