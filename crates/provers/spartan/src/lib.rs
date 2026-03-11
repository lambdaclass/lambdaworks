//! Spartan: Efficient and general-purpose zkSNARKs without trusted setup.
//!
//! This crate implements the Spartan proof system (Setty 2019) for proving
//! R1CS satisfiability using two nested sumcheck protocols over multilinear
//! extensions (MLEs).
//!
//! # Protocol overview
//!
//! Given R1CS matrices A, B, C ∈ F^{m×n} and witness z = (1, x, w) ∈ F^n:
//!
//! 1. **Commit**: Commit to witness polynomial z̃
//! 2. **Outer sumcheck**: Prove ∑_{x ∈ {0,1}^s} eq(τ,x)·[ÃZ(x)·B̃Z(x) − C̃Z(x)] = 0
//! 3. **Inner sumcheck**: Reduce matrix-vector product claims to z̃ evaluation
//! 4. **PCS opening**: Open z̃ at inner sumcheck challenge point r_y
//!
//! # References
//!
//! - Setty, "Spartan: Efficient and general-purpose zkSNARKs without trusted setup"
//!   <https://eprint.iacr.org/2019/550>

pub mod errors;
pub mod mle;
pub mod pcs;
pub mod prover;
pub mod r1cs;
mod transcript;
pub mod verifier;

pub use errors::SpartanError;
pub use pcs::trivial::{TrivialCommitment, TrivialPCS, TrivialProof};
pub use prover::{SpartanProof, SpartanProver};
pub use r1cs::R1CS;
pub use verifier::SpartanVerifier;

use lambdaworks_math::field::{
    element::FieldElement, traits::HasDefaultTranscript, traits::IsField,
};
use lambdaworks_math::traits::ByteConversion;
use pcs::IsMultilinearPCS;

/// Convenience function: prove R1CS satisfiability using Spartan.
pub fn spartan_prove<F, PCS>(
    r1cs: &R1CS<F>,
    public_inputs: &[FieldElement<F>],
    witness: &[FieldElement<F>],
    pcs: PCS,
) -> Result<SpartanProof<F, PCS>, SpartanError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    PCS: IsMultilinearPCS<F>,
    PCS::Error: 'static,
{
    let prover = SpartanProver::new(pcs);
    prover.prove(r1cs, public_inputs, witness)
}

/// Convenience function: verify a Spartan proof.
pub fn spartan_verify<F, PCS>(
    r1cs: &R1CS<F>,
    public_inputs: &[FieldElement<F>],
    proof: &SpartanProof<F, PCS>,
    pcs: PCS,
) -> Result<bool, SpartanError>
where
    F: IsField + HasDefaultTranscript,
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
    PCS: IsMultilinearPCS<F>,
    PCS::Error: 'static,
{
    let verifier = SpartanVerifier::new(pcs);
    verifier.verify(r1cs, public_inputs, proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    fn fe(n: u64) -> FE {
        FE::from(n)
    }

    fn zero() -> FE {
        FE::zero()
    }

    fn one() -> FE {
        FE::one()
    }

    /// Build the R1CS for: a * b = c
    /// Variables: [1, c, a, b] (constant, output, left input, right input)
    /// Constraint:
    ///   A[0] = [0, 0, 1, 0]  (selects a)
    ///   B[0] = [0, 0, 0, 1]  (selects b)
    ///   C[0] = [0, 1, 0, 0]  (selects c)
    fn mul_r1cs(a_val: u64, b_val: u64) -> (R1CS<F>, Vec<FE>) {
        let c_val = (a_val * b_val) % MODULUS;

        let a_mat = vec![vec![zero(), zero(), one(), zero()]];
        let b_mat = vec![vec![zero(), zero(), zero(), one()]];
        let c_mat = vec![vec![zero(), one(), zero(), zero()]];

        let r1cs = R1CS::new(a_mat, b_mat, c_mat, 1).unwrap();

        // witness = [1, c, a, b]
        let witness = vec![one(), fe(c_val), fe(a_val), fe(b_val)];
        (r1cs, witness)
    }

    // -------------------------------------------------------------------------
    // Test 1: R1CS satisfaction check
    // -------------------------------------------------------------------------
    #[test]
    fn test_r1cs_satisfaction() {
        let (r1cs, witness) = mul_r1cs(3, 3);
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_r1cs_not_satisfied() {
        let (r1cs, mut witness) = mul_r1cs(3, 3);
        witness[1] = fe(7); // wrong output
        assert!(!r1cs.is_satisfied(&witness));
    }

    // -------------------------------------------------------------------------
    // Test 2: MLE encoding correctness
    // -------------------------------------------------------------------------
    #[test]
    fn test_mle_encoding() {
        use crate::mle::encode_witness;

        let z = vec![fe(1), fe(7), fe(3), fe(5)];
        let z_mle = encode_witness(&z);

        assert_eq!(z_mle.evaluate(vec![FE::zero(), FE::zero()]).unwrap(), fe(1));
        assert_eq!(z_mle.evaluate(vec![FE::zero(), FE::one()]).unwrap(), fe(7));
        assert_eq!(z_mle.evaluate(vec![FE::one(), FE::zero()]).unwrap(), fe(3));
        assert_eq!(z_mle.evaluate(vec![FE::one(), FE::one()]).unwrap(), fe(5));
    }

    // -------------------------------------------------------------------------
    // Test 3: eq_poly correctness
    // -------------------------------------------------------------------------
    #[test]
    fn test_eq_poly() {
        use crate::mle::eq_poly;

        let a = fe(3);
        let b = fe(5);
        let tau = vec![a.clone(), b.clone()];
        let eq = eq_poly(&tau);

        let evals = eq.evals();
        let one = FE::one();

        // (0,0) -> (1-a)(1-b)
        assert_eq!(evals[0], (&one - &a) * (&one - &b));
        // (0,1) -> (1-a)*b
        assert_eq!(evals[1], (&one - &a) * &b);
        // (1,0) -> a*(1-b)
        assert_eq!(evals[2], &a * (&one - &b));
        // (1,1) -> a*b
        assert_eq!(evals[3], &a * &b);
    }

    // -------------------------------------------------------------------------
    // Test 4: Matrix-vector product MLE
    // -------------------------------------------------------------------------
    #[test]
    fn test_matrix_vector_product_mle() {
        use crate::mle::matrix_vector_product_mle;

        let two = fe(2);
        let a = vec![vec![one(), zero()], vec![zero(), two.clone()]];

        // r_x = [0] -> selects row 0 weighted by eq([0], i)
        let r_x = vec![FE::zero()];
        let mz = matrix_vector_product_mle(&a, 2, 2, &r_x).unwrap();

        // MZ([0])[0] = A[0][0]*1 + A[1][0]*0 = 1
        assert_eq!(mz.evaluate(vec![FE::zero()]).unwrap(), one());
        // MZ([0])[1] = A[0][1]*1 + A[1][1]*0 = 0
        assert_eq!(mz.evaluate(vec![FE::one()]).unwrap(), zero());

        // r_x = [1] -> selects row 1
        let r_x1 = vec![FE::one()];
        let mz1 = matrix_vector_product_mle(&a, 2, 2, &r_x1).unwrap();

        // MZ([1])[0] = A[0][0]*0 + A[1][0]*1 = 0
        assert_eq!(mz1.evaluate(vec![FE::zero()]).unwrap(), zero());
        // MZ([1])[1] = A[0][1]*0 + A[1][1]*1 = 2
        assert_eq!(mz1.evaluate(vec![FE::one()]).unwrap(), two);
    }

    // -------------------------------------------------------------------------
    // Test 5: TrivialPCS round trip
    // -------------------------------------------------------------------------
    #[test]
    fn test_trivial_pcs_round_trip() {
        use crate::pcs::trivial::TrivialPCS;
        use crate::pcs::IsMultilinearPCS;
        use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

        let pcs = TrivialPCS;
        let poly = DenseMultilinearPolynomial::new(vec![fe(1), fe(2), fe(3), fe(4)]);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![fe(5), fe(7)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let ok = pcs.verify(&commitment, &point, &value, &proof).unwrap();
        assert!(ok);
    }

    // -------------------------------------------------------------------------
    // Test 6: Full Spartan prove/verify
    // -------------------------------------------------------------------------
    #[test]
    fn test_spartan_prove_verify() {
        // Circuit: a * b = c, with a=2, b=3, c=6
        let (r1cs, witness) = mul_r1cs(2, 3);
        assert!(r1cs.is_satisfied(&witness));

        let public_inputs = vec![fe(6)]; // c is public

        let proof = spartan_prove(&r1cs, &public_inputs, &witness, TrivialPCS).unwrap();
        let ok = spartan_verify(&r1cs, &public_inputs, &proof, TrivialPCS).unwrap();
        assert!(ok, "Valid proof should verify");
    }

    // -------------------------------------------------------------------------
    // Test 7: Soundness — corrupted witness
    // -------------------------------------------------------------------------
    #[test]
    fn test_spartan_soundness_corrupted_witness() {
        // Same circuit but with wrong witness (7 ≠ 2*3)
        let (r1cs, _) = mul_r1cs(2, 3);

        // Wrong witness: c=7 instead of c=6
        let wrong_witness = vec![one(), fe(7), fe(2), fe(3)];
        assert!(!r1cs.is_satisfied(&wrong_witness));

        // The prover will produce a proof with an incorrect claimed sum for the outer sumcheck
        // (claimed_sum != 0 because R1CS is not satisfied).
        // The verifier should reject it.
        let result = spartan_prove(&r1cs, &[fe(6)], &wrong_witness, TrivialPCS);

        match result {
            Ok(proof) => {
                // If prove succeeded, verify should fail
                let ok = spartan_verify(&r1cs, &[fe(7)], &proof, TrivialPCS).unwrap_or(false);
                assert!(!ok, "Corrupted witness proof should not verify");
            }
            Err(_) => {
                // Prover failed, which is also acceptable (won't produce valid proof)
            }
        }
    }

    // -------------------------------------------------------------------------
    // Test 8: Multiple constraints
    // -------------------------------------------------------------------------
    #[test]
    fn test_spartan_multiple_constraints() {
        // Circuit: a*b = c AND c*d = e
        // Variables: [1, c, e, a, b, d]
        //   index:    0  1  2  3  4  5
        //
        // Constraint 0: a * b = c  -> A[0]*z*B[0]*z = C[0]*z
        //   A[0] = [0, 0, 0, 1, 0, 0]  (a)
        //   B[0] = [0, 0, 0, 0, 1, 0]  (b)
        //   C[0] = [0, 1, 0, 0, 0, 0]  (c)
        //
        // Constraint 1: c * d = e  -> A[1]*z*B[1]*z = C[1]*z
        //   A[1] = [0, 1, 0, 0, 0, 0]  (c)
        //   B[1] = [0, 0, 0, 0, 0, 1]  (d)
        //   C[1] = [0, 0, 1, 0, 0, 0]  (e)
        //
        // With a=2, b=3, c=6, d=4, e=24

        let a_val = 2u64;
        let b_val = 3u64;
        let c_val = (a_val * b_val) % MODULUS; // 6
        let d_val = 4u64;
        let e_val = (c_val * d_val) % MODULUS; // 24

        let zero = zero();
        let one = one();

        // 6 variables: [1, c, e, a, b, d]
        let a_mat = vec![
            vec![
                zero.clone(),
                zero.clone(),
                zero.clone(),
                one.clone(),
                zero.clone(),
                zero.clone(),
            ],
            vec![
                zero.clone(),
                one.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
            ],
        ];
        let b_mat = vec![
            vec![
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                one.clone(),
                zero.clone(),
            ],
            vec![
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                one.clone(),
            ],
        ];
        let c_mat = vec![
            vec![
                zero.clone(),
                one.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
            ],
            vec![
                zero.clone(),
                zero.clone(),
                one.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
            ],
        ];

        let r1cs = R1CS::new(a_mat, b_mat, c_mat, 2).unwrap();

        // witness = [1, c, e, a, b, d]
        let witness = vec![
            one.clone(),
            fe(c_val),
            fe(e_val),
            fe(a_val),
            fe(b_val),
            fe(d_val),
        ];

        assert!(
            r1cs.is_satisfied(&witness),
            "Multi-constraint R1CS should be satisfied"
        );

        let public_inputs = vec![fe(c_val), fe(e_val)];

        let proof = spartan_prove(&r1cs, &public_inputs, &witness, TrivialPCS).unwrap();
        let ok = spartan_verify(&r1cs, &public_inputs, &proof, TrivialPCS).unwrap();
        assert!(ok, "Multi-constraint proof should verify");
    }

    // -------------------------------------------------------------------------
    // Test 9: Wrong public inputs — transcript diverges, verification fails
    // -------------------------------------------------------------------------
    #[test]
    fn test_spartan_wrong_public_inputs() {
        let (r1cs, witness) = mul_r1cs(2, 3);
        let correct_public_inputs = vec![fe(6)]; // c = 2*3 = 6
        let wrong_public_inputs = vec![fe(7)]; // wrong

        // Proof generated with correct public inputs
        let proof = spartan_prove(&r1cs, &correct_public_inputs, &witness, TrivialPCS).unwrap();

        // Verifying with wrong public inputs should fail: the transcripts diverge
        // because public inputs are absorbed before drawing tau.
        let ok = spartan_verify(&r1cs, &wrong_public_inputs, &proof, TrivialPCS).unwrap_or(false);
        assert!(!ok, "Proof with wrong public inputs should not verify");
    }

    // -------------------------------------------------------------------------
    // Additional: test that MLE sum = 0 for satisfied R1CS
    // -------------------------------------------------------------------------
    #[test]
    fn test_outer_sum_is_zero_for_satisfied_r1cs() {
        use crate::mle::eq_poly;

        let (r1cs, witness) = mul_r1cs(2, 3);
        assert!(r1cs.is_satisfied(&witness));

        let num_constraints_padded = crate::mle::next_power_of_two(r1cs.num_constraints);
        let log_constraints = {
            let mut k = 0;
            let mut n = num_constraints_padded;
            while n > 1 {
                k += 1;
                n >>= 1;
            }
            k
        };

        // Use a fixed tau for testing
        let tau: Vec<FE> = (0..log_constraints)
            .map(|i| fe((i as u64 + 2) * 7 % MODULUS))
            .collect();

        // Compute combined sum: ∑_i eq(τ,i) * (AZ[i]*BZ[i] - CZ[i])
        let eq_ev = eq_poly(&tau);

        let mut sum = FE::zero();
        for i in 0..r1cs.num_constraints {
            let az_i: FE = r1cs.a[i]
                .iter()
                .zip(witness.iter())
                .map(|(a, z)| a * z)
                .fold(FE::zero(), |acc, x| acc + x);
            let bz_i: FE = r1cs.b[i]
                .iter()
                .zip(witness.iter())
                .map(|(b, z)| b * z)
                .fold(FE::zero(), |acc, x| acc + x);
            let cz_i: FE = r1cs.c[i]
                .iter()
                .zip(witness.iter())
                .map(|(c, z)| c * z)
                .fold(FE::zero(), |acc, x| acc + x);

            sum = sum + &eq_ev.evals()[i] * &(az_i * bz_i - cz_i);
        }

        assert_eq!(sum, FE::zero(), "Outer sum should be 0 for satisfied R1CS");
    }

    // -------------------------------------------------------------------------
    // Integration test: full Spartan prove+verify with ZeromorphPCS over BLS12-381
    // -------------------------------------------------------------------------
    #[test]
    fn test_spartan_prove_verify_zeromorph() {
        use crate::pcs::zeromorph::ZeromorphPCS;
        use lambdaworks_crypto::commitments::kzg::{
            KateZaveruchaGoldberg, StructuredReferenceString,
        };
        use lambdaworks_math::cyclic_group::IsGroup;
        use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve,
            default_types::{FrElement, FrField},
            pairing::BLS12381AtePairing,
            twist::BLS12381TwistCurve,
        };
        use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassJacobianPoint;
        use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
        use lambdaworks_math::field::element::FieldElement;

        type G1 = ShortWeierstrassJacobianPoint<BLS12381Curve>;
        type KZG = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;
        type ZM = ZeromorphPCS<4, FrField, BLS12381AtePairing>;
        type FE = FrElement;

        // Build a simple multiplication circuit: x * y = z over BLS12-381 Fr.
        // witness = [1, 6, 2, 3]  (constant=1, output=6, x=2, y=3)
        // A[0] = [0, 0, 1, 0]  (picks x)
        // B[0] = [0, 0, 0, 1]  (picks y)
        // C[0] = [0, 1, 0, 0]  (picks output)
        let zero = FE::zero();
        let one = FE::one();
        let a = vec![vec![zero.clone(), zero.clone(), one.clone(), zero.clone()]];
        let b = vec![vec![zero.clone(), zero.clone(), zero.clone(), one.clone()]];
        let c = vec![vec![zero.clone(), one.clone(), zero.clone(), zero.clone()]];
        let r1cs = R1CS::new(a, b, c, 1).unwrap();

        let witness: Vec<FE> = vec![
            FieldElement::one(),
            FE::from(6u64),
            FE::from(2u64),
            FE::from(3u64),
        ];
        let public_inputs = vec![FE::from(6u64)];
        assert!(r1cs.is_satisfied(&witness));

        // SRS needs ≥ 4 powers (witness has 4 elements, padded to 2^2 = 4)
        let toxic = FE::from(13u64);
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let powers: Vec<G1> = (0..8)
            .map(|i| g1.operate_with_self(toxic.pow(i as u128).canonical()))
            .collect();
        let g2_powers = [g2.clone(), g2.operate_with_self(toxic.canonical())];
        let srs = StructuredReferenceString::new(&powers, &g2_powers);
        let pcs = ZM::new(KZG::new(srs));

        let proof = spartan_prove(&r1cs, &public_inputs, &witness, pcs.clone()).unwrap();
        let ok = spartan_verify(&r1cs, &public_inputs, &proof, pcs).unwrap();
        assert!(ok, "Spartan+Zeromorph should verify for satisfied R1CS");
    }
}
