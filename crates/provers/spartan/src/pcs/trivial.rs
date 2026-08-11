//! Trivial PCS for testing Spartan correctness.
//!
//! TrivialPCS is NOT hiding or succinct — it simply stores all evaluations
//! as the commitment and re-evaluates for verification.
//! Used for unit testing the Spartan sumcheck protocol independently of PCS security.

use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use lambdaworks_math::traits::ByteConversion;

use super::{IsMultilinearPCS, PcsError};

/// Trivial PCS: commitment = all evaluations of the polynomial.
///
/// Proof = just the claimed evaluation value.
/// Verification re-evaluates from the stored evals.
#[derive(Clone, Debug, Default)]
pub struct TrivialPCS;

/// Commitment for TrivialPCS: stores the full evaluation vector.
#[derive(Clone, Debug, PartialEq)]
pub struct TrivialCommitment<F: IsField>
where
    F::BaseType: Send + Sync,
{
    pub evals: Vec<FieldElement<F>>,
    pub num_vars: usize,
}

/// Opening proof for TrivialPCS: just the claimed value.
#[derive(Clone, Debug, PartialEq)]
pub struct TrivialProof<F: IsField>
where
    F::BaseType: Send + Sync,
{
    pub value: FieldElement<F>,
}

impl<F: IsField> IsMultilinearPCS<F> for TrivialPCS
where
    F::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    type Commitment = TrivialCommitment<F>;
    type Proof = TrivialProof<F>;
    type Error = PcsError;

    fn commit(
        &self,
        poly: &DenseMultilinearPolynomial<F>,
    ) -> Result<Self::Commitment, Self::Error> {
        Ok(TrivialCommitment {
            evals: poly.evals().to_vec(),
            num_vars: poly.num_vars(),
        })
    }

    fn open(
        &self,
        poly: &DenseMultilinearPolynomial<F>,
        point: &[FieldElement<F>],
    ) -> Result<(FieldElement<F>, Self::Proof), Self::Error> {
        let value = poly
            .evaluate(point.to_vec())
            .map_err(|e| PcsError(format!("{e:?}")))?;
        Ok((value.clone(), TrivialProof { value }))
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        point: &[FieldElement<F>],
        value: &FieldElement<F>,
        proof: &Self::Proof,
    ) -> Result<bool, Self::Error> {
        // Recompute from commitment's evals
        let poly = DenseMultilinearPolynomial::new(commitment.evals.clone());
        let computed = poly
            .evaluate(point.to_vec())
            .map_err(|e| PcsError(format!("{e:?}")))?;
        Ok(&computed == value && &proof.value == value)
    }

    /// Serialize the commitment by concatenating all evaluation bytes (big-endian).
    fn serialize_commitment(commitment: &Self::Commitment) -> Vec<u8> {
        commitment
            .evals
            .iter()
            .flat_map(|e| e.to_bytes_be())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn test_trivial_pcs_round_trip() {
        let pcs = TrivialPCS;
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(5), FE::from(7)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let ok = pcs.verify(&commitment, &point, &value, &proof).unwrap();
        assert!(ok);
    }

    #[test]
    fn test_trivial_pcs_wrong_value() {
        let pcs = TrivialPCS;
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]);
        let commitment = pcs.commit(&poly).unwrap();
        let point = vec![FE::from(5), FE::from(7)];
        let (value, proof) = pcs.open(&poly, &point).unwrap();
        let wrong_value = value + FE::one();
        let ok = pcs
            .verify(&commitment, &point, &wrong_value, &proof)
            .unwrap();
        assert!(!ok);
    }
}
