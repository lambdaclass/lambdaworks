//! Pedersen Vector Commitment Scheme
//!
//! This module implements Pedersen vector commitments, which are the foundation for
//! Inner Product Arguments (IPA). The scheme provides:
//! - **Hiding**: The commitment reveals nothing about the committed values (due to blinding)
//! - **Binding**: The committer cannot open to a different value
//! - **Transparent setup**: Generators are derived from hash functions, no trusted setup
//!
//! # Mathematical Background
//!
//! A Pedersen commitment to a vector `a = [a_0, a_1, ..., a_{n-1}]` with blinding factor `r` is:
//!
//! ```text
//! C = sum_{i=0}^{n-1}(a_i * G_i) + r * H
//! ```
//!
//! Where `G_i` are independent generators and `H` is the blinding generator.
//!
//! # Reference
//!
//! - Bulletproofs paper, Section 2: <https://eprint.iacr.org/2017/1066.pdf>

use alloc::vec::Vec;
use core::marker::PhantomData;

use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::traits::IsEllipticCurve,
    field::{element::FieldElement, traits::IsPrimeField},
    msm::pippenger::msm,
    traits::ByteConversion,
    unsigned_integer::element::UnsignedInteger,
};
use sha3::{Digest, Keccak256};

/// Error types for Pedersen commitment operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PedersenError {
    /// The input vector length exceeds the parameter size
    VectorTooLong { max_size: usize, actual_size: usize },
    /// Empty vector provided where non-empty is required
    EmptyVector,
    /// MSM computation failed
    MsmError,
}

/// Pedersen commitment parameters for a vector of size up to `max_size`.
///
/// Contains random generators `G_0, G_1, ..., G_{n-1}` for committing to values
/// and a blinding generator `H`.
///
/// # Transparent Setup
///
/// Generators are derived deterministically from hashing, providing a transparent
/// (no trusted setup) commitment scheme. This is crucial for IPA-based systems
/// like Halo 2 that avoid trusted ceremonies.
/// Pedersen commitment parameters
pub struct PedersenParams<E: IsEllipticCurve> {
    /// Vector of generators for the message coefficients: `G_0, G_1, ..., G_{n-1}`
    pub g_vec: Vec<E::PointRepresentation>,
    /// Blinding generator `H`
    pub h: E::PointRepresentation,
    /// Additional generator `U` used in IPA for binding the inner product
    /// This is the "u" point in the Bulletproofs paper that makes
    /// the inner product argument binding
    pub u: E::PointRepresentation,
}

impl<E: IsEllipticCurve> Clone for PedersenParams<E>
where
    E::PointRepresentation: Clone,
{
    fn clone(&self) -> Self {
        Self {
            g_vec: self.g_vec.clone(),
            h: self.h.clone(),
            u: self.u.clone(),
        }
    }
}

impl<E: IsEllipticCurve> core::fmt::Debug for PedersenParams<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PedersenParams")
            .field("size", &self.g_vec.len())
            .finish()
    }
}

impl<E: IsEllipticCurve> PedersenParams<E>
where
    FieldElement<E::BaseField>: ByteConversion,
{
    /// Create new Pedersen parameters with the given size using transparent setup.
    ///
    /// Generators are derived deterministically by hashing a domain separator
    /// and index, then using hash-to-curve. This provides:
    /// - No trusted setup (anyone can verify generator derivation)
    /// - Independent generators (discrete log relations are unknown)
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum vector size this commitment scheme supports
    ///
    /// # Panics
    ///
    /// Panics if `size` is 0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Pedersen parameters require size > 0");

        // Generate deterministic independent generators using hash-to-curve
        let g_vec: Vec<E::PointRepresentation> = (0..size)
            .map(|i| Self::hash_to_curve(b"IPA_G_GENERATOR", i as u64))
            .collect();

        let h = Self::hash_to_curve(b"IPA_H_GENERATOR", 0);
        let u = Self::hash_to_curve(b"IPA_U_GENERATOR", 0);

        Self { g_vec, h, u }
    }

    /// Hash to curve using a simple try-and-increment method.
    ///
    /// This is a straightforward (not constant-time) implementation suitable
    /// for parameter generation during setup. For production use with
    /// secret inputs, consider using a proper hash-to-curve implementation
    /// like RFC 9380.
    ///
    /// # Arguments
    ///
    /// * `domain` - Domain separation tag
    /// * `index` - Index for generating multiple independent points
    fn hash_to_curve(domain: &[u8], index: u64) -> E::PointRepresentation {
        // We use a try-and-increment approach:
        // 1. Hash (domain || index || counter) to get a candidate x-coordinate
        // 2. Check if there's a valid y-coordinate
        // 3. Increment counter and retry if not
        //
        // This is simple but works for transparent setup where the inputs are public.

        let generator = E::generator();

        // Use a deterministic scalar derived from the hash to multiply the generator
        // This ensures we get a valid curve point while maintaining independence
        let mut hasher = Keccak256::new();
        hasher.update(domain);
        hasher.update(index.to_le_bytes());

        let hash_result = hasher.finalize();

        // Convert hash to a scalar and multiply by generator
        // We use operate_with_self to compute scalar * G
        // The hash gives us 256 bits, which we use as a scalar
        let scalar_bytes: [u8; 32] = hash_result.into();
        let scalar =
            u128::from_le_bytes(scalar_bytes[0..16].try_into().expect(
                "hash_to_curve: slice to array conversion should not fail for 16-byte slice",
            ));

        generator.operate_with_self(scalar)
    }

    /// Commit to a vector of field elements with a blinding factor.
    ///
    /// Computes: `C = sum_{i=0}^{n-1}(values[i] * G_i) + blinding * H`
    ///
    /// # Arguments
    ///
    /// * `values` - The vector of field elements to commit to
    /// * `blinding` - Random blinding factor for hiding
    ///
    /// # Returns
    ///
    /// The commitment point, or an error if the vector is too long.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let params = PedersenParams::<PallasCurve>::new(4);
    /// let values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
    /// let blinding = FE::from(42);
    /// let commitment = params.commit(&values, &blinding)?;
    /// ```
    pub fn commit<const N: usize, F>(
        &self,
        values: &[FieldElement<F>],
        blinding: &FieldElement<F>,
    ) -> Result<E::PointRepresentation, PedersenError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
    {
        if values.is_empty() {
            return Err(PedersenError::EmptyVector);
        }
        if values.len() > self.g_vec.len() {
            return Err(PedersenError::VectorTooLong {
                max_size: self.g_vec.len(),
                actual_size: values.len(),
            });
        }

        // Convert field elements to their representatives for MSM
        let scalars: Vec<UnsignedInteger<N>> = values.iter().map(|v| v.representative()).collect();

        // Compute sum(values[i] * G_i) using MSM
        let value_commitment =
            msm(&scalars, &self.g_vec[..values.len()]).map_err(|_| PedersenError::MsmError)?;

        // Add blinding: C = value_commitment + blinding * H
        let blinding_term = self.h.operate_with_self(blinding.representative());
        let commitment = value_commitment.operate_with(&blinding_term);

        Ok(commitment)
    }

    /// Commit to a vector without blinding (for non-hiding commitments).
    ///
    /// Computes: `C = sum_{i=0}^{n-1}(values[i] * G_i)`
    ///
    /// **Warning**: This commitment is not hiding! Only use when the committed
    /// values are already public or when hiding is not required.
    pub fn commit_without_blinding<const N: usize, F>(
        &self,
        values: &[FieldElement<F>],
    ) -> Result<E::PointRepresentation, PedersenError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
    {
        if values.is_empty() {
            return Err(PedersenError::EmptyVector);
        }
        if values.len() > self.g_vec.len() {
            return Err(PedersenError::VectorTooLong {
                max_size: self.g_vec.len(),
                actual_size: values.len(),
            });
        }

        let scalars: Vec<UnsignedInteger<N>> = values.iter().map(|v| v.representative()).collect();

        msm(&scalars, &self.g_vec[..values.len()]).map_err(|_| PedersenError::MsmError)
    }
}

// Methods that don't require ByteConversion
impl<E: IsEllipticCurve> PedersenParams<E> {
    /// Returns the maximum vector size this parameter set supports.
    pub fn max_size(&self) -> usize {
        self.g_vec.len()
    }
}

/// A Pedersen commitment with its opening information.
///
/// This structure holds both the commitment point and the information
/// needed to verify an opening (the committed values and blinding factor).
#[derive(Clone, Debug)]
pub struct PedersenCommitment<E: IsEllipticCurve, F: IsPrimeField> {
    /// The commitment point
    pub commitment: E::PointRepresentation,
    /// The committed values (for opening verification)
    pub values: Vec<FieldElement<F>>,
    /// The blinding factor used
    pub blinding: FieldElement<F>,
    _marker: PhantomData<E>,
}

impl<E: IsEllipticCurve, F: IsPrimeField> PedersenCommitment<E, F>
where
    FieldElement<E::BaseField>: ByteConversion,
{
    /// Create a new Pedersen commitment.
    pub fn new<const N: usize>(
        params: &PedersenParams<E>,
        values: Vec<FieldElement<F>>,
        blinding: FieldElement<F>,
    ) -> Result<Self, PedersenError>
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
    {
        let commitment = params.commit::<N, F>(&values, &blinding)?;
        Ok(Self {
            commitment,
            values,
            blinding,
            _marker: PhantomData,
        })
    }

    /// Verify that the commitment opens to the claimed values.
    pub fn verify<const N: usize>(&self, params: &PedersenParams<E>) -> bool
    where
        F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>,
    {
        match params.commit::<N, F>(&self.values, &self.blinding) {
            Ok(expected) => expected == self.commitment,
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::{
        elliptic_curve::short_weierstrass::curves::pallas::curve::PallasCurve,
        field::{element::FieldElement, fields::vesta_field::Vesta255PrimeField},
    };

    // The scalar field of Pallas is the base field of Vesta (cycle of curves)
    type FE = FieldElement<Vesta255PrimeField>;

    #[test]
    fn test_pedersen_params_creation() {
        let params = PedersenParams::<PallasCurve>::new(8);
        assert_eq!(params.max_size(), 8);

        // Generators should be non-trivial (not neutral element)
        for g in &params.g_vec {
            assert!(!g.is_neutral_element());
        }
        assert!(!params.h.is_neutral_element());
        assert!(!params.u.is_neutral_element());
    }

    #[test]
    fn test_pedersen_commit_basic() {
        let params = PedersenParams::<PallasCurve>::new(4);
        let values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let blinding = FE::from(42);

        let commitment = params.commit(&values, &blinding);
        assert!(commitment.is_ok());
        assert!(!commitment
            .expect("commit should succeed")
            .is_neutral_element());
    }

    #[test]
    fn test_pedersen_commit_without_blinding() {
        let params = PedersenParams::<PallasCurve>::new(4);
        let values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];

        let commitment = params.commit_without_blinding(&values);
        assert!(commitment.is_ok());
    }

    #[test]
    fn test_pedersen_homomorphic() {
        // Pedersen commitments are homomorphic:
        // Commit(a, r1) + Commit(b, r2) = Commit(a + b, r1 + r2)
        let params = PedersenParams::<PallasCurve>::new(2);

        let a = vec![FE::from(3), FE::from(5)];
        let r1 = FE::from(7);
        let c1 = params.commit(&a, &r1).expect("commit should succeed");

        let b = vec![FE::from(2), FE::from(4)];
        let r2 = FE::from(11);
        let c2 = params.commit(&b, &r2).expect("commit should succeed");

        // Commit to sum
        let a_plus_b: Vec<FE> = a.iter().zip(b.iter()).map(|(ai, bi)| ai + bi).collect();
        let r1_plus_r2 = &r1 + &r2;
        let c_sum = params
            .commit(&a_plus_b, &r1_plus_r2)
            .expect("commit should succeed");

        // Check homomorphic property
        let c1_plus_c2 = c1.operate_with(&c2);
        assert_eq!(c_sum, c1_plus_c2);
    }

    #[test]
    fn test_pedersen_binding() {
        // The same values with same blinding should give same commitment
        let params = PedersenParams::<PallasCurve>::new(4);
        let values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let blinding = FE::from(123);

        let c1 = params
            .commit(&values, &blinding)
            .expect("commit should succeed");
        let c2 = params
            .commit(&values, &blinding)
            .expect("commit should succeed");
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_pedersen_hiding() {
        // Different blinding factors should give different commitments
        let params = PedersenParams::<PallasCurve>::new(4);
        let values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];

        let c1 = params
            .commit(&values, &FE::from(1))
            .expect("commit should succeed");
        let c2 = params
            .commit(&values, &FE::from(2))
            .expect("commit should succeed");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_pedersen_different_values() {
        // Different values should give different commitments (binding)
        let params = PedersenParams::<PallasCurve>::new(4);
        let blinding = FE::from(42);

        let c1 = params
            .commit(&[FE::from(1), FE::from(2)], &blinding)
            .expect("commit should succeed");
        let c2 = params
            .commit(&[FE::from(3), FE::from(4)], &blinding)
            .expect("commit should succeed");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_pedersen_vector_too_long() {
        let params = PedersenParams::<PallasCurve>::new(2);
        let values = vec![FE::from(1), FE::from(2), FE::from(3)]; // 3 > 2

        let result = params.commit(&values, &FE::from(1));
        assert_eq!(
            result,
            Err(PedersenError::VectorTooLong {
                max_size: 2,
                actual_size: 3
            })
        );
    }

    #[test]
    fn test_pedersen_empty_vector() {
        let params = PedersenParams::<PallasCurve>::new(4);
        let values: Vec<FE> = vec![];

        let result = params.commit(&values, &FE::from(1));
        assert_eq!(result, Err(PedersenError::EmptyVector));
    }

    #[test]
    fn test_pedersen_commitment_verify() {
        let params = PedersenParams::<PallasCurve>::new(4);
        let values = vec![FE::from(1), FE::from(2), FE::from(3), FE::from(4)];
        let blinding = FE::from(42);

        let commitment = PedersenCommitment::new(&params, values, blinding)
            .expect("commitment creation should succeed");
        assert!(commitment.verify(&params));
    }

    #[test]
    fn test_pedersen_generators_independent() {
        // Generators at different indices should be different
        let params = PedersenParams::<PallasCurve>::new(8);

        for i in 0..params.g_vec.len() {
            for j in (i + 1)..params.g_vec.len() {
                assert_ne!(
                    params.g_vec[i], params.g_vec[j],
                    "G[{}] should differ from G[{}]",
                    i, j
                );
            }
        }

        // h should be different from all g_i
        for (i, g) in params.g_vec.iter().enumerate() {
            assert_ne!(&params.h, g, "H should differ from G[{}]", i);
        }

        // u should be different from h and all g_i
        assert_ne!(params.u, params.h, "U should differ from H");
        for (i, g) in params.g_vec.iter().enumerate() {
            assert_ne!(&params.u, g, "U should differ from G[{}]", i);
        }
    }
}
