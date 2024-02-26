use lambdaworks_math::{
    elliptic_curve::traits::IsEllipticCurve, field::{element::FieldElement, traits::{IsField, IsPrimeField}}, msm::pippenger::msm, polynomial::dense_multilinear_poly::{compute_factored_chis, DenseMultilinearPolynomial}, unsigned_integer::element::UnsignedInteger
};

use crate::commitments::pedersen::PedersenCommitment;

use super::traits::IsPolynomialCommitmentScheme;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct Hyrax<G: IsEllipticCurve> {
    gens: Vec<G::PointRepresentation>
}

impl<G: IsEllipticCurve> Hyrax<G> 
where 
    <G::BaseField as IsField>::BaseType: Send + Sync,
{
    fn new(num_vars: usize, gens: Vec<G::PointRepresentation>) -> Self {
        let (_left_num_vars, right_num_vars) = matrix_dimensions(num_vars);
        let r_size = 2_u32.pow(right_num_vars as u32) as usize;
        let gens = gens[..r_size].to_vec();
        Self { gens }
    }

    fn vector_matrix_product(poly: &DenseMultilinearPolynomial<G::BaseField>, l_vector: &[FieldElement<G::BaseField>]) -> Vec<FieldElement<G::BaseField>> {
        let (_left_num_vars, right_num_vars) = matrix_dimensions(poly.num_vars());
        let r_size = 2_u32.pow(right_num_vars as u32) as usize;

        #[cfg(feature = "parallel")]
        let iter = poly.evals()
            .par_chunks(r_size);
        #[cfg(not(feature = "parallel"))]
        let iter = poly.evals().chunks(r_size);
        let res = iter.enumerate()
            .map(|(i, row)| {
                row.iter()
                    .map(|x| &l_vector[i] * x)
                    .collect::<Vec<FieldElement<G::BaseField>>>()
            });
        #[cfg(not(feature = "parallel"))]
        let res = res.fold(vec![FieldElement::zero(); r_size], 
            |mut acc: Vec<_>, row| {
                acc.iter_mut().zip(row).for_each(|(x,y)| *x += y);
                acc
            });
        #[cfg(feature = "parallel")]
        let res = res.reduce(
                || vec![FieldElement::zero(); r_size],
                |mut acc: Vec<_>, row| {
                    acc.iter_mut().zip(row).for_each(|(x, y)| *x += y);
                    acc
                },
            );
        res
    }
}

impl<const NUM_LIMBS: usize, G: IsEllipticCurve> IsPolynomialCommitmentScheme<G::BaseField> for Hyrax<G> 
where
    G::BaseField: IsPrimeField + IsField<BaseType = UnsignedInteger<NUM_LIMBS>>,
    //FieldElement<<G as IsEllipticCurve>::BaseField>: IsUnsignedInteger
{
    type Polynomial = DenseMultilinearPolynomial<G::BaseField>;

    type Point = Vec<FieldElement<G::BaseField>>;

    type Commitment = Vec<G::PointRepresentation>;

    type Proof = Vec<FieldElement<G::BaseField>>;

    fn commit(&self, p: &Self::Polynomial) -> Self::Commitment {
        let n = p.len();
        let ell = p.num_vars();
        //TODO: make error
        assert_eq!(n, 2_u32.pow(ell as u32) as usize);

        let (left_num_vars, right_num_vars) = matrix_dimensions(p.num_vars());
        //TODO: address whether size and num_vars constants should be explicit primitive in LW not usize
        let l_size = 2_u32.pow(left_num_vars as u32) as usize;
        let r_size = 2_u32.pow(right_num_vars as u32) as usize;
        assert_eq!(r_size * l_size, n);

        // compute the L and R vectors
        // compute vector-matrix product between L and Z viewed as a matrix
        #[cfg(feature = "parallel")]
        let row_commitments = p.evals().par_chunks(r_size).map(|row| PedersenCommitment::commit_vector(row, self.gens).unwrap()).collect();
        #[cfg(not(feature = "parallel"))]
        let row_commitments = p.evals().chunks(r_size).map(|row| PedersenCommitment::<G>::commit(row, &self.gens).unwrap()).collect::<Vec<_>>();

        row_commitments
    }

    fn open(
        &self,
        point: &Self::Point,
        _eval: &FieldElement<G::BaseField>,
        poly: &Self::Polynomial,
        _transcript: &Option<&mut dyn crate::fiat_shamir::is_transcript::IsTranscript<G::BaseField>>,
    ) -> Self::Proof {
        assert_eq!(poly.num_vars(), point.len());

        let (left_num_vars, right_num_vars) = matrix_dimensions(poly.num_vars());
        //TODO: address whether size and num_vars constants should be explicit primitive in LW not usize
        let l_size = 2_u32.pow(left_num_vars as u32) as usize;
        let r_size = 2_u32.pow(right_num_vars as u32) as usize;

        // compute L and R vectors
        let (l, r) = compute_factored_chis(point.as_slice());
        assert_eq!(l.len(), l_size);
        assert_eq!(r.len(), r_size);

        // compute vector-matrix product between L and Z viewed as a matrix
        Self::vector_matrix_product(poly, &l)
    }

    fn open_batch(
        &self,
        points: &[Self::Point],
        evals: &[FieldElement<G::BaseField>],
        polys: &[Self::Polynomial],
        _transcript: &Option<&mut dyn crate::fiat_shamir::is_transcript::IsTranscript<G::BaseField>>,
    ) -> Vec<Self::Proof> {
        assert_eq!(polys.len(), points.len());
        assert_eq!(polys.len(), evals.len());

        let mut proofs = Vec::new();
        for (i, p) in polys.into_iter().enumerate() {
            proofs.push(self.open(&points[i], &evals[i], &p, _transcript));
        }
        proofs
    }

    fn verify(
        &self,
        point: &Self::Point,
        eval: &FieldElement<G::BaseField>,
        p_commitment: &Self::Commitment,
        proof: &Self::Proof,
        _transcript: &Option<&mut dyn crate::fiat_shamir::is_transcript::IsTranscript<G::BaseField>>,
    ) -> bool {
        // Copmute L and R
        let (l, r) = compute_factored_chis(point.as_slice());

        // Verifier-derived commitment to u * a = prod( Com(u_j)^{a_j})
        let homomorphic_comm = msm(&l, p_commitment).unwrap();

        let product_comm = msm(proof, &self.gens).unwrap();

        let dot_product = compute_dotproduct(&proof, &r);

        if (homomorphic_comm == product_comm) && (dot_product == *eval) {
            return true
        }
        false
    }

    fn verify_batch(
        &self,
        points: &[Self::Point],
        evals: &[FieldElement<G::BaseField>],
        p_commitments: &[Self::Commitment],
        proofs: &[Self::Proof],
        _transcript: &Option<&mut dyn crate::fiat_shamir::is_transcript::IsTranscript<G::BaseField>>,
    ) -> Vec<bool> {
        let mut res = Vec::new();
        for (i, p) in p_commitments.into_iter().enumerate() {
            res.push(self.verify(&points[i], &evals[i], &p, &proofs[i], _transcript));
        }
        res
    }
}


pub fn matrix_dimensions(num_vars: usize) -> (usize, usize) {
    (num_vars / 2, num_vars - num_vars / 2)
}

pub fn compute_dotproduct<F: IsField>(a: &[FieldElement<F>], b: &[FieldElement<F>]) -> FieldElement<F> {
    assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a_i, b_i)| *a_i * b_i)
        .sum()
}

#[cfg(test)]
mod tests {

    use lambdaworks_math::{cyclic_group::IsGroup, elliptic_curve::short_weierstrass::curves::bn_254::curve::{BN254Curve, BN254FieldElement}};

    use super::*;

    type FE = BN254FieldElement;

    #[test]
    fn prove_verify() {
        let z = vec![
            FE::one(),
            FE::from(2u64),
            FE::one(),
            FE::from(4u64),
        ];
        let poly = DenseMultilinearPolynomial::new(z);

        // r = [4,3]
        let point = vec![FieldElement::from(4u64), FieldElement::from(3u64)];
        let eval = poly.evaluate(point.clone()).unwrap();
        assert_eq!(eval, FieldElement::from(28u64));

        let gens = (0..(1u64 << 8u64)).map(|i| {BN254Curve::generator().operate_with_self(i)}).collect::<Vec<_>>();
        let hyrax = Hyrax::<BN254Curve>::new(poly.num_vars(), gens);
        let poly_commitment = hyrax.commit(&poly);

        let proof = hyrax.open(&point, &eval, &poly, &None);


        assert!(
            hyrax.verify(&point, &eval, &poly_commitment, &proof, &None))
    }
}