use super::traits::IsCommitmentScheme;
use lambdaworks_math::{
    field::{element::FieldElement, traits::{IsPrimeField, IsField}},
    polynomial::Polynomial, unsigned_integer::element::UnsignedInteger,
};
use std::marker::PhantomData;

#[derive(Clone, Debug, Default)]
pub struct Params<C: IsField + IsPrimeField> {
    pub k: u32,
    pub n: u64,
    pub g: Vec<C>,
    pub g_lagrange: Vec<C>,
    pub w: C,
    pub u: C,
}

#[derive(Clone)]
pub struct IPA<F: IsField + IsPrimeField> {
    params: Params<F>, 
    r: FieldElement<F>,
    w: FieldElement<F>,
    phantom: PhantomData<F>,
}

impl<F: IsField + IsPrimeField> IPA<F> {
    pub fn new() -> Self {
        Self { 
            params: todo!(),
            r: FieldElement::<F>::default(),
            w: FieldElement::<F>::default(),
            phantom: PhantomData,

        }
    }
}

impl<const N: usize, F: IsPrimeField<RepresentativeType = UnsignedInteger<N>>> IsCommitmentScheme<F> for IPA<F> {
    type Commitment = FieldElement<F>;
    
    // reference https://github.com/zcash/halo2/blob/main/halo2_proofs/src/poly/commitment.rs#L119 
    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment {
        let mut tmp_scalars = Vec::with_capacity(p.coeff_len() + 1);
        let mut tmp_bases = Vec::with_capacity(p.coeff_len() + 1);

        tmp_scalars.extend(p.coefficients().iter());
        //TODO: move reference to params type
        tmp_scalars.push(&self.r);

        tmp_scalars.extend(self.params.g.iter());
        //TODO: move reference to params type
        tmp_scalars.push(&self.w);
    
        //TODO add bucketed multiexponentiation
        FieldElement::zero()
    }

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment {
        todo!()
    }

    fn open_batch(
        &self,
        x: &FieldElement<F>,
        y: &[FieldElement<F>],
        p: &[Polynomial<FieldElement<F>>],
        upsilon: &FieldElement<F>,
    ) -> Self::Commitment {
        todo!()
    }

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p_commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool {
        todo!()
    }

    fn verify_batch(
        &self,
        x: &FieldElement<F>,
        ys: &[FieldElement<F>],
        p_commitments: &[Self::Commitment],
        proof: &Self::Commitment,
        upsilon: &FieldElement<F>,
    ) -> bool {
        todo!()
    }
}
