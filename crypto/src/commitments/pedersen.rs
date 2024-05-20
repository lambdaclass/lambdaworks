use core::{fmt::Display, marker::PhantomData};

use lambdaworks_math::{
    elliptic_curve::traits::IsEllipticCurve, field::{element::FieldElement, traits::{IsField, IsPrimeField}}, msm::pippenger::msm, unsigned_integer::element::UnsignedInteger
};

pub struct PedersenCommitment<G: IsEllipticCurve> {
    _phantom: PhantomData<G>,
}

impl<const NUM_LIMBS: usize, G: IsEllipticCurve> PedersenCommitment<G> 
where
    G::BaseField: IsPrimeField + IsField<BaseType = UnsignedInteger<NUM_LIMBS>>,
{

    pub fn commit(vals: &[FieldElement<G::BaseField>], gens: &[G::PointRepresentation]) -> Result<G::PointRepresentation, PedersenCommitmentError>
    where
    {
        if gens.len() != vals.len() {
            return Err(PedersenCommitmentError::GeneratorLengthError(gens.len()))
        }
        Ok(msm(&vals, &gens).unwrap())
    }
}

#[derive(Debug)]
pub enum PedersenCommitmentError {
    GeneratorLengthError(usize)
}

impl Display for PedersenCommitmentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            &PedersenCommitmentError::GeneratorLengthError(len) => write!(f, "Number of Pedersen Generators does not match the number of values to commit: {:?}", len)
        }
    }
}
