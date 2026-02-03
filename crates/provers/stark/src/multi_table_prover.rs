use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    proof::stark::StarkProof,
    prover::{IsStarkProver, Prover, ProvingError},
    trace::TraceTable,
    traits::AIR,
};

// List of airs and their associated table
type Airs<'a, F, E, PI> = Vec<(
    &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    &'a mut TraceTable<F, E>,
)>;

pub fn multi_prove<F: IsSubFieldOf<E> + IsFFTField + Send + Sync, E: Send + Sync + IsFFTField, PI>(
    airs: Airs<F, E, PI>,
    transcript: &mut impl IsStarkTranscript<E, F>,
) -> Result<StarkProof<F, E>, ProvingError>
where
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
{
    if airs.is_empty() {
        return Err(ProvingError::EmptyAirs);
    }
    let mut proof = None;
    for (air, table) in airs {
        let _ = proof.insert(Prover::<F, E, PI>::prove(air, table, transcript)?);
    }
    Ok(proof.expect("proof is Some because airs is non-empty after early return check"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcript::StoneProverTranscript;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

    #[test]
    fn test_multi_prove_empty_airs_returns_error() {
        let airs: Airs<Stark252PrimeField, Stark252PrimeField, ()> = vec![];
        let mut transcript = StoneProverTranscript::new(&[]);

        let result = multi_prove(airs, &mut transcript);

        assert!(matches!(result, Err(ProvingError::EmptyAirs)));
    }
}
