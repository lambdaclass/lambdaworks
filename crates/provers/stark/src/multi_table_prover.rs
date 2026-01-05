use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    domain::new_domain,
    proof::stark::StarkProof,
    prover::{IsStarkProver, Prover, ProvingError, Round1},
    trace::TraceTable,
    traits::AIR,
};

// List of airs and their associated table
type Airs<'a, F, E, PI> = Vec<(
    &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    &'a mut TraceTable<F, E>,
)>;
pub fn multi_prove<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: Send + Sync + IsFFTField,
    PI: Send + Sync,
>(
    airs: &mut Airs<F, E, PI>,
    transcript: &mut impl IsStarkTranscript<E, F>,
) -> Result<Vec<StarkProof<F, E>>, ProvingError>
where
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
{
    let mut proofs = Vec::new();

    let mut round_1_results: Vec<Round1<F, E>> = Vec::new();
    let mut domains = Vec::new();

    for (air, table) in &mut *airs {
        let domain = new_domain(*air);
        let round_1_result = Prover::<F, E, PI>::round_1_randomized_air_with_preprocessing(
            *air, table, &domain, transcript,
        )?;
        round_1_results.push(round_1_result);
        domains.push(domain);
    }

    for (((air, _), round_1_result), domain) in airs
        .into_iter()
        .zip(round_1_results)
        .into_iter()
        .zip(domains)
    {
        let proof = Prover::<F, E, PI>::single_table_prove(
            *air,
            &round_1_result,
            transcript,
            &domain,
        )?;
        proofs.push(proof);
    }
    Ok(proofs)
}
